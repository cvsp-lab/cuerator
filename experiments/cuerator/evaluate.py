"""Stage 3 oracle search: per-iteration Sobol grid on train split + ablations
+ raw per-sample arrays + per-modality votes.

Per-modality voting arrays are computed under each sample's per-sample best
oracle params:
    oracle_audio_votes  (N, T) int32: per frame, the highest-sim audio category
                                       among those passing the audio threshold,
                                       else -1.
    oracle_visual_votes (N, T) int32: same for visual.
These let the Oracle agent diagnose "modality bypass" (e.g., visual
threshold collapsed to always-pass, letting audio carry the joint decision).

Val and test are computed only in the in-loop policy training step
(run_policy.py). This script does not touch them.

Usage:
    python -m experiments.cuerator.evaluate \\
        --formulation <path> --config <path> --output <path> \\
        [--protocol <path>] [--raw <path>]
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import ovavel.data as dataset_utils  # noqa: E402
from step2_formulation.oracle import (  # noqa: E402
    generate_ablation_grid,
    run_ablation,
    run_oracle,
)


def load_formulation_metadata(formulation_path: Path):
    spec = importlib.util.spec_from_file_location("oracle_baseline_formulation", formulation_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load formulation from {formulation_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {
        "name": getattr(mod, "NAME", ""),
        "description": getattr(mod, "DESCRIPTION", ""),
        "num_params": int(getattr(mod, "NUM_PARAMS")),
        "param_ranges": [list(t) for t in getattr(mod, "PARAM_RANGES")],
        "param_names": list(getattr(mod, "PARAM_NAMES", [])),
    }


def write_stub_result(output_path: Path, formulation_file: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"formulation_file": formulation_file, "stage": "oracle_search"}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_train_inputs(encoder: str):
    bg_id = dataset_utils.get_bg_id("train")
    sim = dataset_utils.load_similarities(encoder, "train", "data/similarities")
    try:
        emb_raw = dataset_utils.load_embeddings(encoder, "train", "data/embeddings")
        a_text = dataset_utils.load_text_embeddings(encoder, "train", "audio", "data/embeddings")
        v_text = dataset_utils.load_text_embeddings(encoder, "train", "visual", "data/embeddings")
        emb = {**emb_raw, "audio_text": a_text, "visual_text": v_text}
    except FileNotFoundError:
        emb = None
    return sim, emb, bg_id


def _modality_vote(sim: torch.Tensor, thresh: torch.Tensor) -> torch.Tensor:
    """Per-frame vote: among categories whose sim exceeds their threshold, the
    highest-sim one; -1 if none exceeds.

    sim:    (T, C)
    thresh: (T, C)
    returns: (T,) int64 with -1 for "no vote".
    """
    passed = sim > thresh  # (T, C)
    masked = torch.where(passed, sim, torch.full_like(sim, -float("inf")))
    top = masked.argmax(dim=-1)  # (T,)
    return torch.where(passed.any(dim=-1), top, torch.full_like(top, -1))


def compute_modality_votes(formulation_path, sim_data, emb_data, per_sample_params, device_str):
    """For each sample, with its per-sample best params, return (audio_votes, visual_votes) (N, T) int32.

    per_sample_params: (N, P) numpy array — output of run_oracle()['best_params'].
    """
    spec = importlib.util.spec_from_file_location("modality_votes_formulation", formulation_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    a_t_sim = sim_data["a_t_sim"]
    v_t_sim = sim_data["v_t_sim"]
    N, T, _ = a_t_sim.shape
    device = torch.device(device_str)

    a_t_emb = emb_data.get("audio_text") if emb_data else None
    v_t_emb = emb_data.get("visual_text") if emb_data else None
    a_emb = emb_data.get("audio") if emb_data else None
    v_emb = emb_data.get("visual") if emb_data else None
    a_t_emb_gpu = torch.from_numpy(a_t_emb).float().to(device) if a_t_emb is not None else None
    v_t_emb_gpu = torch.from_numpy(v_t_emb).float().to(device) if v_t_emb is not None else None

    audio_votes = np.full((N, T), -1, dtype=np.int32)
    visual_votes = np.full((N, T), -1, dtype=np.int32)

    for i in range(N):
        a_sim = torch.from_numpy(a_t_sim[i]).float().to(device)
        v_sim = torch.from_numpy(v_t_sim[i]).float().to(device)
        a_emb_i = torch.from_numpy(a_emb[i]).float().to(device) if a_emb is not None else None
        v_emb_i = torch.from_numpy(v_emb[i]).float().to(device) if v_emb is not None else None
        params_i = torch.from_numpy(per_sample_params[i].astype(np.float32)).to(device).unsqueeze(0)
        with torch.no_grad():
            a_ths, v_ths = mod.params_to_thresholds_batch(
                a_emb_i, v_emb_i, a_t_emb_gpu, v_t_emb_gpu, a_sim, v_sim, params_i
            )
            a_vote = _modality_vote(a_sim, a_ths[0])
            v_vote = _modality_vote(v_sim, v_ths[0])
        audio_votes[i] = a_vote.cpu().numpy().astype(np.int32)
        visual_votes[i] = v_vote.cpu().numpy().astype(np.int32)
    return audio_votes, visual_votes


def summarize_marginal(marginal: np.ndarray) -> dict:
    """Compute summary stats of per-sample marginal value distribution."""
    if marginal.size == 0:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "fraction_zero": 0.0}
    return {
        "mean": float(marginal.mean()),
        "median": float(np.median(marginal)),
        "p10": float(np.percentile(marginal, 10)),
        "p90": float(np.percentile(marginal, 90)),
        "fraction_zero": float((marginal <= 1e-6).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formulation", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--protocol", default="", type=str,
                        help="Optional path to eval_protocol JSON. If absent or empty ablations, skip ablations.")
    parser.add_argument("--raw", default="", type=str,
                        help="Optional path to write raw per-sample .npz (full+ablation per-sample scores, category_ids).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    oracle_cfg = cfg["oracle"]

    formulation_path = Path(args.formulation).resolve()
    output_path = Path(args.output).resolve()

    metadata = load_formulation_metadata(formulation_path)
    write_stub_result(output_path, formulation_path.name)

    print(f"  [evaluate-oracle] loading train inputs (encoder={oracle_cfg['encoder']})...", flush=True)
    train_sim, train_emb, train_bg_id = load_train_inputs(oracle_cfg["encoder"])

    print("  [evaluate-oracle] running oracle search on train split...", flush=True)
    train_oracle = run_oracle(
        str(formulation_path),
        train_sim,
        train_emb,
        train_bg_id,
        num_candidates=oracle_cfg["num_candidates"],
        chunk_size=oracle_cfg["chunk_size"],
        num_gpus=oracle_cfg["num_gpus"],
        oracle_metric="frame_acc",
    )
    full_per_sample_scores = train_oracle["per_sample_best_scores"]

    ablation_results = []
    raw_ablation_names = []
    raw_ablation_scores = []  # list of (N,) arrays, one per ablation
    raw_ablation_marginals = []
    if args.protocol and Path(args.protocol).is_file():
        with open(args.protocol, encoding="utf-8") as f:
            protocol = json.load(f)
        ablations = protocol.get("ablations", [])
        candidates_per_ablation = int(protocol.get("candidates_per_ablation", 8192))
        if ablations:
            print(f"  [evaluate-ablations] running {len(ablations)} ablations "
                  f"(candidates_per_ablation={candidates_per_ablation})...", flush=True)
        for a in ablations:
            name = a["name"]
            params_zero = list(a["params_zero"])
            intent = a["intent"]
            grid = generate_ablation_grid(
                metadata["param_ranges"], params_zero, candidates_per_ablation
            )
            print(f"  [evaluate-ablations]   {name}: params_zero={params_zero} grid={grid.shape}", flush=True)
            per_sample_ab, ab_elapsed = run_ablation(
                str(formulation_path),
                train_sim,
                train_emb,
                train_bg_id,
                grid,
                chunk_size=oracle_cfg["chunk_size"],
                num_gpus=oracle_cfg["num_gpus"],
            )
            marginal = np.maximum(full_per_sample_scores - per_sample_ab, 0.0)
            ablation_results.append({
                "name": name,
                "params_zero": params_zero,
                "intent": intent,
                "candidates": int(grid.shape[0]),
                "ablation_frame_acc": float(per_sample_ab.mean()),
                "marginal_value_distribution": summarize_marginal(marginal),
                "elapsed_sec": float(ab_elapsed),
            })
            raw_ablation_names.append(name)
            raw_ablation_scores.append(per_sample_ab.astype(np.float32))
            raw_ablation_marginals.append(marginal.astype(np.float32))
            print(f"  [evaluate-ablations]   {name}: ablation_acc={per_sample_ab.mean():.4f} "
                  f"marginal_mean={marginal.mean():.4f} ({ab_elapsed:.1f}s)", flush=True)

    final_payload = {
        "formulation_file": formulation_path.name,
        "stage": "oracle_search",
        "oracle_metric": "frame_acc",  # required by step3 train.py if Stage 2 reuses this dir
        "name": metadata["name"],
        "description": metadata["description"],
        "num_params": metadata["num_params"],
        "param_ranges": metadata["param_ranges"],
        "param_names": metadata["param_names"],
        "best_train": {
            "split": "train",
            "metrics": train_oracle["oracle_metrics"],
            "param_stats": train_oracle["param_stats"],
            "elapsed_sec": train_oracle["elapsed"],
        },
        "ablations": ablation_results,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)
    print(f"  [evaluate-oracle] result saved to {output_path}", flush=True)

    if args.raw:
        raw_path = Path(args.raw).resolve()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        category_ids = np.asarray(train_sim["category_ids"]).astype(np.int32)
        npz_payload = {
            "full_per_sample_scores": full_per_sample_scores.astype(np.float32),
            "category_ids": category_ids,
            "ablation_names": np.array(raw_ablation_names, dtype=object),
        }
        if raw_ablation_scores:
            npz_payload["ablation_per_sample_scores"] = np.stack(raw_ablation_scores, axis=0)
            npz_payload["ablation_marginal"] = np.stack(raw_ablation_marginals, axis=0)
        else:
            npz_payload["ablation_per_sample_scores"] = np.zeros((0, full_per_sample_scores.shape[0]), dtype=np.float32)
            npz_payload["ablation_marginal"] = np.zeros((0, full_per_sample_scores.shape[0]), dtype=np.float32)

        # v4: per-modality votes under per-sample best oracle params.
        print("  [evaluate-oracle] computing per-modality votes (oracle params)...", flush=True)
        device_str = f"cuda:0" if oracle_cfg.get("num_gpus", 1) >= 1 else "cpu"
        oracle_audio_votes, oracle_visual_votes = compute_modality_votes(
            str(formulation_path), train_sim, train_emb,
            train_oracle["best_params"], device_str,
        )
        npz_payload["oracle_audio_votes"] = oracle_audio_votes
        npz_payload["oracle_visual_votes"] = oracle_visual_votes

        np.savez_compressed(raw_path, **npz_payload)
        print(f"  [evaluate-oracle] raw arrays saved to {raw_path} (with modality votes)", flush=True)


if __name__ == "__main__":
    main()
