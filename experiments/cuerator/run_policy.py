"""Run policy training (step3_bandit.train) for one iteration, then post-process
into agent-visible artifacts.

Outputs:
  <policy_run_dir>/                    full step3_bandit.train output (test included; agent NOT exposed)
  <policy_results_path>                sanitized JSON: val metrics, best_val_*, param_names, param_ranges (no test fields, no pre-computed param summaries)
  <policy_raw_path>                    .npz: per-sample raw arrays on val (params, votes, sims) for the Policy agent

Usage:
    python -m experiments.cuerator.run_policy \\
        --formulation <path> --config <path> \\
        --policy-run-dir <session>/policy_runs/iter_NNN \\
        --policy-results <session>/agent_context/policy_results/iter_NNN.json \\
        --policy-raw     <session>/policy_raw/iter_NNN.npz
"""

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import ovavel.data as dataset_utils  # noqa: E402
from ovavel.predict import predict_and_gpu  # noqa: E402


# Columns in epoch_summary.csv that contain test-split metrics — stripped from
# the agent-visible policy_results payload to preserve test isolation.
TEST_COLS_PREFIX = "test_"


def load_formulation(path: Path):
    spec = importlib.util.spec_from_file_location("policy_formulation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_epoch_summary(csv_path: Path):
    """Return (val_curve_rows, test_cols_present)."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def run_step3_train(formulation_path: Path, run_dir: Path, cfg: dict):
    """Invoke step3_bandit.train. Returns the run_dir path it wrote into."""
    s3 = cfg["step3_final"]
    save_dir = run_dir.parent
    run_name = run_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "step3_bandit.train",
        "--formulation", str(formulation_path),
        "--encoder", s3["encoder"],
        "--epochs", str(s3["epochs"]),
        "--batch_size", str(s3["batch_size"]),
        "--num_samples", str(s3["num_samples"]),
        "--fixed_action_std", str(s3["fixed_action_std"]),
        "--device", s3["device"],
        "--seed", str(s3["policy_seed"]),
        "--train_reward_metric", s3["selection_metric"],
        "--no_train_steps_log",
        "--no_last_ckpt",
        "--per_epoch_test",
        "--save_dir", str(save_dir),
        "--run_name", run_name,
    ]
    print(f"  [run-policy] cmd: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"step3_bandit.train failed (exit {r.returncode})")


def build_val_curve(epoch_rows):
    """Sanitized per-epoch curve: keep only train_* and val_* fields, drop test_*."""
    curve = []
    for row in epoch_rows:
        clean = {}
        for k, v in row.items():
            if k.startswith(TEST_COLS_PREFIX):
                continue
            try:
                clean[k] = float(v) if v not in ("", None) else None
            except ValueError:
                clean[k] = v
        curve.append(clean)
    return curve


def read_best_val_predictions(jsonl_path: Path, param_names: list[str]):
    """Load per-sample val predictions.

    Returns:
        per_sample_frame_acc: (N,) float32
        video_ids: (N,) object
        params_per_sample: (N, P) float32 — policy's per-sample output for every val sample
    """
    accs = []
    vids = []
    params_rows = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            accs.append(float(rec["frame_acc"]))
            vids.append(rec["video_id"])
            p = rec.get("params", {})
            if isinstance(p, dict):
                params_rows.append([float(p[n]) for n in param_names])
            elif isinstance(p, list):
                params_rows.append([float(x) for x in p])
            else:
                params_rows.append([0.0] * len(param_names))
    return (
        np.array(accs, dtype=np.float32),
        np.array(vids, dtype=object),
        np.array(params_rows, dtype=np.float32),
    )


def _modality_vote(sim, thresh):
    """Per-frame vote: highest-sim category among those passing their threshold, else -1. (T,) int64."""
    passed = sim > thresh
    masked = torch.where(passed, sim, torch.full_like(sim, -float("inf")))
    top = masked.argmax(dim=-1)
    return torch.where(passed.any(dim=-1), top, torch.full_like(top, -1))


def load_val_inputs(encoder: str):
    bg_id = dataset_utils.get_bg_id("val")
    sim = dataset_utils.load_similarities(encoder, "val", "data/similarities")
    try:
        emb_raw = dataset_utils.load_embeddings(encoder, "val", "data/embeddings")
        a_text = dataset_utils.load_text_embeddings(encoder, "val", "audio", "data/embeddings")
        v_text = dataset_utils.load_text_embeddings(encoder, "val", "visual", "data/embeddings")
        emb = {**emb_raw, "audio_text": a_text, "visual_text": v_text}
    except FileNotFoundError:
        emb = None
    return sim, emb, bg_id


def compute_val_per_sample_outputs(
    formulation_path: Path,
    params_per_sample: np.ndarray,
    encoder: str,
    device_str: str,
):
    """Apply each val sample's actual policy-output params to compute per-sample
    thresholds + per-modality votes + joint predictions.

    Args:
        params_per_sample: (N_val, P) — the policy's actual per-sample output
            (read from best_val_predictions.jsonl).

    Returns:
        per_sample_pred  (N, T)  int64   — joint AND-rule predictions
        per_sample_acc   (N,)    float32 — per-sample frame_acc
        audio_votes      (N, T)  int32   — highest-sim passing audio cat, else -1
        visual_votes     (N, T)  int32   — highest-sim passing visual cat, else -1
        category_ids     (N,)    int32
        avc_labels       (N, T)  float32
    """
    sim_data, emb_data, bg_id = load_val_inputs(encoder)
    formulation = load_formulation(formulation_path)
    a_t_sim = sim_data["a_t_sim"]
    v_t_sim = sim_data["v_t_sim"]
    avc_labels = sim_data["avc_labels"]
    category_ids = sim_data["category_ids"]
    N, T, _ = a_t_sim.shape
    gt_labels = np.where(avc_labels > 0, category_ids[:, None], bg_id).astype(np.int64)

    device = torch.device(device_str)
    a_t_emb = emb_data.get("audio_text") if emb_data else None
    v_t_emb = emb_data.get("visual_text") if emb_data else None
    a_emb = emb_data.get("audio") if emb_data else None
    v_emb = emb_data.get("visual") if emb_data else None
    a_t_emb_gpu = torch.from_numpy(a_t_emb).float().to(device) if a_t_emb is not None else None
    v_t_emb_gpu = torch.from_numpy(v_t_emb).float().to(device) if v_t_emb is not None else None

    per_sample_acc = np.zeros(N, dtype=np.float32)
    per_sample_pred = np.zeros((N, T), dtype=np.int64)
    audio_votes = np.full((N, T), -1, dtype=np.int32)
    visual_votes = np.full((N, T), -1, dtype=np.int32)
    for i in range(N):
        a_sim = torch.from_numpy(a_t_sim[i]).float().to(device)
        v_sim = torch.from_numpy(v_t_sim[i]).float().to(device)
        gt_i = torch.from_numpy(gt_labels[i]).long().to(device)
        a_emb_i = torch.from_numpy(a_emb[i]).float().to(device) if a_emb is not None else None
        v_emb_i = torch.from_numpy(v_emb[i]).float().to(device) if v_emb is not None else None
        params_i = torch.from_numpy(params_per_sample[i:i + 1]).float().to(device)  # (1, P)
        with torch.no_grad():
            a_ths, v_ths = formulation.params_to_thresholds_batch(
                a_emb_i, v_emb_i, a_t_emb_gpu, v_t_emb_gpu, a_sim, v_sim, params_i
            )  # (1, T, C)
            pred = predict_and_gpu(a_sim, v_sim, a_ths, v_ths, bg_id)
            a_vote = _modality_vote(a_sim, a_ths[0])
            v_vote = _modality_vote(v_sim, v_ths[0])
        per_sample_pred[i] = pred[0].cpu().numpy()
        per_sample_acc[i] = float((pred[0] == gt_i).float().mean().cpu())
        audio_votes[i] = a_vote.cpu().numpy().astype(np.int32)
        visual_votes[i] = v_vote.cpu().numpy().astype(np.int32)
    return (
        per_sample_pred,
        per_sample_acc,
        audio_votes,
        visual_votes,
        category_ids.astype(np.int32),
        avc_labels.astype(np.float32),
    )


def build_policy_results(epoch_rows, best_val_summary_path, param_names, param_ranges):
    """Build sanitized policy_results.json payload (no test fields).

    Schema is intentionally minimal: training trajectory + best-val metrics +
    parameter metadata only. The Policy agent is expected to derive any
    summaries (mean/std/min/max, modality balance, etc.) from the per-sample
    raw arrays in `policy_raw.npz`.
    """
    val_curve = build_val_curve(epoch_rows)
    with open(best_val_summary_path) as f:
        bvs = json.load(f)
    best_metrics = {}
    for split in ("frame_acc", "seg_f1", "eve_f1", "avg"):
        v = bvs.get(split) or bvs.get(f"val_{split}") or bvs.get("metrics", {}).get(split)
        if v is not None:
            best_metrics[split] = float(v)
    grouped = {}
    for grp in ("all", "close", "open"):
        sub = {}
        for met in ("frame_acc", "seg_f1", "eve_f1", "avg"):
            v = bvs.get(f"val_{grp}_{met}") or bvs.get("metrics", {}).get(f"{grp}_{met}")
            if v is None and "group_metrics" in bvs:
                v = bvs.get("group_metrics", {}).get(grp, {}).get(met)
            if v is not None:
                sub[met] = float(v)
        if sub:
            grouped[grp] = sub
    payload = {
        "best_val_epoch": int(bvs.get("epoch") or bvs.get("best_epoch") or -1),
        "best_val_metrics": best_metrics,
        "best_val_group_metrics": grouped,
        "val_curve": val_curve,
        "param_names": list(param_names),
        "param_ranges": [[float(lo), float(hi)] for (lo, hi) in param_ranges],
    }
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formulation", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--policy-run-dir", required=True,
                    help="Where step3_bandit.train will write its full output (NOT in agent_context).")
    ap.add_argument("--policy-results", required=True,
                    help="Sanitized policy_results.json (val-only, agent-visible).")
    ap.add_argument("--policy-raw", required=True,
                    help="policy_raw.npz (per-sample arrays for the Policy agent).")
    args = ap.parse_args()

    formulation_path = Path(args.formulation).resolve()
    run_dir = Path(args.policy_run_dir).resolve()
    policy_results_path = Path(args.policy_results).resolve()
    policy_raw_path = Path(args.policy_raw).resolve()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    s3 = cfg["step3_final"]

    # Stage A: training
    print(f"  [run-policy] training (epochs={s3['epochs']}, seed={s3['policy_seed']})...", flush=True)
    run_step3_train(formulation_path, run_dir, cfg)

    epoch_rows = parse_epoch_summary(run_dir / "epoch_summary.csv")
    best_val_summary = run_dir / "best_val_summary.json"
    val_pred_jsonl = run_dir / "best_val_predictions.jsonl"

    # Determine param metadata from the formulation file.
    formulation = load_formulation(formulation_path)
    param_names = list(getattr(formulation, "PARAM_NAMES", [f"p{i}" for i in range(int(formulation.NUM_PARAMS))]))
    param_ranges = list(formulation.PARAM_RANGES)

    # Read per-sample policy outputs (sample-conditioned: each val sample has
    # its own (P,) vector). No pre-computed mean/std/min/max — agent computes
    # whatever summaries it needs from the raw array.
    val_per_sample_acc, val_video_ids, val_per_sample_params = read_best_val_predictions(
        val_pred_jsonl, param_names
    )

    # Stage B: apply each val sample's actual params → per-sample thresholds,
    # per-modality votes, joint predictions. This replaces the v5 "mean params
    # applied to all train samples" simulation, which misrepresented the
    # sample-conditioned policy.
    print("  [run-policy] computing per-sample val thresholds + votes...", flush=True)
    (
        val_per_sample_pred,
        val_per_sample_acc_recompute,
        val_audio_votes,
        val_visual_votes,
        val_category_ids,
        val_avc_labels,
    ) = compute_val_per_sample_outputs(
        formulation_path, val_per_sample_params, cfg["oracle"]["encoder"], s3["device"]
    )

    # Sanity-check the re-derived per-sample frame_acc matches the trainer's
    # value (small mismatch indicates a formulation determinism issue).
    diff = np.abs(val_per_sample_acc_recompute - val_per_sample_acc).max()
    if diff > 1e-3:
        print(
            f"  [run-policy] WARNING: per-sample frame_acc recompute differs from trainer "
            f"by max {diff:.6f}",
            flush=True,
        )

    # Stage C: write sanitized policy_results.json
    payload = build_policy_results(epoch_rows, best_val_summary, param_names, param_ranges)
    policy_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(policy_results_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  [run-policy] policy_results saved to {policy_results_path}", flush=True)

    # Stage D: write policy_raw.npz with raw per-sample arrays only.
    policy_raw_path.parent.mkdir(parents=True, exist_ok=True)
    sim_data, _, bg_id = load_val_inputs(cfg["oracle"]["encoder"])
    npz_payload = {
        # Parameter metadata
        "param_names": np.array(param_names, dtype=object),
        "param_ranges": np.array(param_ranges, dtype=np.float32),
        "best_val_epoch": np.int32(payload["best_val_epoch"]),
        "bg_id": np.int32(bg_id),
        # Actual per-sample policy outputs on val
        "val_per_sample_params": val_per_sample_params,                 # (N, P)
        "val_per_sample_frame_acc": val_per_sample_acc,                 # (N,)
        "val_per_sample_pred": val_per_sample_pred,                     # (N, T)
        "val_audio_votes": val_audio_votes,                             # (N, T) int32, -1 = no vote
        "val_visual_votes": val_visual_votes,                           # (N, T)
        "val_video_ids": val_video_ids,                                 # (N,)
        "val_category_ids": val_category_ids,                           # (N,)
        "val_avc_labels": val_avc_labels,                               # (N, T)
        # Similarities (provided so agent can recompute thresholds without
        # re-loading data; same as data/similarities/<encoder>/val.npz)
        "val_a_t_sim": sim_data["a_t_sim"].astype(np.float32),          # (N, T, C)
        "val_v_t_sim": sim_data["v_t_sim"].astype(np.float32),          # (N, T, C)
    }
    # Also stack val_curve numerical arrays for easy loading
    val_keys = [k for k in payload["val_curve"][0].keys() if k != "epoch"] if payload["val_curve"] else []
    for k in val_keys:
        try:
            arr = np.array(
                [row.get(k) if row.get(k) is not None else np.nan for row in payload["val_curve"]],
                dtype=np.float32,
            )
            npz_payload[f"val_curve_{k}"] = arr
        except Exception:
            pass
    npz_payload["val_curve_epoch"] = np.array(
        [int(row.get("epoch", i + 1)) for i, row in enumerate(payload["val_curve"])], dtype=np.int32,
    )
    np.savez_compressed(policy_raw_path, **npz_payload)
    print(f"  [run-policy] policy_raw saved to {policy_raw_path}", flush=True)


if __name__ == "__main__":
    main()
