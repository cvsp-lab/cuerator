from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch

from step3_bandit.common import (
    get_formulation_metadata,
    load_formulation,
    normalize_encoder_name,
    save_json,
    suppress_known_pytorch_warnings,
    write_jsonl,
)
from step3_bandit.data import load_split_data
from step3_bandit.evaluator import RewardEvaluator, evaluate_policy_on_split
from step3_bandit.policy import CrossAttentionBanditPolicy


def print_group_metrics(prefix: str, group_metrics):
    for group_name in ("all", "close", "open"):
        metrics = group_metrics[group_name]
        print(
            f"{prefix}[{group_name}] "
            f"count={metrics['count']} "
            f"frame_acc={metrics['frame_acc']:.4f} "
            f"seg_f1={metrics['seg_f1']:.4f} "
            f"eve_f1={metrics['eve_f1']:.4f} "
            f"avg={metrics['avg']:.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--split", default="test", type=str, choices=["train", "val", "test"])
    parser.add_argument("--formulation", default="", type=str)
    parser.add_argument("--encoder", default="", type=str)
    parser.add_argument("--embeddings_dir", default="", type=str)
    parser.add_argument("--similarities_dir", default="", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--output_dir", default="", type=str)
    return parser.parse_args()


def resolve_device(device_arg: str):
    if device_arg.startswith("cuda"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"CUDA initialization: Unexpected error from cudaGetDeviceCount.*",
            )
            if not torch.cuda.is_available():
                return torch.device("cpu")
    return torch.device(device_arg)


def main():
    suppress_known_pytorch_warnings()
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    formulation_path = args.formulation.strip() or config["formulation_path"].strip()
    formulation = load_formulation(formulation_path)
    metadata = get_formulation_metadata(formulation)

    encoder = normalize_encoder_name(args.encoder.strip() or config["encoder"])
    embeddings_dir = args.embeddings_dir.strip() or config["embeddings_dir"]
    similarities_dir = args.similarities_dir.strip() or config["similarities_dir"]

    split_data = load_split_data(
        encoder,
        args.split,
        embeddings_dir=embeddings_dir,
        similarities_dir=similarities_dir,
    )
    policy = CrossAttentionBanditPolicy(
        input_dim=int(config["input_dim"]),
        action_dim=int(config["action_dim"]),
        sequence_length=int(config["sequence_length"]),
        video_dim=int(config["video_dim"]),
        audio_dim=int(config["audio_dim"]),
        fixed_action_std=float(config["fixed_action_std"]),
        model_dim=int(config["policy_model_dim"]),
        transformer_nhead=int(config["policy_transformer_nhead"]),
        transformer_ff_dim=int(config["policy_transformer_ff_dim"]),
        dropout=float(config["policy_dropout"]),
    ).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])

    evaluator = RewardEvaluator(split_data, formulation, device)
    result = evaluate_policy_on_split(
        policy=policy,
        split_data=split_data,
        evaluator=evaluator,
        param_ranges=metadata["param_ranges"],
        param_names=metadata["param_names"],
        device=device,
    )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir.strip()
        else checkpoint_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / f"eval_{args.split}_summary.json",
        {
            "metrics": result["metrics"],
            "group_metrics": result["group_metrics"],
        },
    )
    write_jsonl(output_dir / f"eval_{args.split}_predictions.jsonl", result["details"])

    print(f"[Eval] split={args.split}")
    print_group_metrics("[Eval]", result["group_metrics"])


if __name__ == "__main__":
    main()
