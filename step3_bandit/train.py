from __future__ import annotations

import argparse
import csv
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F


def _cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

from step3_bandit.common import (
    append_jsonl,
    build_run_dir,
    get_formulation_metadata,
    load_oracle_targets_by_video_id,
    load_formulation,
    load_step2_result_payload,
    map_action_to_ranges,
    normalize_encoder_name,
    params_row_to_dict,
    save_json,
    set_random_seed,
    suppress_known_pytorch_warnings,
    write_jsonl,
)
from step3_bandit.data import load_split_data
from step3_bandit.evaluator import METRIC_KEYS, RewardEvaluator, evaluate_policy_on_split
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


def mean_metric_rows(metric_rows):
    if not metric_rows:
        return {metric_name: 0.0 for metric_name in METRIC_KEYS}
    return {
        metric_name: float(sum(row[metric_name] for row in metric_rows) / len(metric_rows))
        for metric_name in METRIC_KEYS
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formulation", required=True, type=str)
    parser.add_argument("--encoder", default="imagebind", type=str)
    parser.add_argument("--embeddings_dir", default="data/embeddings", type=str)
    parser.add_argument("--similarities_dir", default="data/similarities", type=str)
    parser.add_argument("--train_split", default="train", type=str, choices=["train", "val", "test"])
    parser.add_argument("--val_split", default="val", type=str, choices=["train", "val", "test"])
    parser.add_argument("--test_split", default="test", type=str, choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--fixed_action_std", default=0.3, type=float)
    parser.add_argument(
        "--num_samples",
        default=4,
        type=int,
        help=(
            "K: number of actions sampled per context for multi-sample REINFORCE. "
            "Baseline = leave-one-out mean over the K rewards (K>=2). "
            "K=1 falls back to zero baseline."
        ),
    )
    parser.add_argument(
        "--train_mode",
        default="rl",
        type=str,
        choices=["rl", "supervised"],
    )
    parser.add_argument(
        "--train_reward_metric",
        default=None,
        type=str,
        choices=list(METRIC_KEYS) + ["fa_seg"],
    )

    parser.add_argument(
        "--no_range_squash",
        action="store_true",
        help=(
            "If set, skip sigmoid+range mapping and use the raw policy output "
            "directly as theta. PARAM_RANGES is ignored at training and inference. "
            "Used for the raw-policy-output ablation."
        ),
    )
    parser.add_argument(
        "--init_mu_head_box_center",
        action="store_true",
        help=(
            "Override mu_head: zero the weight and set bias to PARAM_RANGES "
            "midpoints, so the policy initially outputs the box center for any "
            "input. Useful for matching the formulation's initial theta when sigmoid+range "
            "mapping is disabled (raw-policy ablation)."
        ),
    )
    parser.add_argument("--policy_model_dim", default=256, type=int)
    parser.add_argument("--policy_transformer_nhead", default=4, type=int)
    parser.add_argument("--policy_transformer_ff_dim", default=512, type=int)
    parser.add_argument("--policy_dropout", default=0.1, type=float)

    parser.add_argument("--save_dir", default="runs/step3_bandit", type=str)
    parser.add_argument("--run_name", default="", type=str)
    parser.add_argument("--log_every", default=20, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--no_train_steps_log", action="store_true",
                        help="Skip writing train_steps.jsonl (~136MB/run).")
    parser.add_argument("--no_last_ckpt", action="store_true",
                        help="Skip writing last.ckpt (~32MB/run); best.ckpt still saved.")
    parser.add_argument(
        "--per_epoch_test",
        action="store_true",
        help="Evaluate test split every epoch and record in epoch_summary.csv (adds ~20-30% training time).",
    )
    parser.add_argument(
        "--per_epoch_full_val",
        action="store_true",
        help="Compute full val metrics (seg_f1/eve_f1/avg + close/open) every epoch; disables fast_frame_acc path.",
    )
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


def ensure_split_compatibility(reference_split, other_split):
    if reference_split.sequence_length != other_split.sequence_length:
        raise ValueError(
            f"Sequence length mismatch: {reference_split.sequence_length} vs {other_split.sequence_length}"
        )
    if reference_split.video_dim != other_split.video_dim:
        raise ValueError(
            f"Video dim mismatch: {reference_split.video_dim} vs {other_split.video_dim}"
        )
    if reference_split.audio_dim != other_split.audio_dim:
        raise ValueError(
            f"Audio dim mismatch: {reference_split.audio_dim} vs {other_split.audio_dim}"
        )


def build_checkpoint_payload(
    args,
    policy,
    optimizer,
    epoch,
    best_val_metric,
    metadata,
    formulation_path,
):
    return {
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "best_val_metric": float(best_val_metric),
        "best_val_metric_name": args.selection_metric_name,
        "best_val_avg": float(best_val_metric) if args.selection_metric_name == "avg" else None,
        "config": {
            "formulation_path": str(formulation_path),
            "encoder": args.encoder,
            "embeddings_dir": args.embeddings_dir,
            "similarities_dir": args.similarities_dir,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "input_dim": int(policy.input_dim),
            "action_dim": int(policy.action_dim),
            "sequence_length": int(policy.sequence_length),
            "video_dim": int(policy.video_dim),
            "audio_dim": int(policy.audio_dim),
            "fixed_action_std": float(args.fixed_action_std),
            "train_mode": args.train_mode,
            "train_reward_metric": args.train_reward_metric,
            "train_reward_metric_override": args.train_reward_metric_override,
            "selection_metric_name": args.selection_metric_name,
            "step2_result_path": args.step2_result_path,
            "step2_oracle_metric": args.step2_oracle_metric,
            "oracle_artifact_path": args.oracle_artifact_path,
            "policy_model_dim": int(args.policy_model_dim),
            "policy_transformer_nhead": int(args.policy_transformer_nhead),
            "policy_transformer_ff_dim": int(args.policy_transformer_ff_dim),
            "policy_dropout": float(args.policy_dropout),
            "seed": int(args.seed),
            "param_names": metadata["param_names"],
            "param_ranges": metadata["param_ranges"],
        },
    }


def save_checkpoint(path, args, policy, optimizer, epoch, best_val_metric, metadata, formulation_path):
    payload = build_checkpoint_payload(
        args=args,
        policy=policy,
        optimizer=optimizer,
        epoch=epoch,
        best_val_metric=best_val_metric,
        metadata=metadata,
        formulation_path=formulation_path,
    )
    torch.save(payload, path)


def main():
    suppress_known_pytorch_warnings()
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if args.fixed_action_std <= 0.0:
        raise ValueError("--fixed_action_std must be > 0")
    if args.log_every <= 0:
        raise ValueError("--log_every must be > 0")

    args.encoder = normalize_encoder_name(args.encoder)
    device = resolve_device(args.device)
    set_random_seed(args.seed)

    formulation_path = Path(args.formulation).expanduser().resolve()
    formulation = load_formulation(str(formulation_path))
    metadata = get_formulation_metadata(formulation)
    param_ranges = metadata["param_ranges"]
    param_names = metadata["param_names"]
    # The search loop stores each formulation next to its oracle result JSON
    # (formulations/fNNN.py <-> results/iter_NNN.json). A formulation taken out of that
    # layout has no such sibling; in rl mode the payload is bookkeeping only, so fall back
    # to an empty one. Supervised training reads oracle targets from it and still requires it.
    try:
        step2_result_path, step2_result_payload = load_step2_result_payload(formulation_path)
    except (ValueError, FileNotFoundError):
        if args.train_mode == "supervised":
            raise
        step2_result_path, step2_result_payload = "", {}
    step2_oracle_metric = str(step2_result_payload.get("oracle_metric", "frame_acc"))
    if step2_oracle_metric not in METRIC_KEYS:
        raise ValueError(
            f"Unsupported step2 oracle_metric={step2_oracle_metric}. "
            f"Expected one of: {', '.join(METRIC_KEYS)}"
        )

    train_reward_metric_override = args.train_reward_metric
    args.train_reward_metric_override = train_reward_metric_override
    args.train_reward_metric = train_reward_metric_override or step2_oracle_metric
    args.selection_metric_name = args.train_reward_metric
    # Fast path: when both reward and selection are frame_acc, skip per-batch
    # seg_f1/eve_f1/avg computation. Final best_val_summary is rebuilt at the
    # end by reloading best.ckpt and running one full eval.
    use_fast_frame_acc = (
        args.train_mode == "rl"
        and args.train_reward_metric == "frame_acc"
        and args.selection_metric_name == "frame_acc"
        and not args.per_epoch_full_val
    )
    args.use_fast_frame_acc = use_fast_frame_acc
    args.step2_result_path = str(step2_result_path)
    args.step2_oracle_metric = step2_oracle_metric
    args.oracle_artifact_path = str(
        step2_result_payload.get("oracle_artifact", {}).get("path", "")
    )

    train_data = load_split_data(
        args.encoder,
        args.train_split,
        embeddings_dir=args.embeddings_dir,
        similarities_dir=args.similarities_dir,
    )
    val_data = load_split_data(
        args.encoder,
        args.val_split,
        embeddings_dir=args.embeddings_dir,
        similarities_dir=args.similarities_dir,
    )
    test_data = load_split_data(
        args.encoder,
        args.test_split,
        embeddings_dir=args.embeddings_dir,
        similarities_dir=args.similarities_dir,
    )
    ensure_split_compatibility(train_data, val_data)
    ensure_split_compatibility(train_data, test_data)

    policy = CrossAttentionBanditPolicy(
        input_dim=train_data.video_dim + train_data.audio_dim,
        action_dim=metadata["num_params"],
        sequence_length=train_data.sequence_length,
        video_dim=train_data.video_dim,
        audio_dim=train_data.audio_dim,
        fixed_action_std=args.fixed_action_std,
        model_dim=args.policy_model_dim,
        transformer_nhead=args.policy_transformer_nhead,
        transformer_ff_dim=args.policy_transformer_ff_dim,
        dropout=args.policy_dropout,
    ).to(device)
    if args.init_mu_head_box_center:
        with torch.no_grad():
            policy.mu_head.weight.zero_()
            box_center = torch.tensor(
                [(lo + hi) / 2.0 for (lo, hi) in param_ranges],
                dtype=policy.mu_head.bias.dtype,
                device=device,
            )
            policy.mu_head.bias.copy_(box_center)
        print(f"  [init] mu_head reset to box-center init (bias={box_center.tolist()})", flush=True)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_oracle_targets = None
    if args.train_mode == "supervised":
        train_oracle_targets_np, oracle_artifact_path = load_oracle_targets_by_video_id(
            step2_result_payload=step2_result_payload,
            step2_result_path=step2_result_path,
            encoder_name=args.encoder,
            split_name=args.train_split,
            video_ids=train_data.video_ids,
            expected_num_params=metadata["num_params"],
        )
        train_oracle_targets = torch.from_numpy(train_oracle_targets_np)
        args.oracle_artifact_path = str(oracle_artifact_path)

    train_evaluator = RewardEvaluator(train_data, formulation, device) if args.train_mode == "rl" else None
    val_evaluator = RewardEvaluator(val_data, formulation, device)
    test_evaluator = RewardEvaluator(test_data, formulation, device)

    run_dir = build_run_dir(args.save_dir, args.run_name)
    config_payload = {
        **vars(args),
        "device_resolved": str(device),
        "formulation_path": str(formulation_path),
        "param_names": param_names,
        "param_ranges": param_ranges,
        "sequence_length": train_data.sequence_length,
        "video_dim": train_data.video_dim,
        "audio_dim": train_data.audio_dim,
    }
    save_json(run_dir / "config.json", config_payload)

    epoch_summary_path = run_dir / "epoch_summary.csv"
    train_steps_path = run_dir / "train_steps.jsonl"
    best_ckpt_path = run_dir / "best.ckpt"
    last_ckpt_path = run_dir / "last.ckpt"

    best_val_metric = float("-inf")
    best_epoch = 0
    group_names = ("all", "close", "open")
    group_metric_keys = ("frame_acc", "seg_f1", "eve_f1", "avg")
    val_group_fields = [f"val_{g}_{m}" for g in group_names for m in group_metric_keys]
    test_group_fields = (
        [f"test_{g}_{m}" for g in group_names for m in group_metric_keys]
        if args.per_epoch_test
        else []
    )
    with epoch_summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "epoch",
                "train_mode",
                "train_reward_metric",
                "train_reward_mean",
                "train_reward_mu_mean",
                "train_adv_mean",
                "train_loss_mean",
                "train_param_mse_mean",
                "val_frame_acc",
                "val_seg_f1",
                "val_eve_f1",
                "val_avg",
                *val_group_fields,
                *test_group_fields,
                "selection_metric_name",
                "val_selection_metric",
                "best_val_metric",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            policy.train()
            permutation = torch.randperm(len(train_data)).tolist()

            epoch_rewards = []
            epoch_rewards_mu = []
            epoch_advantages = []
            epoch_losses = []
            epoch_param_mse = []

            # Timing accumulators (seconds) for this epoch
            t_ctx_transfer = 0.0
            t_forward = 0.0
            t_reward_loop = 0.0
            t_backward = 0.0
            t_bookkeeping = 0.0

            epoch_train_start = time.perf_counter()

            for batch_idx, start in enumerate(range(0, len(permutation), args.batch_size), start=1):
                batch_indices = permutation[start : start + args.batch_size]

                _cuda_sync(device)
                _t0 = time.perf_counter()
                contexts = train_data.contexts[batch_indices].to(device)
                _cuda_sync(device)
                t_ctx_transfer += time.perf_counter() - _t0

                if args.train_mode == "rl":
                    _t0 = time.perf_counter()
                    K = int(args.num_samples)
                    sample_out = policy.sample(contexts, num_samples=K)
                    # raw_action: (B, K, P); mu: (B, P); log_prob: (B, K)
                    B_here = sample_out["raw_action"].shape[0]
                    raw_action_flat = sample_out["raw_action"].reshape(B_here * K, -1)
                    if args.no_range_squash:
                        mapped_sample_flat = raw_action_flat
                    else:
                        mapped_sample_flat = map_action_to_ranges(raw_action_flat, param_ranges)
                    mapped_sample = mapped_sample_flat.view(B_here, K, -1)  # (B, K, P)
                    _cuda_sync(device)
                    t_forward += time.perf_counter() - _t0

                    _t0 = time.perf_counter()
                    batch_rewards = []  # (B, K)
                    batch_metrics = []  # (B, K) list of dicts
                    batch_param_rows = []  # (B, K) list of dicts

                    for local_idx, sample_idx in enumerate(batch_indices):
                        # K param sets for this sample → single batched eval call.
                        params_k = mapped_sample[local_idx]  # (K, P)
                        rewards_k, _, metrics_k = train_evaluator.evaluate_params_stacked(
                            sample_idx,
                            params_k,
                            reward_key=args.train_reward_metric,
                            return_metrics=True,
                            frame_acc_only=use_fast_frame_acc,
                        )
                        batch_rewards.append(rewards_k)
                        batch_metrics.append(metrics_k)
                        batch_param_rows.append(
                            [params_row_to_dict(params_k[k], param_names) for k in range(K)]
                        )
                    _cuda_sync(device)
                    t_reward_loop += time.perf_counter() - _t0

                    _t0 = time.perf_counter()
                    reward_tensor = torch.tensor(
                        batch_rewards,
                        dtype=sample_out["log_prob"].dtype,
                        device=device,
                    )  # (B, K)
                    if K >= 2:
                        # Leave-one-out baseline: baseline[b, k] = mean_{k' != k} R[b, k']
                        row_sum = reward_tensor.sum(dim=1, keepdim=True)
                        baseline = (row_sum - reward_tensor) / (K - 1)  # (B, K)
                    else:
                        baseline = torch.zeros_like(reward_tensor)
                    advantages = reward_tensor - baseline  # (B, K)
                    loss = -(advantages.detach() * sample_out["log_prob"]).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    _cuda_sync(device)
                    t_backward += time.perf_counter() - _t0
                    _t0 = time.perf_counter()

                    # Flatten per-sample rewards/metrics for epoch-level aggregation.
                    flat_rewards = [r for row in batch_rewards for r in row]
                    flat_metrics = [m for row in batch_metrics for m in row]
                    flat_baseline = baseline.detach().cpu().flatten().tolist()
                    flat_adv = advantages.detach().cpu().flatten().tolist()

                    epoch_rewards.extend(flat_rewards)
                    epoch_rewards_mu.extend(flat_baseline)
                    epoch_advantages.extend(flat_adv)
                    epoch_losses.append(float(loss.detach().cpu().item()))

                    step_payload = {
                        "epoch": epoch,
                        "batch": batch_idx,
                        "train_mode": args.train_mode,
                        "reward_metric": args.train_reward_metric,
                        "num_samples_K": K,
                        "reward_mean": float(reward_tensor.mean().detach().cpu().item()),
                        "reward_mu_mean": float(baseline.mean().detach().cpu().item()),
                        "adv_mean": float(advantages.mean().detach().cpu().item()),
                        "adv_std": float(advantages.std().detach().cpu().item()) if K >= 2 else 0.0,
                        "param_mse": None,
                        "loss": float(loss.detach().cpu().item()),
                        "sampled_metric_means": mean_metric_rows(flat_metrics),
                        "mu_metric_means": None,
                        "params_sampled": batch_param_rows,
                        "params_mu": None,
                        "target_params": None,
                    }
                else:
                    mu, _ = policy(contexts)
                    if args.no_range_squash:
                        mapped_mu = mu
                    else:
                        mapped_mu = map_action_to_ranges(mu, param_ranges)
                    target_params = train_oracle_targets[batch_indices].to(device)
                    loss = F.mse_loss(mapped_mu, target_params)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_value = float(loss.detach().cpu().item())
                    epoch_losses.append(loss_value)
                    epoch_param_mse.append(loss_value)

                    batch_param_rows_mu = [
                        params_row_to_dict(mapped_mu[local_idx], param_names)
                        for local_idx in range(len(batch_indices))
                    ]
                    batch_target_rows = [
                        params_row_to_dict(target_params[local_idx], param_names)
                        for local_idx in range(len(batch_indices))
                    ]
                    step_payload = {
                        "epoch": epoch,
                        "batch": batch_idx,
                        "train_mode": args.train_mode,
                        "reward_metric": args.train_reward_metric,
                        "reward_mean": None,
                        "reward_mu_mean": None,
                        "adv_mean": None,
                        "param_mse": loss_value,
                        "loss": loss_value,
                        "sampled_metric_means": None,
                        "mu_metric_means": None,
                        "params_sampled": None,
                        "params_mu": batch_param_rows_mu,
                        "target_params": batch_target_rows,
                    }
                if not args.no_train_steps_log:
                    append_jsonl(train_steps_path, step_payload)
                t_bookkeeping += time.perf_counter() - _t0

                # if batch_idx % args.log_every == 0:
                #     print(
                #         f"[Epoch {epoch}] batch={batch_idx} "
                #         f"reward_mean={step_payload['reward_mean']:.4f} "
                #         f"reward_mu_mean={step_payload['reward_mu_mean']:.4f} "
                #         f"adv_mean={step_payload['adv_mean']:.4f} "
                #         f"loss={step_payload['loss']:.4f}"
                #     )

            epoch_train_elapsed = time.perf_counter() - epoch_train_start

            _cuda_sync(device)
            _val_t0 = time.perf_counter()
            val_result = evaluate_policy_on_split(
                policy=policy,
                split_data=val_data,
                evaluator=val_evaluator,
                param_ranges=param_ranges,
                param_names=param_names,
                device=device,
                frame_acc_only=use_fast_frame_acc,
                no_range_squash=args.no_range_squash,
            )
            _cuda_sync(device)
            t_val = time.perf_counter() - _val_t0

            # Timing summary for this epoch
            t_train_accounted = (
                t_ctx_transfer + t_forward + t_reward_loop + t_backward + t_bookkeeping
            )
            t_other = max(epoch_train_elapsed - t_train_accounted, 0.0)
            t_total = epoch_train_elapsed + t_val
            print(
                f"[Epoch {epoch}] [Timing] "
                f"total={t_total:.2f}s "
                f"(train={epoch_train_elapsed:.2f}s, val={t_val:.2f}s) | "
                f"ctx_xfer={t_ctx_transfer:.2f}s "
                f"fwd={t_forward:.2f}s "
                f"reward_loop={t_reward_loop:.2f}s "
                f"bwd={t_backward:.2f}s "
                f"book={t_bookkeeping:.2f}s "
                f"other={t_other:.2f}s"
            )
            val_metrics = val_result["metrics"]
            val_selection_metric = float(val_metrics[args.selection_metric_name])
            is_best = val_selection_metric > best_val_metric

            if is_best:
                best_val_metric = val_selection_metric
                best_epoch = epoch
                save_checkpoint(
                    best_ckpt_path,
                    args=args,
                    policy=policy,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_metric=best_val_metric,
                    metadata=metadata,
                    formulation_path=formulation_path,
                )
                if not use_fast_frame_acc:
                    # Full val metrics already computed this epoch — write now.
                    save_json(
                        run_dir / "best_val_summary.json",
                        {
                            "epoch": epoch,
                            "selection_metric_name": args.selection_metric_name,
                            "selection_metric_value": val_selection_metric,
                            "metrics": val_metrics,
                            "group_metrics": val_result["group_metrics"],
                        },
                    )
                    write_jsonl(run_dir / "best_val_predictions.jsonl", val_result["details"])
                # else: deferred until end-of-training final eval below.

            if not args.no_last_ckpt:
                save_checkpoint(
                    last_ckpt_path,
                    args=args,
                    policy=policy,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_metric=best_val_metric,
                    metadata=metadata,
                    formulation_path=formulation_path,
                )

            # Per-epoch group breakdown (frame_acc always valid; seg/eve/avg zero in fast mode)
            val_group = val_result["group_metrics"]
            val_group_row = {
                f"val_{g}_{m}": float(val_group[g][m])
                for g in group_names for m in group_metric_keys
            }

            # Optional: per-epoch test evaluation (always full metrics, even if val is fast)
            test_group_row = {}
            if args.per_epoch_test:
                _cuda_sync(device)
                _test_t0 = time.perf_counter()
                test_epoch_result = evaluate_policy_on_split(
                    policy=policy,
                    split_data=test_data,
                    evaluator=test_evaluator,
                    param_ranges=param_ranges,
                    param_names=param_names,
                    device=device,
                    frame_acc_only=False,
                )
                _cuda_sync(device)
                t_test = time.perf_counter() - _test_t0
                print(f"[Epoch {epoch}] [Test] elapsed={t_test:.2f}s")
                test_group_row = {
                    f"test_{g}_{m}": float(test_epoch_result["group_metrics"][g][m])
                    for g in group_names for m in group_metric_keys
                }

            summary_row = {
                "epoch": epoch,
                "train_mode": args.train_mode,
                "train_reward_metric": args.train_reward_metric,
                "train_reward_mean": (
                    sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else None
                ),
                "train_reward_mu_mean": (
                    sum(epoch_rewards_mu) / len(epoch_rewards_mu) if epoch_rewards_mu else None
                ),
                "train_adv_mean": (
                    sum(epoch_advantages) / len(epoch_advantages) if epoch_advantages else None
                ),
                "train_loss_mean": sum(epoch_losses) / len(epoch_losses),
                "train_param_mse_mean": (
                    sum(epoch_param_mse) / len(epoch_param_mse) if epoch_param_mse else None
                ),
                "val_frame_acc": float(val_metrics["frame_acc"]),
                "val_seg_f1": float(val_metrics["seg_f1"]),
                "val_eve_f1": float(val_metrics["eve_f1"]),
                "val_avg": float(val_metrics["avg"]),
                **val_group_row,
                **test_group_row,
                "selection_metric_name": args.selection_metric_name,
                "val_selection_metric": val_selection_metric,
                "best_val_metric": best_val_metric,
            }
            writer.writerow(summary_row)
            csv_file.flush()
            # print(
            #     f"[Epoch {epoch}] "
            #     f"train_reward_mean={summary_row['train_reward_mean']:.4f} "
            #     f"train_reward_mu_mean={summary_row['train_reward_mu_mean']:.4f} "
            #     f"val_frame_acc={summary_row['val_frame_acc']:.4f} "
            #     f"best_val_avg={best_val_avg:.4f}"
            # )
            print(
                f"[Epoch {epoch}] [Val] "
                f"frame_acc={summary_row['val_frame_acc']:.4f} "
                f"seg_f1={summary_row['val_seg_f1']:.4f} "
                f"eve_f1={summary_row['val_eve_f1']:.4f} "
                f"avg={summary_row['val_avg']:.4f} "
                f"{args.selection_metric_name}={val_selection_metric:.4f} "
                f"best_epoch={best_epoch} "
                f"best_{args.selection_metric_name}={best_val_metric:.4f}"
                + (" saved=best.ckpt" if is_best else "")
            )
            print_group_metrics(f"[Epoch {epoch}] [Val]", val_result["group_metrics"])

    best_checkpoint = torch.load(best_ckpt_path, map_location=device)
    policy.load_state_dict(best_checkpoint["policy_state_dict"])

    if use_fast_frame_acc:
        # Per-epoch val skipped seg/eve to save time; rebuild full best_val_summary
        # and best_val_predictions now from best.ckpt.
        print("[Final val] Computing full validation metrics from best.ckpt...")
        val_result_full = evaluate_policy_on_split(
            policy=policy,
            split_data=val_data,
            evaluator=val_evaluator,
            param_ranges=param_ranges,
            param_names=param_names,
            device=device,
            frame_acc_only=False,
        )
        save_json(
            run_dir / "best_val_summary.json",
            {
                "epoch": best_epoch,
                "selection_metric_name": args.selection_metric_name,
                "selection_metric_value": best_val_metric,
                "metrics": val_result_full["metrics"],
                "group_metrics": val_result_full["group_metrics"],
            },
        )
        write_jsonl(run_dir / "best_val_predictions.jsonl", val_result_full["details"])
        print_group_metrics("[Final val]", val_result_full["group_metrics"])

    test_result = evaluate_policy_on_split(
        policy=policy,
        split_data=test_data,
        evaluator=test_evaluator,
        param_ranges=param_ranges,
        param_names=param_names,
        device=device,
    )
    save_json(
        run_dir / "test_summary.json",
        {
            "metrics": test_result["metrics"],
            "group_metrics": test_result["group_metrics"],
            "best_epoch": best_epoch,
            "best_val_metric_name": args.selection_metric_name,
            "best_val_metric": best_val_metric,
            "step2_oracle_metric": args.step2_oracle_metric,
        },
    )
    write_jsonl(run_dir / "test_predictions.jsonl", test_result["details"])

    print(f"[Done] run_dir={run_dir}")
    print(f"[Best] epoch={best_epoch} {args.selection_metric_name}={best_val_metric:.4f}")
    print_group_metrics("[Test]", test_result["group_metrics"])


if __name__ == "__main__":
    main()
