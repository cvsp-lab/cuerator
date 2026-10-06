from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ovavel.metrics as metrics_module
from ovavel.predict import predict_and_gpu

from step3_bandit.common import map_action_to_ranges, params_row_to_dict


METRIC_KEYS = ("frame_acc", "seg_f1", "eve_f1", "avg")
# fa_seg = 0.5*(frame_acc + segment_macro_f1): a fast, GPU/numpy-vectorized proxy
# for `avg` (drops only the expensive eve_f1). Usable as reward/selection metric.
ALLOWED_REWARDS = METRIC_KEYS + ("fa_seg",)
GROUP_ORDER = ("all", "close", "open")


def _placeholder_metrics(fa_seg):
    """Metrics dict with the standard keys (for mean_metric_rows) + fa_seg."""
    return {"frame_acc": 0.0, "seg_f1": 0.0, "eve_f1": 0.0, "avg": 0.0, "fa_seg": float(fa_seg)}


class RewardEvaluator:
    def __init__(self, split_data, formulation, device):
        self.split_data = split_data
        self.formulation = formulation
        self.device = torch.device(device)
        self.audio_text_embeddings = torch.from_numpy(split_data.audio_text_embeddings).float().to(
            self.device
        )
        self.visual_text_embeddings = torch.from_numpy(split_data.visual_text_embeddings).float().to(
            self.device
        )
        # Pre-load per-sample tensors to device once (avoids repeated H2D transfers per call).
        self.audio_embeddings_gpu = torch.from_numpy(split_data.audio_embeddings).float().to(
            self.device
        )
        self.visual_embeddings_gpu = torch.from_numpy(split_data.visual_embeddings).float().to(
            self.device
        )
        self.a_t_sim_gpu = torch.from_numpy(split_data.a_t_sim).float().to(self.device)
        self.v_t_sim_gpu = torch.from_numpy(split_data.v_t_sim).float().to(self.device)

    def _predict_params(self, index: int, params: torch.Tensor, return_thresholds: bool = False):
        params = params.detach().to(self.device).float()
        a_emb = self.audio_embeddings_gpu[index]
        v_emb = self.visual_embeddings_gpu[index]
        a_sim = self.a_t_sim_gpu[index]
        v_sim = self.v_t_sim_gpu[index]

        with torch.no_grad():
            a_thresh, v_thresh = self.formulation.params_to_thresholds(
                a_emb,
                v_emb,
                self.audio_text_embeddings,
                self.visual_text_embeddings,
                a_sim,
                v_sim,
                params,
            )
            pred = predict_and_gpu(
                a_sim,
                v_sim,
                a_thresh.unsqueeze(0),
                v_thresh.unsqueeze(0),
                self.split_data.bg_id,
            )[0]
        if return_thresholds:
            # Effective thresholds, not raw params: most formulation terms are
            # coefficients on relative quantities, so a coefficient change does
            # not tell you which way the threshold moved.
            cat = int(self.split_data.category_ids[index])
            stats = {
                "a_thr_mean": float(a_thresh.mean()),
                "v_thr_mean": float(v_thresh.mean()),
                "a_thr_gt": float(a_thresh[:, cat].mean()),
                "v_thr_gt": float(v_thresh[:, cat].mean()),
            }
            return pred.detach().cpu().numpy(), stats
        return pred.detach().cpu().numpy()

    def _predict_params_stacked(self, index: int, params_batch: torch.Tensor):
        """Evaluate B param sets on one sample with a single feature computation.

        Args:
            index: sample index
            params_batch: (B, P) tensor of B candidate parameter sets.
        Returns:
            preds_np: (B, T) int64 numpy array of predictions.
        """
        params_batch = params_batch.detach().to(self.device).float()
        a_emb = self.audio_embeddings_gpu[index]
        v_emb = self.visual_embeddings_gpu[index]
        a_sim = self.a_t_sim_gpu[index]
        v_sim = self.v_t_sim_gpu[index]

        with torch.no_grad():
            a_thresh, v_thresh = self.formulation.params_to_thresholds_batch(
                a_emb,
                v_emb,
                self.audio_text_embeddings,
                self.visual_text_embeddings,
                a_sim,
                v_sim,
                params_batch,
            )  # both (B, T, C)
            pred = predict_and_gpu(
                a_sim,
                v_sim,
                a_thresh,
                v_thresh,
                self.split_data.bg_id,
            )  # (B, T)
        return pred.detach().cpu().numpy()

    def _evaluate_prediction(self, index: int, pred: np.ndarray, frame_acc_only: bool = False):
        if frame_acc_only:
            avc = np.asarray(self.split_data.avc_labels[index])
            cat_id = int(self.split_data.category_ids[index])
            bg = int(self.split_data.bg_id)
            gt = np.where(avc > 0, cat_id, bg)
            fa = float(np.mean(pred == gt))
            return {"frame_acc": fa, "seg_f1": 0.0, "eve_f1": 0.0, "avg": 0.0}
        metrics = metrics_module.evaluate_sample(
            pred,
            int(self.split_data.category_ids[index]),
            np.asarray(self.split_data.avc_labels[index]),
            int(self.split_data.bg_id),
        )
        return {metric_name: float(metrics[metric_name]) for metric_name in METRIC_KEYS}

    def evaluate_params(
        self,
        index: int,
        params: torch.Tensor,
        reward_key: str = "frame_acc",
        return_metrics: bool = False,
    ):
        if reward_key not in ALLOWED_REWARDS:
            raise ValueError(
                f"Unsupported reward_key={reward_key}. Valid choices: {', '.join(ALLOWED_REWARDS)}"
            )

        pred = self._predict_params(index, params)
        if reward_key == "fa_seg":
            gt = np.where(np.asarray(self.split_data.avc_labels[index]) > 0,
                          int(self.split_data.category_ids[index]), int(self.split_data.bg_id))
            r = metrics_module.frame_seg_score_np(pred, gt, int(self.split_data.bg_id) + 1)
            return (r, pred, _placeholder_metrics(r)) if return_metrics else (r, pred)
        metrics = self._evaluate_prediction(index, pred)
        reward = float(metrics[reward_key])

        if return_metrics:
            return reward, pred, metrics
        return reward, pred

    def evaluate_params_stacked(
        self,
        index: int,
        params_batch: torch.Tensor,
        reward_key: str = "frame_acc",
        return_metrics: bool = False,
        frame_acc_only: bool = False,
    ):
        """Evaluate B parameter sets for one sample in a single batched call.

        Saves redundant _compute_features work vs calling evaluate_params B times.

        If frame_acc_only=True, skips seg_f1/eve_f1 computation; reward_key
        must then be 'frame_acc' (and the returned metrics dict will have
        seg_f1/eve_f1/avg as 0.0 placeholders).

        Args:
            index: sample index
            params_batch: (B, P) tensor of B candidate parameter sets.
        Returns:
            rewards: list[float] length B
            preds: (B, T) int64 numpy array
            metrics_list: list[dict] length B (only if return_metrics=True)
        """
        if reward_key not in ALLOWED_REWARDS:
            raise ValueError(
                f"Unsupported reward_key={reward_key}. Valid choices: {', '.join(ALLOWED_REWARDS)}"
            )
        if frame_acc_only and reward_key != "frame_acc":
            raise ValueError("frame_acc_only=True requires reward_key='frame_acc'")
        preds = self._predict_params_stacked(index, params_batch)  # (B, T)
        B = preds.shape[0]
        if reward_key == "fa_seg":
            # Fast vectorized reward over the whole candidate batch (no per-b
            # evaluate_sample, no eve_f1 / process pool).
            gt = np.where(np.asarray(self.split_data.avc_labels[index]) > 0,
                          int(self.split_data.category_ids[index]), int(self.split_data.bg_id))
            r = metrics_module.frame_seg_score_np(preds, gt, int(self.split_data.bg_id) + 1)
            rewards = [float(x) for x in r]
            if return_metrics:
                return rewards, preds, [_placeholder_metrics(x) for x in rewards]
            return rewards, preds
        rewards = []
        metrics_list = [] if return_metrics else None
        for b in range(B):
            m = self._evaluate_prediction(index, preds[b], frame_acc_only=frame_acc_only)
            rewards.append(float(m[reward_key]))
            if return_metrics:
                metrics_list.append(m)
        if return_metrics:
            return rewards, preds, metrics_list
        return rewards, preds


def _evaluate_group(all_preds, category_ids, avc_labels, bg_id, frame_acc_only: bool = False):
    count = int(len(all_preds))
    if count == 0:
        return {
            "frame_acc": 0.0,
            "seg_f1": 0.0,
            "eve_f1": 0.0,
            "avg": 0.0,
            "count": 0,
        }

    if frame_acc_only:
        gt_labels = np.where(avc_labels > 0, category_ids[:, None], bg_id)
        fa = float(np.mean(all_preds == gt_labels))
        return {"frame_acc": fa, "seg_f1": 0.0, "eve_f1": 0.0, "avg": 0.0,
                "fa_seg": 0.5 * fa, "count": count}

    metrics = metrics_module.evaluate_dataset(all_preds, category_ids, avc_labels, bg_id)
    metrics["fa_seg"] = 0.5 * (metrics["frame_acc"] + metrics["seg_f1"])
    metrics["count"] = count
    return metrics


def compute_group_metrics(split_data, all_preds_np: np.ndarray, frame_acc_only: bool = False):
    cls_types = np.asarray(split_data.cls_types, dtype=object)
    group_indices = {
        "all": np.arange(len(split_data.video_ids)),
        "close": np.flatnonzero(cls_types == "close"),
        "open": np.flatnonzero(cls_types == "open"),
    }

    group_metrics = {}
    for group_name in GROUP_ORDER:
        indices = group_indices[group_name]
        group_metrics[group_name] = _evaluate_group(
            all_preds_np[indices],
            split_data.category_ids[indices],
            split_data.avc_labels[indices],
            split_data.bg_id,
            frame_acc_only=frame_acc_only,
        )
    return group_metrics


def _per_sample_metrics_chunk(args):
    preds_chunk, category_ids_chunk, avc_labels_chunk, bg_id = args
    results = []
    for p, c, a in zip(preds_chunk, category_ids_chunk, avc_labels_chunk):
        m = metrics_module.evaluate_sample(p, int(c), np.asarray(a), int(bg_id))
        results.append({k: float(m[k]) for k in METRIC_KEYS})
    return results


def compute_per_sample_metrics_parallel(
    all_preds_np: np.ndarray,
    category_ids: np.ndarray,
    avc_labels: np.ndarray,
    bg_id: int,
    num_workers: int = 8,
):
    N = int(len(all_preds_np))
    if N == 0:
        return []
    workers = min(num_workers, N)
    chunks = np.array_split(np.arange(N), workers)
    args_list = [
        (
            all_preds_np[idx],
            np.asarray(category_ids)[idx],
            np.asarray(avc_labels)[idx],
            bg_id,
        )
        for idx in chunks
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        chunk_results = list(executor.map(_per_sample_metrics_chunk, args_list))
    flat = []
    for res in chunk_results:
        flat.extend(res)
    return flat


def evaluate_policy_on_split(
    policy,
    split_data,
    evaluator: RewardEvaluator,
    param_ranges,
    param_names,
    device,
    policy_forward_chunk: int = 4096,
    metric_workers: int = 8,
    frame_acc_only: bool = False,
    no_range_squash: bool = False,
):
    device = torch.device(device)
    policy.eval()

    N = len(split_data.video_ids)

    with torch.no_grad():
        # --- (A) Batch policy forward over all samples (chunked to cap memory) ---
        mu_parts = []
        contexts_cpu = split_data.contexts  # torch.Tensor on CPU
        for start in range(0, N, policy_forward_chunk):
            end = min(start + policy_forward_chunk, N)
            ctx_chunk = contexts_cpu[start:end].to(device)
            mu_chunk, _ = policy(ctx_chunk)
            mu_parts.append(mu_chunk)
        mu_all = torch.cat(mu_parts, dim=0) if len(mu_parts) > 1 else mu_parts[0]
        if no_range_squash:
            params_all = mu_all  # raw policy output, no box mapping
        else:
            params_all = map_action_to_ranges(mu_all, param_ranges)  # (N, P)

        # --- Predict per sample (features not batch-safe across samples) ---
        # Collect predictions only; defer metric computation to after the loop.
        all_preds = []
        threshold_stats = None if frame_acc_only else []
        for index in range(N):
            if frame_acc_only:
                pred = evaluator._predict_params(index, params_all[index])
            else:
                pred, stats = evaluator._predict_params(
                    index, params_all[index], return_thresholds=True
                )
                threshold_stats.append(stats)
            all_preds.append(pred)

    all_preds_np = np.stack(all_preds, axis=0)

    if frame_acc_only:
        # Fast path: skip per-sample metrics + seg/eve f1 entirely.
        # Caller (e.g. per-epoch val) only needs frame_acc for selection.
        group_metrics = compute_group_metrics(split_data, all_preds_np, frame_acc_only=True)
        metrics = {
            metric_name: float(group_metrics["all"][metric_name]) for metric_name in METRIC_KEYS
        }
        metrics["fa_seg"] = float(group_metrics["all"].get("fa_seg", 0.5 * metrics["frame_acc"]))
        return {
            "metrics": metrics,
            "group_metrics": group_metrics,
            "details": [],
            "predictions": all_preds_np,
            "frame_acc_mean": float(group_metrics["all"]["frame_acc"]),
        }

    # --- (B) Compute per-sample metrics in parallel on CPU ---
    per_sample_metrics = compute_per_sample_metrics_parallel(
        all_preds_np,
        split_data.category_ids,
        split_data.avc_labels,
        split_data.bg_id,
        num_workers=metric_workers,
    )

    # Build details + per-sample rewards list
    params_cpu = params_all.detach().cpu()
    details = []
    rewards = []
    for index, (video_id, sample_metrics) in enumerate(
        zip(split_data.video_ids, per_sample_metrics)
    ):
        rewards.append(sample_metrics["frame_acc"])
        details.append(
            {
                "video_id": video_id,
                **sample_metrics,
                "params": params_row_to_dict(params_cpu[index], param_names),
                **threshold_stats[index],
            }
        )

    group_metrics = compute_group_metrics(split_data, all_preds_np)
    metrics = {
        metric_name: float(group_metrics["all"][metric_name]) for metric_name in METRIC_KEYS
    }
    metrics["fa_seg"] = float(group_metrics["all"].get(
        "fa_seg", 0.5 * (metrics["frame_acc"] + metrics["seg_f1"])))
    return {
        "metrics": metrics,
        "group_metrics": group_metrics,
        "details": details,
        "predictions": all_preds_np,
        "frame_acc_mean": float(np.mean(rewards)) if rewards else 0.0,
    }
