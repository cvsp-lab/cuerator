"""OV-AVEL evaluation metrics.

Implements frame accuracy, segment-level F1, and event-level F1
following the official ov-avel evaluation protocol (jasongief/ov-avel).

All F1 metrics are computed per-sample then averaged across the dataset.
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor


# ---------------------------------------------------------------------------
# Event extraction helpers
# ---------------------------------------------------------------------------

def _extract_events(seq):
    """Extract contiguous event spans from a binary sequence.

    Args:
        seq: 1D array of length T with binary values.

    Returns:
        List of 1D binary arrays, each representing one contiguous event.
    """
    T = len(seq)
    events = []
    i = 0
    while i < T:
        if seq[i] == 1:
            start = i
            while i < T and seq[i] == 1:
                i += 1
            event = np.zeros(T)
            event[start:i] = 1
            events.append(event)
        else:
            i += 1
    return events


def _event_wise_metric(events_pred, events_gt):
    """Compute event-level TP, FP, FN using IoU >= 0.5 matching.

    Uses greedy first-match (same as official implementation).
    """
    TP, FP, FN = 0, 0, 0

    if events_pred:
        for ep in events_pred:
            matched = False
            if events_gt:
                for eg in events_gt:
                    intersection = np.sum(ep * eg)
                    union = np.sum(ep + eg - ep * eg)
                    if intersection >= 0.5 * union:
                        TP += 1
                        matched = True
                        break
            if not matched:
                FP += 1

    if events_gt:
        for eg in events_gt:
            matched = False
            if events_pred:
                for ep in events_pred:
                    intersection = np.sum(ep * eg)
                    union = np.sum(ep + eg - ep * eg)
                    if intersection >= 0.5 * union:
                        matched = True
                        break
            if not matched:
                FN += 1

    return TP, FP, FN


# ---------------------------------------------------------------------------
# Per-sample binary matrix construction
# ---------------------------------------------------------------------------

def _build_matrices(pred_labels, category_id, avc_label, bg_id):
    """Convert per-frame predictions and ground truth to class-wise binary matrices.

    Args:
        pred_labels: (T,) int — predicted class per frame.
        category_id: int — ground truth category for this sample.
        avc_label: (T,) binary — which frames contain the event.
        bg_id: int — background class ID (= number of foreground classes).

    Returns:
        pred_mat: (K, T) binary matrix where K = bg_id + 1.
        gt_mat: (K, T) binary matrix.
    """
    T = len(pred_labels)
    K = bg_id + 1  # foreground classes + 1 background class

    pred_mat = np.zeros((K, T))
    for t in range(T):
        c = pred_labels[t]
        if c < K:
            pred_mat[c, t] = 1

    gt_mat = np.zeros((K, T))
    if category_id != bg_id:
        gt_mat[category_id] = avc_label
        gt_mat[-1] = 1 - avc_label
    else:
        gt_mat[-1] = 1 - avc_label

    return pred_mat, gt_mat


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------

def _segment_level_f1(pred_mat, gt_mat):
    """Segment-level F1 for one sample. Macro-averaged across active classes."""
    TP = np.sum(pred_mat * gt_mat, axis=1)
    FP = np.sum(pred_mat * (1 - gt_mat), axis=1)
    FN = np.sum((1 - pred_mat) * gt_mat, axis=1)

    f1_scores = []
    for c in range(len(TP)):
        if (TP[c] + FP[c]) != 0 or (TP[c] + FN[c]) != 0:
            f1_scores.append(2 * TP[c] / (2 * TP[c] + FP[c] + FN[c]))

    return sum(f1_scores) / len(f1_scores) if f1_scores else 1.0


def _event_level_f1(pred_mat, gt_mat):
    """Event-level F1 for one sample. Macro-averaged across active classes."""
    N = pred_mat.shape[0]
    TP_all = np.zeros(N)
    FP_all = np.zeros(N)
    FN_all = np.zeros(N)

    for c in range(N):
        events_pred = _extract_events(pred_mat[c]) if np.any(pred_mat[c]) else []
        events_gt = _extract_events(gt_mat[c]) if np.any(gt_mat[c]) else []
        tp, fp, fn = _event_wise_metric(events_pred, events_gt)
        TP_all[c] += tp
        FP_all[c] += fp
        FN_all[c] += fn

    f1_scores = []
    for c in range(N):
        if (TP_all[c] + FP_all[c]) != 0 or (TP_all[c] + FN_all[c]) != 0:
            f1_scores.append(2 * TP_all[c] / (2 * TP_all[c] + FP_all[c] + FN_all[c]))

    return sum(f1_scores) / len(f1_scores) if f1_scores else 1.0


# ---------------------------------------------------------------------------
# Fast vectorized surrogates (frame_acc + segment-macro-F1)
#
# Both pred_mat and gt_mat in _build_matrices are one-hots of the label
# sequence (gt one-hot == _build_matrices' gt_mat exactly), so segment-level
# macro-F1 is one_hot(pred) vs one_hot(gt) class-macro F1 — fully vectorizable.
# These reproduce _segment_level_f1 / frame_accuracy WITHOUT event extraction
# or a process pool, so they are cheap enough for oracle scoring + policy reward.
# eve_f1 is intentionally omitted (it needs IoU matching); the blend
# 0.5*(frame_acc + seg_f1) is a light proxy for avg used as the search target.
# ---------------------------------------------------------------------------

def _seg_f1_from_counts(TP, FP, FN):
    active = ((TP + FP) > 0) | ((TP + FN) > 0)          # (..., K)
    denom = 2 * TP + FP + FN
    import numpy as _np
    f1 = _np.where(denom > 0, 2 * TP / _np.maximum(denom, 1), 0.0)
    na = active.sum(-1)
    seg = _np.where(na > 0, (f1 * active).sum(-1) / _np.maximum(na, 1), 1.0)
    return seg


def frame_seg_score_np(pred, gt, K):
    """0.5*(frame_acc + segment_macro_f1), vectorized. Matches the official
    per-sample frame_acc / _segment_level_f1 exactly.

    pred: (B, T) or (T,) int; gt: (T,) int; K = bg_id + 1. Returns (B,) or scalar."""
    p = np.asarray(pred); g = np.asarray(gt)
    single = p.ndim == 1
    if single:
        p = p[None]
    fa = (p == g[None]).mean(axis=1)                     # (B,)
    pm = (p[:, :, None] == np.arange(K)[None, None]).astype(np.float64)   # (B,T,K)
    gm = (g[:, None] == np.arange(K)[None]).astype(np.float64)            # (T,K)
    TP = (pm * gm[None]).sum(1)
    FP = (pm * (1 - gm[None])).sum(1)
    FN = ((1 - pm) * gm[None]).sum(1)
    seg = _seg_f1_from_counts(TP, FP, FN)
    out = 0.5 * (fa + seg)
    return float(out[0]) if single else out


def frame_seg_score_torch(pred, gt, K):
    """GPU version of frame_seg_score_np. pred: (B, T) long tensor, gt: (T,)
    long tensor. Returns (B,) numpy float array (for oracle scoring)."""
    import torch
    pm = torch.nn.functional.one_hot(pred, K).to(torch.float32)          # (B,T,K)
    gm = torch.nn.functional.one_hot(gt, K).to(torch.float32).unsqueeze(0)  # (1,T,K)
    TP = (pm * gm).sum(1); FP = (pm * (1 - gm)).sum(1); FN = ((1 - pm) * gm).sum(1)  # (B,K)
    active = ((TP + FP) > 0) | ((TP + FN) > 0)
    denom = (2 * TP + FP + FN).clamp(min=1)
    f1 = 2 * TP / denom
    na = active.sum(1)
    seg = torch.where(na > 0, (f1 * active).sum(1) / na.clamp(min=1),
                      torch.ones_like(f1[:, 0]))
    fa = (pred == gt.unsqueeze(0)).float().mean(1)
    return (0.5 * (fa + seg)).cpu().numpy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def frame_accuracy(pred_labels, gt_labels):
    """Compute frame-level accuracy.

    Args:
        pred_labels: (N, T) int — predicted class per frame.
        gt_labels: (N, T) int — ground truth class per frame.

    Returns:
        float accuracy in [0, 1].
    """
    return float(np.mean(pred_labels == gt_labels))


def evaluate_sample(pred_labels, category_id, avc_label, bg_id):
    """Compute all three metrics for a single sample.

    Args:
        pred_labels: (T,) int — predicted class per frame.
        category_id: int — ground truth category.
        avc_label: (T,) binary — event presence per frame.
        bg_id: int — background class ID.

    Returns:
        dict with frame_acc, seg_f1, eve_f1, avg.
    """
    gt_labels = np.where(avc_label > 0, category_id, bg_id)
    fa = float(np.mean(pred_labels == gt_labels))

    pred_mat, gt_mat = _build_matrices(pred_labels, category_id, avc_label, bg_id)
    sf1 = _segment_level_f1(pred_mat, gt_mat)
    ef1 = _event_level_f1(pred_mat, gt_mat)
    avg = (fa + sf1 + ef1) / 3.0

    return {"frame_acc": fa, "seg_f1": sf1, "eve_f1": ef1, "avg": avg}


def _score_chunk(args):
    all_preds_chunk, category_ids_chunk, avc_labels_chunk, bg_id = args
    seg_f1_sum = 0.0
    eve_f1_sum = 0.0
    for i in range(len(all_preds_chunk)):
        pred_mat, gt_mat = _build_matrices(
            all_preds_chunk[i], category_ids_chunk[i], avc_labels_chunk[i], bg_id
        )
        seg_f1_sum += _segment_level_f1(pred_mat, gt_mat)
        eve_f1_sum += _event_level_f1(pred_mat, gt_mat)
    return seg_f1_sum, eve_f1_sum


def evaluate_dataset(all_preds, category_ids, avc_labels, bg_id, num_workers=8):
    """Compute metrics across the full dataset (per-sample then average).

    Args:
        all_preds: (N, T) int — predicted class per frame.
        category_ids: (N,) int — ground truth category per sample.
        avc_labels: (N, T) binary — event presence per frame.
        bg_id: int — background class ID.
        num_workers: number of parallel workers for F1 computation.

    Returns:
        dict with frame_acc, seg_f1, eve_f1, avg.
    """
    N = len(all_preds)
    gt_labels = np.where(avc_labels > 0, category_ids[:, None], bg_id)
    fa = float(np.mean(all_preds == gt_labels))

    chunks = np.array_split(np.arange(N), num_workers)
    args = [
        (all_preds[idx], category_ids[idx], avc_labels[idx], bg_id)
        for idx in chunks
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_score_chunk, args))

    seg_f1 = sum(r[0] for r in results) / N
    eve_f1 = sum(r[1] for r in results) / N
    avg = (fa + seg_f1 + eve_f1) / 3.0

    return {"frame_acc": fa, "seg_f1": seg_f1, "eve_f1": eve_f1, "avg": avg}
