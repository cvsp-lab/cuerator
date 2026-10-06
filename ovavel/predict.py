"""Prediction rules for converting thresholds to class predictions.

Prediction rules are decoupled from formulations: formulations produce
thresholds, prediction rules consume them to produce per-frame class labels.
"""

import numpy as np
import torch


def predict_and(a_sim, v_sim, a_thresh, v_thresh, bg_id):
    """Strict AND top-1 prediction rule (A-method).

    For each frame:
    1. Find audio top-1 among categories exceeding audio threshold.
    2. Find visual top-1 among categories exceeding visual threshold.
    3. If both agree on the same category -> that category.
    4. Otherwise (disagree or no category passes) -> background.

    Args:
        a_sim: (T, C) audio-text similarities.
        v_sim: (T, C) visual-text similarities.
        a_thresh: (T, C) audio thresholds.
        v_thresh: (T, C) visual thresholds.
        bg_id: int — background class ID.

    Returns:
        pred: (T,) int — predicted class per frame.
    """
    T, C = a_sim.shape
    pred = np.full(T, bg_id, dtype=np.int64)

    a_pass = a_sim > a_thresh  # (T, C)
    v_pass = v_sim > v_thresh  # (T, C)

    for t in range(T):
        a_candidates = np.where(a_pass[t])[0]
        v_candidates = np.where(v_pass[t])[0]

        if len(a_candidates) == 0 or len(v_candidates) == 0:
            continue

        a_top1 = a_candidates[np.argmax(a_sim[t, a_candidates])]
        v_top1 = v_candidates[np.argmax(v_sim[t, v_candidates])]

        if a_top1 == v_top1:
            pred[t] = a_top1

    return pred


def predict_and_batch(a_sim, v_sim, a_thresh, v_thresh, bg_id):
    """Batched version of predict_and for oracle evaluation.

    Args:
        a_sim: (T, C) audio-text similarities (single sample).
        v_sim: (T, C) visual-text similarities (single sample).
        a_thresh: (B, T, C) audio thresholds (B candidates).
        v_thresh: (B, T, C) visual thresholds (B candidates).
        bg_id: int — background class ID.

    Returns:
        pred: (B, T) int — predicted class per frame per candidate.
    """
    B, T, C = a_thresh.shape
    pred = np.full((B, T), bg_id, dtype=np.int64)

    # Broadcast sim to (B, T, C)
    a_sim_b = np.broadcast_to(a_sim[None], (B, T, C))
    v_sim_b = np.broadcast_to(v_sim[None], (B, T, C))

    a_pass = a_sim_b > a_thresh  # (B, T, C)
    v_pass = v_sim_b > v_thresh  # (B, T, C)

    # Mask similarities: set non-passing to -inf
    a_masked = np.where(a_pass, a_sim_b, -np.inf)  # (B, T, C)
    v_masked = np.where(v_pass, v_sim_b, -np.inf)  # (B, T, C)

    a_top1 = np.argmax(a_masked, axis=-1)  # (B, T)
    v_top1 = np.argmax(v_masked, axis=-1)  # (B, T)

    # Check that at least one category passes in each modality
    a_any = np.any(a_pass, axis=-1)  # (B, T)
    v_any = np.any(v_pass, axis=-1)  # (B, T)

    # Agreement mask
    agree = (a_top1 == v_top1) & a_any & v_any  # (B, T)
    pred[agree] = a_top1[agree]

    return pred


def predict_and_gpu(a_sim, v_sim, a_thresh, v_thresh, bg_id):
    """GPU-accelerated batched AND top-1 prediction (A-method).

    Args:
        a_sim: (T, C) tensor on GPU.
        v_sim: (T, C) tensor on GPU.
        a_thresh: (B, T, C) tensor on GPU.
        v_thresh: (B, T, C) tensor on GPU.
        bg_id: int — background class ID.

    Returns:
        pred: (B, T) tensor of int64 class IDs on GPU.
    """
    B, T, C = a_thresh.shape

    a_pass = a_sim.unsqueeze(0) > a_thresh  # (B, T, C)
    v_pass = v_sim.unsqueeze(0) > v_thresh

    a_masked = torch.where(a_pass, a_sim.unsqueeze(0).expand(B, T, C),
                           torch.tensor(-float('inf'), device=a_thresh.device))
    v_masked = torch.where(v_pass, v_sim.unsqueeze(0).expand(B, T, C),
                           torch.tensor(-float('inf'), device=v_thresh.device))

    a_top1 = a_masked.argmax(dim=-1)  # (B, T)
    v_top1 = v_masked.argmax(dim=-1)

    a_any = a_pass.any(dim=-1)  # (B, T)
    v_any = v_pass.any(dim=-1)

    agree = (a_top1 == v_top1) & a_any & v_any
    pred = torch.full((B, T), bg_id, dtype=torch.long, device=a_thresh.device)
    pred[agree] = a_top1[agree]

    return pred
