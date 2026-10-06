import torch


NAME = "Blended Contenders With Sustained Burst Control"
DESCRIPTION = "Each modality keeps a simple opposite-modality cross-check, but replaces raw same-frame self-rivalry with a contender term on blended own-plus-shared support. The temporal slots are separated into clip-level sustained-support rivalry and a signed burstiness term that penalizes frame spikes unsupported by persistent evidence."
NUM_PARAMS = 10
PARAM_RANGES = [
    (-0.1, 0.5),
    (0.0, 2.0),
    (0.0, 2.0),
    (0.0, 1.8),
    (0.0, 1.8),
    (0.0, 0.8),
    (0.0, 2.1),
    (0.0, 2.0),
    (0.0, 2.0),
    (0.0, 1.8),
]
PARAM_NAMES = [
    "a_bias",
    "a_blended_contender_penalty",
    "a_visual_cross_check_penalty",
    "a_sustained_support_penalty",
    "a_burstiness_penalty",
    "v_bias",
    "v_blended_contender_penalty",
    "v_sustained_support_penalty",
    "v_burstiness_penalty",
    "v_audio_cross_check_penalty",
]


def _shared_support(a_sim, v_sim):
    return torch.minimum(a_sim, v_sim)


def _rival_gap(sim):
    top2 = sim.topk(2, dim=-1)
    top1 = top2.values[..., :1]
    top2v = top2.values[..., 1:2]
    winner = torch.nn.functional.one_hot(top2.indices[..., 0], num_classes=sim.shape[-1]).to(sim.dtype)
    return top1 + (top2v - top1) * winner - sim


def _temporal_rival(sim):
    clip = sim.mean(dim=-2, keepdim=True)
    return _rival_gap(clip)


def _burst_gap(sim, support):
    sustain = support.mean(dim=-2, keepdim=True)
    return sim - sustain


def params_to_thresholds(a_emb, v_emb, a_t_emb, v_t_emb, a_sim, v_sim, params):
    """Single-sample version (correctness reference).

    Args:
        a_emb: (T, D) audio embeddings
        v_emb: (T, D) visual embeddings
        a_t_emb: (C, D) text embeddings aligned with audio space
        v_t_emb: (C, D) text embeddings aligned with visual space
        a_sim: (T, C) audio-text cosine similarities
        v_sim: (T, C) visual-text cosine similarities
        params: (P,) tensor of parameters

    Returns:
        a_thresh: (T, C) audio thresholds
        v_thresh: (T, C) visual thresholds
    """
    joint = _shared_support(a_sim, v_sim)
    a_support = 0.5 * (a_sim + joint)
    v_support = 0.5 * (v_sim + joint)
    a_contend = _rival_gap(a_support)
    a_cross = _rival_gap(v_sim)
    a_sustain = _temporal_rival(a_support)
    a_burst = _burst_gap(a_sim, a_support)
    v_contend = _rival_gap(v_support)
    v_sustain = _temporal_rival(v_support)
    v_burst = _burst_gap(v_sim, v_support)
    v_cross = _rival_gap(a_sim)
    a_thresh = params[0] + params[1] * a_contend + params[2] * a_cross + params[3] * a_sustain + params[4] * a_burst
    v_thresh = params[5] + params[6] * v_contend + params[7] * v_sustain + params[8] * v_burst + params[9] * v_cross
    return a_thresh, v_thresh


def params_to_thresholds_batch(a_emb, v_emb, a_t_emb, v_t_emb, a_sim, v_sim, params):
    """Batched version for speed (must match single version exactly).

    Args:
        params: (B, P) tensor of B candidate parameter sets

    Returns:
        a_thresh: (B, T, C) audio thresholds
        v_thresh: (B, T, C) visual thresholds
    """
    joint = _shared_support(a_sim, v_sim)
    a_support = 0.5 * (a_sim + joint)
    v_support = 0.5 * (v_sim + joint)
    a_contend = _rival_gap(a_support)
    a_cross = _rival_gap(v_sim)
    a_sustain = _temporal_rival(a_support)
    a_burst = _burst_gap(a_sim, a_support)
    v_contend = _rival_gap(v_support)
    v_sustain = _temporal_rival(v_support)
    v_burst = _burst_gap(v_sim, v_support)
    v_cross = _rival_gap(a_sim)
    p = params.unsqueeze(-1).unsqueeze(-1)
    a_thresh = p[:, 0] + p[:, 1] * a_contend + p[:, 2] * a_cross + p[:, 3] * a_sustain + p[:, 4] * a_burst
    v_thresh = p[:, 5] + p[:, 6] * v_contend + p[:, 7] * v_sustain + p[:, 8] * v_burst + p[:, 9] * v_cross
    return a_thresh, v_thresh
