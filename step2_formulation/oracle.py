"""GPU-parallel oracle optimization.

For each sample, exhaustively search a Sobol parameter grid to find the
parameters that maximize frame accuracy under the given formulation.

Usage:
    from step2_formulation.oracle import run_oracle
    results = run_oracle(formulation_module, sim_data, emb_data, config)
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ovavel.metrics as metrics_module
from ovavel.predict import predict_and_gpu


def load_formulation(path):
    """Dynamically load a formulation module from file path."""
    spec = importlib.util.spec_from_file_location("formulation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_sobol_grid(param_ranges, num_candidates):
    """Generate quasi-random parameter candidates via Sobol sequence.

    Args:
        param_ranges: list of (low, high) tuples, length P.
        num_candidates: number of candidates to generate.

    Returns:
        (num_candidates, P) numpy array.
    """
    P = len(param_ranges)
    sampler = qmc.Sobol(d=P, scramble=True, seed=42)
    m = int(np.ceil(np.log2(num_candidates)))
    samples = sampler.random_base2(m)[:num_candidates]

    lows = np.array([r[0] for r in param_ranges])
    highs = np.array([r[1] for r in param_ranges])
    return (samples * (highs - lows) + lows).astype(np.float32)


def _score_predictions(pred, gt_i, category_id, avc_label, bg_id, metric):
    """Score a batch of predictions for one sample.

    Args:
        pred: (B, T) tensor on GPU.
        gt_i: (T,) tensor on GPU (frame-level ground truth).
        category_id: int.
        avc_label: (T,) numpy array.
        bg_id: int.
        metric: 'frame_acc' | 'seg_f1' | 'eve_f1' | 'avg'.

    Returns:
        scores: (B,) numpy array of float scores.
    """
    B = pred.shape[0]

    if metric == "frame_acc":
        # Fast GPU path
        return (pred == gt_i.unsqueeze(0)).float().mean(dim=1).cpu().numpy()

    if metric == "fa_seg":
        # Fast GPU path: 0.5*(frame_acc + segment_macro_f1), no eve_f1 / CPU pool.
        return metrics_module.frame_seg_score_torch(pred, gt_i, bg_id + 1)

    # Slow CPU path for seg_f1, eve_f1, avg
    pred_np = pred.cpu().numpy()
    scores = np.zeros(B, dtype=np.float32)
    for b in range(B):
        m = metrics_module.evaluate_sample(pred_np[b], category_id, avc_label, bg_id)
        if metric == "avg":
            scores[b] = m["avg"]
        else:
            scores[b] = m[metric]
    return scores


def _global_oracle_worker(gpu_id, sample_indices, data_dict, formulation_path,
                          param_grid_np, chunk_size, bg_id, result_dict):
    """Worker process: accumulate per-candidate frame_acc sum across assigned
    samples (for finding the single best GLOBAL parameter vector)."""
    device = torch.device(f'cuda:{gpu_id}')

    formulation = load_formulation(formulation_path)
    param_grid = torch.from_numpy(param_grid_np).float().to(device)
    B = param_grid.shape[0]

    a_t_sim = data_dict["a_t_sim"]
    v_t_sim = data_dict["v_t_sim"]
    gt_labels = data_dict["gt_labels"]

    a_emb = data_dict.get("a_emb")
    v_emb = data_dict.get("v_emb")
    a_t_emb = data_dict.get("a_t_emb")
    v_t_emb = data_dict.get("v_t_emb")

    a_t_emb_gpu = torch.from_numpy(a_t_emb).float().to(device) if a_t_emb is not None else None
    v_t_emb_gpu = torch.from_numpy(v_t_emb).float().to(device) if v_t_emb is not None else None

    # Per-candidate accumulators
    candidate_score_sum = torch.zeros(B, dtype=torch.float64, device=device)
    n_processed = 0

    for i in sample_indices:
        a_sim = torch.from_numpy(a_t_sim[i]).float().to(device)
        v_sim = torch.from_numpy(v_t_sim[i]).float().to(device)
        gt_i = torch.from_numpy(gt_labels[i]).long().to(device)

        a_emb_i = torch.from_numpy(a_emb[i]).float().to(device) if a_emb is not None else None
        v_emb_i = torch.from_numpy(v_emb[i]).float().to(device) if v_emb is not None else None

        for cs in range(0, B, chunk_size):
            ce = min(cs + chunk_size, B)
            params_chunk = param_grid[cs:ce]

            with torch.no_grad():
                a_ths, v_ths = formulation.params_to_thresholds_batch(
                    a_emb_i, v_emb_i, a_t_emb_gpu, v_t_emb_gpu, a_sim, v_sim, params_chunk
                )
                pred = predict_and_gpu(a_sim, v_sim, a_ths, v_ths, bg_id)
                # Frame accuracy per candidate (no CPU transfer)
                scores_gpu = (pred == gt_i.unsqueeze(0)).float().mean(dim=1)
                candidate_score_sum[cs:ce] += scores_gpu.double()

        n_processed += 1

    result_dict[gpu_id] = (
        candidate_score_sum.cpu().numpy(),
        n_processed,
    )


def run_global_oracle(formulation_path, sim_data, emb_data, bg_id,
                      num_candidates=100_000, chunk_size=200_000, num_gpus=4):
    """Find the SINGLE BEST parameter vector that maximizes mean frame_acc
    across all training samples (global aggregation, not per-sample best).

    Returns:
        dict with:
            'best_global_params': (P,) array — single best parameter vector.
            'best_global_frame_acc': float — its mean frame_acc across samples.
            'best_global_metrics': dict (frame_acc, seg_f1, eve_f1, avg) under that param.
            'elapsed': float, seconds.
    """
    formulation = load_formulation(formulation_path)
    param_grid = generate_sobol_grid(formulation.PARAM_RANGES, num_candidates)
    print(f"  Global Sobol grid: {param_grid.shape} candidates, {formulation.NUM_PARAMS} params")

    a_t_sim = sim_data["a_t_sim"]
    category_ids = sim_data["category_ids"]
    avc_labels = sim_data["avc_labels"]
    N, T, C = a_t_sim.shape
    gt_labels = np.where(avc_labels > 0, category_ids[:, None], bg_id).astype(np.int64)

    data_dict = {
        "a_t_sim": a_t_sim,
        "v_t_sim": sim_data["v_t_sim"],
        "gt_labels": gt_labels,
        "category_ids": category_ids,
        "avc_labels": avc_labels,
        "a_emb": emb_data.get("audio") if emb_data else None,
        "v_emb": emb_data.get("visual") if emb_data else None,
        "a_t_emb": emb_data.get("audio_text") if emb_data else None,
        "v_t_emb": emb_data.get("visual_text") if emb_data else None,
    }

    all_indices = list(range(N))
    chunks = np.array_split(all_indices, num_gpus)

    t0 = time.time()

    if num_gpus <= 1:
        result_dict = {}
        _global_oracle_worker(0, all_indices, data_dict, formulation_path,
                              param_grid, chunk_size, bg_id, result_dict)
    else:
        manager = mp.Manager()
        result_dict = manager.dict()
        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=_global_oracle_worker,
                args=(gpu_id, chunks[gpu_id].tolist(), data_dict,
                      formulation_path, param_grid, chunk_size, bg_id,
                      result_dict),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    # Aggregate
    total_sum = np.zeros(param_grid.shape[0], dtype=np.float64)
    total_n = 0
    for gpu_id in range(num_gpus if num_gpus > 1 else 1):
        scores_sum, n_proc = result_dict[gpu_id]
        total_sum += scores_sum
        total_n += n_proc
    candidate_mean = total_sum / total_n
    best_idx = int(np.argmax(candidate_mean))
    best_global_params = param_grid[best_idx].copy()
    best_global_frame_acc = float(candidate_mean[best_idx])

    elapsed = time.time() - t0

    # Compute full metrics under the best global param (apply to all samples)
    device = torch.device('cuda:0')
    formulation_loaded = load_formulation(formulation_path)
    a_t_emb = emb_data.get("audio_text") if emb_data else None
    v_t_emb = emb_data.get("visual_text") if emb_data else None
    a_t_emb_gpu = torch.from_numpy(a_t_emb).float().to(device) if a_t_emb is not None else None
    v_t_emb_gpu = torch.from_numpy(v_t_emb).float().to(device) if v_t_emb is not None else None
    best_param_gpu = torch.from_numpy(best_global_params).float().to(device).unsqueeze(0)  # (1, P)

    all_preds = np.zeros((N, T), dtype=np.int64)
    a_emb = emb_data.get("audio") if emb_data else None
    v_emb = emb_data.get("visual") if emb_data else None
    for i in range(N):
        a_sim = torch.from_numpy(a_t_sim[i]).float().to(device)
        v_sim = torch.from_numpy(sim_data["v_t_sim"][i]).float().to(device)
        a_emb_i = torch.from_numpy(a_emb[i]).float().to(device) if a_emb is not None else None
        v_emb_i = torch.from_numpy(v_emb[i]).float().to(device) if v_emb is not None else None
        with torch.no_grad():
            a_ths, v_ths = formulation_loaded.params_to_thresholds_batch(
                a_emb_i, v_emb_i, a_t_emb_gpu, v_t_emb_gpu, a_sim, v_sim, best_param_gpu
            )
            pred = predict_and_gpu(a_sim, v_sim, a_ths, v_ths, bg_id)
        all_preds[i] = pred[0].cpu().numpy()

    full_metrics = metrics_module.evaluate_dataset(
        all_preds, category_ids, avc_labels, bg_id
    )

    print(f"  Global Oracle: frame_acc={full_metrics['frame_acc']:.4f} "
          f"seg_f1={full_metrics['seg_f1']:.4f} "
          f"eve_f1={full_metrics['eve_f1']:.4f} "
          f"avg={full_metrics['avg']:.4f} ({elapsed:.1f}s)")

    return {
        "best_global_params": best_global_params.tolist(),
        "best_global_frame_acc": best_global_frame_acc,
        "best_global_metrics": full_metrics,
        "best_global_preds": all_preds,
        "elapsed": elapsed,
    }


def _oracle_worker(gpu_id, sample_indices, data_dict, formulation_path,
                   param_grid_np, chunk_size, bg_id, oracle_metric, result_dict):
    """Worker process: find oracle params for assigned samples on one GPU."""
    device = torch.device(f'cuda:{gpu_id}')

    formulation = load_formulation(formulation_path)
    param_grid = torch.from_numpy(param_grid_np).float().to(device)
    B = param_grid.shape[0]

    a_t_sim = data_dict["a_t_sim"]
    v_t_sim = data_dict["v_t_sim"]
    gt_labels = data_dict["gt_labels"]
    category_ids = data_dict["category_ids"]
    avc_labels = data_dict["avc_labels"]

    a_emb = data_dict.get("a_emb")
    v_emb = data_dict.get("v_emb")
    a_t_emb = data_dict.get("a_t_emb")
    v_t_emb = data_dict.get("v_t_emb")

    a_t_emb_gpu = torch.from_numpy(a_t_emb).float().to(device) if a_t_emb is not None else None
    v_t_emb_gpu = torch.from_numpy(v_t_emb).float().to(device) if v_t_emb is not None else None

    best_params_local = {}
    best_preds_local = {}
    best_scores_local = {}

    for i in sample_indices:
        a_sim = torch.from_numpy(a_t_sim[i]).float().to(device)
        v_sim = torch.from_numpy(v_t_sim[i]).float().to(device)
        gt_i = torch.from_numpy(gt_labels[i]).long().to(device)

        a_emb_i = torch.from_numpy(a_emb[i]).float().to(device) if a_emb is not None else None
        v_emb_i = torch.from_numpy(v_emb[i]).float().to(device) if v_emb is not None else None

        best_score = -1.0
        best_idx = 0
        best_pred = None

        for cs in range(0, B, chunk_size):
            ce = min(cs + chunk_size, B)
            params_chunk = param_grid[cs:ce]

            with torch.no_grad():
                a_ths, v_ths = formulation.params_to_thresholds_batch(
                    a_emb_i, v_emb_i, a_t_emb_gpu, v_t_emb_gpu, a_sim, v_sim, params_chunk
                )
                pred = predict_and_gpu(a_sim, v_sim, a_ths, v_ths, bg_id)

            scores = _score_predictions(
                pred, gt_i, category_ids[i], avc_labels[i], bg_id, oracle_metric
            )
            chunk_best_idx = scores.argmax()
            chunk_best_score = scores[chunk_best_idx]

            if chunk_best_score > best_score:
                best_score = chunk_best_score
                best_idx = cs + chunk_best_idx
                best_pred = pred[chunk_best_idx].cpu().numpy()

        best_params_local[i] = param_grid_np[best_idx]
        best_preds_local[i] = best_pred
        best_scores_local[i] = float(best_score)

    result_dict[gpu_id] = (best_params_local, best_preds_local, best_scores_local)


def run_oracle(formulation_path, sim_data, emb_data, bg_id,
               num_candidates=5_000_000, chunk_size=200_000, num_gpus=4,
               oracle_metric="frame_acc"):
    """Run oracle optimization across all samples.

    Args:
        formulation_path: path to formulation .py file.
        sim_data: dict with 'a_t_sim' (N,T,C), 'v_t_sim', 'category_ids', 'avc_labels'.
        emb_data: dict with 'audio' (N,T,D), 'visual' (N,T,D), 'text' (C,D) or None.
        bg_id: background class ID.
        num_candidates: Sobol grid size.
        chunk_size: GPU batch size for candidates.
        num_gpus: number of GPUs to use.
        oracle_metric: 'frame_acc' | 'seg_f1' | 'eve_f1' | 'avg'.

    Returns:
        dict with:
            'best_params': (N, P) array of oracle parameters.
            'best_preds': (N, T) array of oracle predictions.
            'oracle_metrics': dict with frame_acc, seg_f1, eve_f1, avg.
            'param_stats': dict of per-param mean/std/min/max.
            'elapsed': float, seconds.
    """
    formulation = load_formulation(formulation_path)

    param_grid = generate_sobol_grid(formulation.PARAM_RANGES, num_candidates)
    print(f"  Sobol grid: {param_grid.shape} candidates, {formulation.NUM_PARAMS} params")
    print(f"  Oracle metric: {oracle_metric}")

    a_t_sim = sim_data["a_t_sim"]
    category_ids = sim_data["category_ids"]
    avc_labels = sim_data["avc_labels"]
    N, T, C = a_t_sim.shape
    gt_labels = np.where(avc_labels > 0, category_ids[:, None], bg_id).astype(np.int64)

    data_dict = {
        "a_t_sim": a_t_sim,
        "v_t_sim": sim_data["v_t_sim"],
        "gt_labels": gt_labels,
        "category_ids": category_ids,
        "avc_labels": avc_labels,
        "a_emb": emb_data.get("audio") if emb_data else None,
        "v_emb": emb_data.get("visual") if emb_data else None,
        "a_t_emb": emb_data.get("audio_text") if emb_data else None,
        "v_t_emb": emb_data.get("visual_text") if emb_data else None,
    }

    all_indices = list(range(N))
    chunks = np.array_split(all_indices, num_gpus)

    t0 = time.time()

    if num_gpus <= 1:
        result_dict = {}
        _oracle_worker(0, all_indices, data_dict, formulation_path,
                       param_grid, chunk_size, bg_id, oracle_metric, result_dict)
    else:
        manager = mp.Manager()
        result_dict = manager.dict()
        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=_oracle_worker,
                args=(gpu_id, chunks[gpu_id].tolist(), data_dict,
                      formulation_path, param_grid, chunk_size, bg_id,
                      oracle_metric, result_dict),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    elapsed = time.time() - t0

    # Reassemble
    best_params = np.zeros((N, formulation.NUM_PARAMS), dtype=np.float32)
    best_preds = np.zeros((N, T), dtype=np.int64)
    per_sample_best_scores = np.zeros(N, dtype=np.float32)

    for gpu_id in range(num_gpus if num_gpus > 1 else 1):
        params_dict, preds_dict, scores_dict = result_dict[gpu_id]
        for i in params_dict:
            best_params[i] = params_dict[i]
            best_preds[i] = preds_dict[i]
            per_sample_best_scores[i] = scores_dict[i]

    # Compute all three metrics on oracle predictions
    oracle_metrics = metrics_module.evaluate_dataset(
        best_preds, category_ids, avc_labels, bg_id
    )

    param_names = getattr(formulation, 'PARAM_NAMES', [f'p{i}' for i in range(formulation.NUM_PARAMS)])
    param_stats = {}
    for j, name in enumerate(param_names):
        param_stats[name] = {
            "mean": float(best_params[:, j].mean()),
            "std": float(best_params[:, j].std()),
            "min": float(best_params[:, j].min()),
            "max": float(best_params[:, j].max()),
        }

    print(f"  Oracle: frame_acc={oracle_metrics['frame_acc']:.4f} "
          f"seg_f1={oracle_metrics['seg_f1']:.4f} "
          f"eve_f1={oracle_metrics['eve_f1']:.4f} "
          f"avg={oracle_metrics['avg']:.4f} ({elapsed:.1f}s)")

    return {
        "best_params": best_params,
        "best_preds": best_preds,
        "per_sample_best_scores": per_sample_best_scores,
        "oracle_metrics": oracle_metrics,
        "param_stats": param_stats,
        "elapsed": elapsed,
    }


def generate_ablation_grid(param_ranges, params_zero, num_candidates, seed=42):
    """Generate a Sobol grid with specified params forced to 0.0.

    Sobol-samples the (P - len(params_zero))-dimensional free subspace within the
    corresponding entries of param_ranges, and inserts 0.0 at the zero-pinned
    indices.
    """
    P = len(param_ranges)
    zero_set = set(int(i) for i in params_zero)
    free_dims = [i for i in range(P) if i not in zero_set]
    d_free = len(free_dims)
    if d_free == 0:
        return np.zeros((1, P), dtype=np.float32)
    n = max(int(num_candidates), 2)
    m = int(np.ceil(np.log2(n)))
    sampler = qmc.Sobol(d=d_free, scramble=True, seed=seed)
    samples = sampler.random_base2(m)[:n]
    grid = np.zeros((n, P), dtype=np.float32)
    for j, dim in enumerate(free_dims):
        low, high = param_ranges[dim]
        grid[:, dim] = (samples[:, j] * (high - low) + low).astype(np.float32)
    return grid


def run_ablation(formulation_path, sim_data, emb_data, bg_id,
                 param_grid_np, chunk_size=200_000, num_gpus=1):
    """Run per-sample best frame_acc on a precomputed param grid.

    Used by the evaluator pipeline to score zero-pinned ablation subsets.
    Returns only per-sample best frame_acc and elapsed seconds — no aggregated
    metrics, no param_stats. The caller compares these to the full-grid
    per_sample_best_scores to compute marginal value distributions.
    """
    a_t_sim = sim_data["a_t_sim"]
    category_ids = sim_data["category_ids"]
    avc_labels = sim_data["avc_labels"]
    N = a_t_sim.shape[0]
    gt_labels = np.where(avc_labels > 0, category_ids[:, None], bg_id).astype(np.int64)

    data_dict = {
        "a_t_sim": a_t_sim,
        "v_t_sim": sim_data["v_t_sim"],
        "gt_labels": gt_labels,
        "category_ids": category_ids,
        "avc_labels": avc_labels,
        "a_emb": emb_data.get("audio") if emb_data else None,
        "v_emb": emb_data.get("visual") if emb_data else None,
        "a_t_emb": emb_data.get("audio_text") if emb_data else None,
        "v_t_emb": emb_data.get("visual_text") if emb_data else None,
    }

    all_indices = list(range(N))
    chunks = np.array_split(all_indices, num_gpus)

    t0 = time.time()
    if num_gpus <= 1:
        result_dict = {}
        _oracle_worker(0, all_indices, data_dict, formulation_path,
                       param_grid_np, chunk_size, bg_id, "frame_acc", result_dict)
    else:
        manager = mp.Manager()
        result_dict = manager.dict()
        procs = []
        for gid in range(num_gpus):
            p = mp.Process(
                target=_oracle_worker,
                args=(gid, chunks[gid].tolist(), data_dict,
                      formulation_path, param_grid_np, chunk_size, bg_id,
                      "frame_acc", result_dict),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    elapsed = time.time() - t0

    per_sample_scores = np.zeros(N, dtype=np.float32)
    for gid in range(num_gpus if num_gpus > 1 else 1):
        _, _, scores_local = result_dict[gid]
        for i, s in scores_local.items():
            per_sample_scores[i] = s
    return per_sample_scores, elapsed
