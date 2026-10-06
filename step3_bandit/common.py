from __future__ import annotations

import importlib.util
import json
import random
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ENCODER_NAME_ALIASES = {
    "imagebind": "imagebind",
    "image_bind": "imagebind",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def suppress_known_pytorch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=(
            r"Converting mask without torch\.bool dtype to bool; "
            r"this will negatively affect performance\..*"
        ),
    )


def normalize_encoder_name(name: str):
    key = str(name).strip().lower()
    if key not in ENCODER_NAME_ALIASES:
        valid = ", ".join(sorted(ENCODER_NAME_ALIASES.keys()))
        raise ValueError(f"Unsupported encoder: {name}. Valid names: {valid}")
    return ENCODER_NAME_ALIASES[key]


def load_formulation(path: str):
    formulation_path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location("step3_formulation", formulation_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Unable to load formulation from {formulation_path}")
    spec.loader.exec_module(module)
    return module


def resolve_project_path(path: str | Path):
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def resolve_step2_result_json_path(formulation_path: str | Path):
    formulation_path = Path(formulation_path).expanduser().resolve()
    match = re.fullmatch(r"f(\d+)", formulation_path.stem)
    if match is None:
        raise ValueError(
            "Unable to infer step2 iteration from formulation name. "
            f"Expected f###.py, got {formulation_path.name}"
        )
    if formulation_path.parent.name != "formulations":
        raise FileNotFoundError(
            "Unable to infer step2 result JSON because formulation is not inside a "
            f"'formulations' directory: {formulation_path}"
        )

    iteration = int(match.group(1))
    result_path = formulation_path.parent.parent / "results" / f"iter_{iteration:03d}.json"
    if not result_path.is_file():
        raise FileNotFoundError(
            f"Missing step2 result JSON for {formulation_path.name}: {result_path}"
        )
    return result_path


def load_step2_result_payload(formulation_path: str | Path):
    result_path = resolve_step2_result_json_path(formulation_path)
    with result_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    formulation_file = payload.get("formulation_file")
    if formulation_file and formulation_file != Path(formulation_path).name:
        raise ValueError(
            "Step2 result JSON does not match formulation file: "
            f"{formulation_file} vs {Path(formulation_path).name}"
        )
    return result_path, payload


def resolve_oracle_artifact_path(step2_result_payload, step2_result_path: str | Path):
    oracle_artifact = step2_result_payload.get("oracle_artifact")
    if not oracle_artifact or not oracle_artifact.get("path"):
        raise ValueError(
            f"Step2 result JSON does not contain oracle_artifact metadata: {step2_result_path}"
        )

    artifact_path = Path(oracle_artifact["path"]).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = resolve_project_path(artifact_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Missing oracle artifact NPZ: {artifact_path}")
    return artifact_path


def _normalize_video_id(raw_value):
    if isinstance(raw_value, bytes):
        return raw_value.decode("utf-8")
    return str(raw_value)


def load_oracle_targets_by_video_id(
    *,
    step2_result_payload,
    step2_result_path: str | Path,
    encoder_name: str,
    split_name: str,
    video_ids,
    expected_num_params: int | None = None,
):
    artifact_path = resolve_oracle_artifact_path(step2_result_payload, step2_result_path)
    oracle_artifact = step2_result_payload["oracle_artifact"]
    key_map = oracle_artifact.get("key_map", {})

    split_key_map = key_map.get(encoder_name, {}).get(split_name, {})
    param_key = split_key_map.get("best_params", f"{encoder_name}__{split_name}__best_params")
    video_ids_key = split_key_map.get("video_ids", f"{encoder_name}__{split_name}__video_ids")

    with np.load(artifact_path, allow_pickle=False) as arrays:
        if param_key not in arrays or video_ids_key not in arrays:
            available = ", ".join(sorted(arrays.files))
            raise KeyError(
                "Required oracle target arrays are missing from artifact. "
                f"encoder={encoder_name} split={split_name} "
                f"param_key={param_key} video_ids_key={video_ids_key} "
                f"available=[{available}]"
            )
        params = np.asarray(arrays[param_key], dtype=np.float32)
        source_video_ids = [_normalize_video_id(v) for v in arrays[video_ids_key].tolist()]

    if params.ndim != 2:
        raise ValueError(f"Oracle params must be 2D, got shape {params.shape}")
    if len(source_video_ids) != params.shape[0]:
        raise ValueError(
            "Oracle artifact video_ids/params length mismatch: "
            f"{len(source_video_ids)} vs {params.shape[0]}"
        )
    if expected_num_params is not None and params.shape[1] != expected_num_params:
        raise ValueError(
            "Oracle artifact param dim does not match formulation: "
            f"{params.shape[1]} vs {expected_num_params}"
        )

    by_video_id = {}
    for idx, video_id in enumerate(source_video_ids):
        if video_id in by_video_id:
            raise ValueError(f"Duplicate video_id in oracle artifact: {video_id}")
        by_video_id[video_id] = params[idx]

    ordered_targets = []
    missing_video_ids = []
    for video_id in video_ids:
        normalized_id = _normalize_video_id(video_id)
        if normalized_id not in by_video_id:
            missing_video_ids.append(normalized_id)
            continue
        ordered_targets.append(by_video_id[normalized_id])

    if missing_video_ids:
        preview = ", ".join(missing_video_ids[:5])
        raise ValueError(
            "Missing oracle targets for requested video_ids "
            f"(showing up to 5): {preview}"
        )

    return np.stack(ordered_targets, axis=0).astype(np.float32), artifact_path


def get_formulation_metadata(formulation):
    param_ranges = getattr(formulation, "PARAM_RANGES", None)
    if param_ranges is None:
        raise ValueError("Formulation is missing PARAM_RANGES")

    num_params = int(getattr(formulation, "NUM_PARAMS", len(param_ranges)))
    if num_params != len(param_ranges):
        raise ValueError(
            f"NUM_PARAMS ({num_params}) does not match PARAM_RANGES ({len(param_ranges)})"
        )

    param_names = list(getattr(formulation, "PARAM_NAMES", [f"p{i}" for i in range(num_params)]))
    if len(param_names) != num_params:
        raise ValueError(
            f"PARAM_NAMES ({len(param_names)}) does not match NUM_PARAMS ({num_params})"
        )

    normalized_ranges = []
    for idx, bounds in enumerate(param_ranges):
        if len(bounds) != 2:
            raise ValueError(f"Invalid PARAM_RANGES[{idx}]: {bounds}")
        low = float(bounds[0])
        high = float(bounds[1])
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError(f"PARAM_RANGES[{idx}] must be finite, got {bounds}")
        if high <= low:
            raise ValueError(f"PARAM_RANGES[{idx}] must satisfy low < high, got {bounds}")
        normalized_ranges.append((low, high))

    return {
        "num_params": num_params,
        "param_names": param_names,
        "param_ranges": normalized_ranges,
    }


def map_action_to_ranges(raw_action: torch.Tensor, param_ranges: list[tuple[float, float]]):
    squeeze_output = False
    if raw_action.dim() == 1:
        raw_action = raw_action.unsqueeze(0)
        squeeze_output = True

    lows = torch.tensor(
        [bounds[0] for bounds in param_ranges],
        dtype=raw_action.dtype,
        device=raw_action.device,
    )
    highs = torch.tensor(
        [bounds[1] for bounds in param_ranges],
        dtype=raw_action.dtype,
        device=raw_action.device,
    )
    mapped = lows.unsqueeze(0) + torch.sigmoid(raw_action) * (highs - lows).unsqueeze(0)
    if squeeze_output:
        return mapped.squeeze(0)
    return mapped


def params_row_to_dict(param_row, param_names):
    if isinstance(param_row, torch.Tensor):
        param_row = param_row.detach().cpu().tolist()
    return {name: float(value) for name, value in zip(param_names, param_row)}


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_run_dir(save_dir: str, run_name: str):
    root = Path(save_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_id = run_name.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: str | Path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str | Path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
