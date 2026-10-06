from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ovavel.data as dataset_utils
from step3_bandit.common import normalize_encoder_name


def _l2_normalize_np(x: np.ndarray, axis: int = -1, eps: float = 1e-6):
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def build_segment_context(audio_embeddings: np.ndarray, visual_embeddings: np.ndarray):
    audio_embeddings = np.asarray(audio_embeddings, dtype=np.float32)
    visual_embeddings = np.asarray(visual_embeddings, dtype=np.float32)
    if audio_embeddings.shape[:2] != visual_embeddings.shape[:2]:
        raise ValueError(
            f"Audio/visual timeline shape mismatch: {audio_embeddings.shape} vs {visual_embeddings.shape}"
        )
    return np.concatenate([visual_embeddings, audio_embeddings], axis=-1).astype(np.float32)
    # visual_norm = _l2_normalize_np(visual_embeddings, axis=-1)
    # audio_norm = _l2_normalize_np(audio_embeddings, axis=-1)
    # return np.concatenate([visual_norm, audio_norm], axis=-1).astype(np.float32)


@dataclass
class SplitData:
    split: str
    encoder: str
    video_ids: list[str]
    cls_types: list[str]
    contexts: torch.Tensor
    audio_embeddings: np.ndarray
    visual_embeddings: np.ndarray
    audio_text_embeddings: np.ndarray
    visual_text_embeddings: np.ndarray
    a_t_sim: np.ndarray
    v_t_sim: np.ndarray
    category_ids: np.ndarray
    avc_labels: np.ndarray
    bg_id: int
    video_dim: int
    audio_dim: int
    sequence_length: int

    def __len__(self):
        return len(self.video_ids)


def load_split_data(
    encoder_name: str,
    split: str,
    embeddings_dir: str = "data/embeddings",
    similarities_dir: str = "data/similarities",
):
    encoder_name = normalize_encoder_name(encoder_name)
    emb = dataset_utils.load_embeddings(encoder_name, split, base_dir=embeddings_dir)
    sim = dataset_utils.load_similarities(encoder_name, split, base_dir=similarities_dir)
    audio_text = dataset_utils.load_text_embeddings(
        encoder_name, split, "audio", base_dir=embeddings_dir
    )
    visual_text = dataset_utils.load_text_embeddings(
        encoder_name, split, "visual", base_dir=embeddings_dir
    )
    meta_df = dataset_utils.load_metadata(split)
    video_ids = [str(v) for v in meta_df["vid_name"].tolist()]
    cls_types = [str(v).strip().lower() for v in meta_df["cls_type"].tolist()]

    audio_embeddings = np.asarray(emb["audio"], dtype=np.float32)
    visual_embeddings = np.asarray(emb["visual"], dtype=np.float32)
    a_t_sim = np.asarray(sim["a_t_sim"], dtype=np.float32)
    v_t_sim = np.asarray(sim["v_t_sim"], dtype=np.float32)
    category_ids = np.asarray(sim["category_ids"], dtype=np.int64)
    avc_labels = np.asarray(sim["avc_labels"], dtype=np.float32)
    audio_text = np.asarray(audio_text, dtype=np.float32)
    visual_text = np.asarray(visual_text, dtype=np.float32)

    num_samples = audio_embeddings.shape[0]
    if len(video_ids) != num_samples:
        raise ValueError(
            f"Metadata/sample mismatch for split={split}: {len(video_ids)} vs {num_samples}"
        )
    if a_t_sim.shape[0] != num_samples or v_t_sim.shape[0] != num_samples:
        raise ValueError(
            f"Similarity/sample mismatch for split={split}: "
            f"{a_t_sim.shape[0]}, {v_t_sim.shape[0]} vs {num_samples}"
        )

    contexts = torch.from_numpy(build_segment_context(audio_embeddings, visual_embeddings))
    return SplitData(
        split=split,
        encoder=encoder_name,
        video_ids=video_ids,
        cls_types=cls_types,
        contexts=contexts,
        audio_embeddings=audio_embeddings,
        visual_embeddings=visual_embeddings,
        audio_text_embeddings=audio_text,
        visual_text_embeddings=visual_text,
        a_t_sim=a_t_sim,
        v_t_sim=v_t_sim,
        category_ids=category_ids,
        avc_labels=avc_labels,
        bg_id=dataset_utils.get_bg_id(split),
        video_dim=int(visual_embeddings.shape[-1]),
        audio_dim=int(audio_embeddings.shape[-1]),
        sequence_length=int(audio_embeddings.shape[1]),
    )
