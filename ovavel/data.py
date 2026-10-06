"""Shared data loading utilities for OV-AVEL.

Handles loading of pre-extracted embeddings, precomputed similarities,
dataset metadata, and category mappings.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataset root and derived paths
# ---------------------------------------------------------------------------

def _root():
    val = os.environ.get("OVAVEL_ROOT")
    if val is None:
        raise EnvironmentError("Environment variable OVAVEL_ROOT is not set. "
                               "Source env.sh before running.")
    return Path(val)


def _data_dir():
    return _root() / "ovave_dataset_preprocessed"

def _meta_dir():
    return _root() / "meta_anno_files"

def _meta_csv():
    return _meta_dir() / "ovave_dataset_meta.csv"

def _anno_json():
    return _meta_dir() / "released_ovavel_dataset_anno.json"

def _train_categories_csv():
    return _meta_dir() / "ovave_train_close_categories.csv"

def _total_categories_csv():
    return _meta_dir() / "ovave_total_close_open_categories.csv"

def _categories_csv(split):
    return _train_categories_csv() if split == "train" else _total_categories_csv()


# ---------------------------------------------------------------------------
# Category loading
# ---------------------------------------------------------------------------

def load_categories(split):
    """Load category names for a split (excluding background 'other').

    Args:
        split: 'train', 'val', or 'test'.

    Returns:
        list of category name strings.
    """
    df = pd.read_csv(_categories_csv(split), header=None)
    return [v for v in df.iloc[:, 0] if str(v).lower() != "other"]


def get_num_classes(split):
    """Get number of foreground classes for a split."""
    return len(load_categories(split))


def get_bg_id(split):
    """Get background class ID for a split (= num foreground classes)."""
    return get_num_classes(split)


# ---------------------------------------------------------------------------
# Metadata & annotations
# ---------------------------------------------------------------------------

def load_metadata(split):
    """Load dataset metadata for a split.

    Returns:
        DataFrame with columns: split, cls_name, cls_type, vid_name
    """
    df = pd.read_csv(_meta_csv())
    return df[df["split"] == split].reset_index(drop=True)


def load_annotations():
    """Load temporal event annotations.

    Returns:
        dict: {vid_name: {"category": str, "label": list[int]}}
    """
    with open(_anno_json()) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def load_embeddings(encoder_name, split, base_dir="data/embeddings"):
    """Load pre-extracted embeddings for a split.

    Args:
        encoder_name: e.g. 'imagebind'.
        split: 'train', 'val', or 'test'.
        base_dir: root directory for embeddings.

    Returns:
        dict with 'audio' (N, T, D), 'visual' (N, T, D).
    """
    path = Path(base_dir) / encoder_name / f"{split}.npz"
    data = np.load(path)
    return {"audio": data["audio"], "visual": data["visual"]}


def load_text_embeddings(encoder_name, split, modality, base_dir="data/embeddings"):
    """Load text embeddings for a specific modality (excluding background).

    For shared-space encoders (ImageBind), both modalities return the same
    embeddings from text.npy. For separate-space encoders (CLIP+CLAP),
    audio_text.npy and visual_text.npy are used respectively.

    Args:
        encoder_name: e.g. 'imagebind'.
        split: 'train', 'val', or 'test'.
        modality: 'audio' or 'visual'.
        base_dir: root directory for embeddings.

    Returns:
        np.ndarray of shape (C, D) where C excludes the background class.
    """
    text_key = "train" if split == "train" else "test"
    enc_dir = Path(base_dir) / encoder_name

    # Try modality-specific file first, fall back to shared
    modality_path = enc_dir / f"{text_key}_{modality}_text.npy"
    shared_path = enc_dir / f"{text_key}_text.npy"

    if modality_path.exists():
        text_all = np.load(modality_path)
    else:
        text_all = np.load(shared_path)

    num_cats = get_num_classes(split)
    return text_all[:num_cats]


# ---------------------------------------------------------------------------
# Similarity loading
# ---------------------------------------------------------------------------

def load_similarities(encoder_name, split, base_dir="data/similarities"):
    """Load precomputed similarities for a split.

    Args:
        encoder_name: e.g. 'imagebind'.
        split: 'train', 'val', or 'test'.
        base_dir: root directory for similarities.

    Returns:
        dict with:
            'a_t_sim': (N, T, C) audio-text similarities
            'v_t_sim': (N, T, C) visual-text similarities
            'category_ids': (N,) ground truth category IDs
            'avc_labels': (N, T) event presence labels
    """
    path = Path(base_dir) / encoder_name / f"{split}_sim.npz"
    data = np.load(path)
    return {
        "a_t_sim": data["a_t_sim"],
        "v_t_sim": data["v_t_sim"],
        "category_ids": data["category_ids"],
        "avc_labels": data["avc_labels"],
    }


# ---------------------------------------------------------------------------
# Dataset paths for raw data (used by step1_prepare)
# ---------------------------------------------------------------------------

def get_audio_paths(split):
    """Get paths to audio files for a split.

    Returns:
        list of (audio_path, category_name, vid_name) tuples.
    """
    meta = load_metadata(split)
    data_dir = _data_dir()
    results = []
    for _, row in meta.iterrows():
        audio_path = data_dir / split / "audio" / row["cls_name"] / f"{row['vid_name']}.wav"
        results.append((str(audio_path), row["cls_name"], row["vid_name"]))
    return results


def get_frame_dirs(split):
    """Get paths to video frame directories for a split.

    Returns:
        list of (frame_dir, category_name, vid_name) tuples.
    """
    meta = load_metadata(split)
    data_dir = _data_dir()
    results = []
    for _, row in meta.iterrows():
        frame_dir = data_dir / split / "video" / row["cls_name"] / row["vid_name"]
        results.append((str(frame_dir), row["cls_name"], row["vid_name"]))
    return results
