"""Precompute cosine similarities from extracted embeddings.

Usage:
    python -m step1_prepare.precompute \
        --embeddings data/embeddings/imagebind \
        --output data/similarities/imagebind \
        --splits train val test
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ovavel.data as dataset_utils


def _load_text_embeddings(embeddings_dir, split):
    """Load audio/visual text embeddings, supporting shared or separate spaces."""
    text_key = "train" if split == "train" else "test"
    shared_path = os.path.join(embeddings_dir, f"{text_key}_text.npy")
    audio_path = os.path.join(embeddings_dir, f"{text_key}_audio_text.npy")
    visual_path = os.path.join(embeddings_dir, f"{text_key}_visual_text.npy")

    num_cats = dataset_utils.get_num_classes(split)
    if os.path.exists(audio_path) and os.path.exists(visual_path):
        audio_text = np.load(audio_path)[:num_cats]
        visual_text = np.load(visual_path)[:num_cats]
        return audio_text, visual_text

    shared_text = np.load(shared_path)[:num_cats]
    return shared_text, shared_text


def precompute_split(split, embeddings_dir, output_dir, device="cuda",
                     chunk_size=512):
    """Compute and save audio-text / visual-text similarities for one split."""
    print(f"\n{'='*60}")
    print(f"Precomputing similarities: {split}")
    print(f"{'='*60}")

    # Load embeddings
    emb_path = os.path.join(embeddings_dir, f"{split}.npz")
    emb = np.load(emb_path)
    audio = torch.from_numpy(emb["audio"]).float()
    visual = torch.from_numpy(emb["visual"]).float()
    if audio.shape[:2] != visual.shape[:2]:
        raise ValueError(
            f"Audio/visual timeline shape mismatch: {audio.shape} vs {visual.shape}"
        )
    N, T, D_audio = audio.shape
    _, _, D_visual = visual.shape
    print(f"  Samples: {N}, T={T}, D_audio={D_audio}, D_visual={D_visual}")

    # Load text embeddings (exclude BG = last row)
    audio_text, visual_text = _load_text_embeddings(embeddings_dir, split)
    audio_text = torch.from_numpy(audio_text).float()
    visual_text = torch.from_numpy(visual_text).float()
    print(
        f"  Categories: {audio_text.shape[0]} (excl BG), "
        f"D_audio_text={audio_text.shape[1]}, D_visual_text={visual_text.shape[1]}"
    )

    # Build category_ids and avc_labels from dataset metadata
    categories = dataset_utils.load_categories(split)
    cat_map = {name: idx for idx, name in enumerate(categories)}
    bg_id = dataset_utils.get_bg_id(split)

    meta_df = dataset_utils.load_metadata(split)
    anno = dataset_utils.load_annotations()

    category_ids = np.array([
        cat_map.get(row["cls_name"], bg_id) for _, row in meta_df.iterrows()
    ], dtype=np.int64)

    avc_labels = np.array([
        json.loads(anno[row["vid_name"]]["label"]) for _, row in meta_df.iterrows()
    ], dtype=np.float32)

    print(f"  Category ID range: [{category_ids.min()}, {category_ids.max()}]")

    # Compute cosine similarities in chunks on GPU
    audio_text_norm = F.normalize(audio_text, p=2, dim=-1).to(device)
    visual_text_norm = F.normalize(visual_text, p=2, dim=-1).to(device)
    a_t_sim_list, v_t_sim_list = [], []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        a_chunk = F.normalize(audio[start:end].to(device), p=2, dim=-1)
        v_chunk = F.normalize(visual[start:end].to(device), p=2, dim=-1)

        # (B, T, D) @ (D, C) -> (B, T, C)
        a_sim = torch.bmm(
            a_chunk, audio_text_norm.T.unsqueeze(0).expand(end - start, -1, -1)
        )
        v_sim = torch.bmm(
            v_chunk, visual_text_norm.T.unsqueeze(0).expand(end - start, -1, -1)
        )

        a_t_sim_list.append(a_sim.cpu().numpy())
        v_t_sim_list.append(v_sim.cpu().numpy())

    a_t_sim = np.concatenate(a_t_sim_list, axis=0)  # (N, T, C)
    v_t_sim = np.concatenate(v_t_sim_list, axis=0)
    print(f"  a_t_sim: {a_t_sim.shape}, v_t_sim: {v_t_sim.shape}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{split}_sim.npz")
    np.savez_compressed(
        out_path,
        a_t_sim=a_t_sim,
        v_t_sim=v_t_sim,
        category_ids=category_ids,
        avc_labels=avc_labels,
    )
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Precompute similarities")
    parser.add_argument("--embeddings", default="data/embeddings/imagebind")
    parser.add_argument("--output", default="data/similarities/imagebind")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk_size", type=int, default=512)
    args = parser.parse_args()

    for split in args.splits:
        precompute_split(split, args.embeddings, args.output,
                        device=args.device, chunk_size=args.chunk_size)

    print("\nDone!")


if __name__ == "__main__":
    main()
