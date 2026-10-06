"""Extract embeddings from raw audio/visual/text data.

Supports multi-GPU parallelism via subprocess workers.

Usage:
    # Single GPU
    python -m step1_prepare.extract --encoder imagebind --device cuda:0

    # Multi-GPU
    python -m step1_prepare.extract --encoder imagebind --num_gpus 4
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ovavel.data as dataset_utils


def build_encoder(encoder_name, pretrained_path=None, device="cuda"):
    """Instantiate an encoder by name."""
    if encoder_name == "imagebind":
        from step1_prepare.encoders.imagebind import ImageBindEncoder

        return ImageBindEncoder(
            pretrained_path=pretrained_path,
            device=device,
        )

    raise ValueError(f"Unknown encoder: {encoder_name}")


def extract_split(encoder, split, output_dir, batch_size, num_workers):
    """Extract and save embeddings for one split (single GPU)."""
    print(f"\n{'='*60}")
    print(f"Extracting {split} split")
    print(f"{'='*60}")

    audio_info = dataset_utils.get_audio_paths(split)
    frame_info = dataset_utils.get_frame_dirs(split)
    audio_paths = [p for p, _, _ in audio_info]
    frame_dirs = [p for p, _, _ in frame_info]

    print(f"  Samples: {len(audio_paths)}")

    audio_embeds, visual_embeds = encoder.extract_audio_visual(
        audio_paths, frame_dirs, batch_size=batch_size, num_workers=num_workers
    )
    print(f"  Audio: {audio_embeds.shape}, Visual: {visual_embeds.shape}")

    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_dir, f"{split}.npz"),
        audio=audio_embeds, visual=visual_embeds,
    )

    size_mb = os.path.getsize(os.path.join(output_dir, f"{split}.npz")) / 1024 / 1024
    print(f"  Saved: {output_dir}/{split}.npz ({size_mb:.1f} MB)")


def extract_split_multigpu(split, output_dir, num_gpus, pretrained_path,
                           batch_size, num_workers):
    """Extract embeddings for one split using multiple GPUs via subprocesses."""
    print(f"\n{'='*60}")
    print(f"Extracting {split} split ({num_gpus} GPUs)")
    print(f"{'='*60}")

    audio_info = dataset_utils.get_audio_paths(split)
    frame_info = dataset_utils.get_frame_dirs(split)
    audio_paths = [p for p, _, _ in audio_info]
    frame_dirs = [p for p, _, _ in frame_info]
    N = len(audio_paths)
    print(f"  Samples: {N}")

    # Ensure pretrained weights exist before spawning workers
    if pretrained_path is None:
        default_path = os.path.join(os.getcwd(), ".checkpoints", "imagebind_huge.pth")
        if not os.path.exists(default_path):
            print("  Downloading ImageBind weights before launching workers...")
            os.makedirs(os.path.dirname(default_path), exist_ok=True)
            import torch
            torch.hub.download_url_to_file(
                "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
                default_path, progress=True,
            )
        pretrained_path = default_path

    # Split indices across GPUs
    chunks = np.array_split(range(N), num_gpus)

    # Write chunk manifests to temp files
    tmpdir = tempfile.mkdtemp(prefix="cuerator_extract_")
    manifest_paths = []
    for gpu_id, indices in enumerate(chunks):
        manifest = {
            "audio_paths": [audio_paths[i] for i in indices],
            "frame_dirs": [frame_dirs[i] for i in indices],
        }
        path = os.path.join(tmpdir, f"manifest_{gpu_id}.json")
        with open(path, "w") as f:
            json.dump(manifest, f)
        manifest_paths.append(path)

    # Launch subprocesses
    out_paths = []
    procs = []
    for gpu_id in range(num_gpus):
        out_path = os.path.join(tmpdir, f"chunk_{gpu_id}.npz")
        out_paths.append(out_path)
        cmd = [
            sys.executable, "-m", "step1_prepare.extract_worker",
            "--manifest", manifest_paths[gpu_id],
            "--output", out_path,
            "--device", "cuda:0",  # CUDA_VISIBLE_DEVICES remaps to cuda:0
            "--batch_size", str(batch_size),
            "--num_workers", str(max(1, num_workers // num_gpus)),
        ]
        if pretrained_path:
            cmd.extend(["--pretrained_path", pretrained_path])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        procs.append(subprocess.Popen(cmd, env=env, cwd=str(Path(__file__).parent.parent)))

    # Wait for all
    for i, p in enumerate(procs):
        ret = p.wait()
        if ret != 0:
            raise RuntimeError(f"GPU {i} worker failed with exit code {ret}")
        print(f"  GPU {i}: done ({len(chunks[i])} samples)")

    # Reassemble
    audio_parts, visual_parts = [], []
    for out_path in out_paths:
        d = np.load(out_path)
        audio_parts.append(d["audio"])
        visual_parts.append(d["visual"])

    audio_embeds = np.concatenate(audio_parts)
    visual_embeds = np.concatenate(visual_parts)
    print(f"  Audio: {audio_embeds.shape}, Visual: {visual_embeds.shape}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_dir, f"{split}.npz"),
        audio=audio_embeds, visual=visual_embeds,
    )
    # Cleanup temp files
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    size_mb = os.path.getsize(os.path.join(output_dir, f"{split}.npz")) / 1024 / 1024
    print(f"  Saved: {output_dir}/{split}.npz ({size_mb:.1f} MB)")


def extract_text(encoder, output_dir):
    """Extract and save text embeddings for train/test category sets."""
    os.makedirs(output_dir, exist_ok=True)

    for text_key, split in [("train", "train"), ("test", "test")]:
        categories = dataset_utils.load_categories(split)
        categories_with_bg = categories + ["other"]

        audio_text = encoder.extract_text_for_audio(categories_with_bg)
        visual_text = encoder.extract_text_for_visual(categories_with_bg)

        # If identical (shared space), save once as text.npy
        if np.array_equal(audio_text, visual_text):
            np.save(os.path.join(output_dir, f"{text_key}_text.npy"), audio_text)
            print(f"  Text ({text_key}): {audio_text.shape} "
                  f"({len(categories)} categories + BG) [shared]")
        else:
            np.save(os.path.join(output_dir, f"{text_key}_audio_text.npy"), audio_text)
            np.save(os.path.join(output_dir, f"{text_key}_visual_text.npy"), visual_text)
            print(f"  Audio text ({text_key}): {audio_text.shape}, "
                  f"Visual text ({text_key}): {visual_text.shape}")



def main():
    parser = argparse.ArgumentParser(description="Extract embeddings")
    parser.add_argument("--encoder", default="imagebind", choices=["imagebind"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--pretrained_path", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text_only", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/embeddings/{args.encoder}"

    if args.text_only or args.num_gpus <= 1:
        encoder = build_encoder(
            args.encoder,
            pretrained_path=args.pretrained_path,
            device=args.device,
        )

        if args.text_only:
            extract_text(encoder, args.output)
        else:
            for split in args.splits:
                extract_split(encoder, split, args.output,
                             args.batch_size, args.num_workers)
            extract_text(encoder, args.output)
    else:
        if args.encoder != "imagebind":
            raise NotImplementedError(
                f"Multi-GPU extraction is not implemented for {args.encoder}. "
                "Use --num_gpus 1."
            )

        # Multi-GPU: use subprocesses for audio/visual, single-GPU for text
        for split in args.splits:
            extract_split_multigpu(split, args.output, args.num_gpus,
                                  args.pretrained_path, args.batch_size,
                                  args.num_workers)

        # Text extraction on single GPU
        encoder = build_encoder(
            args.encoder,
            pretrained_path=args.pretrained_path,
            device=args.device,
        )
        extract_text(encoder, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
