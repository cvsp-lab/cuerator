"""Worker process for multi-GPU embedding extraction.

Called by extract.py via subprocess. Each worker handles a chunk of samples
on a single GPU.
"""

import argparse
import json

import numpy as np

from step1_prepare.encoders.imagebind import ImageBindEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrained_path", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    encoder = ImageBindEncoder(
        pretrained_path=args.pretrained_path,
        device=args.device,
    )

    audio_embeds, visual_embeds = encoder.extract_audio_visual(
        manifest["audio_paths"],
        manifest["frame_dirs"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    np.savez(args.output, audio=audio_embeds, visual=visual_embeds)


if __name__ == "__main__":
    main()
