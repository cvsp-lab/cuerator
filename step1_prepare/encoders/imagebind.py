"""ImageBind Huge encoder implementation."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .base import AudioVisualTextEncoder
from ._pytorchvideo_compat import patch_torchvision_for_pytorchvideo

patch_torchvision_for_pytorchvideo()  # must run before imagebind_lib.data imports pytorchvideo

from .imagebind_lib import data as ib_data  # noqa: E402
from .imagebind_lib import imagebind_model, ModalityType


class _AudioVisualDataset(Dataset):
    """Dataset that loads raw audio + visual frames for a list of samples."""

    def __init__(self, audio_paths, frame_dirs):
        self.audio_paths = audio_paths
        self.frame_dirs = frame_dirs

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio = ib_data.load_and_transform_audio_data([self.audio_paths[idx]])
        frame_dir = self.frame_dirs[idx]
        frame_paths = sorted(
            str(p) for p in Path(frame_dir).iterdir() if p.suffix in ('.jpg', '.png')
        )
        visual = ib_data.load_and_transform_vision_data(frame_paths)
        return audio, visual


class ImageBindEncoder(AudioVisualTextEncoder):
    """ImageBind Huge encoder (1024-dim shared embedding space)."""

    def __init__(self, pretrained_path=None, device="cuda"):
        self._pretrained_path = pretrained_path
        self._device = device

        print(f"Loading ImageBind Huge model on {device}...")
        self._model = imagebind_model.imagebind_huge(
            pretrained=True, pretrained_path=pretrained_path
        )
        self._model.eval()
        self._model.to(device)
        print("Model loaded.")

    def extract_audio_visual(self, audio_paths, frame_dirs, batch_size=32,
                             num_workers=8):
        """Extract audio and visual embeddings.

        Args:
            audio_paths: List of audio file paths.
            frame_dirs: List of video frame directories.
            batch_size: Batch size for DataLoader.
            num_workers: Number of DataLoader workers.

        Returns:
            audio_embeds: (N, T, D) numpy array.
            visual_embeds: (N, T, D) numpy array.
        """
        dataset = _AudioVisualDataset(audio_paths, frame_dirs)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        audio_list, visual_list = [], []
        with torch.no_grad():
            for audio, visual in tqdm(loader, desc=f"Extracting ({self._device})"):
                audio = audio.squeeze(1).to(self._device)
                visual = visual.to(self._device)
                embeddings = self._model({
                    ModalityType.AUDIO: audio,
                    ModalityType.VISION: visual,
                })
                audio_list.append(embeddings["audio"].cpu().numpy())
                visual_list.append(embeddings["vision"].cpu().numpy())

        return np.concatenate(audio_list), np.concatenate(visual_list)

    def extract_audio(self, audio_paths, batch_size=32, num_workers=8):
        raise NotImplementedError("Use extract_audio_visual()")

    def extract_visual(self, frame_dirs, batch_size=32, num_workers=8):
        raise NotImplementedError("Use extract_audio_visual()")

    def _extract_text(self, texts):
        text_inputs = ib_data.load_and_transform_text(texts)
        with torch.no_grad():
            text_inputs = text_inputs.to(self._device)
            embeddings = self._model({ModalityType.TEXT: text_inputs})
        return embeddings["text"].cpu().numpy()

    def extract_text_for_audio(self, texts):
        return self._extract_text(texts)

    def extract_text_for_visual(self, texts):
        return self._extract_text(texts)

    @property
    def audio_embed_dim(self) -> int:
        return 1024

    @property
    def visual_embed_dim(self) -> int:
        return 1024
