"""Abstract base class for audio-visual-text encoders."""

from abc import ABC, abstractmethod
import numpy as np


class AudioVisualTextEncoder(ABC):
    """Base encoder interface for multimodal embedding extraction.

    Supports both shared-space encoders (ImageBind) and separate-space
    encoders (CLIP+CLAP) through modality-specific text extraction.
    """

    @abstractmethod
    def extract_audio(self, audio_paths: list[str]) -> np.ndarray:
        """Extract audio embeddings.

        Returns:
            Embeddings of shape (N, T, D_audio).
        """

    @abstractmethod
    def extract_visual(self, frame_dirs: list[str]) -> np.ndarray:
        """Extract visual embeddings.

        Returns:
            Embeddings of shape (N, T, D_visual).
        """

    @abstractmethod
    def extract_text_for_audio(self, texts: list[str]) -> np.ndarray:
        """Extract text embeddings aligned with the audio space.

        Returns:
            Embeddings of shape (C, D_audio).
        """

    @abstractmethod
    def extract_text_for_visual(self, texts: list[str]) -> np.ndarray:
        """Extract text embeddings aligned with the visual space.

        Returns:
            Embeddings of shape (C, D_visual).
        """

    @property
    @abstractmethod
    def audio_embed_dim(self) -> int:
        """Dimensionality of the audio embedding space."""

    @property
    @abstractmethod
    def visual_embed_dim(self) -> int:
        """Dimensionality of the visual embedding space."""
