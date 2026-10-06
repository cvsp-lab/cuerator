#!/usr/bin/env python3
"""
Text-Only ImageBind Model

This module provides a lightweight text-only version of ImageBind that:
- Loads only text-related parameters (~300M vs ~1.2B parameters)
- Provides a simple interface for text embedding extraction
- Uses the same pretrained weights as the full ImageBind model

Usage:
    from .models import imagebind_text_only

    model = imagebind_text_only.imagebind_text_only_huge(pretrained=True)
    model.eval()
    model.to(device)

    # Extract text embeddings
    text_embeds = model(text_inputs)  # Returns [N, 1024] tensor
"""

import os
from functools import partial

import torch
import torch.nn as nn

from .helpers import (
    EinOpsRearrange,
    LearnableLogitScaling,
    Normalize,
    SelectEOSAndProject,
)
from .multimodal_preprocessors import TextPreprocessor
from .transformer import MultiheadAttention, SimpleTransformer


class TextOnlyImageBind(nn.Module):
    """
    Lightweight text-only version of ImageBind.

    This class contains only the components necessary for text embedding extraction:
    - Text preprocessor (tokenizer + embedding)
    - Text transformer trunk
    - Text projection head
    - Text postprocessor (normalization + scaling)

    Args:
        text_embed_dim: Text transformer embedding dimension (default: 1024 for huge model)
        text_num_blocks: Number of transformer blocks (default: 24 for huge model)
        text_num_heads: Number of attention heads (default: 16 for huge model)
        out_embed_dim: Output embedding dimension (default: 1024)
    """

    def __init__(
        self,
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        out_embed_dim=1024,
    ):
        super().__init__()

        # Text preprocessor
        self.text_preprocessor = TextPreprocessor(
            context_length=77,
            vocab_size=49408,
            embed_dim=text_embed_dim,
            causal_masking=True,
        )

        # Text transformer trunk
        self.text_trunk = SimpleTransformer(
            embed_dim=text_embed_dim,
            num_blocks=text_num_blocks,
            ffn_dropout_rate=0.0,
            drop_path_rate=0.0,
            attn_target=partial(
                MultiheadAttention,
                embed_dim=text_embed_dim,
                num_heads=text_num_heads,
                bias=True,
                add_bias_kv=False,
            ),
            pre_transformer_layer=nn.Sequential(
                EinOpsRearrange("b l d -> l b d"),
            ),
            post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
        )

        # Text projection head
        self.text_head = SelectEOSAndProject(
            proj=nn.Sequential(
                nn.LayerNorm(normalized_shape=text_embed_dim, eps=1e-6),
                nn.Linear(text_embed_dim, out_embed_dim, bias=False),
            )
        )

        # Text postprocessor
        self.text_postprocessor = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(learnable=True)
        )

    def forward(self, text_input):
        """
        Extract text embeddings.

        Args:
            text_input: Tokenized text input [N, 77] (from .data.load_and_transform_text)

        Returns:
            text_embeds: Text embeddings [N, 1024]
        """
        # Preprocess
        preprocessed = self.text_preprocessor(text=text_input)
        trunk_inputs = preprocessed["trunk"]
        head_inputs = preprocessed["head"]

        # Trunk (Transformer)
        trunk_output = self.text_trunk(**trunk_inputs)

        # Head (Select EOS + Project)
        head_output = self.text_head(trunk_output, **head_inputs)

        # Postprocess (Normalize + Scale)
        output = self.text_postprocessor(head_output)

        return output


def imagebind_text_only_huge(pretrained=False, pretrained_path=None):
    """
    Create text-only ImageBind huge model.

    Args:
        pretrained: If True, load pretrained weights from the full ImageBind model
        pretrained_path: Path to pretrained weights. If None, downloads to .checkpoints/

    Returns:
        model: TextOnlyImageBind instance
    """
    model = TextOnlyImageBind(
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        out_embed_dim=1024,
    )

    if pretrained:
        if pretrained_path is None:
            pretrained_path = ".checkpoints/imagebind_huge.pth"
            if not os.path.exists(pretrained_path):
                print(
                    "Downloading imagebind weights to .checkpoints/imagebind_huge.pth ..."
                )
                os.makedirs(".checkpoints", exist_ok=True)
                torch.hub.download_url_to_file(
                    "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
                    pretrained_path,
                    progress=True,
                )

        print(f"Loading text-only weights from {pretrained_path}...")

        # Load full state dict
        full_state_dict = torch.load(pretrained_path, map_location='cpu')

        # Filter and rename text-only parameters
        text_state_dict = {}
        for key, value in full_state_dict.items():
            if key.startswith('modality_preprocessors.text.'):
                new_key = key.replace('modality_preprocessors.text.', 'text_preprocessor.')
                text_state_dict[new_key] = value
            elif key.startswith('modality_trunks.text.'):
                new_key = key.replace('modality_trunks.text.', 'text_trunk.')
                text_state_dict[new_key] = value
            elif key.startswith('modality_heads.text.'):
                new_key = key.replace('modality_heads.text.', 'text_head.')
                text_state_dict[new_key] = value
            elif key.startswith('modality_postprocessors.text.'):
                new_key = key.replace('modality_postprocessors.text.', 'text_postprocessor.')
                text_state_dict[new_key] = value

        # Load filtered state dict
        model.load_state_dict(text_state_dict)
        print(f"Loaded {len(text_state_dict)} text-only parameters (filtered from {len(full_state_dict)} total parameters)")

    return model
