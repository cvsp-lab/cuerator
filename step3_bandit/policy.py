from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal


class _SelfAttnBlock(nn.Module):
    def __init__(self, model_dim: int, nhead: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            model_dim,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
        )
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask=None):
        attn_out, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class _CrossAttnBlock(nn.Module):
    def __init__(self, model_dim: int, nhead: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            model_dim,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
        )
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_padding_mask=None,
        context_padding_mask=None,
    ):
        attn_out, _ = self.attn(
            query,
            context,
            context,
            key_padding_mask=context_padding_mask,
            need_weights=False,
        )
        x = self.norm1(query + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        if query_padding_mask is not None:
            x = x.masked_fill(query_padding_mask.unsqueeze(-1), 0.0)
        return x


class CrossAttentionBanditPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        sequence_length: int,
        video_dim: int,
        audio_dim: int,
        fixed_action_std: float = 0.3,
        model_dim: int = 256,
        transformer_nhead: int = 4,
        transformer_ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if action_dim <= 0:
            raise ValueError("action_dim must be > 0")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if fixed_action_std <= 0.0:
            raise ValueError("fixed_action_std must be > 0")
        if video_dim <= 0 or audio_dim <= 0:
            raise ValueError("video_dim/audio_dim must be > 0")
        if video_dim + audio_dim != input_dim:
            raise ValueError(
                f"Expected video_dim + audio_dim == input_dim, got "
                f"{video_dim} + {audio_dim} != {input_dim}"
            )
        if model_dim <= 0:
            raise ValueError("model_dim must be > 0")
        if transformer_nhead <= 0 or model_dim % transformer_nhead != 0:
            raise ValueError("transformer_nhead must divide model_dim")
        if transformer_ff_dim <= 0:
            raise ValueError("transformer_ff_dim must be > 0")

        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.sequence_length = int(sequence_length)
        self.video_dim = int(video_dim)
        self.audio_dim = int(audio_dim)
        self.fixed_action_std = float(fixed_action_std)

        self.video_in_proj = nn.Linear(self.video_dim, model_dim)
        self.audio_in_proj = nn.Linear(self.audio_dim, model_dim)
        self.video_self = _SelfAttnBlock(model_dim, transformer_nhead, transformer_ff_dim, dropout)
        self.audio_self = _SelfAttnBlock(model_dim, transformer_nhead, transformer_ff_dim, dropout)
        self.video_cross = _CrossAttnBlock(model_dim, transformer_nhead, transformer_ff_dim, dropout)
        self.audio_cross = _CrossAttnBlock(model_dim, transformer_nhead, transformer_ff_dim, dropout)
        self.fuse = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(model_dim, self.action_dim)

        fixed_log_std = math.log(self.fixed_action_std)
        self.register_buffer("fixed_log_std", torch.full((self.action_dim,), fixed_log_std))

    def _encode(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError(f"Expected input shape [B,T,D], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got last_dim={x.shape[-1]}"
            )
        if x.shape[1] > self.sequence_length:
            x = x[:, : self.sequence_length, :]

        token_padding_mask = x.abs().sum(dim=-1) <= 1e-12

        xv = x[:, :, : self.video_dim]
        xa = x[:, :, self.video_dim : self.video_dim + self.audio_dim]

        xv = self.video_in_proj(xv)
        xa = self.audio_in_proj(xa)

        xv = self.video_self(xv, key_padding_mask=token_padding_mask)
        xa = self.audio_self(xa, key_padding_mask=token_padding_mask)

        xv = self.video_cross(
            xv,
            xa,
            query_padding_mask=token_padding_mask,
            context_padding_mask=token_padding_mask,
        )
        xa = self.audio_cross(
            xa,
            xv,
            query_padding_mask=token_padding_mask,
            context_padding_mask=token_padding_mask,
        )

        valid = (~token_padding_mask).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        xv_pool = (xv * valid).sum(dim=1) / denom
        xa_pool = (xa * valid).sum(dim=1) / denom

        h = torch.cat([xv_pool, xa_pool], dim=-1)
        return self.fuse(h)

    def forward(self, x: torch.Tensor):
        h = self._encode(x)
        mu = self.mu_head(h)
        log_std = self.fixed_log_std.unsqueeze(0).expand_as(mu)
        return mu, log_std

    def sample(self, x: torch.Tensor, num_samples: int = 1):
        """Draw num_samples actions per context from the Gaussian policy.

        Args:
            x: (B, T, F) context batch.
            num_samples: K (default 1).
        Returns:
            dict with:
                mu: (B, P) — deterministic mean action (shared across K).
                log_std: (B, P)
                std: (B, P)
                raw_action: (B, K, P) detached — sampled actions.
                log_prob: (B, K) — log π(a_k | context), gradient-carrying.
        """
        mu, log_std = self.forward(x)
        std = torch.exp(log_std)
        # Expand mu/std to (B, K, P) for K-sample noise.
        mu_exp = mu.unsqueeze(1).expand(-1, num_samples, -1)
        std_exp = std.unsqueeze(1).expand(-1, num_samples, -1)
        eps = torch.randn_like(mu_exp)
        raw_action = (mu_exp + std_exp * eps).detach()  # (B, K, P)

        dist = Normal(mu_exp, std_exp)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)  # (B, K)
        return {
            "mu": mu,
            "log_std": log_std,
            "std": std,
            "raw_action": raw_action,
            "log_prob": log_prob,
        }
