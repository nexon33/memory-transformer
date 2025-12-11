"""Token and positional embeddings for Memory Transformer."""

import math
import torch
import torch.nn as nn
from typing import Optional

from .config import MemoryTransformerConfig


class TokenEmbedding(nn.Module):
    """Token embedding with optional scaling."""

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.scale = math.sqrt(config.d_model)
        self.init_std = config.init_std

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token indices [batch, seq_len]

        Returns:
            Token embeddings [batch, seq_len, d_model]
        """
        return self.embedding(x) * self.scale


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encodings."""

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len
        self.dropout = nn.Dropout(config.dropout)

        # Create positional encodings
        pe = torch.zeros(config.max_seq_len, config.d_model)
        position = torch.arange(0, config.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.d_model, 2).float() * (-math.log(10000.0) / config.d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Position-encoded tensor [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learnable positional embeddings."""

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.init_std = config.init_std

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=self.init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Position-encoded tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embedding(positions)  # [seq_len, d_model]
        x = x + pos_emb.unsqueeze(0)  # Broadcast over batch
        return self.dropout(x)


class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE) for better length generalization."""

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.dim = config.head_dim
        self.max_seq_len = config.max_seq_len

        # Compute inverse frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build cache for cos and sin values
        self._build_cache()

    def _build_cache(self):
        """Pre-compute cos and sin values."""
        seq = torch.arange(self.max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", seq, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> tuple:
        """
        Get cos and sin values for RoPE.

        Args:
            x: Input tensor (used only for device/dtype)
            seq_len: Sequence length

        Returns:
            Tuple of (cos, sin) tensors
        """
        if seq_len is None:
            seq_len = x.size(1)

        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor [batch, n_heads, seq_len, head_dim]
        k: Key tensor [batch, n_heads, seq_len, head_dim]
        cos: Cosine values [seq_len, head_dim]
        sin: Sine values [seq_len, head_dim]

    Returns:
        Rotated (query, key) tensors
    """
    # Reshape cos/sin for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class TransformerEmbedding(nn.Module):
    """Combined token + positional embedding."""

    def __init__(self, config: MemoryTransformerConfig, use_rope: bool = False):
        super().__init__()
        self.token_embedding = TokenEmbedding(config)

        if use_rope:
            # RoPE is applied in attention, not here
            self.pos_encoding = None
            self.rope = RotaryPositionalEncoding(config)
        else:
            self.pos_encoding = LearnedPositionalEncoding(config)
            self.rope = None

        self.use_rope = use_rope

    def forward(
        self,
        x: torch.Tensor,
        return_rope: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Token indices [batch, seq_len]
            return_rope: If True and using RoPE, also return cos/sin

        Returns:
            Embedded tokens [batch, seq_len, d_model]
            Optionally: (embeddings, cos, sin) if return_rope=True
        """
        emb = self.token_embedding(x)

        if self.use_rope:
            if return_rope:
                cos, sin = self.rope(emb)
                return emb, cos, sin
            return emb
        else:
            return self.pos_encoding(emb)
