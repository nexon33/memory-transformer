"""Multi-head self-attention for Memory Transformer."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import MemoryTransformerConfig
from .embeddings import apply_rotary_pos_emb


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional RoPE support."""

    def __init__(self, config: MemoryTransformerConfig, is_cross_attention: bool = False):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.is_cross_attention = is_cross_attention
        self.dropout = config.attention_dropout
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Query, Key, Value projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self._init_weights(config.init_std)

    def _init_weights(self, std: float):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        x: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Query tensor [batch, seq_len, d_model]
            kv: Key/Value tensor for cross-attention [batch, kv_len, d_model]
            mask: Attention mask [batch, 1, seq_len, kv_len] or [1, 1, seq_len, kv_len]
            cos: Cosine values for RoPE [seq_len, head_dim]
            sin: Sine values for RoPE [seq_len, head_dim]
            return_attention: Whether to return attention weights

        Returns:
            Output tensor [batch, seq_len, d_model]
            Optional attention weights [batch, n_heads, seq_len, kv_len]
        """
        batch_size, seq_len, _ = x.shape

        # For self-attention, kv is the same as x
        if kv is None:
            kv = x
        kv_len = kv.size(1)

        # Project to queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        # Reshape for multi-head attention: [batch, n_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if provided
        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask
        if mask is not None:
            attn_weights = attn_weights + mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights_dropped = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights_dropped, v)

        # Reshape back: [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Output projection
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)

        if return_attention:
            return output, attn_weights
        return output, None


class CausalSelfAttention(MultiHeadAttention):
    """Self-attention with causal masking for autoregressive models."""

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__(config, is_cross_attention=False)

        # Create causal mask buffer
        mask = torch.triu(
            torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1
        ).bool()
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.max_seq_len, config.max_seq_len),
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            cos: Cosine values for RoPE
            sin: Sine values for RoPE
            return_attention: Whether to return attention weights

        Returns:
            Output tensor [batch, seq_len, d_model]
            Optional attention weights
        """
        seq_len = x.size(1)

        # Get causal mask for current sequence length
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        # Convert bool mask to additive mask (True -> -inf)
        mask = causal_mask.float() * -1e9

        return super().forward(
            x=x,
            kv=None,
            mask=mask,
            cos=cos,
            sin=sin,
            return_attention=return_attention,
        )


class MemoryAttention(nn.Module):
    """Attention mechanism for reading from memory bank.

    Supports both single query attention and continuous attention
    over all memory slots.
    """

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_memory = config.d_memory if not config.use_compressed else config.d_compressed
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Query projection (from hidden state)
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Key/Value projections (from memory)
        # Note: memory dimension might differ from d_model
        self.k_proj = nn.Linear(self.d_memory, config.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_memory, config.d_model, bias=False)

        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.memory_dropout)

        self._init_weights(config.init_std)

    def _init_weights(self, std: float):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: Query tensor [batch, seq_len, d_model]
            memory: Memory bank [batch, memory_size, d_memory]
            memory_mask: Mask for valid memory slots [batch, memory_size]
            return_attention: Whether to return attention weights

        Returns:
            Retrieved content [batch, seq_len, d_model]
            Optional attention weights [batch, n_heads, seq_len, memory_size]
        """
        batch_size, seq_len, _ = query.shape
        memory_size = memory.size(1)

        # Project query, keys, values
        q = self.q_proj(query)
        k = self.k_proj(memory)
        v = self.v_proj(memory)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, memory_size, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, memory_size, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply memory mask if provided
        if memory_mask is not None:
            # memory_mask: [batch, memory_size] -> [batch, 1, 1, memory_size]
            mask = (~memory_mask).unsqueeze(1).unsqueeze(2).float() * -1e9
            attn_weights = attn_weights + mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights_dropped, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Output projection
        output = self.out_proj(attn_output)

        if return_attention:
            return output, attn_weights
        return output, None


class FeedForward(nn.Module):
    """Position-wise feedforward network."""

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

        self._init_weights(config.init_std)

    def _init_weights(self, std: float):
        nn.init.normal_(self.fc1.weight, mean=0.0, std=std)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=std)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm for transformers.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
