"""Differentiable memory bank for storing and addressing snapshots."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .config import MemoryTransformerConfig


class MemoryBank(nn.Module):
    """Differentiable memory storage with soft addressing.

    Stores both full and optionally compressed snapshots of hidden states.
    Supports content-based addressing and temporal ordering.
    """

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.memory_size = config.memory_size
        self.d_memory = config.d_memory
        self.d_compressed = config.d_compressed
        self.use_compressed = config.use_compressed
        self.use_both = config.use_both

        # Content-based addressing keys
        self.key_proj = nn.Linear(config.d_memory, config.d_memory, bias=False)

        # Temporal encoding for memory slots
        self.temporal_encoding = nn.Embedding(config.memory_size, config.d_memory)

        # Usage decay factor (learned)
        self.usage_decay = nn.Parameter(torch.tensor(0.99))

        self._init_weights(config.init_std)

    def _init_weights(self, std: float):
        nn.init.normal_(self.key_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.temporal_encoding.weight, mean=0.0, std=std)

    def init_memory(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, torch.Tensor]:
        """Initialize empty memory state.

        Args:
            batch_size: Batch size
            device: Device to create tensors on
            dtype: Data type

        Returns:
            Dictionary containing memory state tensors
        """
        state = {
            # Full memory bank
            "memory": torch.zeros(
                batch_size, self.memory_size, self.d_memory,
                device=device, dtype=dtype
            ),
            # Compressed memory bank
            "memory_compressed": torch.zeros(
                batch_size, self.memory_size, self.d_compressed,
                device=device, dtype=dtype
            ),
            # Write position (soft)
            "write_position": torch.zeros(
                batch_size, self.memory_size,
                device=device, dtype=dtype
            ),
            # Usage weights (for LRU-style replacement)
            "usage": torch.zeros(
                batch_size, self.memory_size,
                device=device, dtype=dtype
            ),
            # Valid mask (which slots have been written)
            "valid_mask": torch.zeros(
                batch_size, self.memory_size,
                device=device, dtype=torch.bool
            ),
            # Current write index (for sequential writing)
            "write_index": torch.zeros(
                batch_size, dtype=torch.long, device=device
            ),
            # Total writes counter
            "total_writes": torch.zeros(
                batch_size, dtype=torch.long, device=device
            ),
        }

        # Initialize write position to first slot
        state["write_position"][:, 0] = 1.0

        return state

    def write(
        self,
        state: Dict[str, torch.Tensor],
        content: torch.Tensor,
        content_compressed: Optional[torch.Tensor] = None,
        write_strength: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Write content to memory.

        Args:
            state: Current memory state
            content: Content to write [batch, d_memory]
            content_compressed: Compressed content [batch, d_compressed]
            write_strength: How strongly to write [batch, 1], range [0, 1]

        Returns:
            Updated memory state
        """
        batch_size = content.size(0)
        device = content.device

        # Default to full write strength
        if write_strength is None:
            write_strength = torch.ones(batch_size, 1, device=device)

        # Get current write position (soft addressing)
        write_pos = state["write_position"]  # [batch, memory_size]

        # Add temporal encoding to content
        temporal_idx = state["write_index"] % self.memory_size
        temporal_emb = self.temporal_encoding(temporal_idx)  # [batch, d_memory]
        content_with_temporal = content + temporal_emb

        # Write to memory using soft addressing
        # Expand for broadcasting: [batch, memory_size, 1]
        write_pos_expanded = write_pos.unsqueeze(-1)
        content_expanded = content_with_temporal.unsqueeze(1)  # [batch, 1, d_memory]
        write_strength_expanded = write_strength.unsqueeze(-1)  # [batch, 1, 1]

        # Soft write: blend new content with existing based on write position and strength
        state["memory"] = (
            state["memory"] * (1 - write_pos_expanded * write_strength_expanded) +
            content_expanded * write_pos_expanded * write_strength_expanded
        )

        # Write compressed version if provided
        if content_compressed is not None:
            compressed_expanded = content_compressed.unsqueeze(1)
            state["memory_compressed"] = (
                state["memory_compressed"] * (1 - write_pos_expanded * write_strength_expanded) +
                compressed_expanded * write_pos_expanded * write_strength_expanded
            )

        # Update valid mask
        write_idx = state["write_index"]
        for b in range(batch_size):
            idx = write_idx[b].item() % self.memory_size
            state["valid_mask"][b, idx] = True

        # Update usage (for LRU replacement)
        # write_strength is [batch, 1], broadcasts to [batch, memory_size]
        state["usage"] = state["usage"] * self.usage_decay + write_pos * write_strength

        # Advance write position (circular buffer)
        state["write_index"] = (state["write_index"] + 1) % self.memory_size
        state["total_writes"] = state["total_writes"] + 1

        # Update soft write position for next write
        new_write_pos = torch.zeros_like(write_pos)
        for b in range(batch_size):
            next_idx = state["write_index"][b].item()
            new_write_pos[b, next_idx] = 1.0
        state["write_position"] = new_write_pos

        return state

    def read(
        self,
        state: Dict[str, torch.Tensor],
        query: torch.Tensor,
        use_compressed: bool = False,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read from memory using content-based addressing.

        Args:
            state: Current memory state
            query: Query vector [batch, d_model] or [batch, seq_len, d_model]
            use_compressed: Whether to read from compressed memory
            top_k: If set, only attend to top-k most similar slots

        Returns:
            Retrieved content [batch, (...), d_memory]
            Attention weights [batch, (...), memory_size]
        """
        memory = state["memory_compressed"] if use_compressed else state["memory"]
        valid_mask = state["valid_mask"]

        # Handle both single query and batched queries
        if query.dim() == 2:
            query = query.unsqueeze(1)  # [batch, 1, d_model]
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, seq_len, _ = query.shape

        # Project memory to keys
        keys = self.key_proj(memory)  # [batch, memory_size, d_memory]

        # Compute attention scores
        # query: [batch, seq_len, d_model], keys: [batch, memory_size, d_memory]
        scores = torch.bmm(query, keys.transpose(1, 2))  # [batch, seq_len, memory_size]
        scores = scores / (self.d_memory ** 0.5)

        # Mask invalid slots
        invalid_mask = ~valid_mask.unsqueeze(1)  # [batch, 1, memory_size]
        scores = scores.masked_fill(invalid_mask, float('-inf'))

        # Optional top-k selection
        if top_k is not None and top_k < self.memory_size:
            top_k_values, _ = scores.topk(top_k, dim=-1)
            threshold = top_k_values[..., -1:].clamp(min=float('-inf'))
            scores = scores.masked_fill(scores < threshold, float('-inf'))

        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Handle case where all slots are masked (return zeros)
        all_invalid = (~valid_mask).all(dim=-1, keepdim=True).unsqueeze(1)
        attn_weights = attn_weights.masked_fill(all_invalid, 0.0)

        # Read from memory
        # attn_weights: [batch, seq_len, memory_size]
        # memory: [batch, memory_size, d_memory]
        retrieved = torch.bmm(attn_weights, memory)  # [batch, seq_len, d_memory]

        if squeeze_output:
            retrieved = retrieved.squeeze(1)
            attn_weights = attn_weights.squeeze(1)

        return retrieved, attn_weights

    def get_all_memories(
        self,
        state: Dict[str, torch.Tensor],
        use_compressed: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all memory contents for continuous attention.

        Args:
            state: Current memory state
            use_compressed: Whether to return compressed memories

        Returns:
            All memories [batch, memory_size, d_memory]
            Valid mask [batch, memory_size]
        """
        memory = state["memory_compressed"] if use_compressed else state["memory"]
        return memory, state["valid_mask"]

    def compute_usage_loss(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss to encourage sparse memory usage.

        Returns:
            Scalar loss value
        """
        # Encourage usage to be sparse (not writing to every slot equally)
        usage = state["usage"]
        # L1 regularization on usage variance encourages differentiated usage
        usage_mean = usage.mean(dim=-1, keepdim=True)
        usage_var = ((usage - usage_mean) ** 2).mean()
        # Negative variance (we want high variance = sparse usage)
        return -usage_var


class MemoryState:
    """Wrapper class for cleaner memory state management."""

    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        self._state = state_dict

    @property
    def memory(self) -> torch.Tensor:
        return self._state["memory"]

    @property
    def memory_compressed(self) -> torch.Tensor:
        return self._state["memory_compressed"]

    @property
    def valid_mask(self) -> torch.Tensor:
        return self._state["valid_mask"]

    @property
    def usage(self) -> torch.Tensor:
        return self._state["usage"]

    @property
    def num_writes(self) -> torch.Tensor:
        return self._state["total_writes"]

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return self._state

    def detach(self) -> "MemoryState":
        """Detach all tensors from computation graph."""
        return MemoryState({
            k: v.detach() if isinstance(v, torch.Tensor) else v
            for k, v in self._state.items()
        })
