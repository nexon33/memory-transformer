"""Snapshot writer module for creating memory snapshots."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import MemoryTransformerConfig


class SnapshotWriter(nn.Module):
    """Learns when and what to write to memory.

    The writer has two components:
    1. Write gate: Decides whether to write (learned soft decision)
    2. Content transform: Transforms hidden state to memory format
    """

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_memory = config.d_memory
        self.d_compressed = config.d_compressed
        self.use_both = config.use_both
        self.snapshot_interval = config.snapshot_interval

        # Write gate: learns when to write
        self.write_gate = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid(),
        )

        # Content transform: transforms hidden state to memory format
        self.content_transform = nn.Sequential(
            nn.Linear(config.d_model, config.d_memory),
            nn.LayerNorm(config.d_memory),
        )

        # Compression transform for ablation
        if config.use_both or config.use_compressed:
            self.compressor = nn.Sequential(
                nn.Linear(config.d_model, config.d_compressed),
                nn.LayerNorm(config.d_compressed),
            )
        else:
            self.compressor = None

        # Position encoding for snapshots (to remember where they came from)
        self.position_encoder = nn.Linear(1, config.d_memory // 4)

        self.dropout = nn.Dropout(config.memory_dropout)
        self._init_weights(config.init_std)

    def _init_weights(self, std: float):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def compute_write_strength(
        self,
        hidden_state: torch.Tensor,
        position: int,
        force_write: bool = False,
    ) -> torch.Tensor:
        """Compute how strongly to write to memory.

        Args:
            hidden_state: Current hidden state [batch, d_model]
            position: Current position in sequence
            force_write: If True, always write (for interval-based writing)

        Returns:
            Write strength [batch, 1], range [0, 1]
        """
        if force_write:
            return torch.ones(hidden_state.size(0), 1, device=hidden_state.device)

        # Learned write gate
        gate_value = self.write_gate(hidden_state)
        return gate_value

    def should_write(self, position: int) -> bool:
        """Check if we should write at this position based on interval."""
        return position > 0 and position % self.snapshot_interval == 0

    def forward(
        self,
        hidden_state: torch.Tensor,
        position: int,
        force_write: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Create a snapshot from hidden state.

        Args:
            hidden_state: Current hidden state [batch, d_model] or [batch, seq_len, d_model]
            position: Current position in sequence
            force_write: If True, always create snapshot

        Returns:
            Tuple of:
                - Full snapshot [batch, d_memory] or None
                - Compressed snapshot [batch, d_compressed] or None
                - Write strength [batch, 1] or None
        """
        # Check if we should write at this position
        if not force_write and not self.should_write(position):
            return None, None, None

        # Handle sequence dimension
        if hidden_state.dim() == 3:
            # Use the hidden state at the snapshot position
            # For sequence processing, we might want the last position or a specific one
            hidden_state = hidden_state[:, -1, :]  # [batch, d_model]

        batch_size = hidden_state.size(0)
        device = hidden_state.device

        # Compute write strength (gated decision)
        write_strength = self.compute_write_strength(
            hidden_state, position, force_write=force_write
        )

        # Transform content to memory format
        content = self.content_transform(hidden_state)  # [batch, d_memory]

        # Add position encoding
        pos_tensor = torch.tensor(
            [[position]], dtype=torch.float32, device=device
        ).expand(batch_size, 1) / 256.0  # Normalize position
        pos_emb = self.position_encoder(pos_tensor)  # [batch, d_memory // 4]

        # Pad position embedding and add
        pos_emb_padded = F.pad(pos_emb, (0, content.size(-1) - pos_emb.size(-1)))
        content = content + pos_emb_padded

        content = self.dropout(content)

        # Create compressed version if needed
        compressed = None
        if self.compressor is not None:
            compressed = self.compressor(hidden_state)
            compressed = self.dropout(compressed)

        return content, compressed, write_strength


class AdaptiveSnapshotWriter(SnapshotWriter):
    """Snapshot writer that learns to adaptively decide when to write.

    Instead of fixed intervals, uses content-based triggers.
    """

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__(config)

        # Content-based trigger (high change = should write)
        self.change_detector = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
            nn.Sigmoid(),
        )

        # Importance scorer (important content = should write)
        self.importance_scorer = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid(),
        )

        # Combine multiple signals
        self.signal_combiner = nn.Linear(3, 1)  # gate + change + importance

    def compute_write_strength(
        self,
        hidden_state: torch.Tensor,
        position: int,
        force_write: bool = False,
        prev_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute adaptive write strength based on multiple signals.

        Args:
            hidden_state: Current hidden state [batch, d_model]
            position: Current position
            force_write: Force writing
            prev_hidden: Previous hidden state for change detection

        Returns:
            Write strength [batch, 1]
        """
        if force_write:
            return torch.ones(hidden_state.size(0), 1, device=hidden_state.device)

        batch_size = hidden_state.size(0)
        device = hidden_state.device

        # Signal 1: Learned gate
        gate_signal = self.write_gate(hidden_state)

        # Signal 2: Change detection (if previous state available)
        if prev_hidden is not None:
            change_input = torch.cat([hidden_state, prev_hidden], dim=-1)
            change_signal = self.change_detector(change_input)
        else:
            change_signal = torch.zeros(batch_size, 1, device=device)

        # Signal 3: Importance scoring
        importance_signal = self.importance_scorer(hidden_state)

        # Combine signals
        signals = torch.cat([gate_signal, change_signal, importance_signal], dim=-1)
        combined = torch.sigmoid(self.signal_combiner(signals))

        return combined

    def forward(
        self,
        hidden_state: torch.Tensor,
        position: int,
        force_write: bool = False,
        prev_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Create snapshot with adaptive writing decision.

        Args:
            hidden_state: Current hidden state
            position: Current position
            force_write: Force writing
            prev_hidden: Previous hidden state for change detection

        Returns:
            Tuple of (full_snapshot, compressed_snapshot, write_strength)
        """
        # Handle sequence dimension
        if hidden_state.dim() == 3:
            hidden_state = hidden_state[:, -1, :]

        batch_size = hidden_state.size(0)
        device = hidden_state.device

        # Compute adaptive write strength
        write_strength = self.compute_write_strength(
            hidden_state, position, force_write, prev_hidden
        )

        # Only write if strength exceeds threshold (for efficiency)
        # But keep it differentiable by using soft thresholding
        threshold = 0.3
        effective_strength = F.relu(write_strength - threshold) / (1 - threshold)

        # Transform content
        content = self.content_transform(hidden_state)

        # Add position encoding
        pos_tensor = torch.tensor(
            [[position]], dtype=torch.float32, device=device
        ).expand(batch_size, 1) / 256.0
        pos_emb = self.position_encoder(pos_tensor)
        pos_emb_padded = F.pad(pos_emb, (0, content.size(-1) - pos_emb.size(-1)))
        content = content + pos_emb_padded
        content = self.dropout(content)

        # Compressed version
        compressed = None
        if self.compressor is not None:
            compressed = self.compressor(hidden_state)
            compressed = self.dropout(compressed)

        return content, compressed, effective_strength
