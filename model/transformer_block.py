"""Transformer block with memory interaction."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from .config import MemoryTransformerConfig
from .attention import CausalSelfAttention, FeedForward, RMSNorm
from .memory_bank import MemoryBank
from .snapshot_writer import SnapshotWriter, AdaptiveSnapshotWriter
from .memory_retrieval import MemoryRetrieval
from .memory_integration import MemoryIntegration


class MemoryTransformerBlock(nn.Module):
    """Single transformer block with memory interaction.

    Architecture:
    1. Self-attention (causal)
    2. Memory snapshot (write to memory)
    3. Memory retrieval (read from memory)
    4. Memory integration (combine with hidden state)
    5. Feed-forward network
    """

    def __init__(
        self,
        config: MemoryTransformerConfig,
        layer_idx: int,
        use_adaptive_writer: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Pre-attention layer norm
        self.norm1 = RMSNorm(config.d_model)

        # Self-attention
        self.self_attention = CausalSelfAttention(config)

        # Memory interaction (only on certain layers for efficiency)
        self.use_memory = layer_idx >= 1  # Skip first layer

        if self.use_memory:
            # Snapshot writer
            if use_adaptive_writer:
                self.snapshot_writer = AdaptiveSnapshotWriter(config)
            else:
                self.snapshot_writer = SnapshotWriter(config)

            # Memory retrieval
            self.memory_retrieval = MemoryRetrieval(config)

            # Memory integration
            self.memory_integration = MemoryIntegration(config)

            # Pre-memory layer norm
            self.norm_memory = RMSNorm(config.d_model)

        # Pre-FFN layer norm
        self.norm2 = RMSNorm(config.d_model)

        # Feed-forward network
        self.ffn = FeedForward(config)

        # Gradient checkpointing flag
        self.gradient_checkpointing = config.gradient_checkpointing

    def _memory_forward(
        self,
        hidden_states: torch.Tensor,
        memory_state: Dict[str, torch.Tensor],
        memory_bank: MemoryBank,
        position: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Process memory interaction.

        Args:
            hidden_states: Current hidden states [batch, seq_len, d_model]
            memory_state: Current memory state
            memory_bank: Memory bank module
            position: Current position in sequence

        Returns:
            Updated hidden states [batch, seq_len, d_model]
            Updated memory state
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Process each position in sequence
        for pos_offset in range(seq_len):
            current_pos = position + pos_offset
            current_hidden = hidden_states[:, pos_offset, :]  # [batch, d_model]

            # Try to write snapshot
            content, compressed, write_strength = self.snapshot_writer(
                current_hidden.unsqueeze(1),
                current_pos,
            )

            if content is not None:
                # Write to memory bank
                memory_state = memory_bank.write(
                    memory_state,
                    content,
                    compressed,
                    write_strength,
                )

        # Retrieve from memory
        normed = self.norm_memory(hidden_states)
        retrieved, _ = self.memory_retrieval(normed, memory_state)

        # Integrate retrieved memory
        hidden_states, _ = self.memory_integration(hidden_states, retrieved)

        return hidden_states, memory_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_state: Optional[Dict[str, torch.Tensor]] = None,
        memory_bank: Optional[MemoryBank] = None,
        position: int = 0,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        """Forward pass through transformer block.

        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            memory_state: Current memory state (if using memory)
            memory_bank: Memory bank module (if using memory)
            position: Starting position in sequence
            cos: Cosine values for RoPE
            sin: Sine values for RoPE
            return_attention: Return attention weights

        Returns:
            Output tensor [batch, seq_len, d_model]
            Updated memory state (or None)
            Optional attention weights
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attn_output, attn_weights = self.self_attention(
            hidden_states, cos=cos, sin=sin, return_attention=return_attention
        )
        hidden_states = residual + attn_output

        # Memory interaction
        if self.use_memory and memory_state is not None and memory_bank is not None:
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing for memory operations
                hidden_states, memory_state = torch.utils.checkpoint.checkpoint(
                    self._memory_forward,
                    hidden_states,
                    memory_state,
                    memory_bank,
                    position,
                    use_reentrant=False,
                )
            else:
                hidden_states, memory_state = self._memory_forward(
                    hidden_states, memory_state, memory_bank, position
                )

        # Feed-forward with residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.ffn(hidden_states)

        return hidden_states, memory_state, attn_weights


class MemoryTransformerBlockNoCheckpoint(MemoryTransformerBlock):
    """Transformer block without gradient checkpointing for inference."""

    def __init__(self, config: MemoryTransformerConfig, layer_idx: int):
        # Temporarily disable checkpointing in config
        orig_checkpointing = config.gradient_checkpointing
        config.gradient_checkpointing = False
        super().__init__(config, layer_idx)
        config.gradient_checkpointing = orig_checkpointing
        self.gradient_checkpointing = False


class TransformerBlockStack(nn.Module):
    """Stack of transformer blocks with shared memory."""

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.config = config
        self.n_layers = config.n_layers

        # Create transformer blocks
        self.blocks = nn.ModuleList([
            MemoryTransformerBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Shared memory bank across all layers
        self.memory_bank = MemoryBank(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_state: Optional[Dict[str, torch.Tensor]] = None,
        position: int = 0,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[list], Optional[list]]:
        """Forward pass through all transformer blocks.

        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            memory_state: Initial memory state (or None to initialize)
            position: Starting position
            cos: Cosine values for RoPE
            sin: Sine values for RoPE
            return_all_hidden_states: Return hidden states from all layers
            return_attention: Return attention weights from all layers

        Returns:
            Final hidden states [batch, seq_len, d_model]
            Final memory state
            Optional list of hidden states from each layer
            Optional list of attention weights from each layer
        """
        batch_size = hidden_states.size(0)
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Initialize memory state if not provided
        if memory_state is None:
            memory_state = self.memory_bank.init_memory(batch_size, device, dtype)

        all_hidden_states = [] if return_all_hidden_states else None
        all_attention_weights = [] if return_attention else None

        # Process through each block
        for block in self.blocks:
            if return_all_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states, memory_state, attn_weights = block(
                hidden_states,
                memory_state=memory_state,
                memory_bank=self.memory_bank,
                position=position,
                cos=cos,
                sin=sin,
                return_attention=return_attention,
            )

            if return_attention and attn_weights is not None:
                all_attention_weights.append(attn_weights)

        if return_all_hidden_states:
            all_hidden_states.append(hidden_states)

        return hidden_states, memory_state, all_hidden_states, all_attention_weights
