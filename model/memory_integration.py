"""Memory integration module for combining retrieved content with hidden states."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import MemoryTransformerConfig


class GatedMemoryIntegration(nn.Module):
    """Gated integration of retrieved memory with current hidden state.

    Uses a learned gate to control how much memory information to incorporate.
    """

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.d_model = config.d_model

        # Gate computation
        self.gate_net = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.Sigmoid(),
        )

        # Transform for memory content
        self.memory_transform = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
        )

        # Final combination
        self.combine = nn.Linear(config.d_model * 2, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.memory_dropout)

        self._init_weights(config.init_std)

    def _init_weights(self, std: float):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_state: torch.Tensor,
        retrieved_memory: torch.Tensor,
        return_gate: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Integrate retrieved memory with hidden state.

        Args:
            hidden_state: Current hidden state [batch, seq_len, d_model]
            retrieved_memory: Retrieved memory content [batch, seq_len, d_model]
            return_gate: Whether to return gate values

        Returns:
            Integrated hidden state [batch, seq_len, d_model]
            Optional gate values [batch, seq_len, d_model]
        """
        # Transform memory
        memory_transformed = self.memory_transform(retrieved_memory)

        # Compute gate
        gate_input = torch.cat([hidden_state, memory_transformed], dim=-1)
        gate = self.gate_net(gate_input)

        # Apply gate to memory
        gated_memory = gate * memory_transformed

        # Combine with hidden state
        combined = torch.cat([hidden_state, gated_memory], dim=-1)
        output = self.combine(combined)
        output = self.dropout(output)

        # Residual connection and norm
        output = self.norm(hidden_state + output)

        if return_gate:
            return output, gate
        return output, None


class CrossAttentionIntegration(nn.Module):
    """Integration using cross-attention between hidden state and memory."""

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim ** -0.5

        # Cross-attention projections
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        # FFN for post-attention processing
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

        self.dropout = nn.Dropout(config.memory_dropout)
        self._init_weights(config.init_std)

    def _init_weights(self, std: float):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_state: torch.Tensor,
        retrieved_memory: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Integrate using cross-attention.

        Args:
            hidden_state: Current hidden state [batch, seq_len, d_model]
            retrieved_memory: Retrieved memory content [batch, seq_len, d_model]
            return_attention: Return attention weights

        Returns:
            Integrated hidden state [batch, seq_len, d_model]
            Optional attention weights
        """
        batch_size, seq_len, _ = hidden_state.shape

        # Pre-norm
        hidden_normed = self.norm1(hidden_state)
        memory_normed = self.norm1(retrieved_memory)

        # Cross-attention: query from hidden, key/value from memory
        q = self.q_proj(hidden_normed)
        k = self.k_proj(memory_normed)
        v = self.v_proj(memory_normed)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights_dropped, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)

        # Residual
        hidden_state = hidden_state + self.dropout(attn_output)

        # FFN with residual
        hidden_state = hidden_state + self.ffn(self.norm2(hidden_state))

        if return_attention:
            return hidden_state, attn_weights
        return hidden_state, None


class MemoryIntegration(nn.Module):
    """Main memory integration module.

    Combines gated integration and cross-attention for robust memory incorporation.
    """

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.config = config

        # Primary integration method
        self.gated_integration = GatedMemoryIntegration(config)

        # Optional cross-attention integration
        self.use_cross_attention = True
        if self.use_cross_attention:
            self.cross_attention_integration = CrossAttentionIntegration(config)

        # Method selector (learned)
        self.method_gate = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 2),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        retrieved_memory: torch.Tensor,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """Integrate memory with hidden state.

        Args:
            hidden_state: Current hidden state [batch, seq_len, d_model]
            retrieved_memory: Retrieved memory content [batch, seq_len, d_model]
            return_details: Return integration details

        Returns:
            Integrated hidden state [batch, seq_len, d_model]
            Optional details dictionary
        """
        # Gated integration
        gated_output, gate_values = self.gated_integration(
            hidden_state, retrieved_memory, return_gate=return_details
        )

        if self.use_cross_attention:
            # Cross-attention integration
            cross_output, cross_attn = self.cross_attention_integration(
                hidden_state, retrieved_memory, return_attention=return_details
            )

            # Learned combination of methods
            method_weights = self.method_gate(hidden_state)  # [batch, seq_len, 2]
            gated_weight = method_weights[..., 0:1]
            cross_weight = method_weights[..., 1:2]

            output = gated_weight * gated_output + cross_weight * cross_output
        else:
            output = gated_output

        if return_details:
            details = {
                "gate_values": gate_values,
                "gated_output": gated_output,
            }
            if self.use_cross_attention:
                details["cross_attention"] = cross_attn
                details["cross_output"] = cross_output
                details["method_weights"] = method_weights
            return output, details

        return output, None
