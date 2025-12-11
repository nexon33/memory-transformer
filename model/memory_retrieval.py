"""Memory retrieval module with multi-hop and continuous attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from .config import MemoryTransformerConfig
from .attention import MemoryAttention


class MultiHopRetrieval(nn.Module):
    """Multi-hop memory retrieval with iterative query refinement.

    Each hop:
    1. Query memory with current query
    2. Retrieve relevant content
    3. Refine query using retrieved content
    4. Repeat
    """

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.n_hops = config.n_retrieval_hops
        self.d_model = config.d_model
        self.d_memory = config.d_memory

        # Per-hop query refinement
        self.query_refiners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model * 2, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.GELU(),
            )
            for _ in range(self.n_hops)
        ])

        # Per-hop memory attention
        self.memory_attentions = nn.ModuleList([
            MemoryAttention(config)
            for _ in range(self.n_hops)
        ])

        # Hop combination (combine all hops into final output)
        self.hop_combiner = nn.Sequential(
            nn.Linear(config.d_model * self.n_hops, config.d_model),
            nn.LayerNorm(config.d_model),
        )

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
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Perform multi-hop retrieval.

        Args:
            query: Initial query [batch, seq_len, d_model]
            memory: Memory bank [batch, memory_size, d_memory]
            memory_mask: Valid memory slots [batch, memory_size]
            return_intermediate: Return intermediate hop results

        Returns:
            Final retrieved content [batch, seq_len, d_model]
            Optional list of intermediate attention weights
        """
        batch_size, seq_len, _ = query.shape
        current_query = query
        hop_outputs = []
        attention_weights = []

        for hop in range(self.n_hops):
            # Retrieve from memory (MemoryAttention already projects to d_model)
            retrieved, attn = self.memory_attentions[hop](
                current_query, memory, memory_mask, return_attention=True
            )

            hop_outputs.append(retrieved)
            if return_intermediate:
                attention_weights.append(attn)

            # Refine query for next hop (except last hop)
            if hop < self.n_hops - 1:
                # Concatenate current query with retrieved content
                combined = torch.cat([current_query, retrieved], dim=-1)
                current_query = self.query_refiners[hop](combined)
                current_query = self.dropout(current_query)

        # Combine all hop outputs
        combined_hops = torch.cat(hop_outputs, dim=-1)  # [batch, seq_len, d_model * n_hops]
        output = self.hop_combiner(combined_hops)

        if return_intermediate:
            return output, attention_weights
        return output, None


class ContinuousAttentionRetrieval(nn.Module):
    """Continuous attention over all memory slots.

    Treats memory as an extended context and attends to all slots
    simultaneously using standard multi-head attention.
    """

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_memory = config.d_memory
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        # Project memory to model dimension
        self.memory_proj = nn.Linear(config.d_memory, config.d_model)

        # Cross-attention from hidden states to memory
        self.cross_attention = MemoryAttention(config)

        # Layer norm
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.memory_dropout)

        self._init_weights(config.init_std)

    def _init_weights(self, std: float):
        nn.init.normal_(self.memory_proj.weight, mean=0.0, std=std)
        if self.memory_proj.bias is not None:
            nn.init.zeros_(self.memory_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Attend to all memory slots.

        Args:
            query: Query tensor [batch, seq_len, d_model]
            memory: Memory bank [batch, memory_size, d_memory]
            memory_mask: Valid memory slots [batch, memory_size]
            return_attention: Return attention weights

        Returns:
            Retrieved content [batch, seq_len, d_model]
            Optional attention weights
        """
        # Cross-attention to memory
        attended, attn_weights = self.cross_attention(
            query, memory, memory_mask, return_attention=return_attention
        )

        # Residual and norm
        output = self.norm(query + self.dropout(attended))

        return output, attn_weights


class HybridRetrieval(nn.Module):
    """Combines multi-hop and continuous attention retrieval.

    Uses a learned gate to blend the two retrieval strategies.
    """

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.multi_hop = MultiHopRetrieval(config)
        self.continuous = ContinuousAttentionRetrieval(config)

        # Learned gate to combine strategies
        self.strategy_gate = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 2),
            nn.Softmax(dim=-1),
        )

        # Final projection
        self.output_proj = nn.Linear(config.d_model, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)

        self._init_weights(config.init_std)

    def _init_weights(self, std: float):
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=std)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Hybrid retrieval combining multi-hop and continuous attention.

        Args:
            query: Query tensor [batch, seq_len, d_model]
            memory: Memory bank [batch, memory_size, d_memory]
            memory_mask: Valid memory slots [batch, memory_size]
            return_details: Return detailed retrieval information

        Returns:
            Retrieved content [batch, seq_len, d_model]
            Optional dictionary with retrieval details
        """
        # Multi-hop retrieval
        multi_hop_out, hop_attns = self.multi_hop(
            query, memory, memory_mask, return_intermediate=return_details
        )

        # Continuous attention retrieval
        continuous_out, cont_attn = self.continuous(
            query, memory, memory_mask, return_attention=return_details
        )

        # Compute strategy weights (content-dependent)
        strategy_weights = self.strategy_gate(query)  # [batch, seq_len, 2]
        multi_hop_weight = strategy_weights[..., 0:1]
        continuous_weight = strategy_weights[..., 1:2]

        # Blend strategies
        combined = multi_hop_weight * multi_hop_out + continuous_weight * continuous_out

        # Final projection
        output = self.output_proj(combined)
        output = self.norm(output)

        if return_details:
            details = {
                "multi_hop_output": multi_hop_out,
                "continuous_output": continuous_out,
                "strategy_weights": strategy_weights,
                "hop_attentions": hop_attns,
                "continuous_attention": cont_attn,
            }
            return output, details

        return output, None


class MemoryRetrieval(nn.Module):
    """Main memory retrieval module.

    Wraps the hybrid retrieval with additional processing.
    """

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.config = config

        # Main retrieval module
        self.retrieval = HybridRetrieval(config)

        # Optional: separate retrieval for compressed memory
        if config.use_both:
            # Create a modified config for compressed retrieval
            compressed_config = MemoryTransformerConfig(
                d_model=config.d_model,
                d_memory=config.d_compressed,
                d_compressed=config.d_compressed,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                d_ff=config.d_ff,
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len,
                memory_size=config.memory_size,
                snapshot_interval=config.snapshot_interval,
                n_retrieval_hops=config.n_retrieval_hops,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                memory_dropout=config.memory_dropout,
                use_compressed=True,
                use_both=False,
            )
            self.compressed_retrieval = HybridRetrieval(compressed_config)

            # Combine full and compressed
            self.combine_gate = nn.Sequential(
                nn.Linear(config.d_model * 2, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )
        else:
            self.compressed_retrieval = None
            self.combine_gate = None

    def forward(
        self,
        query: torch.Tensor,
        memory_state: Dict[str, torch.Tensor],
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Retrieve from memory.

        Args:
            query: Query tensor [batch, seq_len, d_model]
            memory_state: Memory state dictionary
            return_details: Return detailed information

        Returns:
            Retrieved content [batch, seq_len, d_model]
            Optional details dictionary
        """
        memory = memory_state["memory"]
        memory_mask = memory_state["valid_mask"]

        # Main retrieval from full memory
        output, details = self.retrieval(
            query, memory, memory_mask, return_details=return_details
        )

        # If using both full and compressed
        if self.compressed_retrieval is not None:
            compressed_memory = memory_state["memory_compressed"]
            compressed_out, compressed_details = self.compressed_retrieval(
                query, compressed_memory, memory_mask, return_details=return_details
            )

            # Combine full and compressed retrievals
            combined_input = torch.cat([output, compressed_out], dim=-1)
            output = self.combine_gate(combined_input)

            if return_details and details is not None:
                details["compressed_retrieval"] = compressed_details

        return output, details
