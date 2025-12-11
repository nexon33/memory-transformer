"""Full Memory-Augmented Transformer model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from .config import MemoryTransformerConfig
from .embeddings import TransformerEmbedding, RotaryPositionalEncoding
from .attention import RMSNorm
from .transformer_block import TransformerBlockStack
from .memory_bank import MemoryBank


class MemoryTransformer(nn.Module):
    """Memory-Augmented Transformer for reasoning and retrieval tasks.

    This model combines a standard transformer with a learned memory system:
    - Periodically creates snapshots of hidden states
    - Learns when and what to write to memory
    - Uses multi-hop retrieval to read from memory
    - Integrates retrieved memory with current computation
    """

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.config = config

        # Token + positional embeddings
        self.embeddings = TransformerEmbedding(config, use_rope=True)

        # Rotary position encoding (for attention)
        self.rope = RotaryPositionalEncoding(config)

        # Transformer blocks with memory
        self.transformer = TransformerBlockStack(config)

        # Output layer norm
        self.norm_out = RMSNorm(config.d_model)

        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embeddings weights with lm_head
        self.lm_head.weight = self.embeddings.token_embedding.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {self.n_params / 1e6:.2f}M parameters")

    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def get_memory_bank(self) -> MemoryBank:
        """Get the memory bank module."""
        return self.transformer.memory_bank

    def init_memory(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, torch.Tensor]:
        """Initialize memory state for a batch."""
        return self.transformer.memory_bank.init_memory(batch_size, device, dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_state: Optional[Dict[str, torch.Tensor]] = None,
        position: int = 0,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        return_attention: bool = False,
        return_memory_state: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            memory_state: Current memory state (or None to initialize)
            position: Starting position in sequence
            labels: Target labels for language modeling loss [batch, seq_len]
            return_hidden_states: Return all hidden states
            return_attention: Return attention weights
            return_memory_state: Return updated memory state

        Returns:
            Dictionary containing:
                - logits: Output logits [batch, seq_len, vocab_size]
                - loss: Language modeling loss (if labels provided)
                - memory_state: Updated memory state (if return_memory_state)
                - hidden_states: All hidden states (if return_hidden_states)
                - attentions: Attention weights (if return_attention)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        # Initialize memory if needed
        if memory_state is None:
            memory_state = self.init_memory(batch_size, device, dtype)

        # Get embeddings (RoPE is applied in attention)
        hidden_states = self.embeddings(input_ids)

        # Get RoPE cos/sin values
        cos, sin = self.rope(hidden_states, seq_len=seq_len)

        # Forward through transformer blocks
        hidden_states, memory_state, all_hidden_states, all_attentions = self.transformer(
            hidden_states,
            memory_state=memory_state,
            position=position,
            cos=cos,
            sin=sin,
            return_all_hidden_states=return_hidden_states,
            return_attention=return_attention,
        )

        # Output norm
        hidden_states = self.norm_out(hidden_states)

        # Language modeling head
        logits = self.lm_head(hidden_states)

        # Build output dictionary
        outputs = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten and compute cross-entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,  # Ignore padding
            )
            outputs["loss"] = loss

        if return_memory_state:
            outputs["memory_state"] = memory_state

        if return_hidden_states and all_hidden_states is not None:
            outputs["hidden_states"] = all_hidden_states

        if return_attention and all_attentions is not None:
            outputs["attentions"] = all_attentions

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        memory_state: Optional[Dict[str, torch.Tensor]] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate text autoregressively.

        Args:
            input_ids: Initial token IDs [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            memory_state: Initial memory state
            eos_token_id: End of sequence token ID

        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
            Final memory state
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        # Initialize memory
        if memory_state is None:
            memory_state = self.init_memory(batch_size, device, dtype)

        # Process initial context
        generated = input_ids.clone()
        position = 0

        # Generate tokens one by one
        for _ in range(max_new_tokens):
            # Get logits for last position
            outputs = self.forward(
                generated,
                memory_state=memory_state,
                position=position,
                return_memory_state=True,
            )

            logits = outputs["logits"][:, -1, :]  # [batch, vocab_size]
            memory_state = outputs["memory_state"]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample next token
            # Clamp logits to prevent numerical issues
            logits = torch.clamp(logits, min=-100, max=100)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
            probs = F.softmax(logits, dim=-1)
            # Ensure no NaN/inf in probs
            probs = torch.nan_to_num(probs, nan=1e-8, posinf=1.0, neginf=0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # Update position
            position = generated.size(1) - 1

        return generated, memory_state

    def get_memory_statistics(
        self,
        memory_state: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Get statistics about memory usage.

        Args:
            memory_state: Current memory state

        Returns:
            Dictionary of memory statistics
        """
        valid_mask = memory_state["valid_mask"]
        usage = memory_state["usage"]

        stats = {
            "num_valid_slots": valid_mask.float().sum(dim=-1).mean().item(),
            "total_writes": memory_state["total_writes"].float().mean().item(),
            "usage_mean": usage.mean().item(),
            "usage_std": usage.std().item(),
            "usage_max": usage.max().item(),
            "usage_sparsity": (usage < 0.1).float().mean().item(),
        }

        return stats


class MemoryTransformerForSequenceClassification(MemoryTransformer):
    """Memory Transformer with a classification head."""

    def __init__(self, config: MemoryTransformerConfig, num_labels: int):
        super().__init__(config)
        self.num_labels = num_labels

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_state: Optional[Dict[str, torch.Tensor]] = None,
        position: int = 0,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for classification.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            memory_state: Current memory state
            position: Starting position
            labels: Classification labels [batch]

        Returns:
            Dictionary with logits and optionally loss
        """
        # Get base model outputs
        outputs = super().forward(
            input_ids,
            memory_state=memory_state,
            position=position,
            return_memory_state=True,
        )

        # Get the last token's hidden state (or mean pool)
        # Here we use mean pooling over sequence
        hidden_states = outputs.get("hidden_states", None)
        if hidden_states is None:
            # Re-run to get hidden states
            outputs = super().forward(
                input_ids,
                memory_state=memory_state,
                position=position,
                return_hidden_states=True,
                return_memory_state=True,
            )

        # Get last layer hidden states before LM head
        # We need to access the norm_out output
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        if memory_state is None:
            memory_state = self.init_memory(batch_size, device, dtype)

        hidden_states = self.embeddings(input_ids)
        cos, sin = self.rope(hidden_states, seq_len=seq_len)

        hidden_states, memory_state, _, _ = self.transformer(
            hidden_states,
            memory_state=memory_state,
            position=position,
            cos=cos,
            sin=sin,
        )
        hidden_states = self.norm_out(hidden_states)

        # Mean pool over sequence
        pooled = hidden_states.mean(dim=1)  # [batch, d_model]

        # Classify
        logits = self.classifier(pooled)

        result = {
            "logits": logits,
            "memory_state": memory_state,
        }

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result["loss"] = loss

        return result
