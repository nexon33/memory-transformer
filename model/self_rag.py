"""Self-RAG: Retrieval-Augmented Generation with same model for embed & generate."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
import random

from .self_rag_config import SelfRAGConfig
from .memory_store import MemoryStore
from .embeddings import RotaryPositionalEncoding, apply_rotary_pos_emb, rotate_half
from .attention import RMSNorm


class BaseTransformer(nn.Module):
    """Simple transformer without memory ops (for Self-RAG)."""

    def __init__(self, config: SelfRAGConfig):
        super().__init__()
        self.config = config

        # Token + positional embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryPositionalEncoding(config)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.n_layers)
        ])

        # Output
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.config.init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=self.config.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            return_hidden_states: Return all layer outputs

        Returns:
            Dict with logits and optionally hidden_states
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        hidden = self.token_embedding(input_ids)

        # RoPE
        cos, sin = self.rope(hidden, seq_len=seq_len)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1
        )

        all_hidden = [hidden] if return_hidden_states else None

        # Transformer layers
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, causal_mask, attention_mask)
            if return_hidden_states:
                all_hidden.append(hidden)

        # Output
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        outputs = {"logits": logits, "last_hidden_state": hidden}
        if return_hidden_states:
            outputs["hidden_states"] = all_hidden

        return outputs

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        pooling: str = "mean",
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get pooled embeddings for retrieval.

        Args:
            input_ids: [batch, seq_len]
            pooling: "mean", "last", "first", "cls"
            attention_mask: [batch, seq_len]

        Returns:
            Embeddings [batch, d_model]
        """
        outputs = self.forward(input_ids, attention_mask, return_hidden_states=False)
        hidden = outputs["last_hidden_state"]

        if pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return hidden.mean(dim=1)
        elif pooling == "last":
            return hidden[:, -1, :]
        elif pooling == "first" or pooling == "cls":
            return hidden[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")


class TransformerLayer(nn.Module):
    """Single transformer layer."""

    def __init__(self, config: SelfRAGConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        # Attention
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # FFN
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

        # Norms
        self.attn_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention
        residual = hidden
        hidden = self.attn_norm(hidden)

        batch, seq_len, _ = hidden.shape

        q = self.q_proj(hidden).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal mask
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        out = self.o_proj(out)

        hidden = residual + self.dropout(out)

        # FFN (SwiGLU)
        residual = hidden
        hidden = self.ffn_norm(hidden)

        gate = F.silu(self.gate_proj(hidden))
        up = self.up_proj(hidden)
        hidden = self.down_proj(gate * up)

        hidden = residual + self.dropout(hidden)

        return hidden


class SelfRAGModel(nn.Module):
    """Self-RAG: Same model for embedding and generation with retrieval augmentation.

    Flow:
    1. Input → Embed → Query vector
    2. Query → Memory search → Top-k chunks
    3. Prepend chunks to input → Augmented input
    4. Augmented input → Generate
    """

    def __init__(
        self,
        config: SelfRAGConfig,
        tokenizer: Any = None,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        # Base transformer
        self.model = BaseTransformer(config)

        # Memory store
        self.memory = MemoryStore(
            d_model=config.d_model,
            max_chunks=config.max_chunks,
        )

        # Special token IDs (set after tokenizer is assigned)
        self.mem_token_id = None
        self.sep_token_id = None

        self.n_params = sum(p.numel() for p in self.parameters())
        print(f"SelfRAGModel initialized with {self.n_params / 1e6:.2f}M parameters")

    def set_tokenizer(self, tokenizer):
        """Set tokenizer and configure special tokens."""
        self.tokenizer = tokenizer

        # Try to get special token IDs
        if hasattr(tokenizer, 'encode'):
            try:
                self.mem_token_id = tokenizer.encode(self.config.memory_token, add_special_tokens=False)[0]
            except:
                self.mem_token_id = tokenizer.unk_token_id or 0

            try:
                self.sep_token_id = tokenizer.encode(self.config.sep_token, add_special_tokens=False)[0]
            except:
                self.sep_token_id = tokenizer.sep_token_id or tokenizer.eos_token_id or 1

    def embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get embedding for retrieval query.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            Query embedding [batch, d_model]
        """
        with torch.no_grad():
            emb = self.model.get_embeddings(
                input_ids,
                pooling=self.config.embed_pooling,
                attention_mask=attention_mask,
            )
        if self.config.normalize_embeddings:
            emb = F.normalize(emb, dim=-1)
        return emb

    def retrieve(
        self,
        query_emb: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> List[List[Tuple[str, List[int], float]]]:
        """Retrieve top-k chunks from memory.

        Args:
            query_emb: [batch, d_model]
            top_k: Number of chunks to retrieve

        Returns:
            List of lists (per batch) of (chunk_text, token_ids, score)
        """
        if self.memory.size == 0:
            return [[] for _ in range(query_emb.size(0))]

        top_k = top_k or self.config.top_k

        results = self.memory.search(query_emb, top_k=top_k)

        # Handle batched vs single result
        if query_emb.dim() == 1 or query_emb.size(0) == 1:
            if isinstance(results[0], tuple):
                results = [results]

        # Extract just (chunk, tokens, score) for each result
        batch_results = []
        for batch_res in results:
            chunks = [(r[1], r[2], r[3]) for r in batch_res]  # (chunk, token_ids, score)
            batch_results.append(chunks)

        return batch_results

    def augment(
        self,
        input_ids: torch.Tensor,
        retrieved: List[List[Tuple[str, List[int], float]]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepend retrieved chunks to input.

        Args:
            input_ids: [batch, seq_len]
            retrieved: Retrieved chunks per batch item
            attention_mask: [batch, seq_len]

        Returns:
            Augmented input_ids and attention_mask
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        augmented_ids = []
        augmented_masks = []

        for b in range(batch_size):
            chunks = retrieved[b]
            prefix_ids = []

            # Prepend each retrieved chunk
            for chunk_text, token_ids, score in chunks:
                if self.mem_token_id is not None:
                    prefix_ids.append(self.mem_token_id)
                prefix_ids.extend(token_ids)

            # Add separator
            if len(prefix_ids) > 0 and self.sep_token_id is not None:
                prefix_ids.append(self.sep_token_id)

            # Combine with original input
            original = input_ids[b].tolist()
            combined = prefix_ids + original

            # Truncate if too long - keep end of original (most recent context)
            if len(combined) > self.config.max_seq_len:
                # Limit prefix to half of max_seq_len
                max_prefix = self.config.max_seq_len // 2
                if len(prefix_ids) > max_prefix:
                    prefix_ids = prefix_ids[:max_prefix]
                max_original = self.config.max_seq_len - len(prefix_ids)
                # Keep end of original (most relevant for generation)
                original = original[-max_original:] if max_original > 0 else []
                combined = prefix_ids + original

            augmented_ids.append(combined)

            # Build attention mask
            if attention_mask is not None:
                orig_mask = attention_mask[b].tolist()
                prefix_mask = [1] * len(prefix_ids)
                combined_mask = prefix_mask + orig_mask[:len(combined) - len(prefix_ids)]
                augmented_masks.append(combined_mask)

        # Pad to same length
        max_len = max(len(ids) for ids in augmented_ids)
        padded_ids = []
        padded_masks = []

        pad_id = self.tokenizer.pad_token_id if self.tokenizer else 0

        for i, ids in enumerate(augmented_ids):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            if augmented_masks:
                padded_masks.append(augmented_masks[i] + [0] * pad_len)

        result_ids = torch.tensor(padded_ids, device=device, dtype=input_ids.dtype)

        if augmented_masks:
            result_mask = torch.tensor(padded_masks, device=device, dtype=torch.bool)
        else:
            result_mask = torch.ones_like(result_ids, dtype=torch.bool)

        return result_ids, result_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_memory: bool = True,
        return_retrieved: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with retrieval augmentation.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, seq_len] for LM loss
            use_memory: Whether to use memory retrieval
            return_retrieved: Return retrieved chunks

        Returns:
            Dict with logits, loss, and optionally retrieved chunks
        """
        retrieved_chunks = None

        # Retrieval augmentation
        if use_memory and self.memory.size > 0:
            # Random dropout during training
            if self.training and random.random() < self.config.memory_dropout:
                use_memory = False

            if use_memory:
                # Get query embedding
                query_emb = self.embed(input_ids, attention_mask)

                # Retrieve
                retrieved_chunks = self.retrieve(query_emb)

                # Augment input
                input_ids, attention_mask = self.augment(
                    input_ids, retrieved_chunks, attention_mask
                )

                # Adjust labels if provided (pad prefix with -100)
                if labels is not None:
                    prefix_len = input_ids.size(1) - labels.size(1)
                    if prefix_len > 0:
                        prefix_labels = torch.full(
                            (labels.size(0), prefix_len),
                            -100,
                            device=labels.device,
                            dtype=labels.dtype
                        )
                        labels = torch.cat([prefix_labels, labels], dim=1)

        # Forward through base model
        outputs = self.model(input_ids, attention_mask)
        logits = outputs["logits"]

        result = {"logits": logits}

        # Compute loss
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        if return_retrieved:
            result["retrieved"] = retrieved_chunks

        return result

    def write_to_memory(
        self,
        text: str,
        token_ids: Optional[List[int]] = None,
        metadata: Optional[Dict] = None,
    ):
        """Store text chunk in memory.

        Args:
            text: Text content
            token_ids: Pre-tokenized IDs (computed if not provided)
            metadata: Optional metadata
        """
        if token_ids is None:
            if self.tokenizer is None:
                raise ValueError("Tokenizer required to encode text")
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        # Get embedding
        ids_tensor = torch.tensor([token_ids], device=next(self.parameters()).device)
        emb = self.embed(ids_tensor)

        self.memory.add(text, token_ids, emb[0], metadata)

    def write_batch_to_memory(
        self,
        texts: List[str],
        token_ids_batch: Optional[List[List[int]]] = None,
        metadata_batch: Optional[List[Dict]] = None,
    ):
        """Store multiple chunks in memory.

        Args:
            texts: List of text chunks
            token_ids_batch: Pre-tokenized IDs
            metadata_batch: Optional metadata per chunk
        """
        if token_ids_batch is None:
            if self.tokenizer is None:
                raise ValueError("Tokenizer required")
            token_ids_batch = [
                self.tokenizer.encode(t, add_special_tokens=False)
                for t in texts
            ]

        # Get embeddings
        device = next(self.parameters()).device
        max_len = max(len(ids) for ids in token_ids_batch)
        padded = [ids + [0] * (max_len - len(ids)) for ids in token_ids_batch]
        ids_tensor = torch.tensor(padded, device=device)

        embs = self.embed(ids_tensor)

        self.memory.add_batch(texts, token_ids_batch, embs, metadata_batch)

    def clear_memory(self):
        """Clear all memory."""
        self.memory.clear()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_memory: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text with retrieval augmentation.

        Args:
            input_ids: [batch, seq_len]
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            use_memory: Use memory retrieval
            eos_token_id: EOS token ID

        Returns:
            Generated token IDs [batch, seq_len + new_tokens]
        """
        # Initial retrieval augmentation
        if use_memory and self.memory.size > 0:
            query_emb = self.embed(input_ids)
            retrieved = self.retrieve(query_emb)
            input_ids, _ = self.augment(input_ids, retrieved)

        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate if too long
            if generated.size(1) > self.config.max_seq_len:
                generated = generated[:, -self.config.max_seq_len:]

            outputs = self.model(generated)
            logits = outputs["logits"][:, -1, :]

            if temperature != 1.0:
                logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated
