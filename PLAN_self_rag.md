# Self-RAG Memory Architecture

## Overview

Replace complex differentiable memory with simple, proven RAG-style retrieval using the same LLM for embedding and generation.

```
┌─────────────────────────────────────────────────────────────────┐
│                       MEMORY STORE                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ chunks: ["chunk1 text", "chunk2 text", ...]                 ││
│  │ embeddings: Tensor[n_chunks, d_model]                       ││
│  │ metadata: [{position, timestamp, source}, ...]              ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                         ↑ write                    ↓ retrieve
┌─────────────────────────────────────────────────────────────────┐
│                          LLM                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Embed   │ →  │ Retrieve │ →  │ Augment  │ →  │ Generate │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       ↓               ↓               ↓               ↓         │
│   query_emb      top-k chunks    [mem][sep][in]    output       │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. MemoryStore

```python
class MemoryStore:
    """Simple vector store for chunks + embeddings."""

    def __init__(self, d_model: int, max_chunks: int = 10000):
        self.chunks: List[str] = []
        self.token_ids: List[List[int]] = []  # Pre-tokenized
        self.embeddings: torch.Tensor = None  # [n, d_model]
        self.metadata: List[Dict] = []
        self.d_model = d_model
        self.max_chunks = max_chunks

    def add(self, chunk: str, embedding: torch.Tensor, meta: Dict = None):
        """Add chunk with precomputed embedding."""

    def search(self, query_emb: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """Cosine similarity search, return chunks + scores."""

    def search_batched(self, query_embs: torch.Tensor, top_k: int = 5):
        """Batch search for efficiency."""
```

### 2. SelfRAGModel

```python
class SelfRAGModel(nn.Module):
    """LLM with self-retrieval from memory."""

    def __init__(self, base_model: nn.Module, memory: MemoryStore, config: SelfRAGConfig):
        self.model = base_model
        self.memory = memory
        self.config = config

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get embedding for retrieval query (last hidden state or pooled)."""
        with torch.no_grad():
            hidden = self.model.forward_embed_only(input_ids)
        return hidden.mean(dim=1)  # [batch, d_model]

    def retrieve(self, query_emb: torch.Tensor, top_k: int = 5) -> List[str]:
        """Retrieve top-k chunks from memory."""
        return self.memory.search(query_emb, top_k)

    def augment(self, input_ids: torch.Tensor, retrieved_chunks: List[str]) -> torch.Tensor:
        """Prepend retrieved chunks to input."""
        # [MEM] chunk1 [MEM] chunk2 [SEP] original_input

    def forward(self, input_ids, labels=None, use_memory=True):
        """Full forward: embed → retrieve → augment → generate."""
        if use_memory and self.memory.size > 0:
            query_emb = self.embed(input_ids)
            chunks = self.retrieve(query_emb, self.config.top_k)
            input_ids = self.augment(input_ids, chunks)

        return self.model(input_ids, labels=labels)

    def write_to_memory(self, text: str):
        """Store text chunk in memory."""
        tokens = self.tokenizer.encode(text)
        emb = self.embed(torch.tensor([tokens]))
        self.memory.add(text, emb)
```

### 3. Config

```python
@dataclass
class SelfRAGConfig:
    # Base model
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8

    # Memory
    max_chunks: int = 10000
    chunk_size: int = 64  # Tokens per chunk
    top_k: int = 5

    # Retrieval
    retrieval_method: str = "cosine"  # or "dot", "learned"
    embed_pooling: str = "mean"  # or "last", "cls"

    # Training
    memory_dropout: float = 0.1  # Randomly skip retrieval
    contrastive_weight: float = 0.1  # Optional contrastive loss
```

## Training Strategy

### Phase 1: Pretrain Base Model (no memory)
```python
# Standard language modeling
for batch in dataloader:
    logits = model(batch.input_ids, use_memory=False)
    loss = cross_entropy(logits, batch.labels)
    loss.backward()
```

### Phase 2: Memory Integration
```python
# Train with retrieval, random memory dropout
for batch in dataloader:
    # Build memory from batch context
    model.memory.clear()
    for chunk in batch.context_chunks:
        model.write_to_memory(chunk)

    # Forward with retrieval (dropout = sometimes skip)
    use_mem = random.random() > config.memory_dropout
    logits = model(batch.input_ids, use_memory=use_mem)
    loss = cross_entropy(logits, batch.labels)

    # Optional: contrastive loss on retrieval
    if config.contrastive_weight > 0:
        loss += config.contrastive_weight * contrastive_loss(...)

    loss.backward()
```

### Phase 3: Position Recall Training (the smoking gun test)
```python
# Explicitly train on position-invariant recall
for batch in position_recall_dataloader:
    # Store N facts at different positions
    # Query each position
    # Loss should be uniform across positions
```

## File Structure

```
memory_transformer/
├── model/
│   ├── self_rag.py          # NEW: SelfRAGModel
│   ├── memory_store.py      # NEW: MemoryStore (replaces memory_bank.py)
│   └── base_transformer.py  # Simplified base model (no memory ops)
├── data/
│   ├── chunking.py          # NEW: Text chunking utilities
│   └── ...
├── training/
│   ├── trainer_rag.py       # NEW: RAG-style trainer
│   └── ...
└── configs/
    └── self_rag.yaml        # NEW: Config for self-RAG
```

## Implementation Steps

### Step 1: MemoryStore (memory_store.py)
- Simple list-based storage
- Cosine similarity search
- Optional FAISS backend for scale
- Circular buffer for max_chunks

### Step 2: Embedding extraction
- Add `forward_embed_only()` to base model
- Pool hidden states to single vector
- Cache embeddings for efficiency

### Step 3: SelfRAGModel wrapper
- Wrap base model
- Implement embed/retrieve/augment/generate
- Handle batched retrieval

### Step 4: Training loop
- Memory population from context
- Retrieval-augmented forward
- Standard LM loss

### Step 5: Position recall benchmark
- Test U-curve vs flat
- Prove memory helps middle positions

## Comparison: Old vs New

| Aspect | Old (Differentiable) | New (Self-RAG) |
|--------|---------------------|----------------|
| Memory content | Hidden states | Tokens + embeddings |
| Retrieval | Learned attention | Cosine similarity |
| Integration | Gated fusion | Context prepending |
| Trainable | End-to-end | Two-phase |
| Interpretable | No | Yes |
| Debuggable | Hard | Easy |
| Proven | Novel | RAG works |

## Why This Is Better

1. **Simpler** - No learned write gates, soft attention, multi-hop
2. **Debuggable** - Can inspect what's retrieved
3. **Proven** - RAG architecture works at scale
4. **Flexible** - Easy to swap retrieval methods
5. **Efficient** - Pre-computed embeddings, fast similarity

## Migration Path

1. Keep base transformer from current code
2. Replace `MemoryBank` → `MemoryStore`
3. Replace `MemoryRetrieval` → simple cosine search
4. Replace `MemoryIntegration` → context prepending
5. Simplify `TransformerBlock` → remove memory ops
6. Update trainer for RAG-style training

## Estimated Timeline

- Step 1 (MemoryStore): 1 hour
- Step 2 (Embedding): 30 min
- Step 3 (SelfRAGModel): 2 hours
- Step 4 (Training): 1 hour
- Step 5 (Benchmark): 1 hour

Total: ~6 hours to working prototype
