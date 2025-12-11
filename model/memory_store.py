"""Simple vector store for Self-RAG memory."""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import json


@dataclass
class MemoryEntry:
    """Single memory entry."""
    chunk: str
    token_ids: List[int]
    embedding: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    """Vector store with cosine similarity search.

    Stores text chunks with their embeddings for fast retrieval.
    Uses simple cosine similarity - can swap to FAISS for scale.
    """

    def __init__(
        self,
        d_model: int,
        max_chunks: int = 10000,
        device: torch.device = None,
    ):
        self.d_model = d_model
        self.max_chunks = max_chunks
        self.device = device or torch.device("cpu")

        # Storage
        self.chunks: List[str] = []
        self.token_ids: List[List[int]] = []
        self.metadata: List[Dict] = []

        # Embeddings tensor (built lazily)
        self._embeddings: Optional[torch.Tensor] = None
        self._embeddings_dirty = True

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def embeddings(self) -> Optional[torch.Tensor]:
        """Get embeddings tensor, rebuilding if dirty."""
        if self._embeddings_dirty and len(self.chunks) > 0:
            self._rebuild_embeddings()
        return self._embeddings

    def _rebuild_embeddings(self):
        """Rebuild embeddings tensor from list."""
        if len(self.chunks) == 0:
            self._embeddings = None
        else:
            # Embeddings stored in metadata temporarily during add
            embs = [m.get("_embedding") for m in self.metadata]
            self._embeddings = torch.stack(embs).to(self.device)
        self._embeddings_dirty = False

    def add(
        self,
        chunk: str,
        token_ids: List[int],
        embedding: torch.Tensor,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Add a chunk to memory.

        Args:
            chunk: Text content
            token_ids: Pre-tokenized IDs
            embedding: Embedding vector [d_model]
            metadata: Optional metadata dict

        Returns:
            Index of added chunk
        """
        # Circular buffer - remove oldest if at capacity
        if len(self.chunks) >= self.max_chunks:
            self.chunks.pop(0)
            self.token_ids.pop(0)
            self.metadata.pop(0)

        # Normalize embedding for cosine similarity
        embedding = F.normalize(embedding.detach().cpu(), dim=-1)

        # Store
        self.chunks.append(chunk)
        self.token_ids.append(token_ids)

        meta = metadata or {}
        meta["_embedding"] = embedding
        meta["position"] = len(self.chunks) - 1
        self.metadata.append(meta)

        self._embeddings_dirty = True

        return len(self.chunks) - 1

    def add_batch(
        self,
        chunks: List[str],
        token_ids_batch: List[List[int]],
        embeddings: torch.Tensor,
        metadata_batch: Optional[List[Dict]] = None,
    ) -> List[int]:
        """Add multiple chunks at once.

        Args:
            chunks: List of text chunks
            token_ids_batch: List of token ID lists
            embeddings: Embeddings tensor [batch, d_model]
            metadata_batch: Optional list of metadata dicts

        Returns:
            List of indices
        """
        indices = []
        metadata_batch = metadata_batch or [None] * len(chunks)

        for chunk, tids, emb, meta in zip(chunks, token_ids_batch, embeddings, metadata_batch):
            idx = self.add(chunk, tids, emb, meta)
            indices.append(idx)

        return indices

    def search(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[int, str, List[int], float, Dict]]:
        """Search for similar chunks.

        Args:
            query_embedding: Query vector [d_model] or [batch, d_model]
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (index, chunk, token_ids, score, metadata) tuples
        """
        if self.size == 0:
            return []

        # Normalize query
        query = F.normalize(query_embedding.detach(), dim=-1)

        # Handle batched vs single query
        if query.dim() == 1:
            query = query.unsqueeze(0)
            single_query = True
        else:
            single_query = False

        query = query.to(self.device)
        embeddings = self.embeddings.to(self.device)

        # Cosine similarity (embeddings already normalized)
        # query: [batch, d_model], embeddings: [n_chunks, d_model]
        similarities = torch.mm(query, embeddings.t())  # [batch, n_chunks]

        # Get top-k
        k = min(top_k, self.size)
        scores, indices = similarities.topk(k, dim=-1)

        if single_query:
            scores = scores[0]
            indices = indices[0]

            results = []
            for score, idx in zip(scores.tolist(), indices.tolist()):
                if score >= threshold:
                    results.append((
                        idx,
                        self.chunks[idx],
                        self.token_ids[idx],
                        score,
                        {k: v for k, v in self.metadata[idx].items() if not k.startswith("_")}
                    ))
            return results
        else:
            # Batched results
            batch_results = []
            for b in range(scores.size(0)):
                results = []
                for score, idx in zip(scores[b].tolist(), indices[b].tolist()):
                    if score >= threshold:
                        results.append((
                            idx,
                            self.chunks[idx],
                            self.token_ids[idx],
                            score,
                            {k: v for k, v in self.metadata[idx].items() if not k.startswith("_")}
                        ))
                batch_results.append(results)
            return batch_results

    def get(self, index: int) -> Tuple[str, List[int], Dict]:
        """Get chunk by index."""
        return self.chunks[index], self.token_ids[index], self.metadata[index]

    def clear(self):
        """Clear all memory."""
        self.chunks = []
        self.token_ids = []
        self.metadata = []
        self._embeddings = None
        self._embeddings_dirty = True

    def save(self, path: str):
        """Save memory to disk."""
        data = {
            "chunks": self.chunks,
            "token_ids": self.token_ids,
            "metadata": [
                {k: v.tolist() if isinstance(v, torch.Tensor) else v
                 for k, v in m.items()}
                for m in self.metadata
            ],
            "d_model": self.d_model,
            "max_chunks": self.max_chunks,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load memory from disk."""
        with open(path, "r") as f:
            data = json.load(f)

        self.chunks = data["chunks"]
        self.token_ids = data["token_ids"]
        self.d_model = data["d_model"]
        self.max_chunks = data["max_chunks"]

        # Rebuild metadata with embeddings as tensors
        self.metadata = []
        for m in data["metadata"]:
            meta = {}
            for k, v in m.items():
                if k == "_embedding":
                    meta[k] = torch.tensor(v)
                else:
                    meta[k] = v
            self.metadata.append(meta)

        self._embeddings_dirty = True

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"MemoryStore(size={self.size}, d_model={self.d_model}, max_chunks={self.max_chunks})"
