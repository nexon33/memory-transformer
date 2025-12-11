"""Configuration for Self-RAG model."""

from dataclasses import dataclass, field
from typing import Optional, List
import yaml


@dataclass
class SelfRAGConfig:
    """Configuration for Self-RAG Memory Transformer.

    Simpler than differentiable memory - uses retrieval augmentation.
    """

    # Base model dimensions
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    d_ff: int = 2048
    vocab_size: int = 32000
    max_seq_len: int = 1024

    # Memory configuration
    max_chunks: int = 10000       # Maximum chunks in memory
    chunk_size: int = 64          # Tokens per chunk
    chunk_overlap: int = 16       # Overlap between chunks
    top_k: int = 5                # Retrieved chunks per query

    # Retrieval settings
    retrieval_method: str = "cosine"  # "cosine", "dot", "learned"
    embed_pooling: str = "mean"       # "mean", "last", "first", "cls"
    normalize_embeddings: bool = True

    # Special tokens
    memory_token: str = "[MEM]"
    sep_token: str = "[SEP]"

    # Training settings
    memory_dropout: float = 0.1   # Prob of skipping retrieval during training
    retrieval_loss_weight: float = 0.0  # Contrastive loss weight (0 = disabled)

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Initialization
    init_std: float = 0.02

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def validate(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.top_k > 0, "top_k must be positive"
        assert self.chunk_size > 0, "chunk_size must be positive"
        assert self.retrieval_method in ["cosine", "dot", "learned"], \
            f"Unknown retrieval method: {self.retrieval_method}"
        assert self.embed_pooling in ["mean", "last", "first", "cls"], \
            f"Unknown pooling method: {self.embed_pooling}"

    def __post_init__(self):
        self.validate()

    @classmethod
    def from_yaml(cls, path: str) -> "SelfRAGConfig":
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: str):
        config_dict = {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "max_chunks": self.max_chunks,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "retrieval_method": self.retrieval_method,
            "embed_pooling": self.embed_pooling,
            "normalize_embeddings": self.normalize_embeddings,
            "memory_token": self.memory_token,
            "sep_token": self.sep_token,
            "memory_dropout": self.memory_dropout,
            "retrieval_loss_weight": self.retrieval_loss_weight,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "init_std": self.init_std,
        }
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# Preset configurations
def tiny_config() -> SelfRAGConfig:
    """Tiny model for testing (~5M params)."""
    return SelfRAGConfig(
        d_model=192,
        n_heads=4,
        n_layers=4,
        d_ff=384,
        vocab_size=4096,
        max_seq_len=256,
        max_chunks=1000,
        chunk_size=32,
        top_k=3,
    )


def small_config() -> SelfRAGConfig:
    """Small model (~50M params)."""
    return SelfRAGConfig(
        d_model=384,
        n_heads=6,
        n_layers=8,
        d_ff=1024,
        vocab_size=16000,
        max_seq_len=512,
        max_chunks=5000,
        chunk_size=48,
        top_k=5,
    )


def medium_config() -> SelfRAGConfig:
    """Medium model for A100 (~180M params)."""
    return SelfRAGConfig(
        d_model=512,
        n_heads=8,
        n_layers=12,
        d_ff=2048,
        vocab_size=32000,
        max_seq_len=1024,
        max_chunks=10000,
        chunk_size=64,
        top_k=5,
    )


def large_config() -> SelfRAGConfig:
    """Large model (~500M params)."""
    return SelfRAGConfig(
        d_model=768,
        n_heads=12,
        n_layers=16,
        d_ff=3072,
        vocab_size=32000,
        max_seq_len=2048,
        max_chunks=20000,
        chunk_size=64,
        top_k=7,
    )
