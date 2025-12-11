"""Configuration for Memory-Augmented Transformer."""

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class MemoryTransformerConfig:
    """Configuration for the Memory-Augmented Transformer model.

    Optimized for mobile/Termux training (~5M parameters).
    """

    # Model dimensions
    d_model: int = 192  # Hidden dimension
    n_heads: int = 4  # Number of attention heads
    n_layers: int = 4  # Number of transformer blocks
    d_ff: int = 384  # Feedforward dimension

    # Vocabulary and sequence
    vocab_size: int = 4096  # Vocabulary size
    max_seq_len: int = 256  # Maximum sequence length

    # Memory parameters
    snapshot_interval: int = 4  # Create snapshot every N tokens
    memory_size: int = 64  # Maximum number of snapshots in bank
    d_memory: int = 192  # Full snapshot dimension
    d_compressed: int = 48  # Compressed snapshot dimension (4x compression)
    n_retrieval_hops: int = 2  # Number of retrieval hops

    # Memory variant (for ablation)
    use_compressed: bool = False  # Whether to use compressed snapshots
    use_both: bool = True  # Use both full and compressed (hybrid)

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    memory_dropout: float = 0.05  # Lower dropout for memory operations

    # Training
    gradient_checkpointing: bool = True  # Save memory on mobile

    # Initialization
    init_std: float = 0.02

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    @property
    def max_memory_slots(self) -> int:
        """Maximum memory slots based on sequence length and interval."""
        return self.max_seq_len // self.snapshot_interval

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.d_memory <= self.d_model, \
            f"d_memory ({self.d_memory}) should not exceed d_model ({self.d_model})"
        # Only validate compressed < memory when using both (not for compressed-only config)
        if not self.use_compressed:
            assert self.d_compressed < self.d_memory, \
                f"d_compressed ({self.d_compressed}) should be less than d_memory ({self.d_memory})"
        assert self.snapshot_interval > 0, \
            f"snapshot_interval must be positive, got {self.snapshot_interval}"

    @classmethod
    def from_yaml(cls, path: str) -> "MemoryTransformerConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "snapshot_interval": self.snapshot_interval,
            "memory_size": self.memory_size,
            "d_memory": self.d_memory,
            "d_compressed": self.d_compressed,
            "n_retrieval_hops": self.n_retrieval_hops,
            "use_compressed": self.use_compressed,
            "use_both": self.use_both,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "memory_dropout": self.memory_dropout,
            "gradient_checkpointing": self.gradient_checkpointing,
            "init_std": self.init_std,
        }
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def __post_init__(self):
        """Validate after initialization."""
        self.validate()


@dataclass
class TrainingConfig:
    """Training configuration optimized for mobile."""

    # Batch settings
    batch_size: int = 2  # Very small for mobile
    gradient_accumulation_steps: int = 8  # Effective batch = 16

    # Learning rate
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 500
    max_steps: int = 50000

    # Loss weights
    lm_loss_weight: float = 1.0
    memory_usage_loss_weight: float = 0.01  # Encourage sparse writes
    retrieval_loss_weight: float = 0.1  # Supervise retrieval

    # Optimization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Checkpointing
    save_every: int = 500
    eval_every: int = 100
    log_every: int = 10

    # Curriculum learning
    curriculum_enabled: bool = True
    stage_thresholds: dict = field(default_factory=lambda: {
        "copy": 0.90,  # 90% accuracy to advance
        "recall": 0.85,
        "arithmetic": 0.80,
    })

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


# Preset configurations
def tiny_full_config() -> MemoryTransformerConfig:
    """Tiny model with full snapshots."""
    return MemoryTransformerConfig(
        use_compressed=False,
        use_both=False,
    )


def tiny_compressed_config() -> MemoryTransformerConfig:
    """Tiny model with compressed snapshots only."""
    return MemoryTransformerConfig(
        use_compressed=True,
        use_both=False,
    )


def tiny_hybrid_config() -> MemoryTransformerConfig:
    """Tiny model with both full and compressed (for ablation)."""
    return MemoryTransformerConfig(
        use_compressed=False,
        use_both=True,
    )
