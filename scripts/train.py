#!/usr/bin/env python3
"""Training script for Memory Transformer."""

import os
import sys
import argparse
import torch
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from memory_transformer.model.config import MemoryTransformerConfig, TrainingConfig
from memory_transformer.model.memory_transformer import MemoryTransformer
from memory_transformer.data.tokenizer import TaskTokenizer
from memory_transformer.data.data_utils import (
    create_tokenizer,
    create_curriculum_dataloaders,
)
from memory_transformer.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Memory Transformer")

    # Model config
    parser.add_argument("--d-model", type=int, default=192, help="Hidden dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--d-ff", type=int, default=384, help="Feedforward dimension")
    parser.add_argument("--vocab-size", type=int, default=4096, help="Vocabulary size")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Maximum sequence length")

    # Memory config
    parser.add_argument("--memory-size", type=int, default=64, help="Memory bank size")
    parser.add_argument("--snapshot-interval", type=int, default=4, help="Snapshot interval")
    parser.add_argument("--n-hops", type=int, default=2, help="Retrieval hops")

    # Training config
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=50000, help="Maximum training steps")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")

    # Data config
    parser.add_argument("--num-samples", type=int, default=10000, help="Samples per dataset")

    # Other
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Memory Transformer Training")
    print("=" * 60)

    # Create model config
    # Note: gradient_checkpointing disabled as it conflicts with mutable memory state
    model_config = MemoryTransformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        memory_size=args.memory_size,
        snapshot_interval=args.snapshot_interval,
        n_retrieval_hops=args.n_hops,
        gradient_checkpointing=False,
    )

    print(f"\nModel Config:")
    print(f"  d_model: {model_config.d_model}")
    print(f"  n_heads: {model_config.n_heads}")
    print(f"  n_layers: {model_config.n_layers}")
    print(f"  d_ff: {model_config.d_ff}")
    print(f"  memory_size: {model_config.memory_size}")

    # Create training config
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
    )

    print(f"\nTraining Config:")
    print(f"  batch_size: {train_config.batch_size}")
    print(f"  gradient_accumulation: {train_config.gradient_accumulation_steps}")
    print(f"  effective_batch_size: {train_config.batch_size * train_config.gradient_accumulation_steps}")
    print(f"  learning_rate: {train_config.learning_rate}")

    # Create tokenizer
    print("\nCreating tokenizer...")
    tokenizer = create_tokenizer(vocab_size=model_config.vocab_size)
    print(f"  Vocabulary size: {len(tokenizer)}")

    # Create model
    print("\nCreating model...")
    model = MemoryTransformer(model_config)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_dataloaders = create_curriculum_dataloaders(
        tokenizer=tokenizer,
        batch_size=train_config.batch_size,
        max_seq_len=model_config.max_seq_len,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    # Create separate eval dataloaders with fewer samples
    eval_dataloaders = create_curriculum_dataloaders(
        tokenizer=tokenizer,
        batch_size=train_config.batch_size,
        max_seq_len=model_config.max_seq_len,
        num_samples=args.num_samples // 10,  # Smaller eval set
        seed=args.seed + 1000,
    )

    print(f"  Stages: {list(train_dataloaders.keys())}")

    # Create trainer
    print("\nCreating trainer...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_config=train_config,
        model_config=model_config,
        output_dir=args.output_dir,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...")
    trainer.train(
        train_dataloaders=train_dataloaders,
        eval_dataloaders=eval_dataloaders,
        num_epochs=args.num_epochs,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
