#!/usr/bin/env python3
"""Evaluation script for Memory Transformer."""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from memory_transformer.model.config import MemoryTransformerConfig
from memory_transformer.model.memory_transformer import MemoryTransformer
from memory_transformer.data.tokenizer import TaskTokenizer
from memory_transformer.data.data_utils import create_tokenizer
from memory_transformer.evaluation.benchmarks import (
    run_full_benchmark,
    print_benchmark_results,
)
from memory_transformer.evaluation.metrics import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Memory Transformer")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--num-samples", type=int, default=100, help="Samples per benchmark")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Reconstruct model config
    model_config_dict = checkpoint["model_config"]
    model_config = MemoryTransformerConfig(**model_config_dict)

    # Create model
    print("Creating model...")
    model = MemoryTransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Create tokenizer
    tokenizer = create_tokenizer(vocab_size=model_config.vocab_size)

    # Run benchmarks
    print("\nRunning benchmarks...")
    results = run_full_benchmark(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_samples=args.num_samples,
    )

    # Print results
    print_benchmark_results(results)

    # Also print training info from checkpoint
    if "global_step" in checkpoint:
        print(f"\nCheckpoint Info:")
        print(f"  Global step: {checkpoint['global_step']}")
        print(f"  Current stage: {checkpoint.get('current_stage', 'unknown')}")
        print(f"  Best loss: {checkpoint.get('best_loss', 'unknown')}")


if __name__ == "__main__":
    main()
