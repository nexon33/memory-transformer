#!/usr/bin/env python3
"""Evaluate position recall - the smoking gun test for memory."""

import os
import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from memory_transformer.model.self_rag import SelfRAGModel
from memory_transformer.model.self_rag_config import SelfRAGConfig
from memory_transformer.evaluation.position_recall import (
    PositionRecallBenchmark,
    plot_position_recall,
    print_results,
    compare_models,
)

try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Position Recall Benchmark")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--baseline-checkpoint", type=str, default=None,
                        help="Optional baseline model for comparison")

    parser.add_argument("--num-positions", type=int, default=50,
                        help="Number of KV pairs to store")
    parser.add_argument("--samples-per-position", type=int, default=10,
                        help="Queries per position")

    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="Output directory for plots")
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    """Load Self-RAG model from checkpoint."""
    checkpoint_dir = Path(checkpoint_path)

    # Load config
    config = SelfRAGConfig.from_yaml(str(checkpoint_dir / "config.yaml"))

    # Load tokenizer
    if HF_AVAILABLE:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        config.vocab_size = len(tokenizer)
    else:
        raise ImportError("transformers required for tokenizer")

    # Create and load model
    model = SelfRAGModel(config, tokenizer)
    model.load_state_dict(
        torch.load(checkpoint_dir / "model.pt", map_location=device)
    )
    model = model.to(device)
    model.eval()

    # Load memory if exists
    memory_path = checkpoint_dir / "memory.json"
    if memory_path.exists():
        model.memory.load(str(memory_path))
        print(f"Loaded memory with {model.memory.size} chunks")

    return model, tokenizer


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("Position Recall Benchmark")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Positions: {args.num_positions}")
    print(f"Samples per position: {args.samples_per_position}")
    print()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, tokenizer = load_model(args.checkpoint, device)

    # Create benchmark
    benchmark = PositionRecallBenchmark(
        tokenizer=tokenizer,
        num_positions=args.num_positions,
        samples_per_position=args.samples_per_position,
    )

    # Run benchmark
    print("\nRunning position recall benchmark...")
    results = benchmark.evaluate_self_rag(model, device)

    # Print results
    print_results(results)

    # Load baseline if provided
    baseline_results = None
    if args.baseline_checkpoint:
        print(f"\nLoading baseline from {args.baseline_checkpoint}...")
        baseline_model, _ = load_model(args.baseline_checkpoint, device)

        print("Running baseline benchmark...")
        baseline_results = benchmark.evaluate_standard_transformer(baseline_model, device)

        print_results(baseline_results)
        compare_models(results, baseline_results)

    # Save plots
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = output_dir / "position_recall.png"
    plot_position_recall(
        results,
        title="Self-RAG Position Recall",
        save_path=str(plot_path),
        reference_results=baseline_results,
    )

    # Save results JSON
    import json
    results_path = output_dir / "position_recall_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "position_accuracies": results.position_accuracies,
            "overall_accuracy": results.overall_accuracy,
            "primacy_accuracy": results.primacy_accuracy,
            "middle_accuracy": results.middle_accuracy,
            "recency_accuracy": results.recency_accuracy,
            "flatness_score": results.flatness_score,
            "num_positions": results.num_positions,
            "samples_per_position": results.samples_per_position,
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if results.flatness_score > 0.8:
        print("✓ EXCELLENT: Very flat recall curve - memory working well!")
    elif results.flatness_score > 0.6:
        print("○ GOOD: Reasonably flat - memory helping")
    elif results.middle_accuracy > 0.5:
        print("○ OKAY: Middle recall acceptable")
    else:
        print("✗ POOR: U-shaped curve - memory not helping middle positions")

    if baseline_results and results.middle_accuracy > baseline_results.middle_accuracy + 0.1:
        print("✓ Beats baseline on middle positions!")


if __name__ == "__main__":
    main()
