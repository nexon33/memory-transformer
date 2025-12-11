"""Position-based recall benchmark - the smoking gun test.

Standard transformers show U-curve (primacy + recency effect).
Memory-augmented should show flat recall across all positions.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import random
import string
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class PositionRecallResult:
    """Results from position recall benchmark."""
    position_accuracies: List[float]
    overall_accuracy: float
    primacy_accuracy: float      # First 10%
    middle_accuracy: float       # Middle 80%
    recency_accuracy: float      # Last 10%
    flatness_score: float        # 1 - (std / mean), higher = flatter
    num_positions: int
    samples_per_position: int


class PositionRecallBenchmark:
    """Benchmark for testing recall accuracy by storage position.

    The Test:
    1. Store N key-value pairs at different positions
    2. Query each position multiple times
    3. Record accuracy per position
    4. Plot: flat = memory works, U-curve = just attention
    """

    def __init__(
        self,
        tokenizer: Any,
        num_positions: int = 50,
        samples_per_position: int = 20,
        key_length: int = 3,
        value_length: int = 4,
        filler_length: int = 20,
        seed: int = 42,
    ):
        """Initialize benchmark.

        Args:
            tokenizer: Tokenizer for encoding
            num_positions: Number of KV pairs to store
            samples_per_position: Queries per position
            key_length: Characters in key
            value_length: Characters in value
            filler_length: Filler tokens between stores
            seed: Random seed
        """
        self.tokenizer = tokenizer
        self.num_positions = num_positions
        self.samples_per_position = samples_per_position
        self.key_length = key_length
        self.value_length = value_length
        self.filler_length = filler_length
        self.seed = seed

        random.seed(seed)

    def generate_kv_pair(self) -> Tuple[str, str]:
        """Generate random key-value pair."""
        key = ''.join(random.choices(string.ascii_lowercase, k=self.key_length))
        value = ''.join(random.choices(string.digits, k=self.value_length))
        return key, value

    def generate_filler(self) -> str:
        """Generate random filler text."""
        return ''.join(random.choices(
            string.ascii_lowercase + string.digits + ' ',
            k=self.filler_length
        ))

    def create_sequence_with_stores(
        self,
        pairs: List[Tuple[str, str]],
    ) -> Tuple[str, List[int]]:
        """Create sequence with stored KV pairs.

        Format: STORE k1=v1 [filler] STORE k2=v2 [filler] ...

        Args:
            pairs: List of (key, value) pairs

        Returns:
            Full sequence text
            Token positions where each store starts
        """
        parts = []
        store_positions = []

        for i, (key, value) in enumerate(pairs):
            store_positions.append(len(' '.join(parts).split()))
            parts.append(f"STORE {key}={value}")
            if i < len(pairs) - 1:
                parts.append(self.generate_filler())

        return ' '.join(parts), store_positions

    def create_query(self, key: str) -> str:
        """Create query for a key."""
        return f" RECALL {key}="

    @torch.no_grad()
    def evaluate_self_rag(
        self,
        model: Any,  # SelfRAGModel
        device: torch.device,
    ) -> PositionRecallResult:
        """Evaluate Self-RAG model on position recall.

        Args:
            model: SelfRAGModel instance
            device: Device to run on

        Returns:
            PositionRecallResult with accuracy by position
        """
        model.eval()
        model.clear_memory()

        # Track accuracy per position
        position_correct = [0] * self.num_positions
        position_total = [0] * self.num_positions

        for sample_idx in range(self.samples_per_position):
            # Generate new KV pairs
            pairs = [self.generate_kv_pair() for _ in range(self.num_positions)]

            # Create sequence and store in memory
            sequence, store_positions = self.create_sequence_with_stores(pairs)

            # Tokenize and store each fact separately
            model.clear_memory()
            for i, (key, value) in enumerate(pairs):
                fact = f"STORE {key}={value}"
                model.write_to_memory(
                    fact,
                    metadata={"position": i, "key": key, "value": value}
                )

            # Query each position
            for query_pos in range(self.num_positions):
                key, expected_value = pairs[query_pos]
                query = self.create_query(key)

                # Tokenize query
                query_ids = self.tokenizer.encode(query, return_tensors="pt").to(device)

                # Generate response
                output_ids = model.generate(
                    query_ids,
                    max_new_tokens=len(expected_value) + 2,
                    temperature=0.1,
                    use_memory=True,
                )

                # Decode and check
                output_text = self.tokenizer.decode(
                    output_ids[0, query_ids.size(1):],
                    skip_special_tokens=True
                ).strip()

                # Check if value is in output
                if expected_value in output_text or output_text.startswith(expected_value):
                    position_correct[query_pos] += 1
                position_total[query_pos] += 1

        # Compute accuracies
        position_accuracies = [
            c / t if t > 0 else 0.0
            for c, t in zip(position_correct, position_total)
        ]

        return self._compute_results(position_accuracies)

    @torch.no_grad()
    def evaluate_standard_transformer(
        self,
        model: Any,
        device: torch.device,
        max_seq_len: int = 1024,
    ) -> PositionRecallResult:
        """Evaluate standard transformer (baseline) on position recall.

        Uses in-context learning - all stores in the prompt.

        Args:
            model: Standard transformer model
            device: Device
            max_seq_len: Maximum sequence length

        Returns:
            PositionRecallResult
        """
        model.eval()

        position_correct = [0] * self.num_positions
        position_total = [0] * self.num_positions

        for sample_idx in range(self.samples_per_position):
            # Generate pairs
            pairs = [self.generate_kv_pair() for _ in range(self.num_positions)]

            # Create full context with all stores
            sequence, _ = self.create_sequence_with_stores(pairs)

            # Query each position
            for query_pos in range(self.num_positions):
                key, expected_value = pairs[query_pos]
                query = self.create_query(key)

                # Full prompt: context + query
                full_prompt = sequence + query

                # Tokenize (truncate from beginning if too long)
                tokens = self.tokenizer.encode(full_prompt)
                if len(tokens) > max_seq_len - 10:
                    tokens = tokens[-(max_seq_len - 10):]

                input_ids = torch.tensor([tokens], device=device)

                # Generate
                if hasattr(model, 'generate'):
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=len(expected_value) + 2,
                        temperature=0.1,
                    )
                    if isinstance(output_ids, tuple):
                        output_ids = output_ids[0]
                else:
                    # Manual generation for basic models
                    for _ in range(len(expected_value) + 2):
                        outputs = model(input_ids)
                        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        input_ids = torch.cat([input_ids, next_token], dim=1)
                    output_ids = input_ids

                output_text = self.tokenizer.decode(
                    output_ids[0, len(tokens):],
                    skip_special_tokens=True
                ).strip()

                if expected_value in output_text or output_text.startswith(expected_value):
                    position_correct[query_pos] += 1
                position_total[query_pos] += 1

        position_accuracies = [
            c / t if t > 0 else 0.0
            for c, t in zip(position_correct, position_total)
        ]

        return self._compute_results(position_accuracies)

    def _compute_results(
        self,
        position_accuracies: List[float],
    ) -> PositionRecallResult:
        """Compute aggregate results from position accuracies."""
        n = len(position_accuracies)

        # Primacy (first 10%), Middle (80%), Recency (last 10%)
        primacy_end = max(1, n // 10)
        recency_start = n - max(1, n // 10)

        primacy = position_accuracies[:primacy_end]
        middle = position_accuracies[primacy_end:recency_start]
        recency = position_accuracies[recency_start:]

        primacy_acc = sum(primacy) / len(primacy) if primacy else 0.0
        middle_acc = sum(middle) / len(middle) if middle else 0.0
        recency_acc = sum(recency) / len(recency) if recency else 0.0
        overall_acc = sum(position_accuracies) / n

        # Flatness score: 1 - (std / mean)
        # Higher = flatter curve
        import statistics
        if overall_acc > 0:
            std = statistics.stdev(position_accuracies) if n > 1 else 0
            flatness = 1 - (std / overall_acc)
        else:
            flatness = 0.0

        return PositionRecallResult(
            position_accuracies=position_accuracies,
            overall_accuracy=overall_acc,
            primacy_accuracy=primacy_acc,
            middle_accuracy=middle_acc,
            recency_accuracy=recency_acc,
            flatness_score=flatness,
            num_positions=n,
            samples_per_position=self.samples_per_position,
        )

    def generate_reference_ucurve(self, n: int) -> List[float]:
        """Generate synthetic U-curve for comparison.

        Based on typical serial position effect in human memory.
        """
        import math
        ucurve = []
        for i in range(n):
            # Primacy effect (exponential decay from start)
            primacy = 0.3 * math.exp(-i / (n * 0.1))
            # Recency effect (exponential growth to end)
            recency = 0.4 * math.exp(-(n - 1 - i) / (n * 0.1))
            # Base accuracy
            base = 0.2
            ucurve.append(min(1.0, primacy + recency + base))
        return ucurve


def plot_position_recall(
    results: PositionRecallResult,
    title: str = "Position-Based Recall Accuracy",
    save_path: Optional[str] = None,
    show_reference: bool = True,
    reference_results: Optional[PositionRecallResult] = None,
):
    """Plot position recall results.

    Args:
        results: Results from benchmark
        title: Plot title
        save_path: Path to save figure
        show_reference: Show reference U-curve
        reference_results: Optional baseline results to compare
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available for plotting")
        print_results(results)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    positions = list(range(results.num_positions))

    # Left plot: Accuracy by position
    ax1.plot(positions, results.position_accuracies, 'b-o',
             label='Model', linewidth=2, markersize=4)

    if reference_results:
        ax1.plot(positions, reference_results.position_accuracies, 'r--s',
                 label='Baseline', linewidth=2, markersize=4, alpha=0.7)
    elif show_reference:
        benchmark = PositionRecallBenchmark(None, num_positions=results.num_positions)
        ucurve = benchmark.generate_reference_ucurve(results.num_positions)
        ax1.plot(positions, ucurve, 'r--',
                 label='Typical U-curve', linewidth=2, alpha=0.5)

    ax1.set_xlabel('Storage Position', fontsize=12)
    ax1.set_ylabel('Recall Accuracy', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Shade regions
    n = results.num_positions
    ax1.axvspan(0, n * 0.1, alpha=0.1, color='green', label='Primacy')
    ax1.axvspan(n * 0.9, n, alpha=0.1, color='blue', label='Recency')

    # Right plot: Summary metrics
    metrics = ['Overall', 'Primacy\n(first 10%)', 'Middle\n(80%)', 'Recency\n(last 10%)', 'Flatness']
    values = [
        results.overall_accuracy,
        results.primacy_accuracy,
        results.middle_accuracy,
        results.recency_accuracy,
        results.flatness_score,
    ]
    colors = ['steelblue', 'green', 'orange', 'blue', 'purple']

    bars = ax2.bar(metrics, values, color=colors, alpha=0.7)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Summary Metrics', fontsize=14)

    # Add value labels
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def print_results(results: PositionRecallResult):
    """Print results to console."""
    print("\n" + "=" * 60)
    print("POSITION RECALL BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Positions: {results.num_positions}")
    print(f"  Samples per position: {results.samples_per_position}")
    print()
    print(f"  Overall Accuracy:  {results.overall_accuracy:.2%}")
    print(f"  Primacy (first 10%): {results.primacy_accuracy:.2%}")
    print(f"  Middle (80%):        {results.middle_accuracy:.2%}")
    print(f"  Recency (last 10%):  {results.recency_accuracy:.2%}")
    print()
    print(f"  Flatness Score: {results.flatness_score:.3f}")
    print("    (1.0 = perfectly flat, lower = more U-shaped)")
    print()

    # Visual bar
    print("  Position accuracies:")
    for i, acc in enumerate(results.position_accuracies):
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"    {i:3d}: {bar} {acc:.0%}")

    print("=" * 60)


def compare_models(
    self_rag_results: PositionRecallResult,
    baseline_results: PositionRecallResult,
    save_path: Optional[str] = None,
):
    """Compare Self-RAG vs baseline results."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(f"{'Metric':<20} {'Self-RAG':>12} {'Baseline':>12} {'Delta':>12}")
    print("-" * 60)

    metrics = [
        ("Overall", self_rag_results.overall_accuracy, baseline_results.overall_accuracy),
        ("Primacy", self_rag_results.primacy_accuracy, baseline_results.primacy_accuracy),
        ("Middle", self_rag_results.middle_accuracy, baseline_results.middle_accuracy),
        ("Recency", self_rag_results.recency_accuracy, baseline_results.recency_accuracy),
        ("Flatness", self_rag_results.flatness_score, baseline_results.flatness_score),
    ]

    for name, self_rag, baseline in metrics:
        delta = self_rag - baseline
        sign = "+" if delta > 0 else ""
        print(f"{name:<20} {self_rag:>11.2%} {baseline:>11.2%} {sign}{delta:>10.2%}")

    print("=" * 60)

    # Verdict
    if self_rag_results.flatness_score > baseline_results.flatness_score + 0.1:
        print("\n✓ Self-RAG shows FLATTER recall curve - memory is working!")
    elif self_rag_results.middle_accuracy > baseline_results.middle_accuracy + 0.1:
        print("\n✓ Self-RAG shows BETTER middle recall - memory helps!")
    else:
        print("\n✗ No significant improvement - check memory retrieval")

    if save_path:
        plot_position_recall(
            self_rag_results,
            title="Self-RAG vs Baseline Position Recall",
            save_path=save_path,
            reference_results=baseline_results,
        )
