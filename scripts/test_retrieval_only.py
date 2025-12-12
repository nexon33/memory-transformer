#!/usr/bin/env python3
"""Test retrieval quality only (no slow generation)."""

import sys
sys.path.insert(0, ".")

import random
import string
from scripts.test_phi2_memory import Phi2WithMemory


def test_retrieval_positions(model, num_positions=10, samples=2):
    """Test if retrieval finds correct fact regardless of storage position."""

    print(f"\nTesting {num_positions} positions, {samples} samples each...")

    position_correct = [0] * num_positions
    position_total = [0] * num_positions

    for sample in range(samples):
        model.clear_memory()

        # Generate random key-value pairs
        pairs = []
        for i in range(num_positions):
            key = "".join(random.choices(string.ascii_lowercase, k=2))
            value = "".join(random.choices(string.digits, k=3))
            pairs.append((key, value))

        # Store all facts
        for i, (key, value) in enumerate(pairs):
            fact = f"The code {key} equals {value}"
            model.store(fact)

        # Query each position
        for pos in range(num_positions):
            key, expected = pairs[pos]
            query = f"What does code {key} equal?"

            retrieved = model.retrieve(query)
            found = any(expected in chunk for chunk, _ in retrieved)

            if found:
                position_correct[pos] += 1
            position_total[pos] += 1

        if (sample + 1) % 5 == 0 or sample == samples - 1:
            print(f"  Progress: {sample+1}/{samples} samples")

    # Results
    accuracies = [c/t for c, t in zip(position_correct, position_total)]

    print("\n" + "="*50)
    print("RETRIEVAL ACCURACY BY POSITION")
    print("="*50)

    for i, acc in enumerate(accuracies):
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {i:2d}: {bar} {acc*100:.0f}%")

    overall = sum(position_correct) / sum(position_total)
    print(f"\nOverall: {overall*100:.1f}%")
    print(f"Min:     {min(accuracies)*100:.0f}%")
    print(f"Max:     {max(accuracies)*100:.0f}%")

    # Check flatness
    variance = sum((a - overall)**2 for a in accuracies) / len(accuracies)
    print(f"Variance: {variance:.4f} (lower = flatter)")

    if min(accuracies) >= 0.5 and variance < 0.05:
        print("\n✓ FLAT CURVE - Memory works regardless of position!")
    elif overall > 0.5:
        print("\n○ Good retrieval but some position variance")
    else:
        print("\n✗ Retrieval needs improvement")

    return accuracies


if __name__ == "__main__":
    print("="*50)
    print("Phi-2 + Memory: Retrieval-Only Test")
    print("="*50)

    model = Phi2WithMemory(top_k=5)  # More results for better recall

    # Quick sanity check
    print("\n--- Sanity Check ---")
    model.clear_memory()
    model.store("The code xy equals 123")
    model.store("The code ab equals 456")
    model.store("The code zz equals 789")

    query = "What does code xy equal?"
    retrieved = model.retrieve(query)
    print(f"Query: {query}")
    for chunk, score in retrieved:
        print(f"  [{score:.3f}] {chunk}")

    top_chunk = retrieved[0][0] if retrieved else ""
    if "123" in top_chunk:
        print("✓ Correct fact retrieved first!")
    else:
        print("✗ Wrong fact retrieved")

    # Full position test - 20 samples for statistical significance
    test_retrieval_positions(model, num_positions=10, samples=20)
