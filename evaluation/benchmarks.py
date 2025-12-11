"""Benchmark evaluation for memory transformer."""

import torch
from typing import Dict, List, Optional
from tqdm import tqdm

from ..data.tokenizer import TaskTokenizer
from ..data.datasets import CopyDataset, RecallDataset, ArithmeticDataset


def evaluate_copy_task(
    model,
    tokenizer: TaskTokenizer,
    device: torch.device,
    num_samples: int = 100,
    lengths: List[int] = None,
) -> Dict[str, float]:
    """Evaluate model on copy task at different lengths.

    Args:
        model: Memory transformer model
        tokenizer: Task tokenizer
        device: Device to run on
        num_samples: Number of samples per length
        lengths: List of content lengths to test

    Returns:
        Dictionary of accuracy by length
    """
    model.eval()
    lengths = lengths or [4, 8, 12, 16, 20]

    results = {}

    with torch.no_grad():
        for length in lengths:
            dataset = CopyDataset(
                tokenizer=tokenizer,
                num_samples=num_samples,
                min_length=length,
                max_length=length,
                max_seq_len=64,
            )

            correct = 0
            total = 0

            for sample in dataset:
                input_ids = sample["input_ids"].unsqueeze(0).to(device)
                labels = sample["labels"].unsqueeze(0).to(device)

                outputs = model(input_ids=input_ids, labels=labels)
                predictions = outputs["logits"].argmax(dim=-1)

                # Check accuracy on target portion only
                mask = labels != -100
                pred_masked = predictions[mask]
                label_masked = labels[mask]

                if (pred_masked == label_masked).all():
                    correct += 1
                total += 1

            results[f"copy_len_{length}"] = correct / total

    results["copy_avg"] = sum(results.values()) / len(results)
    return results


def evaluate_recall_task(
    model,
    tokenizer: TaskTokenizer,
    device: torch.device,
    num_samples: int = 100,
    distractor_lengths: List[int] = None,
) -> Dict[str, float]:
    """Evaluate model on recall task with varying distractor lengths.

    Args:
        model: Memory transformer model
        tokenizer: Task tokenizer
        device: Device to run on
        num_samples: Number of samples per configuration
        distractor_lengths: List of distractor lengths to test

    Returns:
        Dictionary of accuracy by distractor length
    """
    model.eval()
    distractor_lengths = distractor_lengths or [5, 10, 20, 40]

    results = {}

    with torch.no_grad():
        for dist_len in distractor_lengths:
            dataset = RecallDataset(
                tokenizer=tokenizer,
                num_samples=num_samples,
                min_distractor_length=dist_len,
                max_distractor_length=dist_len,
                max_seq_len=128,
            )

            correct = 0
            total = 0

            for sample in dataset:
                input_ids = sample["input_ids"].unsqueeze(0).to(device)
                labels = sample["labels"].unsqueeze(0).to(device)

                outputs = model(input_ids=input_ids, labels=labels)
                predictions = outputs["logits"].argmax(dim=-1)

                mask = labels != -100
                pred_masked = predictions[mask]
                label_masked = labels[mask]

                if (pred_masked == label_masked).all():
                    correct += 1
                total += 1

            results[f"recall_dist_{dist_len}"] = correct / total

    results["recall_avg"] = sum(results.values()) / len(results)
    return results


def evaluate_arithmetic_task(
    model,
    tokenizer: TaskTokenizer,
    device: torch.device,
    num_samples: int = 100,
    num_steps_list: List[int] = None,
) -> Dict[str, float]:
    """Evaluate model on arithmetic task with varying complexity.

    Args:
        model: Memory transformer model
        tokenizer: Task tokenizer
        device: Device to run on
        num_samples: Number of samples per configuration
        num_steps_list: List of number of steps to test

    Returns:
        Dictionary of accuracy by number of steps
    """
    model.eval()
    num_steps_list = num_steps_list or [1, 2, 3]

    results = {}

    with torch.no_grad():
        for num_steps in num_steps_list:
            dataset = ArithmeticDataset(
                tokenizer=tokenizer,
                num_samples=num_samples,
                num_steps=num_steps,
                max_seq_len=128,
            )

            correct = 0
            total = 0

            for sample in dataset:
                input_ids = sample["input_ids"].unsqueeze(0).to(device)
                labels = sample["labels"].unsqueeze(0).to(device)

                outputs = model(input_ids=input_ids, labels=labels)
                predictions = outputs["logits"].argmax(dim=-1)

                mask = labels != -100
                pred_masked = predictions[mask]
                label_masked = labels[mask]

                if (pred_masked == label_masked).all():
                    correct += 1
                total += 1

            results[f"arith_steps_{num_steps}"] = correct / total

    results["arith_avg"] = sum(results.values()) / len(results)
    return results


def run_full_benchmark(
    model,
    tokenizer: TaskTokenizer,
    device: torch.device,
    num_samples: int = 100,
) -> Dict[str, float]:
    """Run full benchmark suite.

    Args:
        model: Memory transformer model
        tokenizer: Task tokenizer
        device: Device to run on
        num_samples: Number of samples per task

    Returns:
        Combined results dictionary
    """
    print("Running copy task benchmark...")
    copy_results = evaluate_copy_task(model, tokenizer, device, num_samples)

    print("Running recall task benchmark...")
    recall_results = evaluate_recall_task(model, tokenizer, device, num_samples)

    print("Running arithmetic task benchmark...")
    arith_results = evaluate_arithmetic_task(model, tokenizer, device, num_samples)

    # Combine results
    results = {}
    results.update(copy_results)
    results.update(recall_results)
    results.update(arith_results)

    # Overall average
    task_avgs = [
        copy_results["copy_avg"],
        recall_results["recall_avg"],
        arith_results["arith_avg"],
    ]
    results["overall_avg"] = sum(task_avgs) / len(task_avgs)

    return results


def print_benchmark_results(results: Dict[str, float]):
    """Pretty print benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    # Copy task results
    print("\nCopy Task:")
    for key, value in results.items():
        if key.startswith("copy"):
            print(f"  {key}: {value:.2%}")

    # Recall task results
    print("\nRecall Task:")
    for key, value in results.items():
        if key.startswith("recall"):
            print(f"  {key}: {value:.2%}")

    # Arithmetic task results
    print("\nArithmetic Task:")
    for key, value in results.items():
        if key.startswith("arith"):
            print(f"  {key}: {value:.2%}")

    # Overall
    print("\n" + "-" * 60)
    print(f"Overall Average: {results['overall_avg']:.2%}")
    print("=" * 60)
