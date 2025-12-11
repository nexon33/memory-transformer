"""Evaluation metrics for memory transformer."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


def compute_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Compute token-level accuracy.

    Args:
        predictions: Predicted token IDs [batch, seq_len]
        labels: Target labels [batch, seq_len]
        ignore_index: Index to ignore in labels

    Returns:
        Accuracy as a float
    """
    mask = labels != ignore_index
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()


def compute_sequence_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Compute sequence-level accuracy (all tokens correct).

    Args:
        predictions: Predicted token IDs [batch, seq_len]
        labels: Target labels [batch, seq_len]
        ignore_index: Index to ignore

    Returns:
        Sequence accuracy as a float
    """
    batch_size = predictions.size(0)
    mask = labels != ignore_index

    # For each sequence, check if all non-masked tokens are correct
    correct_per_seq = []
    for i in range(batch_size):
        seq_mask = mask[i]
        seq_pred = predictions[i][seq_mask]
        seq_label = labels[i][seq_mask]
        correct_per_seq.append((seq_pred == seq_label).all().item())

    return sum(correct_per_seq) / len(correct_per_seq)


def compute_perplexity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Compute perplexity.

    Args:
        logits: Model logits [batch, seq_len, vocab_size]
        labels: Target labels [batch, seq_len]
        ignore_index: Index to ignore

    Returns:
        Perplexity as a float
    """
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute cross-entropy
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )

    perplexity = torch.exp(loss)
    return perplexity.item()


def compute_memory_usage(
    memory_state: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Compute memory usage statistics.

    Args:
        memory_state: Memory state dictionary

    Returns:
        Dictionary of memory statistics
    """
    valid_mask = memory_state["valid_mask"].float()
    usage = memory_state["usage"]

    stats = {
        "slots_used_mean": valid_mask.sum(dim=-1).mean().item(),
        "slots_used_std": valid_mask.sum(dim=-1).std().item(),
        "usage_mean": usage.mean().item(),
        "usage_std": usage.std().item(),
        "usage_max": usage.max().item(),
        "usage_min": usage.min().item(),
        "sparsity": (usage < 0.1).float().mean().item(),
        "total_writes": memory_state["total_writes"].float().mean().item(),
    }

    return stats


def compute_retrieval_accuracy(
    attention_weights: torch.Tensor,
    target_indices: torch.Tensor,
    top_k: int = 1,
) -> float:
    """Compute retrieval accuracy (if target indices are known).

    Args:
        attention_weights: Retrieval attention [batch, seq_len, memory_size]
        target_indices: Ground truth memory indices [batch]
        top_k: Consider correct if target in top-k

    Returns:
        Retrieval accuracy as float
    """
    # Get top-k indices from attention
    # Average attention over sequence positions
    avg_attention = attention_weights.mean(dim=1)  # [batch, memory_size]
    top_k_indices = avg_attention.topk(top_k, dim=-1).indices  # [batch, top_k]

    # Check if target is in top-k
    target_expanded = target_indices.unsqueeze(-1)  # [batch, 1]
    correct = (top_k_indices == target_expanded).any(dim=-1).float()

    return correct.mean().item()


def compute_task_specific_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    task_type: str,
    tokenizer,
) -> Dict[str, float]:
    """Compute task-specific metrics.

    Args:
        predictions: Predicted tokens [batch, seq_len]
        labels: Target labels [batch, seq_len]
        task_type: Type of task ("copy", "recall", "arithmetic")
        tokenizer: Tokenizer for decoding

    Returns:
        Dictionary of task-specific metrics
    """
    metrics = {}

    # Token accuracy
    metrics["token_accuracy"] = compute_accuracy(predictions, labels)

    # Sequence accuracy
    metrics["sequence_accuracy"] = compute_sequence_accuracy(predictions, labels)

    if task_type == "copy":
        # For copy tasks, check exact match of copied content
        metrics["exact_copy_rate"] = compute_sequence_accuracy(predictions, labels)

    elif task_type == "recall":
        # For recall, check if recalled value matches
        metrics["recall_accuracy"] = compute_sequence_accuracy(predictions, labels)

    elif task_type == "arithmetic":
        # For arithmetic, check if numerical result is correct
        # This would require decoding and parsing - simplified here
        metrics["arithmetic_accuracy"] = compute_sequence_accuracy(predictions, labels)

    return metrics


class MetricTracker:
    """Track metrics over training."""

    def __init__(self):
        self.metrics_history = []
        self.running_metrics = {}

    def update(self, metrics: Dict[str, float]):
        """Update with new metrics."""
        for key, value in metrics.items():
            if key not in self.running_metrics:
                self.running_metrics[key] = []
            self.running_metrics[key].append(value)

    def compute_average(self) -> Dict[str, float]:
        """Compute average of running metrics."""
        avg = {}
        for key, values in self.running_metrics.items():
            avg[key] = np.mean(values)
        return avg

    def reset(self):
        """Reset running metrics."""
        self.running_metrics = {}

    def commit(self):
        """Commit current averages to history."""
        avg = self.compute_average()
        self.metrics_history.append(avg)
        self.reset()

    def get_history(self) -> List[Dict[str, float]]:
        """Get full metrics history."""
        return self.metrics_history

    def get_best(self, metric_name: str, higher_is_better: bool = True) -> Dict[str, float]:
        """Get best metrics by specified metric."""
        if not self.metrics_history:
            return {}

        if higher_is_better:
            best_idx = max(
                range(len(self.metrics_history)),
                key=lambda i: self.metrics_history[i].get(metric_name, float("-inf"))
            )
        else:
            best_idx = min(
                range(len(self.metrics_history)),
                key=lambda i: self.metrics_history[i].get(metric_name, float("inf"))
            )

        return self.metrics_history[best_idx]


def evaluate_model(
    model,
    dataloader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Comprehensive model evaluation.

    Args:
        model: Memory transformer model
        dataloader: Evaluation dataloader
        device: Device to run on
        max_batches: Optional maximum batches

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    all_metrics = MetricTracker()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                return_memory_state=True,
            )

            # Get predictions
            logits = outputs["logits"]
            predictions = logits.argmax(dim=-1)

            # Compute metrics
            batch_metrics = {
                "loss": outputs.get("loss", torch.tensor(0.0)).item(),
                "token_accuracy": compute_accuracy(predictions, labels),
                "sequence_accuracy": compute_sequence_accuracy(predictions, labels),
                "perplexity": compute_perplexity(logits, labels),
            }

            # Memory metrics
            if "memory_state" in outputs:
                memory_metrics = compute_memory_usage(outputs["memory_state"])
                batch_metrics.update({f"mem_{k}": v for k, v in memory_metrics.items()})

            all_metrics.update(batch_metrics)

    return all_metrics.compute_average()
