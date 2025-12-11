"""Data loading utilities."""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any

from .tokenizer import TaskTokenizer
from .datasets import (
    CopyDataset,
    RecallDataset,
    ArithmeticDataset,
    MultiKeyRecallDataset,
    CurriculumDataset,
    collate_fn,
)


def create_tokenizer(vocab_size: int = 4096) -> TaskTokenizer:
    """Create and return a task tokenizer."""
    return TaskTokenizer(vocab_size=vocab_size)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a DataLoader from a dataset.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle (only for map-style datasets)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory (for GPU)

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not isinstance(dataset, CurriculumDataset),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


def create_curriculum_dataloaders(
    tokenizer: TaskTokenizer,
    batch_size: int = 2,
    max_seq_len: int = 256,
    num_samples: int = 10000,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """Create dataloaders for each curriculum stage.

    Args:
        tokenizer: Task tokenizer
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        num_samples: Samples per dataset
        seed: Random seed

    Returns:
        Dictionary mapping stage names to DataLoaders
    """
    dataloaders = {}

    # Copy dataset (Stage 1)
    copy_dataset = CopyDataset(
        tokenizer=tokenizer,
        num_samples=num_samples,
        max_seq_len=max_seq_len,
        seed=seed,
    )
    dataloaders["copy"] = create_dataloader(
        copy_dataset, batch_size=batch_size, shuffle=True
    )

    # Recall dataset (Stage 2)
    recall_dataset = RecallDataset(
        tokenizer=tokenizer,
        num_samples=num_samples,
        max_seq_len=max_seq_len,
        seed=seed + 1,
    )
    dataloaders["recall"] = create_dataloader(
        recall_dataset, batch_size=batch_size, shuffle=True
    )

    # Arithmetic dataset (Stage 3)
    arithmetic_dataset = ArithmeticDataset(
        tokenizer=tokenizer,
        num_samples=num_samples,
        max_seq_len=max_seq_len,
        seed=seed + 2,
    )
    dataloaders["arithmetic"] = create_dataloader(
        arithmetic_dataset, batch_size=batch_size, shuffle=True
    )

    # Multi-key recall (advanced)
    multi_recall_dataset = MultiKeyRecallDataset(
        tokenizer=tokenizer,
        num_samples=num_samples,
        max_seq_len=max_seq_len,
        seed=seed + 3,
    )
    dataloaders["multi_recall"] = create_dataloader(
        multi_recall_dataset, batch_size=batch_size, shuffle=True
    )

    return dataloaders


def create_curriculum_iterator(
    tokenizer: TaskTokenizer,
    batch_size: int = 2,
    max_seq_len: int = 256,
    seed: int = 42,
) -> tuple:
    """Create a curriculum dataset iterator.

    Args:
        tokenizer: Task tokenizer
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        seed: Random seed

    Returns:
        Tuple of (CurriculumDataset, DataLoader)
    """
    dataset = CurriculumDataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        seed=seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    return dataset, dataloader


class InfiniteDataLoader:
    """DataLoader that cycles infinitely."""

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch

    def reset(self):
        """Reset the iterator."""
        self.iterator = iter(self.dataloader)
