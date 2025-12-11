"""Datasets for curriculum learning with memory transformer."""

import random
import string
import torch
from torch.utils.data import Dataset, IterableDataset
from typing import List, Dict, Optional, Tuple, Generator
import json

from .tokenizer import TaskTokenizer


class CopyDataset(Dataset):
    """Dataset for copy tasks (Stage 1 of curriculum).

    The model must learn to copy input sequences, which requires
    learning basic memory read/write operations.
    """

    def __init__(
        self,
        tokenizer: TaskTokenizer,
        num_samples: int = 10000,
        min_length: int = 4,
        max_length: int = 16,
        max_seq_len: int = 64,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.min_length = min_length
        self.max_length = max_length
        self.max_seq_len = max_seq_len
        self.seed = seed

        # Pre-generate samples for reproducibility
        self.samples = self._generate_samples()

    def _generate_samples(self) -> List[Tuple[List[int], List[int]]]:
        """Generate copy task samples."""
        random.seed(self.seed)
        samples = []

        # Character set for copy content
        chars = string.ascii_lowercase + string.digits

        for _ in range(self.num_samples):
            # Random length
            length = random.randint(self.min_length, self.max_length)

            # Random content
            content = "".join(random.choices(chars, k=length))

            # Encode
            input_ids, labels = self.tokenizer.encode_copy_task(content)

            # Truncate if needed
            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len]
                labels = labels[:self.max_seq_len]

            samples.append((input_ids, labels))

        return samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids, labels = self.samples[idx]

        # Pad to max_seq_len
        pad_length = self.max_seq_len - len(input_ids)
        if pad_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
            labels = labels + [-100] * pad_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class RecallDataset(Dataset):
    """Dataset for store/recall tasks (Stage 2 of curriculum).

    The model must store key-value pairs and recall them after
    distractor tokens, requiring learned memory retrieval.
    """

    def __init__(
        self,
        tokenizer: TaskTokenizer,
        num_samples: int = 10000,
        key_length: int = 2,
        value_length: int = 4,
        min_distractor_length: int = 5,
        max_distractor_length: int = 20,
        max_seq_len: int = 128,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.key_length = key_length
        self.value_length = value_length
        self.min_distractor_length = min_distractor_length
        self.max_distractor_length = max_distractor_length
        self.max_seq_len = max_seq_len
        self.seed = seed

        self.samples = self._generate_samples()

    def _generate_samples(self) -> List[Tuple[List[int], List[int]]]:
        """Generate store/recall samples."""
        random.seed(self.seed)
        samples = []

        chars = string.ascii_lowercase
        digits = string.digits

        for _ in range(self.num_samples):
            # Generate key (letters) and value (digits)
            key = "".join(random.choices(chars, k=self.key_length))
            value = "".join(random.choices(digits, k=self.value_length))

            # Generate distractor (mixed)
            distractor_length = random.randint(
                self.min_distractor_length, self.max_distractor_length
            )
            distractor = "".join(random.choices(chars + digits, k=distractor_length))

            # Encode
            input_ids, labels = self.tokenizer.encode_recall_task(
                key, value, distractor
            )

            # Truncate if needed
            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len]
                labels = labels[:self.max_seq_len]

            samples.append((input_ids, labels))

        return samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids, labels = self.samples[idx]

        # Pad
        pad_length = self.max_seq_len - len(input_ids)
        if pad_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
            labels = labels + [-100] * pad_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class ArithmeticDataset(Dataset):
    """Dataset for multi-step arithmetic (Stage 3 of curriculum).

    The model must solve arithmetic problems that require storing
    and retrieving intermediate results.
    """

    def __init__(
        self,
        tokenizer: TaskTokenizer,
        num_samples: int = 10000,
        max_operand: int = 99,
        num_steps: int = 2,
        operations: List[str] = None,
        max_seq_len: int = 128,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_operand = max_operand
        self.num_steps = num_steps
        self.operations = operations or ["+", "-", "*"]
        self.max_seq_len = max_seq_len
        self.seed = seed

        self.samples = self._generate_samples()

    def _generate_samples(self) -> List[Tuple[List[int], List[int]]]:
        """Generate arithmetic samples with intermediate steps."""
        random.seed(self.seed)
        samples = []

        for _ in range(self.num_samples):
            # Generate a multi-step problem
            expression, result, steps = self._generate_problem()

            # Encode
            input_ids, labels = self.tokenizer.encode_arithmetic_task(
                expression, str(result), steps
            )

            # Truncate if needed
            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len]
                labels = labels[:self.max_seq_len]

            samples.append((input_ids, labels))

        return samples

    def _generate_problem(self) -> Tuple[str, int, List[Tuple[str, str]]]:
        """Generate a single arithmetic problem.

        Returns:
            Tuple of (expression, result, intermediate_steps)
        """
        var_names = list(string.ascii_uppercase[:26])
        steps = []

        # Start with first operand
        current_value = random.randint(1, self.max_operand)
        expression_parts = [str(current_value)]

        for i in range(self.num_steps):
            op = random.choice(self.operations)
            operand = random.randint(1, self.max_operand)

            # Compute result
            if op == "+":
                new_value = current_value + operand
            elif op == "-":
                new_value = current_value - operand
            else:  # "*"
                new_value = current_value * operand

            # Record intermediate step
            var_name = var_names[i]
            steps.append((var_name, str(current_value)))

            expression_parts.append(op)
            expression_parts.append(str(operand))

            current_value = new_value

        expression = "".join(expression_parts)
        return expression, current_value, steps

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids, labels = self.samples[idx]

        # Pad
        pad_length = self.max_seq_len - len(input_ids)
        if pad_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
            labels = labels + [-100] * pad_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class MultiKeyRecallDataset(Dataset):
    """Extended recall dataset with multiple key-value pairs."""

    def __init__(
        self,
        tokenizer: TaskTokenizer,
        num_samples: int = 10000,
        num_pairs: int = 3,
        key_length: int = 2,
        value_length: int = 3,
        max_seq_len: int = 256,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.num_pairs = num_pairs
        self.key_length = key_length
        self.value_length = value_length
        self.max_seq_len = max_seq_len
        self.seed = seed

        self.samples = self._generate_samples()

    def _generate_samples(self) -> List[Tuple[List[int], List[int]]]:
        """Generate multi-key recall samples."""
        random.seed(self.seed)
        samples = []

        chars = string.ascii_lowercase
        digits = string.digits

        for _ in range(self.num_samples):
            # Generate multiple key-value pairs
            pairs = []
            used_keys = set()

            for _ in range(self.num_pairs):
                # Ensure unique keys
                while True:
                    key = "".join(random.choices(chars, k=self.key_length))
                    if key not in used_keys:
                        used_keys.add(key)
                        break

                value = "".join(random.choices(digits, k=self.value_length))
                pairs.append((key, value))

            # Build sequence: store all pairs, then recall one
            input_parts = []
            for key, value in pairs:
                input_parts.append(f"{self.tokenizer.STORE_TOKEN}{key}={value}")

            # Add separator and some distractor
            distractor = "".join(random.choices(chars + digits, k=random.randint(5, 15)))
            input_parts.append(self.tokenizer.SEP_TOKEN + distractor)

            # Randomly select a key to recall
            recall_idx = random.randint(0, self.num_pairs - 1)
            recall_key, recall_value = pairs[recall_idx]

            input_parts.append(f"{self.tokenizer.RECALL_TOKEN}{recall_key}{self.tokenizer.EQUALS_TOKEN}")

            input_text = "".join(input_parts)
            target_text = recall_value

            input_ids = self.tokenizer.encode(input_text, add_bos=True, add_eos=False)
            target_ids = self.tokenizer.encode(target_text, add_bos=False, add_eos=True)

            full_ids = input_ids + target_ids
            labels = [-100] * len(input_ids) + target_ids

            if len(full_ids) > self.max_seq_len:
                full_ids = full_ids[:self.max_seq_len]
                labels = labels[:self.max_seq_len]

            samples.append((full_ids, labels))

        return samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids, labels = self.samples[idx]

        pad_length = self.max_seq_len - len(input_ids)
        if pad_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
            labels = labels + [-100] * pad_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class CurriculumDataset(IterableDataset):
    """Curriculum learning dataset that progresses through stages.

    Automatically advances stages based on model performance.
    """

    STAGES = ["copy", "recall", "arithmetic", "mixed"]

    def __init__(
        self,
        tokenizer: TaskTokenizer,
        max_seq_len: int = 256,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.current_stage = 0
        self.stage_accuracies = {stage: 0.0 for stage in self.STAGES}

        # Stage thresholds for advancement
        self.thresholds = {
            "copy": 0.90,
            "recall": 0.85,
            "arithmetic": 0.80,
            "mixed": 1.0,  # Never advance past mixed
        }

        # Initialize random generator
        self.rng = random.Random(seed)

    def set_stage(self, stage: int):
        """Manually set curriculum stage."""
        self.current_stage = min(stage, len(self.STAGES) - 1)

    def update_accuracy(self, stage: str, accuracy: float):
        """Update accuracy for a stage and potentially advance."""
        self.stage_accuracies[stage] = accuracy

        # Check if we should advance
        current_stage_name = self.STAGES[self.current_stage]
        if accuracy >= self.thresholds[current_stage_name]:
            if self.current_stage < len(self.STAGES) - 1:
                self.current_stage += 1
                print(f"Advanced to stage: {self.STAGES[self.current_stage]}")

    def _generate_copy_sample(self) -> Dict[str, torch.Tensor]:
        """Generate a copy task sample."""
        chars = string.ascii_lowercase + string.digits
        length = self.rng.randint(4, 16)
        content = "".join(self.rng.choices(chars, k=length))

        input_ids, labels = self.tokenizer.encode_copy_task(content)
        return self._pad_sample(input_ids, labels)

    def _generate_recall_sample(self) -> Dict[str, torch.Tensor]:
        """Generate a recall task sample."""
        chars = string.ascii_lowercase
        digits = string.digits

        key = "".join(self.rng.choices(chars, k=2))
        value = "".join(self.rng.choices(digits, k=4))
        distractor_length = self.rng.randint(5, 20)
        distractor = "".join(self.rng.choices(chars + digits, k=distractor_length))

        input_ids, labels = self.tokenizer.encode_recall_task(key, value, distractor)
        return self._pad_sample(input_ids, labels)

    def _generate_arithmetic_sample(self) -> Dict[str, torch.Tensor]:
        """Generate an arithmetic task sample."""
        # Generate 2-step problem
        a = self.rng.randint(1, 50)
        b = self.rng.randint(1, 50)
        c = self.rng.randint(1, 20)

        op1 = self.rng.choice(["+", "-"])
        op2 = self.rng.choice(["+", "-", "*"])

        if op1 == "+":
            intermediate = a + b
        else:
            intermediate = a - b

        if op2 == "+":
            result = intermediate + c
        elif op2 == "-":
            result = intermediate - c
        else:
            result = intermediate * c

        expression = f"{a}{op1}{b}{op2}{c}"
        steps = [("A", str(intermediate))]

        input_ids, labels = self.tokenizer.encode_arithmetic_task(
            expression, str(result), steps
        )
        return self._pad_sample(input_ids, labels)

    def _generate_mixed_sample(self) -> Dict[str, torch.Tensor]:
        """Generate a random sample from any task type."""
        task_type = self.rng.choice(["copy", "recall", "arithmetic"])
        if task_type == "copy":
            return self._generate_copy_sample()
        elif task_type == "recall":
            return self._generate_recall_sample()
        else:
            return self._generate_arithmetic_sample()

    def _pad_sample(
        self,
        input_ids: List[int],
        labels: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Pad a sample to max_seq_len."""
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]

        pad_length = self.max_seq_len - len(input_ids)
        if pad_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
            labels = labels + [-100] * pad_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __iter__(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Generate samples based on current curriculum stage."""
        while True:
            stage_name = self.STAGES[self.current_stage]

            if stage_name == "copy":
                yield self._generate_copy_sample()
            elif stage_name == "recall":
                yield self._generate_recall_sample()
            elif stage_name == "arithmetic":
                yield self._generate_arithmetic_sample()
            else:  # mixed
                yield self._generate_mixed_sample()


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }
