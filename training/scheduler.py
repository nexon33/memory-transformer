"""Learning rate schedulers for training."""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional


class WarmupCosineScheduler(_LRScheduler):
    """Cosine learning rate scheduler with warmup.

    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps
        min_lr: Minimum learning rate (default: 0)
        last_step: Last step (for resuming)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
        last_step: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch=last_step)

    def get_lr(self):
        """Calculate learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            scale = self.last_epoch / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            scale = 0.5 * (1 + math.cos(math.pi * progress))

        return [
            self.min_lr + (base_lr - self.min_lr) * scale
            for base_lr in self.base_lrs
        ]


class WarmupLinearScheduler(_LRScheduler):
    """Linear learning rate scheduler with warmup.

    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps
        min_lr: Minimum learning rate
        last_step: Last step (for resuming)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
        last_step: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch=last_step)

    def get_lr(self):
        """Calculate learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            scale = self.last_epoch / max(1, self.warmup_steps)
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            scale = max(0, 1 - progress)

        return [
            self.min_lr + (base_lr - self.min_lr) * scale
            for base_lr in self.base_lrs
        ]


class CurriculumScheduler(_LRScheduler):
    """Learning rate scheduler that adapts to curriculum stages.

    Can increase LR when advancing to new stages, or reset warmup.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        stage_steps: int,
        min_lr: float = 0.0,
        lr_bump_factor: float = 1.2,
        last_step: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.stage_steps = stage_steps
        self.min_lr = min_lr
        self.lr_bump_factor = lr_bump_factor
        self.current_stage = 0
        self.stage_start_step = 0
        super().__init__(optimizer, last_epoch=last_step)

    def advance_stage(self):
        """Called when advancing to next curriculum stage."""
        self.current_stage += 1
        self.stage_start_step = self.last_epoch

        # Bump base learning rates
        self.base_lrs = [lr * self.lr_bump_factor for lr in self.base_lrs]

    def get_lr(self):
        """Calculate learning rate for current step."""
        steps_in_stage = self.last_epoch - self.stage_start_step

        if steps_in_stage < self.warmup_steps:
            # Warmup within stage
            scale = steps_in_stage / max(1, self.warmup_steps)
        else:
            # Cosine decay within stage
            progress = (steps_in_stage - self.warmup_steps) / max(
                1, self.stage_steps - self.warmup_steps
            )
            progress = min(1.0, progress)
            scale = 0.5 * (1 + math.cos(math.pi * progress))

        return [
            self.min_lr + (base_lr - self.min_lr) * scale
            for base_lr in self.base_lrs
        ]


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    warmup_steps: int = 500,
    max_steps: int = 50000,
    min_lr: float = 1e-5,
) -> _LRScheduler:
    """Create a learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ("cosine", "linear", "curriculum")
        warmup_steps: Number of warmup steps
        max_steps: Total number of steps
        min_lr: Minimum learning rate

    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "cosine":
        return WarmupCosineScheduler(
            optimizer, warmup_steps, max_steps, min_lr
        )
    elif scheduler_type == "linear":
        return WarmupLinearScheduler(
            optimizer, warmup_steps, max_steps, min_lr
        )
    elif scheduler_type == "curriculum":
        return CurriculumScheduler(
            optimizer, warmup_steps, stage_steps=max_steps // 4, min_lr=min_lr
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
