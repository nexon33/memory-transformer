"""Custom loss functions for memory transformer training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MemoryLoss(nn.Module):
    """Combined loss for memory-augmented transformer training.

    Combines:
    1. Language modeling loss (cross-entropy)
    2. Memory usage regularization (encourage sparse, meaningful writes)
    3. Optional retrieval supervision
    """

    def __init__(
        self,
        vocab_size: int,
        lm_weight: float = 1.0,
        memory_usage_weight: float = 0.01,
        retrieval_weight: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.lm_weight = lm_weight
        self.memory_usage_weight = memory_usage_weight
        self.retrieval_weight = retrieval_weight

        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )

    def compute_lm_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute language modeling loss.

        Args:
            logits: Model outputs [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len]

        Returns:
            Cross-entropy loss
        """
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten and compute loss
        loss = self.ce_loss(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )

        return loss

    def compute_memory_usage_loss(
        self,
        memory_state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute memory usage regularization loss.

        Encourages:
        - Sparse memory writes (not writing everything)
        - Differentiated usage (not all slots equally used)
        - Meaningful write patterns

        Args:
            memory_state: Memory state dictionary

        Returns:
            Regularization loss
        """
        usage = memory_state["usage"]
        valid_mask = memory_state["valid_mask"].float()

        # 1. Encourage sparsity in usage (L1 on usage)
        sparsity_loss = usage.abs().mean()

        # 2. Encourage differentiated usage (negative variance)
        usage_mean = usage.mean(dim=-1, keepdim=True)
        variance_loss = -((usage - usage_mean) ** 2).mean()

        # 3. Penalize writing to too many slots
        write_rate = valid_mask.mean()
        write_penalty = F.relu(write_rate - 0.5)  # Penalize if >50% slots used

        total = sparsity_loss + 0.5 * variance_loss + write_penalty

        return total

    def compute_retrieval_loss(
        self,
        retrieval_details: Optional[Dict],
        target_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute retrieval supervision loss.

        Used when we know which memory slots should be retrieved.

        Args:
            retrieval_details: Dictionary with attention weights
            target_indices: Known correct memory indices to retrieve

        Returns:
            Retrieval supervision loss
        """
        if retrieval_details is None or target_indices is None:
            return torch.tensor(0.0)

        # Get attention weights from retrieval
        if "continuous_attention" in retrieval_details:
            attn_weights = retrieval_details["continuous_attention"]
        elif "hop_attentions" in retrieval_details:
            # Use last hop attention
            attn_weights = retrieval_details["hop_attentions"][-1]
        else:
            return torch.tensor(0.0)

        # Supervised retrieval: encourage attending to target indices
        batch_size, seq_len, memory_size = attn_weights.shape
        device = attn_weights.device

        # Create target distribution (one-hot or smoothed)
        target_dist = torch.zeros(batch_size, seq_len, memory_size, device=device)
        target_dist.scatter_(2, target_indices.unsqueeze(1).expand(-1, seq_len, -1), 1.0)

        # KL divergence loss
        log_attn = torch.log(attn_weights + 1e-10)
        loss = F.kl_div(log_attn, target_dist, reduction="batchmean")

        return loss

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        memory_state: Optional[Dict[str, torch.Tensor]] = None,
        retrieval_details: Optional[Dict] = None,
        target_memory_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            logits: Model outputs [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len]
            memory_state: Memory state for regularization
            retrieval_details: Retrieval information for supervision
            target_memory_indices: Known correct memory indices

        Returns:
            Dictionary with individual and total losses
        """
        losses = {}

        # Language modeling loss
        lm_loss = self.compute_lm_loss(logits, labels)
        losses["lm_loss"] = lm_loss

        # Memory usage loss
        if memory_state is not None and self.memory_usage_weight > 0:
            mem_loss = self.compute_memory_usage_loss(memory_state)
            losses["memory_usage_loss"] = mem_loss
        else:
            mem_loss = torch.tensor(0.0, device=logits.device)
            losses["memory_usage_loss"] = mem_loss

        # Retrieval loss
        if self.retrieval_weight > 0:
            ret_loss = self.compute_retrieval_loss(
                retrieval_details, target_memory_indices
            )
            losses["retrieval_loss"] = ret_loss
        else:
            ret_loss = torch.tensor(0.0, device=logits.device)
            losses["retrieval_loss"] = ret_loss

        # Combined loss
        total_loss = (
            self.lm_weight * lm_loss +
            self.memory_usage_weight * mem_loss +
            self.retrieval_weight * ret_loss
        )
        losses["total_loss"] = total_loss

        return losses


class CurriculumLoss(MemoryLoss):
    """Loss function that adapts to curriculum stage."""

    def __init__(
        self,
        vocab_size: int,
        lm_weight: float = 1.0,
        memory_usage_weight: float = 0.01,
        retrieval_weight: float = 0.1,
    ):
        super().__init__(
            vocab_size=vocab_size,
            lm_weight=lm_weight,
            memory_usage_weight=memory_usage_weight,
            retrieval_weight=retrieval_weight,
        )

        # Stage-specific weight adjustments
        self.stage_weights = {
            "copy": {"lm": 1.0, "memory": 0.1, "retrieval": 0.0},
            "recall": {"lm": 1.0, "memory": 0.05, "retrieval": 0.2},
            "arithmetic": {"lm": 1.0, "memory": 0.02, "retrieval": 0.1},
            "mixed": {"lm": 1.0, "memory": 0.01, "retrieval": 0.05},
        }

        self.current_stage = "copy"

    def set_stage(self, stage: str):
        """Set the current curriculum stage."""
        if stage in self.stage_weights:
            self.current_stage = stage
            weights = self.stage_weights[stage]
            self.lm_weight = weights["lm"]
            self.memory_usage_weight = weights["memory"]
            self.retrieval_weight = weights["retrieval"]


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in predictions."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Predictions [batch, seq_len, vocab_size]
            labels: Targets [batch, seq_len]

        Returns:
            Focal loss
        """
        # Flatten
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)

        # Mask out ignore index
        mask = labels != self.ignore_index
        logits = logits[mask]
        labels = labels[mask]

        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device)

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1 - probs) ** self.gamma

        # Cross-entropy
        ce_loss = F.cross_entropy(logits, labels, reduction="none")

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        return focal_loss.mean()
