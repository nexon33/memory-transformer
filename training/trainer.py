"""Training loop for memory transformer."""

import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, List
from tqdm import tqdm

from ..model.config import MemoryTransformerConfig, TrainingConfig
from ..model.memory_transformer import MemoryTransformer
from ..data.tokenizer import TaskTokenizer
from .losses import MemoryLoss, CurriculumLoss
from .scheduler import create_scheduler


class Trainer:
    """Trainer for memory-augmented transformer.

    Handles:
    - Training loop with gradient accumulation
    - Curriculum learning stage management
    - Checkpointing and logging
    - Evaluation
    """

    def __init__(
        self,
        model: MemoryTransformer,
        tokenizer: TaskTokenizer,
        train_config: TrainingConfig,
        model_config: MemoryTransformerConfig,
        output_dir: str = "checkpoints",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_config = train_config
        self.model_config = model_config
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )

        # Scheduler
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type="cosine",
            warmup_steps=train_config.warmup_steps,
            max_steps=train_config.max_steps,
            min_lr=train_config.min_learning_rate,
        )

        # Loss function
        self.loss_fn = CurriculumLoss(
            vocab_size=model_config.vocab_size,
            lm_weight=train_config.lm_loss_weight,
            memory_usage_weight=train_config.memory_usage_loss_weight,
            retrieval_weight=train_config.retrieval_loss_weight,
        )

        # Training state
        self.global_step = 0
        self.current_stage = "copy"
        self.stage_accuracies = {}
        self.best_loss = float("inf")

        # Logging
        self.log_history = []

    def save_checkpoint(self, name: str = "checkpoint"):
        """Save training checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "current_stage": self.current_stage,
            "stage_accuracies": self.stage_accuracies,
            "best_loss": self.best_loss,
            "model_config": vars(self.model_config),
            "train_config": vars(self.train_config),
        }

        path = os.path.join(self.output_dir, f"{name}.pt")
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.current_stage = checkpoint.get("current_stage", "copy")
        self.stage_accuracies = checkpoint.get("stage_accuracies", {})
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        print(f"Loaded checkpoint from {path} at step {self.global_step}")

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Perform a single training step.

        Args:
            batch: Dictionary with input_ids and labels

        Returns:
            Dictionary of loss values
        """
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            return_memory_state=True,
        )

        logits = outputs["logits"]
        memory_state = outputs.get("memory_state")

        # Compute losses
        losses = self.loss_fn(
            logits=logits,
            labels=labels,
            memory_state=memory_state,
        )

        # Scale loss for gradient accumulation
        loss = losses["total_loss"] / self.train_config.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        return {k: v.item() for k, v in losses.items()}

    def train_epoch(
        self,
        dataloader: DataLoader,
        max_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """Train for one epoch or specified steps.

        Args:
            dataloader: Training data loader
            max_steps: Optional maximum steps (otherwise full epoch)

        Returns:
            Average losses for the epoch
        """
        total_losses = {}
        num_batches = 0
        accumulated_steps = 0

        pbar = tqdm(dataloader, desc=f"Training (Stage: {self.current_stage})")

        for batch in pbar:
            # Training step
            losses = self.train_step(batch)

            # Accumulate losses for logging
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v

            num_batches += 1
            accumulated_steps += 1

            # Gradient accumulation
            if accumulated_steps >= self.train_config.gradient_accumulation_steps:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.max_grad_norm,
                )

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                accumulated_steps = 0

                # Logging
                if self.global_step % self.train_config.log_every == 0:
                    avg_loss = total_losses.get("total_loss", 0) / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}",
                        "step": self.global_step,
                    })

                # Save checkpoint
                if self.global_step % self.train_config.save_every == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}")

                # Check max steps
                if max_steps and self.global_step >= max_steps:
                    break

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}

        return avg_losses

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate the model.

        Args:
            dataloader: Evaluation data loader
            max_batches: Optional maximum batches to evaluate

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_tokens = 0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                labels=labels,
            )

            # Loss
            if "loss" in outputs:
                total_loss += outputs["loss"].item()

            # Accuracy (only on non-masked tokens)
            logits = outputs["logits"]
            predictions = logits.argmax(dim=-1)

            # Shift for comparison
            shift_preds = predictions[..., :-1]
            shift_labels = labels[..., 1:]

            # Mask
            mask = shift_labels != -100
            correct = (shift_preds == shift_labels) & mask

            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

            num_batches += 1

            if max_batches and num_batches >= max_batches:
                break

        metrics = {
            "eval_loss": total_loss / num_batches if num_batches > 0 else 0,
            "accuracy": total_correct / total_tokens if total_tokens > 0 else 0,
        }

        return metrics

    def check_stage_advancement(
        self,
        accuracy: float,
        current_stage: str,
    ) -> bool:
        """Check if we should advance to next curriculum stage.

        Args:
            accuracy: Current accuracy
            current_stage: Current stage name

        Returns:
            True if advanced, False otherwise
        """
        thresholds = self.train_config.stage_thresholds
        stages = ["copy", "recall", "arithmetic", "mixed"]

        if current_stage not in thresholds:
            return False

        if accuracy >= thresholds[current_stage]:
            stage_idx = stages.index(current_stage)
            if stage_idx < len(stages) - 1:
                self.current_stage = stages[stage_idx + 1]
                self.loss_fn.set_stage(self.current_stage)
                print(f"\nAdvancing to stage: {self.current_stage}")
                return True

        return False

    def train(
        self,
        train_dataloaders: Dict[str, DataLoader],
        eval_dataloaders: Optional[Dict[str, DataLoader]] = None,
        num_epochs: int = 10,
    ):
        """Full training loop with curriculum learning.

        Args:
            train_dataloaders: Dictionary mapping stage names to dataloaders
            eval_dataloaders: Optional evaluation dataloaders
            num_epochs: Number of epochs per stage
        """
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {self.model.n_params / 1e6:.2f}M")

        # Training loop
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Current stage: {self.current_stage}")
            print(f"{'='*50}")

            # Get dataloader for current stage
            if self.current_stage in train_dataloaders:
                train_dl = train_dataloaders[self.current_stage]
            else:
                # Default to mixed if stage not found
                train_dl = train_dataloaders.get("mixed", list(train_dataloaders.values())[0])

            # Train epoch
            train_losses = self.train_epoch(train_dl)

            print(f"\nTraining losses: {train_losses}")

            # Evaluate
            if eval_dataloaders:
                eval_dl = eval_dataloaders.get(
                    self.current_stage,
                    list(eval_dataloaders.values())[0]
                )
                eval_metrics = self.evaluate(eval_dl, max_batches=100)

                print(f"Evaluation metrics: {eval_metrics}")

                # Record stage accuracy
                self.stage_accuracies[self.current_stage] = eval_metrics["accuracy"]

                # Check for stage advancement
                if self.train_config.curriculum_enabled:
                    self.check_stage_advancement(
                        eval_metrics["accuracy"],
                        self.current_stage,
                    )

                # Save best model
                if eval_metrics["eval_loss"] < self.best_loss:
                    self.best_loss = eval_metrics["eval_loss"]
                    self.save_checkpoint("best_model")

            # Log
            log_entry = {
                "epoch": epoch + 1,
                "global_step": self.global_step,
                "stage": self.current_stage,
                "train_losses": train_losses,
            }
            if eval_dataloaders:
                log_entry["eval_metrics"] = eval_metrics

            self.log_history.append(log_entry)

            # Save logs
            with open(os.path.join(self.output_dir, "training_log.json"), "w") as f:
                json.dump(self.log_history, f, indent=2)

        # Final checkpoint
        self.save_checkpoint("final_model")

        print("\nTraining complete!")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Final stage: {self.current_stage}")
        print(f"Stage accuracies: {self.stage_accuracies}")

    @torch.no_grad()
    def generate_sample(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
    ) -> str:
        """Generate a sample from a prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        self.model.eval()

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Generate
        output_ids, _ = self.model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode
        output_text = self.tokenizer.decode(output_ids[0].tolist())

        return output_text
