"""Trainer for Self-RAG model."""

import os
import json
import time
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..model.self_rag import SelfRAGModel
from ..model.self_rag_config import SelfRAGConfig


@dataclass
class RAGTrainingConfig:
    """Training configuration for Self-RAG."""

    # Batch settings
    batch_size: int = 8
    gradient_accumulation_steps: int = 4

    # Learning rate
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    warmup_steps: int = 500
    max_steps: int = 50000

    # Optimization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    # Memory training
    memory_warmup_steps: int = 1000  # Steps before enabling memory
    memory_populate_batch: bool = True  # Populate memory from batch context

    # Checkpointing
    save_every: int = 1000
    eval_every: int = 500
    log_every: int = 50

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"


class RAGTrainer:
    """Trainer for Self-RAG model.

    Two-phase training:
    1. Warmup: Train base LM without memory
    2. Memory: Train with retrieval augmentation
    """

    def __init__(
        self,
        model: SelfRAGModel,
        config: RAGTrainingConfig,
        tokenizer: Any,
        output_dir: str = "outputs",
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set tokenizer
        model.set_tokenizer(tokenizer)

        # Device
        self.device = next(model.parameters()).device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
        self.amp_dtype = getattr(torch, config.amp_dtype) if config.use_amp else torch.float32

        # State
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_log = []

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup and cosine decay."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            progress = (step - self.config.warmup_steps) / max(
                1, self.config.max_steps - self.config.warmup_steps
            )
            return max(
                self.config.min_learning_rate / self.config.learning_rate,
                0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item())
            )
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def populate_memory_from_batch(self, batch: Dict[str, torch.Tensor]):
        """Populate memory from batch context.

        Splits batch into chunks and adds to memory.
        """
        input_ids = batch["input_ids"]  # [batch, seq_len]
        chunk_size = self.model.config.chunk_size
        overlap = self.model.config.chunk_overlap

        for b in range(input_ids.size(0)):
            seq = input_ids[b].tolist()

            # Remove padding
            if self.tokenizer.pad_token_id is not None:
                while seq and seq[-1] == self.tokenizer.pad_token_id:
                    seq.pop()

            # Split into chunks
            pos = 0
            while pos < len(seq):
                chunk_ids = seq[pos:pos + chunk_size]
                if len(chunk_ids) < chunk_size // 2:  # Skip tiny chunks
                    break

                # Decode chunk
                chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)

                # Add to memory with position metadata
                self.model.write_to_memory(
                    chunk_text,
                    token_ids=chunk_ids,
                    metadata={"position": pos, "batch_idx": b}
                )

                pos += chunk_size - overlap

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        use_memory: bool = True,
    ) -> Dict[str, float]:
        """Single training step.

        Args:
            batch: Input batch with input_ids and labels
            use_memory: Whether to use memory retrieval

        Returns:
            Dict of loss values
        """
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids.clone()).to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Forward with optional AMP
        with torch.cuda.amp.autocast(enabled=self.config.use_amp, dtype=self.amp_dtype):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_memory=use_memory,
            )
            loss = outputs["loss"] / self.config.gradient_accumulation_steps

        # Backward
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {"loss": loss.item() * self.config.gradient_accumulation_steps}

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        """Full training loop.

        Args:
            train_dataloader: Training data
            eval_dataloader: Optional evaluation data
        """
        print("=" * 60)
        print("Self-RAG Training")
        print("=" * 60)
        print(f"  Steps: {self.config.max_steps}")
        print(f"  Batch: {self.config.batch_size} x {self.config.gradient_accumulation_steps}")
        print(f"  LR: {self.config.learning_rate}")
        print(f"  Memory warmup: {self.config.memory_warmup_steps} steps")
        print("=" * 60)

        pbar = tqdm(total=self.config.max_steps, desc="Training")
        pbar.update(self.global_step)

        running_loss = 0.0
        data_iter = iter(train_dataloader)

        while self.global_step < self.config.max_steps:
            # Get batch (restart iterator if needed)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = next(data_iter)

            # Determine if using memory
            use_memory = self.global_step >= self.config.memory_warmup_steps

            # Populate memory from batch context
            if use_memory and self.config.memory_populate_batch:
                self.populate_memory_from_batch(batch)

            # Training step
            metrics = self.train_step(batch, use_memory=use_memory)
            running_loss += metrics["loss"]

            # Gradient accumulation
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            self.global_step += 1
            pbar.update(1)

            # Logging
            if self.global_step % self.config.log_every == 0:
                avg_loss = running_loss / self.config.log_every
                running_loss = 0.0

                lr = self.scheduler.get_last_lr()[0]
                mem_size = self.model.memory.size

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{lr:.2e}",
                    "mem": mem_size,
                    "phase": "memory" if use_memory else "warmup"
                })

                self.training_log.append({
                    "step": self.global_step,
                    "loss": avg_loss,
                    "lr": lr,
                    "memory_size": mem_size,
                    "use_memory": use_memory,
                })

            # Evaluation
            if eval_dataloader and self.global_step % self.config.eval_every == 0:
                eval_metrics = self.evaluate(eval_dataloader)
                print(f"\n[Step {self.global_step}] Eval loss: {eval_metrics['loss']:.4f}")

                if eval_metrics["loss"] < self.best_loss:
                    self.best_loss = eval_metrics["loss"]
                    self.save_checkpoint("best")

            # Checkpointing
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint(f"step-{self.global_step}")

        pbar.close()

        # Final save
        self.save_checkpoint("final")
        self._save_training_log()

        print("\nTraining complete!")
        print(f"Best loss: {self.best_loss:.4f}")

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = 50,
    ) -> Dict[str, float]:
        """Evaluate model.

        Args:
            dataloader: Evaluation data
            max_batches: Maximum batches to evaluate

        Returns:
            Dict of evaluation metrics
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            if max_batches and num_batches >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids.clone()).to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.config.use_amp, dtype=self.amp_dtype):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_memory=True,
                )

            total_loss += outputs["loss"].item()
            num_batches += 1

        return {"loss": total_loss / max(1, num_batches)}

    def save_checkpoint(self, name: str):
        """Save model checkpoint.

        Args:
            name: Checkpoint name
        """
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Model state
        torch.save(
            self.model.state_dict(),
            checkpoint_dir / "model.pt"
        )

        # Optimizer state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_loss": self.best_loss,
        }, checkpoint_dir / "optimizer.pt")

        # Config
        self.model.config.to_yaml(str(checkpoint_dir / "config.yaml"))

        # Memory (optional - can be large)
        self.model.memory.save(str(checkpoint_dir / "memory.json"))

        print(f"Saved checkpoint: {checkpoint_dir}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint.

        Args:
            path: Checkpoint directory path
        """
        checkpoint_dir = Path(path)

        # Model state
        self.model.load_state_dict(
            torch.load(checkpoint_dir / "model.pt", map_location=self.device)
        )

        # Optimizer state
        opt_state = torch.load(checkpoint_dir / "optimizer.pt", map_location=self.device)
        self.optimizer.load_state_dict(opt_state["optimizer"])
        self.scheduler.load_state_dict(opt_state["scheduler"])
        self.global_step = opt_state["global_step"]
        self.best_loss = opt_state["best_loss"]

        # Memory
        memory_path = checkpoint_dir / "memory.json"
        if memory_path.exists():
            self.model.memory.load(str(memory_path))

        print(f"Loaded checkpoint from {path} at step {self.global_step}")

    def _save_training_log(self):
        """Save training log to JSON."""
        log_path = self.output_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)
