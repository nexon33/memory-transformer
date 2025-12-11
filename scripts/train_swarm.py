#!/usr/bin/env python3
"""Swarm training for Memory Transformer.

Sequential version optimized for Android/Termux (no multiprocessing).
Trains multiple model variants sequentially with evolutionary selection.
"""

import os
import sys
import argparse
import torch
from pathlib import Path
import time
import random
from tqdm import tqdm
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from memory_transformer.model.config import MemoryTransformerConfig, TrainingConfig
from memory_transformer.model.memory_transformer import MemoryTransformer
from memory_transformer.data.data_utils import create_tokenizer, create_curriculum_dataloaders
from memory_transformer.training.losses import CurriculumLoss


class SwarmWorker:
    """A single worker in the training swarm."""

    def __init__(
        self,
        worker_id: int,
        config: MemoryTransformerConfig,
        train_config: TrainingConfig,
        lr_scale: float = 1.0,
        dropout_scale: float = 1.0,
        seed: int = 42,
    ):
        self.worker_id = worker_id
        self.config = config
        self.train_config = train_config
        self.seed = seed + worker_id * 1000
        self.lr_scale = lr_scale
        self.dropout_scale = dropout_scale

    def train(
        self,
        dataloader,
        num_steps: int,
        initial_state: Optional[Dict] = None,
        show_progress: bool = True,
    ) -> Dict:
        """Train for specified steps and return results."""
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        # Create model with worker-specific variations
        config = MemoryTransformerConfig(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            d_ff=self.config.d_ff,
            vocab_size=self.config.vocab_size,
            max_seq_len=self.config.max_seq_len,
            memory_size=self.config.memory_size,
            snapshot_interval=self.config.snapshot_interval,
            n_retrieval_hops=self.config.n_retrieval_hops,
            dropout=min(0.3, self.config.dropout * self.dropout_scale),
            gradient_checkpointing=False,
        )

        model = MemoryTransformer(config)

        # Load initial state if provided (warm start from best)
        if initial_state is not None:
            model.load_state_dict(initial_state)

        # Optimizer with worker-specific LR
        lr = self.train_config.learning_rate * self.lr_scale
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # Loss function
        loss_fn = CurriculumLoss(vocab_size=config.vocab_size)

        # Train
        model.train()
        total_loss = 0
        step = 0
        data_iter = iter(dataloader)

        pbar = tqdm(
            total=num_steps,
            desc=f"Worker {self.worker_id} (lr={self.lr_scale:.2f})",
            disable=not show_progress,
            leave=False,
        )

        while step < num_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"]
            labels = batch["labels"]

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels, return_memory_state=True)

            losses = loss_fn(
                logits=outputs["logits"],
                labels=labels,
                memory_state=outputs.get("memory_state"),
            )

            loss = losses["total_loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        pbar.close()
        avg_loss = total_loss / num_steps

        return {
            "worker_id": self.worker_id,
            "avg_loss": avg_loss,
            "lr_scale": self.lr_scale,
            "dropout_scale": self.dropout_scale,
            "state_dict": model.state_dict(),
        }


def mutate_hyperparams(lr_scale: float, dropout_scale: float, mutation_rate: float = 0.3) -> tuple:
    """Mutate hyperparameters with some randomness."""
    if random.random() < mutation_rate:
        lr_scale = lr_scale * random.uniform(0.7, 1.4)
        lr_scale = max(0.1, min(5.0, lr_scale))  # Clamp

    if random.random() < mutation_rate:
        dropout_scale = dropout_scale * random.uniform(0.8, 1.2)
        dropout_scale = max(0.3, min(2.0, dropout_scale))  # Clamp

    return lr_scale, dropout_scale


def train_swarm(
    num_workers: int = 4,
    steps_per_round: int = 50,
    num_rounds: int = 10,
    output_dir: str = "swarm_checkpoints",
    resume: Optional[str] = None,
):
    """Train using a swarm of workers (sequential execution)."""

    os.makedirs(output_dir, exist_ok=True)

    # Base config
    config = MemoryTransformerConfig(gradient_checkpointing=False)
    train_config = TrainingConfig(batch_size=2)

    # Create tokenizer and dataloader (shared across workers)
    tokenizer = create_tokenizer(vocab_size=config.vocab_size)
    dataloaders = create_curriculum_dataloaders(
        tokenizer=tokenizer,
        batch_size=train_config.batch_size,
        max_seq_len=config.max_seq_len,
        num_samples=1000,
        seed=42,
    )
    dataloader = dataloaders["copy"]  # Start with copy task

    # Initialize population with diverse hyperparameters
    population = []
    for i in range(num_workers):
        population.append({
            "lr_scale": random.uniform(0.5, 2.0),
            "dropout_scale": random.uniform(0.5, 1.5),
        })

    best_loss = float("inf")
    best_state = None
    best_hyperparams = {"lr_scale": 1.0, "dropout_scale": 1.0}

    # Resume from checkpoint if provided
    if resume and os.path.exists(resume):
        checkpoint = torch.load(resume, map_location="cpu")
        best_state = checkpoint.get("state_dict")
        best_loss = checkpoint.get("loss", float("inf"))
        best_hyperparams = checkpoint.get("hyperparams", best_hyperparams)
        print(f"Resumed from {resume} with loss {best_loss:.4f}")

    print(f"\n{'='*60}")
    print(f"Swarm Training (Sequential)")
    print(f"{'='*60}")
    print(f"Workers per round: {num_workers}")
    print(f"Steps per worker: {steps_per_round}")
    print(f"Total rounds: {num_rounds}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for round_num in range(num_rounds):
        print(f"\n{'='*50}")
        print(f"Round {round_num + 1}/{num_rounds}")
        print(f"{'='*50}")

        results = []

        # Run each worker sequentially
        for i, params in enumerate(population):
            worker = SwarmWorker(
                worker_id=i,
                config=config,
                train_config=train_config,
                lr_scale=params["lr_scale"],
                dropout_scale=params["dropout_scale"],
                seed=42 + round_num * 1000 + i * 100,
            )

            # Warm start from best model (if we have one)
            result = worker.train(
                dataloader=dataloader,
                num_steps=steps_per_round,
                initial_state=best_state,
                show_progress=True,
            )
            results.append(result)
            print(f"  Worker {i}: loss={result['avg_loss']:.4f} "
                  f"(lr={params['lr_scale']:.2f}, dropout={params['dropout_scale']:.2f})")

        # Find best worker this round
        best_result = min(results, key=lambda x: x["avg_loss"])
        print(f"\nBest this round: Worker {best_result['worker_id']} "
              f"with loss {best_result['avg_loss']:.4f}")

        # Update global best
        if best_result["avg_loss"] < best_loss:
            best_loss = best_result["avg_loss"]
            best_state = best_result["state_dict"]
            best_hyperparams = {
                "lr_scale": best_result["lr_scale"],
                "dropout_scale": best_result["dropout_scale"],
            }

            # Save checkpoint
            torch.save({
                "state_dict": best_state,
                "loss": best_loss,
                "hyperparams": best_hyperparams,
                "config": {
                    "d_model": config.d_model,
                    "n_heads": config.n_heads,
                    "n_layers": config.n_layers,
                    "d_ff": config.d_ff,
                    "vocab_size": config.vocab_size,
                    "max_seq_len": config.max_seq_len,
                    "memory_size": config.memory_size,
                    "snapshot_interval": config.snapshot_interval,
                    "n_retrieval_hops": config.n_retrieval_hops,
                    "dropout": config.dropout,
                },
                "round": round_num + 1,
            }, os.path.join(output_dir, "best_model.pt"))
            print(f"  -> New best! Saved to {output_dir}/best_model.pt")

        # Evolve population: keep top half, mutate to create new variants
        results.sort(key=lambda x: x["avg_loss"])
        top_half = results[:num_workers // 2]

        new_population = []
        # Keep best performers
        for r in top_half:
            new_population.append({
                "lr_scale": r["lr_scale"],
                "dropout_scale": r["dropout_scale"],
            })

        # Create mutated variants from best
        while len(new_population) < num_workers:
            parent = random.choice(top_half)
            lr, dropout = mutate_hyperparams(
                parent["lr_scale"],
                parent["dropout_scale"],
                mutation_rate=0.5,
            )
            new_population.append({
                "lr_scale": lr,
                "dropout_scale": dropout,
            })

        population = new_population

        elapsed = time.time() - start_time
        print(f"\nElapsed: {elapsed/60:.1f} min | Best overall: {best_loss:.4f}")

    print(f"\n{'='*60}")
    print(f"Swarm training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Best hyperparams: lr_scale={best_hyperparams['lr_scale']:.2f}, "
          f"dropout_scale={best_hyperparams['dropout_scale']:.2f}")
    print(f"Total time: {(time.time() - start_time)/60:.1f} min")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Swarm Training for Memory Transformer")
    parser.add_argument("--num-workers", type=int, default=4, help="Workers per round")
    parser.add_argument("--steps-per-round", type=int, default=50, help="Steps per worker per round")
    parser.add_argument("--num-rounds", type=int, default=10, help="Number of evolution rounds")
    parser.add_argument("--output-dir", type=str, default="swarm_checkpoints", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    train_swarm(
        num_workers=args.num_workers,
        steps_per_round=args.steps_per_round,
        num_rounds=args.num_rounds,
        output_dir=args.output_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
