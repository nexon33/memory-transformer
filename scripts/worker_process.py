#!/usr/bin/env python3
"""Worker subprocess for parallel swarm training.

This script is called by train_swarm_parallel.py for each worker.
It trains a single model variant and saves results to a file.
"""

import os
import sys
import argparse
import json
import torch
import random
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from memory_transformer.model.config import MemoryTransformerConfig, TrainingConfig
from memory_transformer.model.memory_transformer import MemoryTransformer
from memory_transformer.training.losses import CurriculumLoss


def train_worker(
    worker_id: int,
    config_path: str,
    data_path: str,
    output_path: str,
    initial_state_path: str,
    num_steps: int,
    lr_scale: float,
    dropout_scale: float,
    seed: int,
):
    """Train a single worker and save results."""
    torch.manual_seed(seed)
    random.seed(seed)

    # Load config
    with open(config_path, "r") as f:
        cfg = json.load(f)

    config = MemoryTransformerConfig(
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        vocab_size=cfg["vocab_size"],
        max_seq_len=cfg["max_seq_len"],
        memory_size=cfg["memory_size"],
        snapshot_interval=cfg["snapshot_interval"],
        n_retrieval_hops=cfg["n_retrieval_hops"],
        dropout=min(0.3, cfg["dropout"] * dropout_scale),
        gradient_checkpointing=False,
    )

    # Create model
    print(f"Worker {worker_id}: Creating model...")
    model = MemoryTransformer(config)

    # Load initial state if provided
    if initial_state_path and os.path.exists(initial_state_path):
        initial_state = torch.load(initial_state_path, map_location="cpu")
        model.load_state_dict(initial_state)
        print(f"Worker {worker_id}: Loaded initial state from {initial_state_path}")

    # Load data batches
    batches = torch.load(data_path, map_location="cpu")
    print(f"Worker {worker_id}: Loaded {len(batches)} data batches")

    # Setup optimizer
    lr = cfg["learning_rate"] * lr_scale
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Loss function
    loss_fn = CurriculumLoss(vocab_size=config.vocab_size)

    # Train
    model.train()
    total_loss = 0

    pbar = tqdm(
        total=num_steps,
        desc=f"Worker {worker_id} (lr={lr_scale:.2f})",
        leave=True,
    )

    for step in range(num_steps):
        batch = batches[step % len(batches)]
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
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    pbar.close()
    avg_loss = total_loss / num_steps

    print(f"Worker {worker_id}: Avg loss = {avg_loss:.4f}")

    # Save results
    result = {
        "worker_id": worker_id,
        "avg_loss": avg_loss,
        "lr_scale": lr_scale,
        "dropout_scale": dropout_scale,
        "state_dict": model.state_dict(),
    }
    torch.save(result, output_path)
    print(f"Worker {worker_id}: Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Worker process for swarm training")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--initial-state-path", type=str, default=None)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--dropout-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_worker(
        worker_id=args.worker_id,
        config_path=args.config_path,
        data_path=args.data_path,
        output_path=args.output_path,
        initial_state_path=args.initial_state_path,
        num_steps=args.num_steps,
        lr_scale=args.lr_scale,
        dropout_scale=args.dropout_scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
