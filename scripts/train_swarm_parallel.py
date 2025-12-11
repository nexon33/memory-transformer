#!/usr/bin/env python3
"""Parallel swarm training for Memory Transformer.

Uses subprocess to spawn separate Python processes for true parallelism.
This approach works better on Android/Termux than multiprocessing.Process.
"""

import os
import sys
import argparse
import torch
import json
import time
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_worker_subprocess(
    worker_id: int,
    config_path: str,
    data_path: str,
    output_path: str,
    initial_state_path: Optional[str],
    num_steps: int,
    lr_scale: float,
    dropout_scale: float,
    seed: int,
):
    """Run a single worker as a subprocess."""
    script_dir = Path(__file__).parent
    worker_script = script_dir / "worker_process.py"

    cmd = [
        sys.executable,
        str(worker_script),
        "--worker-id", str(worker_id),
        "--config-path", config_path,
        "--data-path", data_path,
        "--output-path", output_path,
        "--num-steps", str(num_steps),
        "--lr-scale", str(lr_scale),
        "--dropout-scale", str(dropout_scale),
        "--seed", str(seed),
    ]

    if initial_state_path:
        cmd.extend(["--initial-state-path", initial_state_path])

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def mutate_hyperparams(lr_scale: float, dropout_scale: float, mutation_rate: float = 0.3) -> tuple:
    """Mutate hyperparameters with some randomness."""
    if random.random() < mutation_rate:
        lr_scale = lr_scale * random.uniform(0.7, 1.4)
        lr_scale = max(0.1, min(5.0, lr_scale))

    if random.random() < mutation_rate:
        dropout_scale = dropout_scale * random.uniform(0.8, 1.2)
        dropout_scale = max(0.3, min(2.0, dropout_scale))

    return lr_scale, dropout_scale


def train_swarm_parallel(
    num_workers: int = 4,
    steps_per_round: int = 50,
    num_rounds: int = 10,
    output_dir: str = "swarm_checkpoints_parallel",
    resume: Optional[str] = None,
    max_parallel: int = 2,  # Limit parallel processes to avoid OOM
):
    """Train using a swarm of workers (parallel subprocess execution)."""
    from memory_transformer.model.config import MemoryTransformerConfig, TrainingConfig
    from memory_transformer.data.data_utils import create_tokenizer, create_curriculum_dataloaders

    os.makedirs(output_dir, exist_ok=True)
    temp_dir = Path(output_dir) / "temp"
    temp_dir.mkdir(exist_ok=True)

    # Base config
    config = MemoryTransformerConfig(gradient_checkpointing=False)
    train_config = TrainingConfig(batch_size=2)

    # Save config for workers
    config_path = temp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
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
            "batch_size": train_config.batch_size,
            "learning_rate": train_config.learning_rate,
        }, f)

    # Create tokenizer and save data samples for workers
    tokenizer = create_tokenizer(vocab_size=config.vocab_size)
    dataloaders = create_curriculum_dataloaders(
        tokenizer=tokenizer,
        batch_size=train_config.batch_size,
        max_seq_len=config.max_seq_len,
        num_samples=1000,
        seed=42,
    )

    # Pre-generate and save data batches
    data_path = temp_dir / "data_batches.pt"
    dataloader = dataloaders["copy"]
    batches = []
    data_iter = iter(dataloader)
    for _ in range(steps_per_round * 2):  # Extra batches for safety
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        batches.append({
            "input_ids": batch["input_ids"],
            "labels": batch["labels"],
        })
    torch.save(batches, data_path)

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
    initial_state_path = None
    if resume and os.path.exists(resume):
        checkpoint = torch.load(resume, map_location="cpu")
        best_state = checkpoint.get("state_dict")
        best_loss = checkpoint.get("loss", float("inf"))
        best_hyperparams = checkpoint.get("hyperparams", best_hyperparams)
        initial_state_path = temp_dir / "initial_state.pt"
        torch.save(best_state, initial_state_path)
        print(f"Resumed from {resume} with loss {best_loss:.4f}")

    print(f"\n{'='*60}")
    print(f"Swarm Training (Parallel Subprocess)")
    print(f"{'='*60}")
    print(f"Workers per round: {num_workers}")
    print(f"Max parallel: {max_parallel}")
    print(f"Steps per worker: {steps_per_round}")
    print(f"Total rounds: {num_rounds}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for round_num in range(num_rounds):
        print(f"\n{'='*50}")
        print(f"Round {round_num + 1}/{num_rounds}")
        print(f"{'='*50}")

        results = []
        output_paths = []

        # Prepare output paths for all workers
        for i in range(num_workers):
            output_path = temp_dir / f"worker_{i}_result.pt"
            output_paths.append(output_path)
            if output_path.exists():
                output_path.unlink()

        # Run workers in batches of max_parallel
        worker_idx = 0
        while worker_idx < num_workers:
            batch_size = min(max_parallel, num_workers - worker_idx)
            processes = []

            print(f"\nStarting workers {worker_idx}-{worker_idx + batch_size - 1}...")

            for i in range(batch_size):
                idx = worker_idx + i
                params = population[idx]

                proc = run_worker_subprocess(
                    worker_id=idx,
                    config_path=str(config_path),
                    data_path=str(data_path),
                    output_path=str(output_paths[idx]),
                    initial_state_path=str(initial_state_path) if initial_state_path else None,
                    num_steps=steps_per_round,
                    lr_scale=params["lr_scale"],
                    dropout_scale=params["dropout_scale"],
                    seed=42 + round_num * 1000 + idx * 100,
                )
                processes.append((idx, proc, params))
                print(f"  Worker {idx} started (lr={params['lr_scale']:.2f}, dropout={params['dropout_scale']:.2f})")

            # Wait for this batch to complete
            for idx, proc, params in processes:
                stdout, stderr = proc.communicate()

                if proc.returncode != 0:
                    print(f"  Worker {idx} FAILED: {stderr.decode()[:200]}")
                    # Use a high loss for failed workers
                    results.append({
                        "worker_id": idx,
                        "avg_loss": float("inf"),
                        "lr_scale": params["lr_scale"],
                        "dropout_scale": params["dropout_scale"],
                        "state_dict": None,
                    })
                else:
                    # Load result from file
                    if output_paths[idx].exists():
                        result = torch.load(output_paths[idx], map_location="cpu")
                        results.append(result)
                        print(f"  Worker {idx}: loss={result['avg_loss']:.4f}")
                    else:
                        print(f"  Worker {idx}: No output file found")
                        results.append({
                            "worker_id": idx,
                            "avg_loss": float("inf"),
                            "lr_scale": params["lr_scale"],
                            "dropout_scale": params["dropout_scale"],
                            "state_dict": None,
                        })

            worker_idx += batch_size

        # Find best worker this round
        valid_results = [r for r in results if r["avg_loss"] < float("inf")]
        if not valid_results:
            print("No valid results this round!")
            continue

        best_result = min(valid_results, key=lambda x: x["avg_loss"])
        print(f"\nBest this round: Worker {best_result['worker_id']} "
              f"with loss {best_result['avg_loss']:.4f}")

        # Update global best
        if best_result["avg_loss"] < best_loss and best_result["state_dict"] is not None:
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

            # Update initial state for next round
            initial_state_path = temp_dir / "initial_state.pt"
            torch.save(best_state, initial_state_path)

        # Evolve population
        valid_results.sort(key=lambda x: x["avg_loss"])
        top_half = valid_results[:max(1, len(valid_results) // 2)]

        new_population = []
        for r in top_half:
            new_population.append({
                "lr_scale": r["lr_scale"],
                "dropout_scale": r["dropout_scale"],
            })

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
    parser = argparse.ArgumentParser(description="Parallel Swarm Training for Memory Transformer")
    parser.add_argument("--num-workers", type=int, default=4, help="Workers per round")
    parser.add_argument("--max-parallel", type=int, default=2, help="Max parallel processes")
    parser.add_argument("--steps-per-round", type=int, default=50, help="Steps per worker per round")
    parser.add_argument("--num-rounds", type=int, default=10, help="Number of evolution rounds")
    parser.add_argument("--output-dir", type=str, default="swarm_checkpoints_parallel", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    train_swarm_parallel(
        num_workers=args.num_workers,
        max_parallel=args.max_parallel,
        steps_per_round=args.steps_per_round,
        num_rounds=args.num_rounds,
        output_dir=args.output_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
