#!/usr/bin/env python3
"""Distributed training script for Memory Transformer using Accelerate.

Run with: accelerate launch --num_processes=8 scripts/train_distributed.py

For 8x A100 setup on GCP:
  accelerate launch --multi_gpu --num_processes=8 scripts/train_distributed.py
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from memory_transformer.model.config import MemoryTransformerConfig, TrainingConfig
from memory_transformer.model.memory_transformer import MemoryTransformer

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("Warning: accelerate not available. pip install accelerate")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError("transformers and datasets required")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str, default="configs/large_a100.yaml")
    parser.add_argument("--batch-size", type=int, default=8)  # Per GPU
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--dataset", type=str, default="c4")
    parser.add_argument("--output-dir", type=str, default="outputs-distributed")
    parser.add_argument("--wandb-project", type=str, default="memory-transformer")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(path: str):
    """Load model config from YAML."""
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / path
    if full_path.exists():
        with open(full_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return MemoryTransformerConfig(**config_dict)
    return MemoryTransformerConfig()


def create_streaming_dataset(dataset_name, tokenizer, max_seq_len):
    """Create streaming dataset with tokenization."""
    if dataset_name == "c4":
        dataset = load_dataset("c4", "en", streaming=True, split="train")
    elif dataset_name == "openwebtext":
        dataset = load_dataset("openwebtext", streaming=True, split="train")
    else:
        dataset = load_dataset(dataset_name, streaming=True, split="train")

    dataset = dataset.shuffle(seed=42, buffer_size=10000)
    return dataset


def collate_fn(batch, tokenizer, max_seq_len):
    """Collate function for streaming data."""
    texts = [item["text"] for item in batch]
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = encodings["input_ids"]
    return {
        "input_ids": input_ids[:, :-1],
        "labels": input_ids[:, 1:],
        "attention_mask": encodings["attention_mask"][:, :-1]
    }


def main():
    args = parse_args()

    if not ACCELERATE_AVAILABLE:
        raise ImportError("accelerate required for distributed training")

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",  # A100 supports bf16
        log_with="wandb" if WANDB_AVAILABLE else None,
    )

    # Set seed
    set_seed(args.seed)

    # Only main process prints
    if accelerator.is_main_process:
        print("=" * 70)
        print("Memory Transformer - Distributed Training")
        print(f"GPUs: {accelerator.num_processes}")
        print("=" * 70)

    # Load config
    model_config = load_config(args.model_config)
    model_config.max_seq_len = args.max_seq_len

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model_config.vocab_size = len(tokenizer)

    # Create model
    model = MemoryTransformer(model_config)

    num_params = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        print(f"\nModel: {num_params/1e6:.1f}M parameters")
        print(f"Per-GPU batch: {args.batch_size}")
        print(f"Global batch: {args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )

    # Learning rate scheduler
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        progress = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
        return max(0.1, 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Prepare for distributed training
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        accelerator.load_state(args.resume)
        # Extract step from checkpoint name
        try:
            start_step = int(Path(args.resume).stem.split("-")[-1])
        except:
            pass
        if accelerator.is_main_process:
            print(f"Resumed from {args.resume} at step {start_step}")

    # Initialize wandb
    if WANDB_AVAILABLE and accelerator.is_main_process:
        run_name = args.run_name or f"mat-dist-{datetime.now().strftime('%m%d-%H%M')}"
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config={
                "model_params": num_params,
                "batch_size_per_gpu": args.batch_size,
                "global_batch_size": args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "max_steps": args.max_steps,
                "num_gpus": accelerator.num_processes,
            },
            init_kwargs={"wandb": {"name": run_name}}
        )

    # Create output directory
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    if accelerator.is_main_process:
        print(f"\nLoading dataset: {args.dataset}")
    dataset = create_streaming_dataset(args.dataset, tokenizer, args.max_seq_len)

    # Training loop
    model.train()
    global_step = start_step
    running_loss = 0.0
    batch_buffer = []

    if accelerator.is_main_process:
        pbar = tqdm(total=args.max_steps - start_step, desc="Training")

    for sample in dataset:
        batch_buffer.append(sample)

        if len(batch_buffer) < args.batch_size:
            continue

        # Collate batch
        batch = collate_fn(batch_buffer, tokenizer, args.max_seq_len)
        batch_buffer = []

        input_ids = batch["input_ids"].to(accelerator.device)
        labels = batch["labels"].to(accelerator.device)

        # Forward pass with gradient accumulation
        with accelerator.accumulate(model):
            outputs = model(input_ids)
            logits = outputs["logits"]
            loss = nn.functional.cross_entropy(
                logits.view(-1, model_config.vocab_size),
                labels.view(-1),
                ignore_index=tokenizer.pad_token_id
            )

            accelerator.backward(loss)

            # Gradient clipping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        running_loss += loss.item()

        # Only count steps when gradients are synced
        if accelerator.sync_gradients:
            global_step += 1

            # Logging
            if global_step % 50 == 0:
                avg_loss = running_loss / 50
                running_loss = 0.0

                if accelerator.is_main_process:
                    lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}"
                    })
                    pbar.update(50)

                    if WANDB_AVAILABLE:
                        accelerator.log({
                            "train/loss": avg_loss,
                            "train/lr": lr,
                        }, step=global_step)

            # Checkpointing
            if global_step % 1000 == 0 and global_step > 0:
                checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                accelerator.save_state(str(checkpoint_dir))
                if accelerator.is_main_process:
                    print(f"\nSaved checkpoint: {checkpoint_dir}")

        if global_step >= args.max_steps:
            break

    # Final checkpoint
    if accelerator.is_main_process:
        pbar.close()
        print("\nSaving final checkpoint...")

    accelerator.save_state(str(output_dir / "checkpoint-final"))

    if WANDB_AVAILABLE:
        accelerator.end_training()

    if accelerator.is_main_process:
        print("\nTraining complete!")


if __name__ == "__main__":
    main()
