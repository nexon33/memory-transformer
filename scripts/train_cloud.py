#!/usr/bin/env python3
"""Cloud training script for Memory Transformer.

Optimized for A100 GPUs with:
- Mixed precision (bf16)
- Wandb logging
- Streaming datasets
- Checkpoint to GCS
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from memory_transformer.model.config import MemoryTransformerConfig, TrainingConfig
from memory_transformer.model.memory_transformer import MemoryTransformer

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train Memory Transformer on Cloud")

    # Config files
    parser.add_argument("--model-config", type=str, default="configs/medium_a100.yaml",
                        help="Model configuration file")
    parser.add_argument("--train-config", type=str, default="configs/training_a100.yaml",
                        help="Training configuration file")

    # Overrides
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps")

    # Dataset
    parser.add_argument("--dataset", type=str, default="c4",
                        choices=["c4", "openwebtext", "pile", "curriculum"],
                        help="Dataset to train on")
    parser.add_argument("--dataset-subset", type=str, default="en",
                        help="Dataset subset (for c4)")
    parser.add_argument("--streaming", action="store_true", default=True,
                        help="Use streaming dataset")

    # Training stage
    parser.add_argument("--stage", type=str, default="language",
                        choices=["curriculum", "language", "finetune"],
                        help="Training stage")

    # Experiment
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Experiment name for wandb")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name for wandb")
    parser.add_argument("--wandb-project", type=str, default="memory-transformer",
                        help="Wandb project name")

    # Checkpointing
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--save-to-gcs", type=str, default=None,
                        help="GCS bucket for saving checkpoints (e.g., gs://my-bucket/checkpoints)")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to train on")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bfloat16 (A100 only)")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Use torch.compile (PyTorch 2.0+)")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test run with minimal steps")

    return parser.parse_args()


def load_config(path: str, config_class):
    """Load configuration from YAML file."""
    if Path(path).exists():
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return config_class(**config_dict)
    else:
        print(f"Warning: Config file {path} not found, using defaults")
        return config_class()


def create_streaming_dataloader(dataset_name, tokenizer, batch_size, max_seq_len, subset=None):
    """Create a streaming dataloader from HuggingFace datasets."""
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library required. pip install datasets")

    if dataset_name == "c4":
        dataset = load_dataset("c4", subset or "en", streaming=True, split="train")
    elif dataset_name == "openwebtext":
        dataset = load_dataset("openwebtext", streaming=True, split="train")
    elif dataset_name == "pile":
        dataset = load_dataset("EleutherAI/pile", streaming=True, split="train")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_tensors="pt"
        )

    def collate_batch(batch):
        texts = [item["text"] for item in batch]
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"]
        # Shift for language modeling
        return {
            "input_ids": input_ids[:, :-1],
            "labels": input_ids[:, 1:],
            "attention_mask": encodings["attention_mask"][:, :-1]
        }

    # Create iterator
    dataset = dataset.shuffle(seed=42, buffer_size=10000)

    return dataset, collate_batch


def save_checkpoint(model, optimizer, scheduler, step, loss, output_dir, gcs_path=None):
    """Save checkpoint locally and optionally to GCS."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "loss": loss,
    }

    local_path = Path(output_dir) / f"checkpoint-{step}.pt"
    torch.save(checkpoint, local_path)
    print(f"Saved checkpoint to {local_path}")

    # Also save best checkpoint
    best_path = Path(output_dir) / "checkpoint-best.pt"
    torch.save(checkpoint, best_path)

    if gcs_path:
        try:
            import subprocess
            gcs_full_path = f"{gcs_path}/checkpoint-{step}.pt"
            subprocess.run(["gsutil", "cp", str(local_path), gcs_full_path], check=True)
            print(f"Uploaded to {gcs_full_path}")
        except Exception as e:
            print(f"Warning: Failed to upload to GCS: {e}")


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 70)
    print("Memory Transformer - Cloud Training")
    print("=" * 70)

    # Determine config paths (relative to script)
    script_dir = Path(__file__).parent.parent
    model_config_path = script_dir / args.model_config
    train_config_path = script_dir / args.train_config

    # Load configurations
    model_config = load_config(str(model_config_path), MemoryTransformerConfig)
    train_config = load_config(str(train_config_path), TrainingConfig)

    # Apply overrides
    if args.batch_size:
        train_config.batch_size = args.batch_size
    if args.learning_rate:
        train_config.learning_rate = args.learning_rate
    if args.max_steps:
        train_config.max_steps = args.max_steps

    if args.dry_run:
        train_config.max_steps = 100
        train_config.eval_every = 20
        train_config.save_every = 50

    print(f"\nModel: {model_config.d_model}d, {model_config.n_layers}L, {model_config.n_heads}H")
    print(f"Memory: {model_config.memory_size} slots, {model_config.snapshot_interval} interval")
    print(f"Batch: {train_config.batch_size} x {train_config.gradient_accumulation_steps} = {train_config.batch_size * train_config.gradient_accumulation_steps}")
    print(f"LR: {train_config.learning_rate}, Max steps: {train_config.max_steps}")

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create tokenizer
    print("\nLoading tokenizer...")
    if TRANSFORMERS_AVAILABLE:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model_config.vocab_size = len(tokenizer)
    else:
        raise ImportError("transformers library required. pip install transformers")

    # Create model
    print("\nCreating model...")
    model = MemoryTransformer(model_config)
    model = model.to(device)

    num_params = count_parameters(model)
    print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # Compile model (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.95)
    )

    # Learning rate scheduler
    def lr_lambda(step):
        if step < train_config.warmup_steps:
            return step / train_config.warmup_steps
        decay_steps = train_config.max_steps - train_config.warmup_steps
        progress = (step - train_config.warmup_steps) / decay_steps
        return max(0.1, 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint["step"]
        print(f"Resumed at step {start_step}")

    # Initialize wandb
    if WANDB_AVAILABLE and not args.dry_run:
        run_name = args.run_name or f"mat-{model_config.d_model}d-{datetime.now().strftime('%Y%m%d-%H%M')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": vars(model_config),
                "training": vars(train_config),
                "dataset": args.dataset,
                "parameters": num_params,
            }
        )
        print(f"\nWandb: {wandb.run.url}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloader
    print(f"\nLoading dataset: {args.dataset}...")
    dataset, collate_fn = create_streaming_dataloader(
        args.dataset,
        tokenizer,
        train_config.batch_size,
        model_config.max_seq_len,
        subset=args.dataset_subset
    )

    # Mixed precision setup
    use_amp = args.bf16 and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = GradScaler() if use_amp and amp_dtype == torch.float16 else None
    print(f"Mixed precision: {amp_dtype if use_amp else 'disabled'}")

    # Training loop
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    model.train()
    global_step = start_step
    running_loss = 0.0
    batch_buffer = []

    pbar = tqdm(total=train_config.max_steps - start_step, desc="Training")

    for sample in dataset:
        batch_buffer.append(sample)

        if len(batch_buffer) < train_config.batch_size:
            continue

        # Collate batch
        batch = collate_fn(batch_buffer)
        batch_buffer = []

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass with mixed precision
        if use_amp:
            with autocast(dtype=amp_dtype):
                outputs = model(input_ids)
                logits = outputs["logits"]
                loss = nn.functional.cross_entropy(
                    logits.view(-1, model_config.vocab_size),
                    labels.view(-1),
                    ignore_index=tokenizer.pad_token_id
                )
                loss = loss / train_config.gradient_accumulation_steps
        else:
            outputs = model(input_ids)
            logits = outputs["logits"]
            loss = nn.functional.cross_entropy(
                logits.view(-1, model_config.vocab_size),
                labels.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
            loss = loss / train_config.gradient_accumulation_steps

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running_loss += loss.item()

        # Gradient accumulation
        if (global_step + 1) % train_config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)

            # Optimizer step
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            # Logging
            if global_step % train_config.log_every == 0:
                avg_loss = running_loss * train_config.gradient_accumulation_steps / train_config.log_every
                running_loss = 0.0

                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{lr:.2e}",
                    "mem": f"{torch.cuda.memory_allocated()/1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
                })

                if WANDB_AVAILABLE and not args.dry_run:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/step": global_step,
                    }, step=global_step)

            # Checkpointing
            if global_step % train_config.save_every == 0 and global_step > 0:
                save_checkpoint(
                    model, optimizer, scheduler,
                    global_step, loss.item(),
                    output_dir, args.save_to_gcs
                )

        global_step += 1
        pbar.update(1)

        if global_step >= train_config.max_steps:
            break

    pbar.close()

    # Final checkpoint
    print("\nSaving final checkpoint...")
    save_checkpoint(
        model, optimizer, scheduler,
        global_step, loss.item(),
        output_dir, args.save_to_gcs
    )

    if WANDB_AVAILABLE and not args.dry_run:
        wandb.finish()

    print("\nTraining complete!")
    print(f"Final loss: {loss.item():.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
