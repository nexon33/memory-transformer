#!/usr/bin/env python3
"""Training script for Self-RAG model."""

import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from memory_transformer.model.self_rag import SelfRAGModel
from memory_transformer.model.self_rag_config import (
    SelfRAGConfig,
    tiny_config,
    small_config,
    medium_config,
)
from memory_transformer.training.trainer_rag import RAGTrainer, RAGTrainingConfig

# Try to import transformers for tokenizer
try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers not available, using basic tokenizer")

# Try to import datasets
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train Self-RAG Model")

    # Model size
    parser.add_argument("--size", type=str, default="tiny",
                        choices=["tiny", "small", "medium"],
                        help="Model size preset")

    # Training
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--memory-warmup", type=int, default=1000,
                        help="Steps before enabling memory")

    # Data
    parser.add_argument("--dataset", type=str, default="wikitext",
                        choices=["wikitext", "c4", "synthetic"])
    parser.add_argument("--max-seq-len", type=int, default=256)

    # Memory
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of chunks to retrieve")
    parser.add_argument("--chunk-size", type=int, default=64)

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/self_rag")
    parser.add_argument("--resume", type=str, default=None)

    # Hardware
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def create_synthetic_data(tokenizer, num_samples=1000, max_len=256):
    """Create synthetic training data for quick testing."""
    import random
    import string

    data = []
    for _ in range(num_samples):
        # Generate random text
        words = []
        for _ in range(random.randint(20, 50)):
            word_len = random.randint(3, 10)
            word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
            words.append(word)
        text = ' '.join(words)

        # Tokenize
        tokens = tokenizer.encode(text, max_length=max_len, truncation=True)
        data.append({"input_ids": tokens})

    return data


def collate_fn(batch, pad_id=0, max_len=256):
    """Collate function for DataLoader."""
    input_ids = [b["input_ids"][:max_len] for b in batch]

    # Pad
    max_batch_len = max(len(ids) for ids in input_ids)
    padded = []
    masks = []
    for ids in input_ids:
        pad_len = max_batch_len - len(ids)
        padded.append(ids + [pad_id] * pad_len)
        masks.append([1] * len(ids) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(padded, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.bool),
        "labels": torch.tensor(padded, dtype=torch.long),
    }


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Self-RAG Training")
    print("=" * 60)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Model config
    if args.size == "tiny":
        config = tiny_config()
    elif args.size == "small":
        config = small_config()
    else:
        config = medium_config()

    # Override config with args
    config.top_k = args.top_k
    config.chunk_size = args.chunk_size
    config.max_seq_len = args.max_seq_len

    print(f"\nModel: {args.size}")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  top_k: {config.top_k}")

    # Tokenizer
    if HF_AVAILABLE:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        config.vocab_size = len(tokenizer)
    else:
        # Basic tokenizer fallback
        from memory_transformer.data.tokenizer import TaskTokenizer
        tokenizer = TaskTokenizer(vocab_size=config.vocab_size)

    print(f"\nVocab size: {config.vocab_size}")

    # Create model
    print("\nCreating model...")
    model = SelfRAGModel(config, tokenizer)
    model = model.to(device)

    # Create data
    print(f"\nLoading dataset: {args.dataset}")

    if args.dataset == "synthetic":
        train_data = create_synthetic_data(tokenizer, num_samples=5000, max_len=args.max_seq_len)
        eval_data = create_synthetic_data(tokenizer, num_samples=500, max_len=args.max_seq_len)

    elif args.dataset == "wikitext" and DATASETS_AVAILABLE:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

        def tokenize(examples):
            return {"input_ids": tokenizer.encode(examples["text"], max_length=args.max_seq_len, truncation=True)}

        train_data = [tokenize({"text": t}) for t in dataset["train"]["text"] if len(t) > 50][:5000]
        eval_data = [tokenize({"text": t}) for t in dataset["validation"]["text"] if len(t) > 50][:500]

    elif args.dataset == "c4" and DATASETS_AVAILABLE:
        dataset = load_dataset("c4", "en", streaming=True, split="train")
        train_data = []
        for i, sample in enumerate(dataset):
            if i >= 5000:
                break
            tokens = tokenizer.encode(sample["text"], max_length=args.max_seq_len, truncation=True)
            train_data.append({"input_ids": tokens})
        eval_data = train_data[:500]  # Simplified for streaming

    else:
        print("Dataset not available, using synthetic data")
        train_data = create_synthetic_data(tokenizer, num_samples=5000, max_len=args.max_seq_len)
        eval_data = create_synthetic_data(tokenizer, num_samples=500, max_len=args.max_seq_len)

    print(f"  Train samples: {len(train_data)}")
    print(f"  Eval samples: {len(eval_data)}")

    # Create dataloaders
    pad_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id, args.max_seq_len),
    )

    eval_loader = DataLoader(
        eval_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id, args.max_seq_len),
    )

    # Training config
    train_config = RAGTrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        memory_warmup_steps=args.memory_warmup,
    )

    # Create trainer
    print("\nCreating trainer...")
    trainer = RAGTrainer(
        model=model,
        config=train_config,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...")
    trainer.train(train_loader, eval_loader)

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
