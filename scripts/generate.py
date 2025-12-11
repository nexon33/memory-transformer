#!/usr/bin/env python3
"""Text generation script for Memory Transformer."""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from memory_transformer.model.config import MemoryTransformerConfig
from memory_transformer.model.memory_transformer import MemoryTransformer
from memory_transformer.data.tokenizer import TaskTokenizer
from memory_transformer.data.data_utils import create_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate with Memory Transformer")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--device", type=str, default="auto", help="Device")

    return parser.parse_args()


def generate_once(
    model,
    tokenizer,
    device,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int = None,
    top_p: float = None,
):
    """Generate text from a prompt."""
    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    print(f"\nPrompt: {prompt}")
    print(f"Input tokens: {len(input_ids)}")

    # Generate
    with torch.no_grad():
        output_ids, memory_state = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    output_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=False)
    generated_text = tokenizer.decode(output_ids[0, len(input_ids):].tolist())

    print(f"\nGenerated: {generated_text}")
    print(f"Full output: {output_text}")

    # Memory stats
    if memory_state:
        stats = model.get_memory_statistics(memory_state)
        print(f"\nMemory Statistics:")
        print(f"  Valid slots: {stats['num_valid_slots']:.1f}")
        print(f"  Total writes: {stats['total_writes']:.1f}")
        print(f"  Usage mean: {stats['usage_mean']:.4f}")

    return generated_text


def interactive_mode(model, tokenizer, device, args):
    """Run interactive generation."""
    print("\n" + "=" * 60)
    print("Interactive Mode - Memory Transformer")
    print("=" * 60)
    print("Enter prompts to generate text. Type 'quit' to exit.")
    print("Special prompts:")
    print("  <copy>text<sep>     - Copy task")
    print("  <store>k=v<sep>...<recall>k<eq> - Recall task")
    print("  expr<eq><store>...<result> - Arithmetic task")
    print()

    while True:
        try:
            prompt = input("\nPrompt> ").strip()
            if prompt.lower() == "quit":
                break
            if not prompt:
                continue

            generate_once(
                model,
                tokenizer,
                device,
                prompt,
                args.max_tokens,
                args.temperature,
                args.top_k,
                args.top_p,
            )

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nGoodbye!")


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Reconstruct model
    model_config_dict = checkpoint["model_config"]
    model_config = MemoryTransformerConfig(**model_config_dict)

    model = MemoryTransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Create tokenizer
    tokenizer = create_tokenizer(vocab_size=model_config.vocab_size)

    if args.interactive:
        interactive_mode(model, tokenizer, device, args)
    elif args.prompt:
        generate_once(
            model,
            tokenizer,
            device,
            args.prompt,
            args.max_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
        )
    else:
        # Demo with example prompts
        print("\nRunning demo with example prompts...")

        examples = [
            # Copy task
            "<copy>hello123<sep>",
            # Recall task
            "<store>ab=1234<sep>xxxxx<recall>ab<eq>",
            # Arithmetic
            "5+3<eq><store>A=8<sep><result>",
        ]

        for prompt in examples:
            print("\n" + "-" * 40)
            generate_once(
                model,
                tokenizer,
                device,
                prompt,
                args.max_tokens,
                args.temperature,
                args.top_k,
                args.top_p,
            )


if __name__ == "__main__":
    main()
