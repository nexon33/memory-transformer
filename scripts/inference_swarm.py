#!/usr/bin/env python3
"""Swarm inference for Memory Transformer.

Runs multiple model instances in parallel with different strategies,
then combines results for more robust outputs.
"""

import os
import sys
import argparse
import torch
import torch.multiprocessing as mp
from pathlib import Path
from queue import Empty
import time
from collections import Counter
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from memory_transformer.model.config import MemoryTransformerConfig
from memory_transformer.model.memory_transformer import MemoryTransformer
from memory_transformer.data.tokenizer import TaskTokenizer
from memory_transformer.data.data_utils import create_tokenizer


class InferenceWorker:
    """Single inference worker with a specific strategy."""

    STRATEGIES = [
        "greedy",           # Always pick highest prob token
        "sample_low",       # Low temperature sampling
        "sample_high",      # High temperature sampling
        "top_k_5",          # Top-5 sampling
        "top_k_10",         # Top-10 sampling
        "nucleus_09",       # Nucleus sampling p=0.9
        "nucleus_095",      # Nucleus sampling p=0.95
        "beam_3",           # Beam search width 3
    ]

    def __init__(
        self,
        worker_id: int,
        model: MemoryTransformer,
        tokenizer: TaskTokenizer,
        strategy: str = "greedy",
    ):
        self.worker_id = worker_id
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
    ) -> Dict:
        """Generate text with this worker's strategy."""
        self.model.eval()

        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        # Strategy-specific generation
        if self.strategy == "greedy":
            output, memory = self._generate_greedy(input_tensor, max_tokens)
        elif self.strategy == "sample_low":
            output, memory = self._generate_sample(input_tensor, max_tokens, temp=0.3)
        elif self.strategy == "sample_high":
            output, memory = self._generate_sample(input_tensor, max_tokens, temp=1.2)
        elif self.strategy == "top_k_5":
            output, memory = self._generate_top_k(input_tensor, max_tokens, k=5)
        elif self.strategy == "top_k_10":
            output, memory = self._generate_top_k(input_tensor, max_tokens, k=10)
        elif self.strategy == "nucleus_09":
            output, memory = self._generate_nucleus(input_tensor, max_tokens, p=0.9)
        elif self.strategy == "nucleus_095":
            output, memory = self._generate_nucleus(input_tensor, max_tokens, p=0.95)
        else:
            output, memory = self._generate_greedy(input_tensor, max_tokens)

        # Decode output
        generated_text = self.tokenizer.decode(output[0, len(input_ids):].tolist())
        full_text = self.tokenizer.decode(output[0].tolist())

        # Get confidence (average max prob)
        logits = self.model(input_tensor)["logits"]
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values.mean().item()

        return {
            "worker_id": self.worker_id,
            "strategy": self.strategy,
            "generated": generated_text,
            "full": full_text,
            "confidence": confidence,
            "memory_stats": self.model.get_memory_statistics(memory) if memory else {},
        }

    def _generate_greedy(self, input_ids, max_tokens):
        return self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.0,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def _generate_sample(self, input_ids, max_tokens, temp):
        return self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temp,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def _generate_top_k(self, input_ids, max_tokens, k):
        return self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_k=k,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def _generate_nucleus(self, input_ids, max_tokens, p):
        return self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=p,
            eos_token_id=self.tokenizer.eos_token_id,
        )


class InferenceSwarm:
    """Manages multiple inference workers."""

    def __init__(
        self,
        checkpoint_path: str,
        num_workers: int = 4,
        strategies: Optional[List[str]] = None,
    ):
        self.num_workers = num_workers

        # Load model config
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "config" in checkpoint:
            config_dict = checkpoint["config"]
            self.config = MemoryTransformerConfig(**config_dict)
        elif "model_config" in checkpoint:
            self.config = MemoryTransformerConfig(**checkpoint["model_config"])
        else:
            self.config = MemoryTransformerConfig()

        # Create tokenizer
        self.tokenizer = create_tokenizer(vocab_size=self.config.vocab_size)

        # Choose strategies
        if strategies is None:
            strategies = InferenceWorker.STRATEGIES[:num_workers]
        self.strategies = strategies[:num_workers]

        # Create workers (each with own model instance for true parallelism)
        self.workers = []
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")

        for i, strategy in enumerate(self.strategies):
            model = MemoryTransformer(self.config)
            if state_dict:
                model.load_state_dict(state_dict)
            model.eval()

            worker = InferenceWorker(
                worker_id=i,
                model=model,
                tokenizer=self.tokenizer,
                strategy=strategy,
            )
            self.workers.append(worker)

        print(f"Initialized swarm with {len(self.workers)} workers")
        print(f"Strategies: {self.strategies}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        combination_method: str = "vote",
    ) -> Dict:
        """Generate using all workers and combine results.

        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate
            combination_method: How to combine results
                - "vote": Majority vote on output
                - "confidence": Pick highest confidence
                - "all": Return all outputs

        Returns:
            Combined result dictionary
        """
        start_time = time.time()

        # Run all workers
        results = []
        for worker in self.workers:
            result = worker.generate(prompt, max_tokens)
            results.append(result)

        elapsed = time.time() - start_time

        # Combine results
        if combination_method == "vote":
            # Majority vote on generated text
            outputs = [r["generated"].strip() for r in results]
            counter = Counter(outputs)
            winner, count = counter.most_common(1)[0]
            consensus = count / len(outputs)

            combined = {
                "output": winner,
                "consensus": consensus,
                "vote_count": count,
                "total_votes": len(outputs),
                "all_outputs": outputs,
            }

        elif combination_method == "confidence":
            # Pick highest confidence output
            best = max(results, key=lambda x: x["confidence"])
            combined = {
                "output": best["generated"].strip(),
                "confidence": best["confidence"],
                "strategy": best["strategy"],
            }

        else:  # "all"
            combined = {
                "outputs": [r["generated"].strip() for r in results],
                "confidences": [r["confidence"] for r in results],
                "strategies": [r["strategy"] for r in results],
            }

        combined["elapsed_ms"] = elapsed * 1000
        combined["workers_used"] = len(self.workers)
        combined["prompt"] = prompt

        return combined


def main():
    parser = argparse.ArgumentParser(description="Swarm Inference for Memory Transformer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens")
    parser.add_argument("--method", type=str, default="vote",
                       choices=["vote", "confidence", "all"],
                       help="Combination method")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    # Create swarm
    swarm = InferenceSwarm(
        checkpoint_path=args.checkpoint,
        num_workers=args.num_workers,
    )

    if args.interactive:
        print("\nSwarm Inference - Interactive Mode")
        print("Type 'quit' to exit\n")

        while True:
            try:
                prompt = input("Prompt> ").strip()
                if prompt.lower() == "quit":
                    break
                if not prompt:
                    continue

                result = swarm.generate(prompt, args.max_tokens, args.method)

                print(f"\nResult ({args.method} method):")
                if args.method == "vote":
                    print(f"  Output: {result['output']}")
                    print(f"  Consensus: {result['consensus']:.0%} ({result['vote_count']}/{result['total_votes']})")
                    if result['consensus'] < 1.0:
                        print(f"  All outputs: {result['all_outputs']}")
                elif args.method == "confidence":
                    print(f"  Output: {result['output']}")
                    print(f"  Confidence: {result['confidence']:.2%}")
                    print(f"  Strategy: {result['strategy']}")
                else:
                    for i, (out, conf, strat) in enumerate(zip(
                        result['outputs'], result['confidences'], result['strategies']
                    )):
                        print(f"  [{strat}] ({conf:.2%}): {out}")

                print(f"  Time: {result['elapsed_ms']:.1f}ms")
                print()

            except KeyboardInterrupt:
                break

    elif args.prompt:
        result = swarm.generate(args.prompt, args.max_tokens, args.method)

        print(f"\nPrompt: {args.prompt}")
        print(f"Method: {args.method}")
        print(f"\nResult:")

        if args.method == "vote":
            print(f"  Output: {result['output']}")
            print(f"  Consensus: {result['consensus']:.0%}")
        elif args.method == "confidence":
            print(f"  Output: {result['output']}")
            print(f"  Confidence: {result['confidence']:.2%}")
        else:
            for out, conf, strat in zip(
                result['outputs'], result['confidences'], result['strategies']
            ):
                print(f"  [{strat}] ({conf:.2%}): {out}")

        print(f"\nTime: {result['elapsed_ms']:.1f}ms")

    else:
        print("Use --prompt or --interactive")


if __name__ == "__main__":
    main()
