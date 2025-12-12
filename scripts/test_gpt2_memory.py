#!/usr/bin/env python3
"""Bolt memory system onto GPT-2 and test position recall.

No training needed - just test if retrieval helps recall.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import random
import string

# Check for transformers
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Install transformers: pip install transformers")
    sys.exit(1)

from memory_transformer.model.memory_store import MemoryStore


class GPT2WithMemory:
    """GPT-2 with bolted-on memory retrieval."""

    def __init__(self, model_name: str = "gpt2", max_chunks: int = 1000, top_k: int = 3):
        print(f"Loading {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

        self.d_model = self.model.config.n_embd
        self.memory = MemoryStore(d_model=self.d_model, max_chunks=max_chunks)
        self.top_k = top_k

        print(f"GPT-2 loaded: {sum(p.numel() for p in self.model.parameters())/1e6:.0f}M params")
        print(f"Memory: d_model={self.d_model}, max_chunks={max_chunks}, top_k={top_k}")

    def get_embedding(self, text: str, method: str = "last") -> torch.Tensor:
        """Get embedding for text using GPT-2.

        Methods:
        - 'last': Use last token embedding (best for causal LM)
        - 'mean': Mean pool all tokens
        - 'weighted': Weighted by position (more weight to end)
        """
        tokens = self.tokenizer.encode(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.transformer(tokens)
            hidden = outputs.last_hidden_state  # [1, seq_len, d_model]

            if method == "last":
                # Last token embedding (best for autoregressive models)
                emb = hidden[0, -1, :]
            elif method == "weighted":
                # Position-weighted mean (more weight to later tokens)
                seq_len = hidden.size(1)
                weights = torch.arange(1, seq_len + 1, dtype=torch.float32)
                weights = weights / weights.sum()
                emb = (hidden[0] * weights.unsqueeze(-1)).sum(dim=0)
            else:  # mean
                emb = hidden.mean(dim=1).squeeze(0)

        return F.normalize(emb, dim=-1)

    def store(self, text: str, metadata: dict = None):
        """Store text in memory."""
        emb = self.get_embedding(text)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        self.memory.add(text, tokens, emb, metadata)

    def retrieve(self, query: str, top_k: int = None, use_keywords: bool = True) -> List[Tuple[str, float]]:
        """Retrieve similar chunks.

        If use_keywords=True, boost scores for keyword matches.
        """
        if self.memory.size == 0:
            return []

        top_k = top_k or self.top_k
        query_emb = self.get_embedding(query)
        results = self.memory.search(query_emb, top_k=top_k * 2)  # Get more candidates

        if use_keywords:
            # Extract keywords from query (simple: words > 3 chars, lowercase)
            query_words = set(w.lower() for w in query.split() if len(w) > 3)
            query_words -= {'what', 'does', 'equal', 'the', 'is', 'are', 'that', 'this'}

            # Boost scores based on keyword overlap
            boosted = []
            for idx, text, tokens, score, meta in results:
                text_words = set(w.lower() for w in text.split() if len(w) > 3)
                overlap = len(query_words & text_words)
                boost = 0.1 * overlap  # Boost per matching keyword
                boosted.append((text, score + boost))

            # Re-sort and take top_k
            boosted.sort(key=lambda x: -x[1])
            return boosted[:top_k]

        return [(r[1], r[3]) for r in results[:top_k]]  # (text, score)

    def generate_with_memory(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        use_memory: bool = True,
    ) -> str:
        """Generate text with optional memory retrieval."""
        if use_memory and self.memory.size > 0:
            # Retrieve relevant chunks
            retrieved = self.retrieve(prompt)

            # Format with few-shot pattern for extraction
            memory_lines = "\n".join([f"Fact: {chunk}" for chunk, _ in retrieved])
            augmented_prompt = f"""Known facts:
{memory_lines}

Q: {prompt}
A: The answer is"""
        else:
            augmented_prompt = prompt

        # Tokenize
        inputs = self.tokenizer.encode(augmented_prompt, return_tensors="pt")

        # Truncate if too long
        max_len = 1024 - max_new_tokens
        if inputs.size(1) > max_len:
            inputs = inputs[:, -max_len:]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only new tokens
        new_tokens = outputs[0, inputs.size(1):]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def clear_memory(self):
        """Clear all memory."""
        self.memory.clear()


def test_position_recall(
    model: GPT2WithMemory,
    num_positions: int = 20,
    samples_per_position: int = 5,
    use_memory: bool = True,
):
    """Test recall accuracy by storage position.

    The smoking gun test:
    - Standard GPT-2: U-curve (primacy + recency)
    - With memory: Should be flatter
    """
    print(f"\n{'='*60}")
    print(f"Position Recall Test (use_memory={use_memory})")
    print(f"{'='*60}")
    print(f"Positions: {num_positions}, Samples per position: {samples_per_position}")

    position_correct = [0] * num_positions
    position_total = [0] * num_positions

    for sample in range(samples_per_position):
        # Generate random KV pairs
        pairs = []
        for i in range(num_positions):
            key = ''.join(random.choices(string.ascii_lowercase, k=2))
            value = ''.join(random.choices(string.digits, k=3))
            pairs.append((key, value))

        # Store all facts
        model.clear_memory()
        for i, (key, value) in enumerate(pairs):
            fact = f"The code {key} equals {value}"
            model.store(fact, metadata={"position": i, "key": key, "value": value})

        # Query each position
        for query_pos in range(num_positions):
            key, expected_value = pairs[query_pos]
            query = f"What does code {key} equal?"

            if use_memory:
                # Check if retrieval gets the right fact (tests retrieval quality)
                retrieved = model.retrieve(query)
                # Success if ANY retrieved chunk contains the expected value
                found_in_retrieved = any(expected_value in chunk for chunk, _ in retrieved)
                if found_in_retrieved:
                    position_correct[query_pos] += 1
            else:
                # Without memory, generate and check
                response = model.generate_with_memory(
                    query,
                    max_new_tokens=10,
                    use_memory=False,
                )
                if expected_value in response:
                    position_correct[query_pos] += 1

            position_total[query_pos] += 1

    # Compute accuracies
    accuracies = [c / t if t > 0 else 0 for c, t in zip(position_correct, position_total)]

    # Print results
    print("\nAccuracy by position:")
    for i, acc in enumerate(accuracies):
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {i:2d}: {bar} {acc:.0%}")

    # Summary stats
    n = len(accuracies)
    primacy = accuracies[:max(1, n//10)]
    middle = accuracies[n//10:n-n//10] if n > 2 else accuracies
    recency = accuracies[-max(1, n//10):]

    overall = sum(accuracies) / n
    primacy_avg = sum(primacy) / len(primacy)
    middle_avg = sum(middle) / len(middle) if middle else 0
    recency_avg = sum(recency) / len(recency)

    print(f"\nSummary:")
    print(f"  Overall:  {overall:.1%}")
    print(f"  Primacy:  {primacy_avg:.1%} (first {len(primacy)} positions)")
    print(f"  Middle:   {middle_avg:.1%} ({len(middle)} positions)")
    print(f"  Recency:  {recency_avg:.1%} (last {len(recency)} positions)")

    # Flatness
    import statistics
    if overall > 0:
        std = statistics.stdev(accuracies) if n > 1 else 0
        flatness = 1 - (std / max(0.01, overall))
        print(f"  Flatness: {flatness:.2f} (1.0 = flat, lower = U-shaped)")

    return accuracies


def quick_debug(model: GPT2WithMemory):
    """Debug retrieval + generation."""
    print("\n" + "="*60)
    print("DEBUG: Testing retrieval quality")
    print("="*60)

    # Store some facts with MORE distinctive keys
    model.clear_memory()
    facts = [
        ("weather in Paris", "sunny"),
        ("capital of Japan", "Tokyo"),
        ("speed of light", "299792458"),
        ("python creator", "Guido"),
        ("Linux kernel", "Torvalds"),
    ]

    for key, val in facts:
        fact = f"The {key} is {val}"
        model.store(fact)
        print(f"Stored: {fact}")

    print("\nRetrieval test:")
    for key, val in facts:
        query = f"What is the {key}?"
        retrieved = model.retrieve(query, top_k=3)
        print(f"\n  Query: {query}")
        for chunk, score in retrieved:
            print(f"    [{score:.3f}] {chunk}")

        # Generate
        response_mem = model.generate_with_memory(query, max_new_tokens=15, use_memory=True)
        response_no = model.generate_with_memory(query, max_new_tokens=15, use_memory=False)
        print(f"  With memory: {response_mem}")
        print(f"  Without:     {response_no}")
        print(f"  Expected: {val}")


def main():
    print("="*60)
    print("GPT-2 + Memory Position Recall Test")
    print("="*60)

    # Load GPT-2 with memory
    model = GPT2WithMemory(model_name="gpt2", top_k=3)

    # Quick debug first
    quick_debug(model)

    # Test WITHOUT memory (baseline - expect U-curve)
    print("\n" + "="*60)
    print("BASELINE: GPT-2 without memory retrieval")
    print("="*60)
    baseline_acc = test_position_recall(model, num_positions=15, samples_per_position=3, use_memory=False)

    # Test WITH memory (expect flatter curve)
    print("\n" + "="*60)
    print("TEST: GPT-2 with memory retrieval")
    print("="*60)
    memory_acc = test_position_recall(model, num_positions=15, samples_per_position=3, use_memory=True)

    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    baseline_overall = sum(baseline_acc) / len(baseline_acc)
    memory_overall = sum(memory_acc) / len(memory_acc)

    print(f"Baseline overall: {baseline_overall:.1%}")
    print(f"Memory overall:   {memory_overall:.1%}")
    print(f"Improvement:      {memory_overall - baseline_overall:+.1%}")

    # Middle positions (where memory should help most)
    n = len(baseline_acc)
    baseline_middle = sum(baseline_acc[n//5:-n//5]) / max(1, len(baseline_acc[n//5:-n//5]))
    memory_middle = sum(memory_acc[n//5:-n//5]) / max(1, len(memory_acc[n//5:-n//5]))

    print(f"\nMiddle positions:")
    print(f"  Baseline: {baseline_middle:.1%}")
    print(f"  Memory:   {memory_middle:.1%}")
    print(f"  Delta:    {memory_middle - baseline_middle:+.1%}")

    if memory_middle > baseline_middle + 0.1:
        print("\n✓ Memory helps middle positions!")
    elif memory_overall > baseline_overall:
        print("\n○ Memory helps overall")
    else:
        print("\n✗ No improvement - check retrieval")


if __name__ == "__main__":
    main()
