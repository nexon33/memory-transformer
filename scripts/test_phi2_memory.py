#!/usr/bin/env python3
"""Test Phi-2 with memory retrieval for position recall."""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional
import random
import string

from model.memory_store import MemoryStore
from model.chunker import TextChunker, chunk_with_tokenizer


class Phi2WithMemory:
    """Phi-2 with external memory for RAG-style retrieval."""

    def __init__(self, max_chunks: int = 1000, top_k: int = 3):
        print("Loading Phi-2...")
        self.model_name = "microsoft/phi-2"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model.eval()

        # Get hidden size from model config
        self.d_model = self.model.config.hidden_size
        self.top_k = top_k
        self.max_chunks = max_chunks
        self.memory = MemoryStore(d_model=self.d_model, max_chunks=max_chunks)

        # Chunking settings
        self.chunk_size = 256      # Tokens per chunk
        self.chunk_overlap = 64    # Overlapping tokens

        print(f"Phi-2 loaded: {sum(p.numel() for p in self.model.parameters())/1e9:.1f}B params")
        print(f"Memory: d_model={self.d_model}, max_chunks={max_chunks}, top_k={top_k}")
        print(f"Chunking: size={self.chunk_size}, overlap={self.chunk_overlap}")

    def get_embedding(self, text: str) -> torch.Tensor:
        """Get embedding using last token hidden state."""
        tokens = self.tokenizer.encode(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]  # Last layer
            emb = hidden[0, -1, :].float()  # Last token
        return F.normalize(emb.cpu(), dim=-1)

    def store(self, text: str, metadata: dict = None):
        """Store text in memory (single chunk)."""
        emb = self.get_embedding(text)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        self.memory.add(text, tokens, emb, metadata)

    def store_document(
        self,
        text: str,
        doc_id: str = "doc",
        metadata: Optional[dict] = None,
    ) -> int:
        """Store a document with proper chunking and overlap.

        Args:
            text: Full document text
            doc_id: Identifier for the document
            metadata: Additional metadata

        Returns:
            Number of chunks stored
        """
        chunks = chunk_with_tokenizer(
            text,
            self.tokenizer,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
            doc_id=doc_id,
        )

        for chunk in chunks:
            chunk_meta = {
                **(metadata or {}),
                "doc_id": chunk.doc_id,
                "chunk_idx": chunk.chunk_idx,
                "char_start": chunk.start_idx,
                "char_end": chunk.end_idx,
            }
            emb = self.get_embedding(chunk.text)
            self.memory.add(chunk.text, chunk.tokens, emb, chunk_meta)

        return len(chunks)

    def clear_memory(self):
        """Clear all stored memories."""
        self.memory = MemoryStore(d_model=self.d_model, max_chunks=self.max_chunks)

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """Retrieve with keyword boosting."""
        if self.memory.size == 0:
            return []

        top_k = top_k or self.top_k
        query_emb = self.get_embedding(query)
        results = self.memory.search(query_emb, top_k=top_k * 2)

        # Keyword boost
        query_words = set(w.lower() for w in query.split() if len(w) > 1)
        query_words -= {"what", "does", "equal", "the", "is", "code"}

        boosted = []
        for idx, text, tokens, score, meta in results:
            text_words = set(w.lower() for w in text.split() if len(w) > 1)
            overlap = len(query_words & text_words)
            boost = 0.15 * overlap
            boosted.append((text, score + boost))

        boosted.sort(key=lambda x: -x[1])
        return boosted[:top_k]

    def generate_with_memory(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        use_memory: bool = True,
    ) -> str:
        """Generate with optional memory retrieval."""
        if use_memory and self.memory.size > 0:
            retrieved = self.retrieve(prompt)
            context_lines = "\n".join([f"- {chunk}" for chunk, _ in retrieved])
            augmented = f"""Context:
{context_lines}

Question: {prompt}
Answer:"""
        else:
            augmented = f"Question: {prompt}\nAnswer:"

        inputs = self.tokenizer(augmented, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the answer part
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        return response


def test_position_recall(
    model: Phi2WithMemory,
    num_positions: int = 10,
    samples_per_position: int = 3,
    use_memory: bool = True,
) -> List[float]:
    """Test recall accuracy by storage position."""

    print(f"\n{'='*60}")
    print(f"Position Recall Test (use_memory={use_memory})")
    print(f"{'='*60}")
    print(f"Positions: {num_positions}, Samples per position: {samples_per_position}")

    position_correct = [0] * num_positions
    position_total = [0] * num_positions

    for sample in range(samples_per_position):
        model.clear_memory()

        # Generate random key-value pairs
        pairs = []
        for i in range(num_positions):
            key = "".join(random.choices(string.ascii_lowercase, k=2))
            value = "".join(random.choices(string.digits, k=3))
            pairs.append((key, value))

        # Store all facts
        for i, (key, value) in enumerate(pairs):
            fact = f"The code {key} equals {value}"
            model.store(fact, metadata={"position": i})

        # Query each position
        for query_pos in range(num_positions):
            key, expected_value = pairs[query_pos]
            query = f"What does code {key} equal?"

            if use_memory:
                # Test retrieval: does top-k contain the answer?
                retrieved = model.retrieve(query)
                found = any(expected_value in chunk for chunk, _ in retrieved)
                if found:
                    position_correct[query_pos] += 1
            else:
                # Without memory: test generation alone
                response = model.generate_with_memory(query, max_new_tokens=10, use_memory=False)
                if expected_value in response:
                    position_correct[query_pos] += 1

            position_total[query_pos] += 1

    # Compute accuracies
    accuracies = [c / t if t > 0 else 0 for c, t in zip(position_correct, position_total)]

    # Display results
    print("\nAccuracy by position:")
    for i, acc in enumerate(accuracies):
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {i:3d}: {bar} {acc*100:.0f}%")

    overall = sum(position_correct) / sum(position_total)
    primacy = accuracies[0] if accuracies else 0
    recency = accuracies[-1] if accuracies else 0
    middle = sum(accuracies[1:-1]) / len(accuracies[1:-1]) if len(accuracies) > 2 else 0

    print(f"\nSummary:")
    print(f"  Overall:  {overall*100:.1f}%")
    print(f"  Primacy:  {primacy*100:.1f}% (first position)")
    print(f"  Middle:   {middle*100:.1f}% ({len(accuracies)-2} positions)")
    print(f"  Recency:  {recency*100:.1f}% (last position)")

    return accuracies


def main():
    print("=" * 60)
    print("Phi-2 + Memory Position Recall Test")
    print("=" * 60)

    model = Phi2WithMemory(top_k=3)

    # Quick sanity check
    print("\n--- Quick Test ---")
    model.clear_memory()
    model.store("The code xy equals 123")
    model.store("The code ab equals 456")
    model.store("The code zz equals 789")

    query = "What does code xy equal?"
    retrieved = model.retrieve(query)
    print(f"Query: {query}")
    for chunk, score in retrieved:
        print(f"  [{score:.3f}] {chunk}")

    response = model.generate_with_memory(query, max_new_tokens=15, use_memory=True)
    print(f"Response: {response}")
    print(f"Expected: 123")

    # Baseline test
    print("\n" + "=" * 60)
    print("BASELINE: Phi-2 without memory")
    print("=" * 60)
    baseline_acc = test_position_recall(model, num_positions=10, samples_per_position=3, use_memory=False)

    # Memory test
    print("\n" + "=" * 60)
    print("TEST: Phi-2 with memory retrieval")
    print("=" * 60)
    memory_acc = test_position_recall(model, num_positions=10, samples_per_position=3, use_memory=True)

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    baseline_overall = sum(baseline_acc) / len(baseline_acc)
    memory_overall = sum(memory_acc) / len(memory_acc)

    print(f"Baseline overall: {baseline_overall*100:.1f}%")
    print(f"Memory overall:   {memory_overall*100:.1f}%")
    print(f"Improvement:      +{(memory_overall-baseline_overall)*100:.1f}%")

    # Check flatness
    if len(memory_acc) > 2:
        middle_acc = memory_acc[1:-1]
        edge_acc = [memory_acc[0], memory_acc[-1]]
        middle_avg = sum(middle_acc) / len(middle_acc)
        edge_avg = sum(edge_acc) / len(edge_acc)

        print(f"\nMiddle positions: {middle_avg*100:.1f}%")
        print(f"Edge positions:   {edge_avg*100:.1f}%")

        if middle_avg > edge_avg * 0.9:  # Middle at least 90% of edges
            print("✓ Flat curve - memory working!")
        elif memory_overall > baseline_overall:
            print("○ Memory helps but curve not flat yet")
        else:
            print("✗ No improvement")


if __name__ == "__main__":
    main()
