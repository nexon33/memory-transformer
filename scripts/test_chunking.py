#!/usr/bin/env python3
"""Test document chunking with overlap."""

import sys
sys.path.insert(0, ".")

from model.chunker import TextChunker, SentenceChunker, chunk_text


def test_basic_chunking():
    """Test basic text chunking with overlap."""
    print("="*60)
    print("TEST: Basic Text Chunking")
    print("="*60)

    # Create a document with numbered sentences for easy tracking
    sentences = [f"This is sentence number {i}." for i in range(20)]
    document = " ".join(sentences)

    print(f"Document: {len(document)} chars, {len(document.split())} words")
    print(f"First 100 chars: {document[:100]}...")

    # Chunk with small size for demo
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)  # 20 words, 5 overlap
    chunks = chunker.chunk_text(document, doc_id="test_doc")

    print(f"\nChunks created: {len(chunks)}")
    print(f"Chunk size: 20 words, Overlap: 5 words")
    print()

    for i, chunk in enumerate(chunks):
        words = chunk.text.split()
        print(f"Chunk {i}: {len(words)} words")
        print(f"  Start: '{' '.join(words[:4])}...'")
        print(f"  End:   '...{' '.join(words[-4:])}'")

        # Show overlap with next chunk
        if i < len(chunks) - 1:
            next_chunk = chunks[i + 1]
            next_words = next_chunk.text.split()
            overlap_words = set(words[-5:]) & set(next_words[:5])
            print(f"  Overlap with next: {len(overlap_words)} words")
        print()


def test_sentence_chunking():
    """Test sentence-based chunking."""
    print("="*60)
    print("TEST: Sentence-Based Chunking")
    print("="*60)

    document = """
    Machine learning is a subset of artificial intelligence. It enables computers to learn from data.
    Deep learning is a type of machine learning. It uses neural networks with many layers.
    Natural language processing deals with text and speech. It powers chatbots and translators.
    Computer vision enables machines to see. It's used in self-driving cars and medical imaging.
    Reinforcement learning learns through trial and error. It's used in game-playing AI and robotics.
    """

    chunker = SentenceChunker(max_chunk_size=30, min_chunk_size=15, overlap_sentences=1)
    chunks = chunker.chunk_text(document.strip(), doc_id="ml_doc")

    print(f"Document: {len(document.split())} words, {len(document.split('.'))-1} sentences")
    print(f"Chunks created: {len(chunks)}")
    print()

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i} ({chunk.metadata.get('sentence_count', '?')} sentences):")
        print(f"  '{chunk.text[:80]}...'")
        print()


def test_overlap_verification():
    """Verify overlap is working correctly."""
    print("="*60)
    print("TEST: Overlap Verification")
    print("="*60)

    # Create document with unique markers
    words = [f"word{i}" for i in range(100)]
    document = " ".join(words)

    chunker = TextChunker(chunk_size=30, chunk_overlap=10)
    chunks = chunker.chunk_text(document)

    print(f"Document: 100 unique words")
    print(f"Chunk size: 30, Overlap: 10")
    print(f"Chunks: {len(chunks)}")
    print()

    for i in range(len(chunks) - 1):
        curr_words = set(chunks[i].text.split())
        next_words = set(chunks[i + 1].text.split())
        overlap = curr_words & next_words

        print(f"Chunks {i} & {i+1}: {len(overlap)} overlapping words")
        if len(overlap) >= 8:  # Should be ~10
            print(f"  ✓ Good overlap")
        else:
            print(f"  ✗ Insufficient overlap!")

    print()


def test_with_phi2():
    """Test chunking with Phi-2 model."""
    print("="*60)
    print("TEST: Document Storage with Phi-2")
    print("="*60)

    from scripts.test_phi2_memory import Phi2WithMemory

    model = Phi2WithMemory(top_k=5)
    model.chunk_size = 128  # Smaller chunks for demo
    model.chunk_overlap = 32

    # Create a document with facts spread throughout
    document = """
    The capital of France is Paris. Paris is known for the Eiffel Tower.
    The population of Paris is about 2 million people in the city proper.

    The capital of Japan is Tokyo. Tokyo is the most populous metropolitan area.
    Mount Fuji is Japan's highest mountain at 3,776 meters.

    The capital of Australia is Canberra. Many people think it's Sydney but that's wrong.
    The Great Barrier Reef is located off the coast of Queensland, Australia.

    The speed of light is 299,792,458 meters per second. Einstein's famous equation is E=mc².
    The theory of relativity changed our understanding of space and time.

    Python was created by Guido van Rossum. It was first released in 1991.
    Python is named after Monty Python, not the snake.
    """

    print(f"\nStoring document ({len(document)} chars)...")
    num_chunks = model.store_document(document, doc_id="facts")
    print(f"Created {num_chunks} chunks")
    print(f"Memory size: {model.memory.size} entries")

    # Test retrieval
    queries = [
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the speed of light?",
        "Who created Python?",
        "Where is the Great Barrier Reef?",
    ]

    print("\nRetrieval test:")
    for query in queries:
        retrieved = model.retrieve(query, top_k=2)
        print(f"\nQ: {query}")
        for chunk, score in retrieved:
            # Truncate long chunks
            display = chunk[:80] + "..." if len(chunk) > 80 else chunk
            print(f"  [{score:.3f}] {display}")


if __name__ == "__main__":
    test_basic_chunking()
    print("\n")
    test_sentence_chunking()
    print("\n")
    test_overlap_verification()
    print("\n")

    # Only run Phi-2 test if requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--phi2":
        test_with_phi2()
    else:
        print("Run with --phi2 to test with Phi-2 model")
