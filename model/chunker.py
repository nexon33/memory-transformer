"""Text chunking utilities for RAG with overlap."""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass


@dataclass
class Chunk:
    """A text chunk with metadata."""
    text: str
    tokens: List[int]
    start_idx: int  # Character start in original
    end_idx: int    # Character end in original
    chunk_idx: int  # Chunk number
    doc_id: str     # Source document ID
    metadata: Dict[str, Any]


class TextChunker:
    """Chunk text with configurable size and overlap."""

    def __init__(
        self,
        chunk_size: int = 256,      # Tokens per chunk
        chunk_overlap: int = 64,     # Overlapping tokens
        tokenizer: Optional[Callable] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer or self._simple_tokenize

    def _simple_tokenize(self, text: str) -> List[int]:
        """Simple whitespace tokenizer (returns word indices)."""
        words = text.split()
        return list(range(len(words)))

    def _simple_detokenize(self, text: str, start: int, end: int) -> str:
        """Get text span from word indices."""
        words = text.split()
        return " ".join(words[start:end])

    def chunk_text(
        self,
        text: str,
        doc_id: str = "doc",
        metadata: Optional[Dict] = None,
    ) -> List[Chunk]:
        """Split text into overlapping chunks.

        Args:
            text: The text to chunk
            doc_id: Identifier for source document
            metadata: Additional metadata to attach to all chunks

        Returns:
            List of Chunk objects
        """
        metadata = metadata or {}
        words = text.split()
        total_words = len(words)

        if total_words == 0:
            return []

        chunks = []
        stride = self.chunk_size - self.chunk_overlap
        stride = max(1, stride)  # Ensure we make progress

        chunk_idx = 0
        start = 0

        while start < total_words:
            end = min(start + self.chunk_size, total_words)
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            # Calculate character positions
            char_start = len(" ".join(words[:start])) + (1 if start > 0 else 0)
            char_end = char_start + len(chunk_text)

            chunk = Chunk(
                text=chunk_text,
                tokens=list(range(start, end)),  # Word indices
                start_idx=char_start,
                end_idx=char_end,
                chunk_idx=chunk_idx,
                doc_id=doc_id,
                metadata={
                    **metadata,
                    "word_start": start,
                    "word_end": end,
                    "total_words": total_words,
                },
            )
            chunks.append(chunk)

            chunk_idx += 1
            start += stride

            # Don't create tiny final chunks
            if total_words - start < self.chunk_overlap:
                break

        return chunks

    def chunk_with_tokenizer(
        self,
        text: str,
        tokenizer,  # HuggingFace tokenizer
        doc_id: str = "doc",
        metadata: Optional[Dict] = None,
    ) -> List[Chunk]:
        """Chunk using a HuggingFace tokenizer for accurate token counts.

        Args:
            text: The text to chunk
            tokenizer: HuggingFace tokenizer
            doc_id: Identifier for source document
            metadata: Additional metadata

        Returns:
            List of Chunk objects with actual token IDs
        """
        metadata = metadata or {}

        # Tokenize entire text
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        token_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]
        total_tokens = len(token_ids)

        if total_tokens == 0:
            return []

        chunks = []
        stride = self.chunk_size - self.chunk_overlap
        stride = max(1, stride)

        chunk_idx = 0
        start = 0

        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)

            # Get token IDs for this chunk
            chunk_token_ids = token_ids[start:end]

            # Get character span
            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]
            chunk_text = text[char_start:char_end]

            chunk = Chunk(
                text=chunk_text,
                tokens=chunk_token_ids,
                start_idx=char_start,
                end_idx=char_end,
                chunk_idx=chunk_idx,
                doc_id=doc_id,
                metadata={
                    **metadata,
                    "token_start": start,
                    "token_end": end,
                    "total_tokens": total_tokens,
                },
            )
            chunks.append(chunk)

            chunk_idx += 1
            start += stride

            # Don't create tiny final chunks
            if total_tokens - start < self.chunk_overlap:
                break

        return chunks


class SentenceChunker:
    """Chunk by sentences with size limits."""

    def __init__(
        self,
        max_chunk_size: int = 512,   # Max tokens per chunk
        min_chunk_size: int = 100,   # Min tokens before starting new chunk
        overlap_sentences: int = 1,   # Sentences to overlap
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_sentences = overlap_sentences

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting."""
        import re
        # Split on .!? followed by space or end
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(
        self,
        text: str,
        doc_id: str = "doc",
        metadata: Optional[Dict] = None,
    ) -> List[Chunk]:
        """Split text into sentence-based chunks."""
        metadata = metadata or {}
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_len = 0
        chunk_idx = 0
        char_pos = 0

        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence.split())  # Word count as proxy for tokens

            # Check if adding this sentence exceeds max
            if current_len + sentence_len > self.max_chunk_size and current_len >= self.min_chunk_size:
                # Create chunk from current sentences
                chunk_text = " ".join(current_sentences)
                chunk = Chunk(
                    text=chunk_text,
                    tokens=[],
                    start_idx=char_pos,
                    end_idx=char_pos + len(chunk_text),
                    chunk_idx=chunk_idx,
                    doc_id=doc_id,
                    metadata={**metadata, "sentence_count": len(current_sentences)},
                )
                chunks.append(chunk)
                chunk_idx += 1

                # Keep overlap sentences
                overlap = current_sentences[-self.overlap_sentences:] if self.overlap_sentences > 0 else []
                char_pos += len(chunk_text) + 1 - sum(len(s) + 1 for s in overlap)
                current_sentences = overlap
                current_len = sum(len(s.split()) for s in overlap)

            current_sentences.append(sentence)
            current_len += sentence_len

        # Final chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk = Chunk(
                text=chunk_text,
                tokens=[],
                start_idx=char_pos,
                end_idx=char_pos + len(chunk_text),
                chunk_idx=chunk_idx,
                doc_id=doc_id,
                metadata={**metadata, "sentence_count": len(current_sentences)},
            )
            chunks.append(chunk)

        return chunks


# Convenience functions
def chunk_text(
    text: str,
    chunk_size: int = 256,
    overlap: int = 64,
    doc_id: str = "doc",
) -> List[Chunk]:
    """Quick chunk text with defaults."""
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    return chunker.chunk_text(text, doc_id=doc_id)


def chunk_with_tokenizer(
    text: str,
    tokenizer,
    chunk_size: int = 256,
    overlap: int = 64,
    doc_id: str = "doc",
) -> List[Chunk]:
    """Quick chunk with HuggingFace tokenizer."""
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    return chunker.chunk_with_tokenizer(text, tokenizer, doc_id=doc_id)
