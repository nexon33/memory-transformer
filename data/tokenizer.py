"""Simple tokenizer for memory transformer training."""

import json
import os
import re
from typing import List, Dict, Optional, Union
from collections import Counter


class SimpleTokenizer:
    """Character-level tokenizer with special tokens.

    For our memory learning tasks, we use a simple tokenizer that
    operates at character level with special task tokens.
    """

    # Special tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    SEP_TOKEN = "<sep>"

    # Task-specific tokens
    COPY_TOKEN = "<copy>"
    STORE_TOKEN = "<store>"
    RECALL_TOKEN = "<recall>"
    EQUALS_TOKEN = "<eq>"
    RESULT_TOKEN = "<result>"

    SPECIAL_TOKENS = [
        PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN,
        COPY_TOKEN, STORE_TOKEN, RECALL_TOKEN, EQUALS_TOKEN, RESULT_TOKEN,
    ]

    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Build initial vocabulary
        self._build_base_vocab()

    def _build_base_vocab(self):
        """Build base vocabulary with special tokens and common characters."""
        idx = 0

        # Add special tokens first
        for token in self.SPECIAL_TOKENS:
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        # Add digits
        for digit in "0123456789":
            self.token_to_id[digit] = idx
            self.id_to_token[idx] = digit
            idx += 1

        # Add lowercase letters
        for char in "abcdefghijklmnopqrstuvwxyz":
            self.token_to_id[char] = idx
            self.id_to_token[idx] = char
            idx += 1

        # Add uppercase letters
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            self.token_to_id[char] = idx
            self.id_to_token[idx] = char
            idx += 1

        # Add common punctuation and symbols
        for char in " .,!?+-*/=()[]{}:;'\"<>@#$%^&_\\|~`\n\t":
            if char not in self.token_to_id:
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char
                idx += 1

        self._next_id = idx

    def add_token(self, token: str) -> int:
        """Add a new token to vocabulary."""
        if token in self.token_to_id:
            return self.token_to_id[token]

        if self._next_id >= self.vocab_size:
            return self.token_to_id[self.UNK_TOKEN]

        self.token_to_id[token] = self._next_id
        self.id_to_token[self._next_id] = token
        self._next_id += 1
        return self._next_id - 1

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text
            add_bos: Add beginning of sequence token
            add_eos: Add end of sequence token

        Returns:
            List of token IDs
        """
        tokens = []

        if add_bos:
            tokens.append(self.token_to_id[self.BOS_TOKEN])

        # Check for special tokens in text
        special_pattern = "|".join(re.escape(t) for t in self.SPECIAL_TOKENS)
        parts = re.split(f"({special_pattern})", text)

        for part in parts:
            if part in self.token_to_id:
                tokens.append(self.token_to_id[part])
            else:
                # Character-level tokenization
                for char in part:
                    if char in self.token_to_id:
                        tokens.append(self.token_to_id[char])
                    else:
                        tokens.append(self.token_to_id[self.UNK_TOKEN])

        if add_eos:
            tokens.append(self.token_to_id[self.EOS_TOKEN])

        return tokens

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output

        Returns:
            Decoded text
        """
        tokens = []
        for tid in token_ids:
            if tid in self.id_to_token:
                token = self.id_to_token[tid]
                if skip_special_tokens and token in self.SPECIAL_TOKENS:
                    continue
                tokens.append(token)
            else:
                if not skip_special_tokens:
                    tokens.append(self.UNK_TOKEN)

        return "".join(tokens)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.UNK_TOKEN]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.BOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]

    @property
    def sep_token_id(self) -> int:
        return self.token_to_id[self.SEP_TOKEN]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def save(self, path: str):
        """Save tokenizer to file."""
        data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "next_id": self._next_id,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        """Load tokenizer from file."""
        with open(path, "r") as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(k): v for k, v in data.get("id_to_token", {}).items()}
        # Rebuild id_to_token if not present
        if not tokenizer.id_to_token:
            tokenizer.id_to_token = {v: k for k, v in tokenizer.token_to_id.items()}
        tokenizer._next_id = data.get("next_id", len(tokenizer.token_to_id))

        return tokenizer


class TaskTokenizer(SimpleTokenizer):
    """Extended tokenizer with task-specific formatting."""

    def encode_copy_task(
        self,
        content: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> tuple:
        """Encode a copy task.

        Format: <bos><copy>content<sep>content<eos>

        Returns:
            Tuple of (input_ids, labels) where labels has -100 for input tokens
        """
        # Build input: <copy>content<sep>
        input_text = f"{self.COPY_TOKEN}{content}{self.SEP_TOKEN}"
        # Build target: content
        target_text = content

        input_ids = self.encode(input_text, add_bos=add_bos, add_eos=False)
        target_ids = self.encode(target_text, add_bos=False, add_eos=add_eos)

        # Full sequence
        full_ids = input_ids + target_ids

        # Labels: -100 for input portion (don't compute loss)
        labels = [-100] * len(input_ids) + target_ids

        return full_ids, labels

    def encode_recall_task(
        self,
        key: str,
        value: str,
        distractor: str = "",
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> tuple:
        """Encode a store/recall task.

        Format: <bos><store>key=value<sep>distractor<recall>key<eq>value<eos>

        Returns:
            Tuple of (input_ids, labels)
        """
        # Build input: <store>key=value<sep>distractor<recall>key<eq>
        input_text = f"{self.STORE_TOKEN}{key}={value}{self.SEP_TOKEN}{distractor}{self.RECALL_TOKEN}{key}{self.EQUALS_TOKEN}"
        # Build target: value
        target_text = value

        input_ids = self.encode(input_text, add_bos=add_bos, add_eos=False)
        target_ids = self.encode(target_text, add_bos=False, add_eos=add_eos)

        full_ids = input_ids + target_ids
        labels = [-100] * len(input_ids) + target_ids

        return full_ids, labels

    def encode_arithmetic_task(
        self,
        expression: str,
        result: str,
        intermediate_steps: Optional[List[tuple]] = None,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> tuple:
        """Encode an arithmetic task with optional intermediate storage.

        Format: <bos>expression<eq><store>A=intermediate<sep>...<result>final<eos>

        Args:
            expression: Math expression (e.g., "2+3*4")
            result: Final result (e.g., "14")
            intermediate_steps: List of (var_name, value) for intermediate results

        Returns:
            Tuple of (input_ids, labels)
        """
        # Build input with intermediate storage
        input_parts = [expression, self.EQUALS_TOKEN]

        if intermediate_steps:
            for var_name, value in intermediate_steps:
                input_parts.extend([self.STORE_TOKEN, f"{var_name}={value}", self.SEP_TOKEN])

        input_parts.append(self.RESULT_TOKEN)
        input_text = "".join(input_parts)

        input_ids = self.encode(input_text, add_bos=add_bos, add_eos=False)
        target_ids = self.encode(result, add_bos=False, add_eos=add_eos)

        full_ids = input_ids + target_ids
        labels = [-100] * len(input_ids) + target_ids

        return full_ids, labels
