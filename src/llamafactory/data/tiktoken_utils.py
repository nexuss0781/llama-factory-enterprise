# Copyright 2025 Nexuss AI Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tiktoken tokenizer wrapper for LLaMA-Factory."""

import json
import os
from typing import Any, Dict, List, Optional, Union

import tiktoken
from transformers import PreTrainedTokenizer


class TiktokenTokenizer(PreTrainedTokenizer):
    """A wrapper around tiktoken to make it compatible with HuggingFace transformers."""

    vocab_files_names = {"vocab_file": "tokenizer.json"}
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        encoding_name: str = "cl100k_base",
        max_length: Optional[int] = None,
        pad_token: str = "<|endoftext|>",
        bos_token: str = "<|endoftext|>",
        eos_token: str = "<|endoftext|>",
        unk_token: Optional[str] = None,
        padding_side: str = "right",
        **kwargs: Any,
    ) -> None:
        """Initialize the TiktokenTokenizer.

        Args:
            vocab_file: Path to a custom tokenizer.json file (optional).
            encoding_name: Name of the tiktoken encoding to use (e.g., 'cl100k_base', 'p50k_base', 'r50k_base').
            max_length: Maximum sequence length.
            pad_token: Padding token string.
            bos_token: Beginning of sequence token string.
            eos_token: End of sequence token string.
            unk_token: Unknown token string.
            padding_side: Side for padding ('left' or 'right').
            **kwargs: Additional keyword arguments passed to PreTrainedTokenizer.
        """
        self.encoding_name = encoding_name
        self.max_length = max_length if max_length is not None else 8192

        # Load tiktoken encoding
        if vocab_file is not None and os.path.exists(vocab_file):
            # Load from custom vocab file
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
            
            # Handle different vocab file formats
            if "vocab" in vocab_data:
                vocab = vocab_data["vocab"]
            elif isinstance(vocab_data, dict):
                vocab = vocab_data
            else:
                raise ValueError(f"Invalid vocab file format: {vocab_file}")
            
            # Create mergeable_ranks for tiktoken
            mergeable_ranks = {k.encode(): v for k, v in vocab.items()}
            self.encoder = tiktoken.Encoding(
                name="custom_nexuss",
                pat_str=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{BLANK}++|(?i:\p{L}++)(?:['\-]?+(?:[dmt]|ll|ve|re))|[\p{BLANK}]++|\p{N}++|[^\p{L}\p{N}]++""",
                mergeable_ranks=mergeable_ranks,
                special_tokens={bos_token: 0, eos_token: 1, pad_token: 2, unk_token: 3} if unk_token else {bos_token: 0, eos_token: 1, pad_token: 2},
            )
        else:
            # Use built-in tiktoken encoding
            try:
                self.encoder = tiktoken.get_encoding(encoding_name)
            except Exception:
                # Fallback to cl100k_base if specified encoding not found
                self.encoder = tiktoken.get_encoding("cl100k_base")

        # Build vocab from encoder
        vocab = {token: idx for token, idx in self.encoder._mergeable_ranks.items()}
        
        # Add special tokens
        special_tokens = {}
        if bos_token:
            special_tokens[bos_token] = len(vocab)
        if eos_token:
            special_tokens[eos_token] = len(vocab) + 1
        if pad_token:
            special_tokens[pad_token] = len(vocab) + 2
        if unk_token:
            special_tokens[unk_token] = len(vocab) + 3

        # Initialize parent class
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            padding_side=padding_side,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self.encoder._mergeable_ranks)

    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary as a dictionary."""
        return dict(self.encoder._mergeable_ranks)

    def _tokenize(self, text: str, **kwargs: Any) -> List[str]:
        """Tokenize text into tokens (not used directly, but required for interface)."""
        # Tiktoken works at byte level, we return individual bytes as tokens
        encoded = self.encoder.encode(text)
        # Convert IDs back to string representation for compatibility
        return [str(id) for id in encoded]

    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token string to its ID."""
        try:
            return int(token)
        except ValueError:
            # If token is not an ID string, try to find it in vocab
            return self.encoder._mergeable_ranks.get(token.encode(), 0)

    def _convert_id_to_token(self, index: int) -> str:
        """Convert an ID to its token string."""
        return str(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a list of token IDs (as strings) back to text."""
        try:
            token_ids = [int(t) for t in tokens]
            return self.encoder.decode(token_ids)
        except (ValueError, AttributeError):
            return " ".join(tokens)

    def prepare_for_tokenization(self, text: str, **kwargs: Any) -> tuple[str, dict]:
        """Prepare text for tokenization."""
        return text, kwargs

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Build model inputs from a sequence or pair of sequences by appending special tokens."""
        bos = [self.bos_token_id] if self.bos_token_id is not None else []
        eos = [self.eos_token_id] if self.eos_token_id is not None else []
        
        if token_ids_1 is None:
            return bos + token_ids_0 + eos
        return bos + token_ids_0 + eos + bos + token_ids_1 + eos

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Create token type IDs from sequences (not used for causal LM)."""
        bos = [self.bos_token_id] if self.bos_token_id is not None else []
        eos = [self.eos_token_id] if self.eos_token_id is not None else []
        
        if token_ids_1 is None:
            return len(bos + token_ids_0 + eos) * [0]
        return len(bos + token_ids_0 + eos) * [0] + len(bos + token_ids_1 + eos) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        """Save the tokenizer vocabulary to a directory."""
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "tokenizer.json")
        
        vocab_dict = {
            "vocab": {k.decode() if isinstance(k, bytes) else k: v for k, v in self.encoder._mergeable_ranks.items()},
            "type": "tiktoken",
            "encoding_name": self.encoding_name,
        }
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, indent=2, ensure_ascii=False)
        
        return (vocab_file,)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any) -> "TiktokenTokenizer":
        """Load tokenizer from a path or model name."""
        # Check if there's a custom tokenizer.json file
        vocab_file = None
        if os.path.isdir(pretrained_model_name_or_path):
            potential_vocab = os.path.join(pretrained_model_name_or_path, "tokenizer.json")
            if os.path.exists(potential_vocab):
                vocab_file = potential_vocab
        
        # Get encoding name from kwargs or use default
        encoding_name = kwargs.pop("encoding_name", "cl100k_base")
        
        return cls(vocab_file=vocab_file, encoding_name=encoding_name, *args, **kwargs)
