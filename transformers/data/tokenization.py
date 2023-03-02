"""Defines a range of tokenization functions and wrappers for different systems."""

from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from glob import glob
from typing import List, Generator, Union, Optional
from typing_extensions import Self

import torch
import tiktoken


class Tokenizer(ABC):
    """Tokenizer BaseClass"""

    @abstractproperty
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary"""
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encodes a string into a list of integers"""
        raise NotImplementedError

    def encode_tensor(self, text: str) -> torch.Tensor:
        """Encodes a string into a tensor of integers"""
        return torch.tensor(self.encode(text))

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decodes a list of integers into a string"""
        raise NotImplementedError

    def decode_tensor(self, tokens: torch.Tensor) -> str:
        """Decodes a tensor of integers into a string"""
        # assert dim
        encoded = tokens.squeeze()
        if encoded.dim() == 2:
            encoded = encoded.argmax(dim=1)
        if encoded.dim() > 1:
            raise ValueError(f"Expected 1D or 2D tensor, got {encoded.dim()}D tensor")

        # run sub decoding
        return self.decode(tokens.tolist())

    @abstractclassmethod
    def create_from_generator(cls, generator: Generator[str, None, None]) -> Self:
        """Create a tokenizer from a generator of strings"""
        raise NotImplementedError

    @classmethod
    def create_from_string(cls, string: str) -> Self:
        """Loads a string as a generator and creates a tokenizer from it"""
        return cls.create_from_generator([string])

    @classmethod
    def create_from_files(cls, files: List[str]) -> Self:
        """Loads files or folders as a generator and creates a tokenizer from it"""
        # create a generator that iterates over all files
        generator = (line for file in files for line in open(file, "r"))
        return cls.create_from_generator(generator)

    @classmethod
    def create_from_folder(
        cls, folder: str, extension: Optional[Union[List[str], str]] = None
    ) -> Self:
        """Loads files or folders as a generator and creates a tokenizer from it"""
        if extension is None:
            extension = ["*"]
        elif isinstance(extension, str):
            extension = [extension]

        files = []
        for ext in extension:
            files += glob(folder + f"/*.{ext}")
        return cls.create_from_files(files)


class CharsetTokenizer(Tokenizer):
    """Simple tokenizer that just indexes all chars."""

    def __init__(self, chars: List[str]) -> None:
        """Initializes the tokenizer with a list of chars"""
        self.chars = sorted(list(chars))

        # convert the data
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> List[int]:
        return [self.char2idx[char] for char in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join([self.idx2char[token] for token in tokens])

    @classmethod
    def create_from_generator(cls, generator: Generator[str, None, None]) -> Self:
        """Create a tokenizer from a generator of strings"""
        # create set of chars from generator
        chars = set()
        for line in generator:
            chars = chars.union(set(line))

        # sort the data
        chars = list(chars)
        return cls(chars)


class GPT2Tokenizer(Tokenizer):
    """Creates a Tokenizer for GPT2 based on TikToken implementation"""

    def __init__(self):
        self.enc: tiktoken.Encoding = tiktoken.get_encoding("gpt2")

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)

    @classmethod
    def create_from_generator(cls, generator: Generator[str, None, None]) -> Self:
        """Create a tokenizer from a generator of strings"""
        return cls()
