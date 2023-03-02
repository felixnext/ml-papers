"""Dataset generator for basic text data (simple snippets of text)"""

import torch
from torch import nn
from torch.utils.data import IterableDataset, Dataset

from typing import List, Tuple, Optional, Type
from typing_extensions import Self

from .tokenization import Tokenizer


class BasicTextDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        tokenizer: Tokenizer,
        block_size: int,
        is_random: bool = True,
        stride: Optional[int] = None,
    ):
        super().__init__()
        self.data = data
        # check the input data
        if data.dim() != 1:
            raise ValueError(f"Expected 1D tensor, got {data.dim()}D tensor")

        # store internal params
        self.block_size = block_size
        self.is_random = is_random
        self.stride = 1 if is_random else stride

        # store the tokenizer for external use
        self.tokenizer = tokenizer

    def __len__(self):
        return (len(self.data) - self.block_size) // (self.stride or self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        # create a random starting point
        # note: we need to have one last char for output
        # note: also need to have block_size difference
        if self.is_random:
            start_idx = torch.randint(0, len(self.data) - self.block_size - 1, (1,))
        else:
            start_idx = idx * (self.stride or self.block_size)

        # retrieve the content length (batchsi)
        content = self.data[start_idx : start_idx + self.block_size + 1]
        # content = torch.stack(
        #    [self.data[i : i + self.block_size + 1] for i in start_idx]
        # )

        # split the content into input and output
        x = content[:-1]
        y = content[1:]
        return x, y

    def collate_fn(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # collate the batch
        x = torch.stack([item[0] for item in batch], dim=0)
        y = torch.stack([item[1] for item in batch], dim=0)
        return x, y

    @classmethod
    def from_file(
        cls,
        path: str,
        block_size: int,
        tokenizer: Type[Tokenizer],
        is_random: bool = True,
        stride: Optional[int] = None,
        encoding: str = "utf-8",
        **kwargs,
    ) -> Self:

        # load the data and convert
        with open(path, "r", encoding=encoding) as f:
            data = f.read()

        # create the dataset
        return cls.from_string(data, block_size, tokenizer, is_random, stride, **kwargs)

    @classmethod
    def from_string(
        cls,
        data: str,
        block_size: int,
        tokenizer: Type[Tokenizer],
        is_random: bool = True,
        stride: Optional[int] = None,
        train_split: Optional[float] = None,
        **kwargs,
    ) -> Tuple[Self, Self]:
        # create the tokenizer
        tok = tokenizer.create_from_string(data)

        # convert the data based on the tokenizer
        data = tok.encode_tensor(data)

        # split the data
        if train_split is not None:
            train_len = int(len(data) * train_split)
            train_data = data[:train_len]
            test_data = data[train_len:]
        else:
            train_data = data
            test_data = None

        # create the dataset
        train = cls(train_data, tok, block_size, is_random, stride, **kwargs)
        test = (
            None
            if test_data is None
            else cls(test_data, tok, block_size, is_random, stride, **kwargs)
        )

        return train, test


# TODO: for larger data define an iteratively loaded dataset
