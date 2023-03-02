"""Defines a Range of base Models that provide inference and generation."""

from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn

from data import Tokenizer


class BaseModel(nn.Module):
    # TODO: define a bunch of save and load functions
    pass


class LanguageModel(BaseModel, ABC):
    def __init__(self, tokenizer: Tokenizer, block_size):
        super().__init__()
        self.tok = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.block_size = block_size

    @property
    def device(self):
        return next(self.parameters()).device

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def generate(self, max_len: int = 100) -> List[int]:
        """Generates a language output from the model"""
        # set model to eval mode
        self.eval()

        # setup output
        out = []
        for i in range(max_len):
            # setup tensor for generation
            prev = torch.tensor(
                (([0] * self.block_size) + out)[-self.block_size :],
                device=self.device,
                dtype=torch.long,
            ).view(1, -1)
            # compute logits
            logits = self.forward(prev)
            # get the last logits and convert to prob
            logits = logits[:, -1]
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            sample = torch.multinomial(probs, 1)
            # append to output
            out.append(sample.detach().cpu().item())

        return out

    def generate_str(self, max_len: int = 100) -> str:
        """Generates a language output from the model"""
        self.tok.decode(self.generate(max_len=max_len))


class SeqToSeqModel(BaseModel, ABC):
    # TODO: implement that later on
    pass
