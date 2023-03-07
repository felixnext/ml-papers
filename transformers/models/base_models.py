"""Defines a Range of base Models that provide inference and generation."""

from abc import ABC, abstractmethod, abstractproperty
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from data import Tokenizer
from layers import MultiAttentionHead


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class LanguageModel(BaseModel, ABC):
    def __init__(self, tokenizer: Tokenizer, block_size):
        super().__init__()
        self.tok = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.block_size = block_size
        self._store_att = False

    @property
    def device(self):
        return next(self.parameters()).device

    @abstractproperty
    def attention_layers(self) -> List[MultiAttentionHead]:
        """Abtract property that returns a list of all attention layers

        Used for attention visualization
        """
        raise NotImplementedError

    @property
    def store_attention(self) -> bool:
        return self._store_att

    @store_attention.setter
    def store_attention(self, value: bool):
        self._store_att = value
        for layer in self.attention_layers:
            layer.store_attention = value

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes a forward pass through the network and returns the attention layers

        This will be used for visualization purposes

        Returns:
            Tensor with attention of shape [LAYERS, HEADS, BATCH, SEQ_LEN, SEQ_LEN]
            Tensor with predicted outputs of shape [BATCH, 1]
        """
        # run a forward pass
        self.eval()
        cur_att = self.store_attention
        self.store_attention = True
        with torch.no_grad():
            # compute forward pass
            logits = self.forward(x)
            logits = logits[:, -1]
            probs = F.softmax(logits, dim=-1)
            sample = torch.multinomial(probs, 1)

            # get the attention layers
            att = torch.stack(
                [torch.stack(head.attention, dim=0) for head in self.attention_layers],
                dim=0,
            )

        # reset the attention
        self.store_attention = cur_att
        return att, sample

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
        return self.tok.decode(self.generate(max_len=max_len))


class SeqToSeqModel(BaseModel, ABC):
    # TODO: implement that later on
    pass
