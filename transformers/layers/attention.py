"""Defines Various Attention related layers"""

from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    def __init__(
        self,
        emb_size: int,
        head_size: int,
        encoder: bool,
        block_size: int = 32,
        dropout: float = None,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.head_size = head_size
        self.encoder = encoder
        self.store_attention = False
        self._att = None

        # define the key, query, and value vectors
        self.k = nn.Linear(emb_size, head_size)
        self.q = nn.Linear(emb_size, head_size)
        self.v = nn.Linear(emb_size, head_size)

        # check for dropout
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

        # create register buffer for the tril
        self.register_buffer(
            "tril",
            torch.tril(torch.ones((block_size, block_size), dtype=torch.float32)),
        )

    @property
    def attention(self) -> Optional[torch.Tensor]:
        """Retrieves the attention from the last iteration"""
        return self._att

    def forward(self, x):
        # compute key and query vectors
        xk = self.k(x)  # (B, T, H)
        xq = self.q(x)  # (B, T, H)

        # compute weight by transpose and matmul product
        wei = xq @ xk.transpose(-1, -2)

        # apply tril mask
        if self.encoder:
            wei = wei.masked_fill(self.tril == 0, float("-inf"))

        # normalize weights and check for dropout
        wei = F.softmax(wei / self.head_size**-0.5, dim=-1)
        # wei is what we want to store as self attention information
        if self.store_attention:
            self._att = wei
        if hasattr(self, "dropout"):
            wei = self.dropout(wei)

        # normalize data and apply softmax
        xv = self.v(x)  # (B, T, H)
        out = wei @ xv

        return out


class MultiAttentionHead(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_size: int,
        head_size: int,
        block_size: int,
        encoders: bool,
        dropout: float = None,
    ) -> None:
        super().__init__()
        self._store_att = False

        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    emb_size,
                    head_size,
                    block_size=block_size,
                    encoder=encoders,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(head_size * num_heads, emb_size)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

    @property
    def attention(self) -> List[torch.Tensor]:
        """Returns the attention weights for each head"""
        return [head.attention for head in self.heads]

    @property
    def store_attention(self) -> bool:
        """Returns the current store_att value"""
        return self._store_att

    @store_attention.setter
    def store_attention(self, value: bool) -> None:
        """Sets the store_att value for all heads"""
        self._store_att = value
        for head in self.heads:
            head.store_attention = value

    def forward(self, x):
        # compute the attention for each head
        outs = [head(x) for head in self.heads]

        # concatenate the outputs
        out = torch.cat(outs, dim=-1)

        # project back to the original embedding size
        out = self.proj(out)

        # apply dropout
        if hasattr(self, "dropout"):
            out = self.dropout(out)

        return out
