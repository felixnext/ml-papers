"""Defines Various Attention related layers"""

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
