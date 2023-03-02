"""Defines the basic encoder/decoder blocks"""

import torch
from torch import nn

from layers import MultiAttentionHead


class TransformerBlock(nn.Module):
    def __init__(
        self,
        emb_size: int,
        num_heads: int,
        encoders: bool,
        block_size: int,
        head_size: int = None,
        proj_size: int = None,
        dropout: float = None,
    ):
        super().__init__()

        # compute the headsize for each head
        head_size = head_size or (emb_size // num_heads)
        proj_size = proj_size or (head_size * num_heads * 4)

        # mutli-head attention
        self.att_head = MultiAttentionHead(
            num_heads,
            emb_size,
            head_size,
            block_size=block_size,
            encoders=encoders,
            dropout=dropout,
        )

        # compute the relu layer
        self.relu_layer = nn.Sequential(
            nn.Linear(head_size * num_heads, proj_size),
            nn.ReLU(),
            # projection layer back to regular size
            nn.Linear(proj_size, emb_size),
            # dropout to regularize
            *([nn.Dropout(dropout)] if dropout is not None else []),
        )

        # normalization
        # TODO: customize normalization
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        # compute the attention
        # out = self.att_head(x)
        out = self.att_head(self.norm1(x)) + x

        # adding
        # out = self.norm1(out + x)

        # compute the relu layer
        # out = self.relu_layer(out)
        out = self.relu_layer(self.norm2(out)) + out

        # adding
        # out = self.norm2(out + x)

        return out


def EncoderBlock() -> TransformerBlock:
    pass


def DecoderBlock() -> TransformerBlock:
    pass
