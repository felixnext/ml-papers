"""Defines some default transformer based models"""

from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from .base_models import LanguageModel
from blocks import TransformerBlock
from data import Tokenizer
from layers import MultiAttentionHead


class LMTransformerLight(LanguageModel):
    def __init__(
        self,
        tokenizer: Tokenizer,
        emb_size: int = 128,
        heads: int = 4,
        depth: int = 6,
        block_size: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__(tokenizer, block_size)

        # store some vars
        self.emb_size = emb_size

        # define the embedding layer
        self.embedding = nn.Embedding(self.vocab_size, emb_size)
        self.pos_embedding = nn.Embedding(block_size, emb_size)
        # self.att_head = AttentionHead(emb_size, vocab_size, encoder=True)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    emb_size,
                    heads,
                    block_size=block_size,
                    encoders=True,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(emb_size)
        self.lm_head = nn.Linear(emb_size, self.vocab_size)

    @property
    def attention_layers(self) -> List[MultiAttentionHead]:
        # retrieve the attention layers
        return [block.att_head for block in self.blocks]

    def forward(self, x):
        # validate device
        if self.device != x.device:
            x = x.to(self.device)

        # get ids
        B, T = x.shape

        # retrieve the embeddings
        chars = self.embedding(x)
        pos = self.pos_embedding(torch.arange(T, device=x.device))
        # add embeddings
        embs = chars + pos

        # compute attention
        att = embs
        for blck in self.blocks:
            att = blck(att)

        # normalize
        att = self.norm(att)

        # compute logits
        logits = self.lm_head(att)
        return logits

    def loss(self, x, y):
        # validate device
        if self.device != y.device:
            y = y.to(self.device)

        # compute the logits
        logits = self.forward(x)

        # requires a reshape from (B, T, vocab_size) to (B*T, vocab_size)
        logits = logits.view(-1, self.vocab_size)
        target = y.view(-1)

        # compute the loss
        loss = F.cross_entropy(logits, target)
        return loss
