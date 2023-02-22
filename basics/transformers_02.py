"""Full Script to train transformer model"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from sklearn.decomposition import PCA
import torch
from torch import nn
from torch.nn import functional as F
from torchviz import make_dot
from typing import List, Callable, Dict, Any, Union, Optional, Tuple, Generator

torch_device = torch.device(
    "mps" if torch.has_mps else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"Using device: {torch_device}")

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path, "data", "tiny_shakespeare.txt")

# read the tiny shakespear dataset
with open(path, "r") as f:
    lines = f.read()

# read some stats
print(f"Number of characters: {len(lines)}")
chars = sorted(list(set(lines)))
print(f"Number of unique characters: {len(chars)}")
print(f"Characters: {chars}")

# build a dict
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}


def encode_tensor(text: str) -> torch.Tensor:
    return torch.tensor([char2idx[ch] for ch in text])


def encode(text: str) -> List[int]:
    return [char2idx[ch] for ch in text]


def decode_tensor(encoded: torch.Tensor) -> str:
    # assert dim
    encoded = encoded.squeeze()
    if encoded.dim() == 2:
        encoded = encoded.argmax(dim=1)
    if encoded.dim() > 1:
        raise ValueError(f"Expected 1D or 2D tensor, got {encoded.dim()}D tensor")
    return "".join([idx2char[idx] for idx in encoded.tolist()])


def decode(encoded: List[int]) -> str:
    return "".join([idx2char[idx] for idx in encoded])


assert decode(encode("hello world")) == "hello world"
assert decode_tensor(encode_tensor("hello world")) == "hello world"

# encode the tensor
data = encode_tensor(lines)

# split the data
split_idx = int(len(data) * 0.9)
train_data = data[:split_idx]
test_data = data[split_idx:]

print(f"Train data: {train_data.shape[0]:08}")
print(f"Test data:  {test_data.shape[0]:08}")

# build a batch generator - this should sample random section from the text
def random_batch(
    data: torch.Tensor, block_size: int, batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Retrieves a random batch from the provided data"""
    # check the input data
    if data.dim() != 1:
        raise ValueError(f"Expected 1D tensor, got {data.dim()}D tensor")

    # create a random starting point
    # note: we need to have one last char for output
    # note: also need to have block_size difference
    start_idx = torch.randint(0, len(data) - block_size - 1, (batch_size,))

    # retrieve the content length (batchsi)
    content = torch.stack([data[i : i + block_size + 1] for i in start_idx])
    x = content[:, :-1]
    y = content[:, 1:]
    return x, y


class LModel(nn.Module):
    def __init__(self, vocab_size, block_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

    @property
    def device(self):
        return next(self.parameters()).device

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
        return "".join([idx2char[idx] for idx in self.generate(max_len=max_len)])


vocab_size = len(chars)


def train_model(model, lr, batches=400, block_size: int = 8, batch_size: int = 4):
    # set model to train mode
    model.train()

    # create the plot
    # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # ax.set_xlabel("Batch")
    # ax.set_ylabel("Loss")
    # ax.set_title("Training loss")

    # train the model
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(batches):
        bx, by = random_batch(train_data, block_size, batch_size)
        loss = model.loss(bx, by)
        loss.backward()
        optim.step()
        optim.zero_grad()

        # plot the loss as a line
        print(f"Batch {epoch}/{batches} - Loss: {loss.item()}")
        # ax.set_title(f"Training loss (batch {epoch}/{batches})")
        # ax.plot(epoch, loss.item(), "r.")
        # clear_output(wait=True)
        # plt.show()

    # generate test loss
    tx, ty = random_batch(test_data, block_size, 100)
    test_loss = model.loss(tx, ty)
    print(f"Test Loss: {test_loss}")


# create an attention head module
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


# build a model around the attention head
class TransformerLight(LModel):
    def __init__(
        self, vocab_size, emb_size=128, heads=4, depth=6, block_size=32, dropout=0.1
    ):
        super().__init__(vocab_size, block_size)

        # define the embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_size)
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
        self.lm_head = nn.Linear(emb_size, vocab_size)

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


# hparams
t_block_size = 256
t_batch_size = 64
t_heads = 8
t_emb_size = 384
t_blocks = 10
t_dropout = 0.2
t_lr = 3e-4

# create the model
model = TransformerLight(
    vocab_size,
    emb_size=t_emb_size,
    heads=t_heads,
    depth=t_blocks,
    block_size=t_block_size,
    dropout=t_dropout,
)
model.to(torch_device)
print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


# train model
train_model(model, t_lr, batches=2000, block_size=t_block_size, batch_size=t_batch_size)
for i in range(5):
    print(f"--- Generated {i} ---")
    print(model.generate_str())
