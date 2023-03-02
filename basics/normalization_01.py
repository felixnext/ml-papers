# %% [markdown]
# # Layer Normalization
#
# Implementation of the Layer normalization layer.

# %%
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Dict, Any, Union, Optional, Tuple, Generator

# %% [markdown]
# Approach is as follows:
# 1. Load a dataset for training (e.g. classification) where we can build a deep network (say ResNet Like)
# 2. Train a basic version of that network (create a test harness that logs the loss, accuracy & training time)
# 3. Train a version with batch normalization
# 4. Implement custom Layer normalization and Test transformation against PyTorch Version
# 5. Train a version with Layer normalization
# 6. Plot the results against each other (loss, accuracy & training time)
#
#
# **Loading a Dataset**

# %%
torch_device = torch.device(
    "mps" if torch.has_mps else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"Using device: {torch_device}")

# read the tiny shakespear dataset
import os

path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/tiny_shakespeare.txt"
)
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


# %% [markdown]
# **Baseline Model**
#
# Create a baseline model that allows injection of normalization layers.

# %%
block_size = 32
vocab_size = len(chars)

# create an attention head module
class AttentionHead(nn.Module):
    def __init__(
        self, emb_size: int, head_size: int, encoder: bool, block_size: int = 32
    ):
        super().__init__()

        # create params
        self.encoder = encoder
        self.head_size = head_size
        self.emb_size = emb_size
        self.k = nn.Linear(emb_size, head_size)
        self.q = nn.Linear(emb_size, head_size)
        self.v = nn.Linear(emb_size, head_size)

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
        if self.encoder:
            wei = wei.masked_fill(self.tril == 0, float("-inf"))

        # normalize weights and apply to value transform
        wei = F.softmax(wei / self.head_size**-0.5, dim=-1)
        return wei @ self.v(x)  # (B, T, H)


class MultiAttentionHead(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_size: int,
        head_size: int,
        block_size: int,
        encoders: bool,
    ) -> None:
        super().__init__()

        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    emb_size, head_size, block_size=block_size, encoder=encoders
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(head_size * num_heads, emb_size)

    def forward(self, x):
        # compute the attention for each head
        outs = [head(x) for head in self.heads]
        out = self.proj(torch.cat(outs, dim=-1))
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        norm: Callable[[int], nn.Module],
        emb_size: int,
        num_heads: int,
        encoders: bool,
        block_size: int,
        head_size: int = None,
        proj_size: int = None,
    ):
        super().__init__()

        # compute the headsize for each head
        head_size = head_size or (emb_size // num_heads)
        proj_size = proj_size or (head_size * num_heads * 4)

        # mutli-head attention
        self.att_head = MultiAttentionHead(
            num_heads, emb_size, head_size, block_size=block_size, encoders=encoders
        )

        # compute the relu layer
        self.relu_layer = nn.Sequential(
            nn.Linear(head_size * num_heads, proj_size),
            nn.ReLU(),
            nn.Linear(proj_size, emb_size),
        )

        # normalization
        self.norm1 = norm(emb_size)
        self.norm2 = norm(emb_size)

    @property
    def norm_mean(self):
        # compute mean of parameters from norm
        n1 = [p.view(-1) for p in self.norm1.parameters()]
        n1 = torch.cat(n1).mean() if len(n1) > 0 else 0
        n2 = [p.view(-1) for p in self.norm2.parameters()]
        n2 = torch.cat(n2).mean() if len(n2) > 0 else 0
        return n1.detach().cpu().item(), n2.detach().cpu().item()

    @property
    def norm_std(self):
        # compute std of parameters from norm
        n1 = [p.view(-1) for p in self.norm1.parameters()]
        n1 = torch.cat(n1).std() if len(n1) > 0 else 0
        n2 = [p.view(-1) for p in self.norm2.parameters()]
        n2 = torch.cat(n2).std() if len(n2) > 0 else 0
        return n1.detach().cpu().item(), n2.detach().cpu().item()

    def forward(self, x):
        out = self.att_head(self.norm1(x)) + x
        out = self.relu_layer(self.norm2(out)) + out
        return out


class TfModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        norm: Callable[[int], nn.Module] = None,
        emb_size: int = 128,
        num_layers: int = 10,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_embedding = nn.Embedding(vocab_size, emb_size)
        self.norm_fct = norm or (lambda x: nn.Identity())

        # generate the transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    norm=self.norm_fct,
                    emb_size=emb_size,
                    num_heads=8,
                    block_size=block_size,
                    encoders=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = self.norm_fct(emb_size)
        self.fc = nn.Linear(emb_size, vocab_size)

    @property
    def norm_means(self):
        return [block.norm_mean for block in self.blocks]

    @property
    def norm_stds(self):
        return [block.norm_std for block in self.blocks]

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def param_count(self) -> int:
        return sum([p.numel() for p in self.parameters()])

    def forward(self, x):
        # validate device
        if self.device != x.device:
            x = x.to(self.device)

        # retrieve the embeddings
        embs = self.embedding(x) + self.pos_embedding(
            torch.arange(block_size, device=x.device)
        )

        # compute attention
        att = embs
        for blck in self.blocks:
            att = blck(att)
        att = self.norm(att)

        # compute logits
        return self.fc(att)


# %% [markdown]
# **Test Harness**
#
# Create a harness function to execute one epoch of train and testing in the entire dataset as well as a fucntion that executes a train regimen on the entire data.

# %%
from time import time


def train_step(
    model: TfModel, opt: torch.optim.Optimizer, batch_size: int = 50
) -> dict:
    start = time()
    bx, by = random_batch(train_data, block_size, batch_size)
    if torch_device != bx.device:
        bx = bx.to(torch_device)
        by = by.to(torch_device)
    batch_time = time()
    out = model(bx)
    loss = F.cross_entropy(out.view(-1, model.vocab_size), by.reshape(-1))
    loss.backward()
    train_time = time()
    opt.step()
    opt.zero_grad()
    opt_time = time()

    return {
        "batch_time": batch_time - start,
        "train_time": train_time - batch_time,
        "opt_time": opt_time - train_time,
        "total_time": opt_time - start,
        "loss": loss.detach().cpu().item(),
        "norm_means": model.norm_means,
        "norm_stds": model.norm_stds,
    }


def test_epoch(model: TfModel, batch_size: int = 50, steps: int = 20):
    # set model to eval
    model.eval()

    # iterate through steps
    loss = []
    start = time()
    for _ in range(steps):
        bx, by = random_batch(test_data, block_size, batch_size)
        if torch_device != bx.device:
            bx = bx.to(torch_device)
            by = by.to(torch_device)
        out = model(bx)
        loss.append(
            F.cross_entropy(out.view(-1, model.vocab_size), by.reshape(-1))
            .detach()
            .cpu()
            .item()
        )

    return {
        "loss": np.mean(loss),
        "time": time() - start,
    }


def train(
    model,
    epochs: int = 5,
    steps_per_epoch: int = 50,
    batch_size: int = 50,
    lr: float = 1e-4,
):
    # create optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # opt = torch.optim.SGD(model.parameters(), lr=lr)

    train_stats = []
    test_stats = []

    # iterate through epochs
    for epoch in range(epochs):
        # set model to train
        model.train()

        # iterate through steps
        for step in range(steps_per_epoch):
            stats = train_step(model, opt, batch_size)
            train_stats.append(stats)
            if step % 10 == 0:
                print(
                    f"Step {step} loss: {stats['loss']:.3f} time: {stats['total_time']:.3f}s"
                )

        # average the loss
        avg_loss = np.mean([s["loss"] for s in train_stats[-steps_per_epoch:]])
        avg_time = np.mean([s["total_time"] for s in train_stats[-steps_per_epoch:]])
        print(f"Epoch {epoch} Train loss: {avg_loss:.3f} | time: {avg_time:.3f}s")

        # run a test
        stats = test_epoch(model)
        test_stats.append(stats)
        print(
            f"Epoch {epoch} Test  loss: {stats['loss']:.3f} | time: {stats['time']:.3f}s"
        )

    return train_stats, test_stats


# execute training
base_model = TfModel(vocab_size, norm=None)
base_model.to(torch_device)

# run the test
# base_train_stats, base_test_stats = train(base_model)
# store stats
import json

stats_path = os.path.join(os.path.dirname(__file__), "stats")
# with open(os.path.join(stats_path, "base_stats.json"), "w") as f:
#    json.dump({"train": base_train_stats, "test": base_test_stats}, f)

# %% [markdown]
# **Train Batch Normalization**
class BN(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(size)
        # register the bn parameters
        self.register_parameter("weight", self.bn.weight)
        self.register_parameter("bias", self.bn.bias)

    def forward(self, x):
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


# %%
bn = lambda x: BN(x)

bn_model = TfModel(vocab_size, norm=bn)
bn_model.to(torch_device)
print(f"BN Model has {bn_model.param_count} parameters")
bn_train_stats, bn_test_stats = train(bn_model)

with open(os.path.join(stats_path, "bn_stats.json"), "w") as f:
    json.dump({"train": bn_train_stats, "test": bn_test_stats}, f)

# %% [markdown]
# **Implement Layer Normalization**
#
# Create a custom module for layer normalization and compare it to torch internal one.

# %%
class LayerNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))  # [C]
        self.beta = nn.Parameter(torch.zeros(num_features))  # [C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [B, C]
        # compute the mean and variance for the current layer
        mean = x.mean(dim=-1, keepdim=True)  # [B, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # [B, 1]

        # apply the normalization
        out = x - mean
        out = out / torch.sqrt(var + self.eps)
        out = out * self.gamma
        out = out + self.beta
        return out


rand = torch.randn(100, 10) * 10 + 5
custom_ln = LayerNorm(10)
torch_ln = nn.LayerNorm(10)

out_custom = custom_ln(rand)
out_torch = torch_ln(rand)

close = torch.allclose(out_custom, out_torch)
diff = torch.abs(out_custom - out_torch).sum()
print(f"Close: {close}, Diff: {diff}")
print(
    f"Torch Var:  {torch.var(out_torch, dim=0).mean():30}, Custom Var:  {torch.var(out_custom, dim=0).mean()}"
)
print(
    f"Torch Mean: {torch.mean(out_torch, dim=0).mean():30}, Custom Mean: {torch.mean(out_custom, dim=0).mean()}"
)

# %% [markdown]
# Next Train a model

# %%
ln = lambda x: LayerNorm(x)

ln_model = TfModel(vocab_size, norm=ln)
ln_model.to(torch_device)
print(f"LN Model has {ln_model.param_count} parameters")
ln_train_stats, ln_test_stats = train(ln_model)

with open(os.path.join(stats_path, "ln_stats.json"), "w") as f:
    json.dump({"train": ln_train_stats, "test": ln_test_stats}, f)

# TODO: visualize the gain and bias modification over time?

# %% [markdown]
# Plot Results in different graphs

# %%
# TODO: generate subgraphs
