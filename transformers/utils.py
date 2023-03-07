"""Bunch of utilities for the library."""

import base64
from dataclasses import dataclass, asdict
from datetime import datetime
import io
import json
import os
from typing import Optional, List

import torch


def check_device() -> torch.device:
    torch_device = torch.device(
        "mps" if torch.has_mps else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {torch_device}")

    return torch_device


@dataclass
class HParams:
    """Hyperparameters for the training process."""

    block_size: int = 256
    batch_size: int = 64
    learning_rate: List[float] = 3e-4
    heads: int = 8
    emb_size: int = 384
    model_depth: Optional[int] = None
    dropout: Optional[float] = None
    optimizer: Optional[str] = None
    others: Optional[dict] = None


@dataclass
class BatchStats:
    """Class to store the stats of the training process."""

    # stats
    loss: float
    time: float
    perplexity: Optional[float] = None
    accuracy: Optional[float] = None


@dataclass
class AttentionStats:

    # text stats
    input_text: str
    input_tokens: List[int]
    output_tokens: int
    output_text: str

    # attention stats
    # att_weights: List[List[List[List[float]]]]
    # base64 byte serialized torch tensor with shape (depth, heads, block_size, block_size)
    att_weights: str

    def get_att_tensor(self) -> torch.Tensor:
        return torch.load(
            io.BytesIO(base64.b64decode(self.att_weights.encode("utf-8")))
        )


@dataclass
class Stats:
    """Class to store the stats of the training process."""

    # training stats
    train_stats: List[BatchStats]
    test_stats: List[BatchStats]
    test_attention: List[List[AttentionStats]]

    # name of the dataset used
    exp_name: Optional[str] = None
    dataset: Optional[str] = None
    tokenizer: Optional[str] = None
    model_name: Optional[str] = None
    model_params: Optional[int] = None

    # size of the different datasets
    train_size: Optional[int] = None
    test_size: Optional[int] = None
    train_test_ratio: Optional[float] = None

    # size of the vocabulary
    vocab_size: Optional[int] = None

    # trained batches
    batches: Optional[int] = None
    test_batches: Optional[int] = None
    test_every: Optional[int] = None
    att_every: Optional[int] = None
    run_completed: bool = False

    # hyperparameters of the model
    hparams: Optional[HParams] = None

    def run_name(self):
        if self.exp_name is None:
            self.exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
        return f"{self.model_name}-{self.exp_name}"

    # function to store the stats
    def store(self, path: str, **kwargs):
        # check if folder exists
        if not os.path.exists(path):
            os.makedirs(path)

        # generate the file path
        path = os.path.join(path, f"{self.run_name()}.json")

        # store the stats
        data = asdict(self)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            data = json.load(f)

        # load sub-elements
        if data.get("hparams") is not None:
            data["hparams"] = HParams(**data["hparams"])
        if data.get("train_stats") is not None:
            data["train_stats"] = [BatchStats(**s) for s in data["train_stats"]]
        if data.get("test_stats") is not None:
            data["test_stats"] = [BatchStats(**s) for s in data["test_stats"]]
        if data.get("test_attention") is not None:
            data["test_attention"] = [
                [AttentionStats(**s) for s in att_ls]
                for att_ls in data["test_attention"]
            ]

        # generate the class
        return cls(**data)

    def show(self):
        print("Model Settings:")
        items = asdict(self)
        print(json.dumps(items, indent=4))
