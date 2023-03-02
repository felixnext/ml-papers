"""Full Script to train transformer model"""

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import fire
import numpy as np

from models import LMTransformerLight, LanguageModel
from utils import check_device
from data import CharsetTokenizer, BasicTextDataset

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(CUR_DIR, "..", "datasets")


def create_dataset(
    name: str,
    block_size: int = 256,
    batch_size: int = 64,
    split: float = 0.9,
) -> Tuple[DataLoader, DataLoader]:
    # load the dataset
    path = os.path.join(DATASET_DIR, name)
    ds_train, ds_test = BasicTextDataset.from_file(
        path, block_size, tokenizer=CharsetTokenizer, train_split=split
    )

    # print stats
    print(f"Train data: {len(ds_train):08}")
    print(f"Test data:  {len(ds_test):08}")
    print(f"Vocab Size: {ds_train.tokenizer.vocab_size}")

    # create the dataset loaders around it
    return (
        DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        ),
        DataLoader(
            ds_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        ),
    )


def train_model(
    model: LanguageModel,
    train_data: DataLoader,
    test_data: DataLoader,
    lr: float,
    batches: int = 400,
):
    # set model to train mode
    model.train()
    dev = model.device

    # train the model
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(batches):
        # load the data
        bx, by = next(iter(train_data))
        bx.to(dev)
        by.to(dev)

        # compute forward and backward pass
        loss = model.loss(bx, by)
        loss.backward()
        optim.step()
        optim.zero_grad()

        # plot the loss as a line
        # TODO: show time per batch
        print(f"Batch {epoch}/{batches} - Loss: {loss.item()}")
        # TODO: generate statistics

    # generate test loss
    test_losses = []
    for bx, by in test_data:
        test_losses.append(model.loss(bx, by).item())
    test_loss = np.mean(test_losses)
    print(f"Test Loss: {test_loss}")


class Trainer:
    """Will contain different training regimes for different models"""

    def __init__(
        self,
        block_size: int = 256,
        batch_size: int = 64,
        heads: int = 8,
        emb_size: int = 384,
        dropout: float = 0.2,
        lr: float = 3e-4,
    ):
        self.block_size = block_size
        self.batch_size = batch_size
        self.heads = heads
        self.emb_size = emb_size
        self.dropout = dropout
        self.lr = lr

    def light(self, batches: int = 200, depth: int = 6):
        """Train a light version of the transformer model"""
        # define device
        torch_device = check_device()

        # load the data
        train_data, test_data = create_dataset(
            name="tiny_shakespeare.txt",
            block_size=self.block_size,
            batch_size=self.batch_size,
        )

        # get the vocab size
        tok = train_data.dataset.tokenizer

        # create the model
        model = LMTransformerLight(
            tok,
            emb_size=self.emb_size,
            heads=self.heads,
            depth=depth,
            block_size=self.block_size,
            dropout=self.dropout,
        )
        model.to(torch_device)
        print(
            f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        # train model
        train_model(
            model,
            train_data,
            test_data,
            self.lr,
            batches=batches,
        )

        # generate some outcomes
        for i in range(5):
            print(f"--- Generated {i} ---")
            print(model.generate_str())


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    fire.Fire(Trainer)
