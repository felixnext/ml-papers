"""Full Script to train transformer model"""

import base64
from datetime import datetime
import os
import io
import time
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import fire
import numpy as np

from models import LMTransformerLight, LanguageModel
from utils import check_device, HParams, Stats, BatchStats, AttentionStats
from data import CharsetTokenizer, BasicTextDataset

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(CUR_DIR, "..", "datasets")
STATS_DIR = os.path.join(CUR_DIR, ".stats")


def create_dataset(
    name: str,
    split: float = 0.9,
    stats: Stats = None,
) -> Tuple[DataLoader, DataLoader, Stats]:
    # load the dataset
    path = os.path.join(DATASET_DIR, name)
    ds_train, ds_test = BasicTextDataset.from_file(
        path,
        stats.hparams.block_size,
        tokenizer=CharsetTokenizer,
        train_split=split,
    )
    stats.dataset = name
    stats.train_test_ratio = split
    stats.tokenizer = ds_train.tokenizer.__class__.__name__
    stats.train_size = len(ds_train)
    stats.test_size = len(ds_test)
    stats.vocab_size = ds_train.tokenizer.vocab_size

    # print stats
    print(f"Train data: {stats.train_size:08}")
    print(f"Test data:  {stats.test_size:08}")
    print(f"Vocab Size: {stats.vocab_size}")

    # create the dataset loaders around it
    return (
        DataLoader(
            ds_train,
            batch_size=stats.hparams.batch_size,
            shuffle=False,
            num_workers=4,
        ),
        DataLoader(
            ds_test,
            batch_size=stats.hparams.batch_size,
            shuffle=False,
            num_workers=4,
        ),
        stats,
    )


def test_model(
    model: LanguageModel,
    test_data: DataLoader,
    stats: Stats = None,
    store_att: bool = False,
) -> Stats:
    # start test
    start = time.time()
    model.eval()

    # retrieve test_batches size
    test_batches = stats.test_batches
    dev = model.device

    # generate test loss (todo: only to specific amount of steps)
    test_losses = []
    for test_step in range(test_batches):
        bx, by = next(iter(test_data))
        bx.to(dev)
        by.to(dev)
        test_losses.append(model.loss(bx, by).item())
    test_loss = np.mean(test_losses)
    test_time = time.time() - start
    print(f"Test Loss: {test_loss} ({test_time:.2f}s)")

    # add to stats
    stats.test_stats.append(BatchStats(test_loss, test_time))

    # compute attention (limit to 10 samples)
    if store_att:
        att_input = next(iter(test_data))[0][:5]
        att_input.to(dev)
        tok = test_data.dataset.tokenizer
        atts, sample = model.compute_attention(att_input)
        # print(f"Attention: {atts.shape}")

        # iterate all batches
        att_list = []
        atts.detach().cpu()
        for b in range(att_input.shape[0]):
            # retrieve current string
            bint = att_input[b].cpu().tolist()
            bstr = tok.decode(bint)
            bout = sample[b].cpu().tolist()
            bout_str = tok.decode(bout)

            # convert the attention weights (store this in object)
            cur_att = atts[:, :, b, :, :]  # (heads, layers, seq, seq)
            # cur_att = cur_att.tolist()
            # serialize tensor in compressed form
            buff = io.BytesIO()
            torch.save(cur_att, buff)
            # buff.seek(0)
            # att_str = buff.read().encode("utf-8")
            # att_str = buff.getvalue().encode("utf-8")
            att_str = base64.b64encode(buff.getvalue()).decode("utf-8")

            att = AttentionStats(bstr, bint, 1, bout_str, att_weights=att_str)
            att_list.append(att)

        stats.test_attention.append(att_list)

    return stats


def train_model(
    model: LanguageModel,
    train_data: DataLoader,
    test_data: DataLoader,
    stats: Stats = None,
) -> Stats:
    # set model to train mode
    model.train()
    dev = model.device

    # retrieve hparams
    lr = stats.hparams.learning_rate[-1]  # TODO: potentially update this
    batches = stats.batches
    test_every = stats.test_every

    # generate optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    stats.hparams.optimizer = optim.__class__.__name__

    # print settings
    stats.show()

    # train the model
    for epoch in range(batches):
        start = time.time()

        # load the data
        bx, by = next(iter(train_data))
        bx.to(dev)
        by.to(dev)

        # compute forward and backward pass
        loss = model.loss(bx, by)
        loss.backward()
        optim.step()
        optim.zero_grad()

        # generate data
        batch_time = time.time() - start
        loss_val = loss.item()
        # FEAT: add more stats
        stats.train_stats.append(BatchStats(loss=loss_val, time=batch_time))

        # print stats
        print(f"Batch {epoch}/{batches} - Loss: {loss_val} ({batch_time:.2f}s)")

        # run the test loop
        if epoch % test_every == 0:
            stats = test_model(
                model, test_data, stats, store_att=(epoch % stats.att_every) == 0
            )
            stats.store(STATS_DIR)

    # final test the model (store the attention weights)
    stats = test_model(model, test_data, stats, store_att=True)
    stats.run_completed = True
    return stats


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
        self.cur_dt = datetime.now().strftime("%Y-%m-%d_%H-%M")
        hparams = HParams(
            block_size=block_size,
            batch_size=batch_size,
            heads=heads,
            emb_size=emb_size,
            dropout=dropout,
            learning_rate=[lr],
        )
        self.stats = Stats([], [], [], hparams=hparams)

    def light(
        self,
        batches: int = 200,
        test_batches: int = 10,
        store_attention_every: int = 100,
        depth: int = 6,
    ):
        """Train a light version of the transformer model"""
        # update hparams and create stats
        self.stats.hparams.model_depth = depth
        self.stats.batches = batches
        self.stats.test_batches = test_batches
        self.stats.test_every = test_batches or (batches // 10)
        self.stats.att_every = store_attention_every
        self.stats.model_name = "transformer_light"

        # define device
        torch_device = check_device()

        # load the data
        train_data, test_data, stats = create_dataset(
            name="tiny_shakespeare.txt",
            stats=self.stats,
        )

        # get the vocab size
        tok = train_data.dataset.tokenizer

        # create the model
        model = LMTransformerLight(
            tok,
            emb_size=self.stats.hparams.emb_size,
            heads=self.stats.hparams.heads,
            depth=depth,
            block_size=self.stats.hparams.block_size,
            dropout=self.stats.hparams.dropout,
        )
        model.to(torch_device)
        self.stats.model_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"Params: {self.stats.model_params}")

        # train model
        self.stats = train_model(
            model,
            train_data,
            test_data,
            stats=self.stats,
        )

        # store the stats (for later visualization)
        self.stats.store(STATS_DIR)
        model.save(os.path.join(STATS_DIR, f"{self.stats.run_name()}.pth"))

        # generate some outcomes
        for i in range(5):
            print(f"--- Generated {i} ---")
            print(model.generate_str())


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    fire.Fire(Trainer)
