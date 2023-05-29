"""Helper functions for loading datasets."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import requests
from io import BytesIO
from PIL import Image
import os


# store the datasets to disk (for later reloading)
home = os.path.expanduser("~")
CACHE_LOCATION = os.path.abspath(f"{home}/.cache/huggingface/custom/laion-art")
print(f"Cache location: {CACHE_LOCATION}")
os.makedirs(CACHE_LOCATION, exist_ok=True)

UNIT_SIZE = (256, 256)


def validate_versions():
    print(f"PyTorch version: {torch.__version__} (Cuda: {torch.version.cuda})")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")

    # validate available devices for pytorch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: None")
    # list mps devices
    try:
        torch.ones(1, device="mps")
        print("MPS: True")
    except RuntimeError:
        print("MPS: False")


class DatasetLoader:
    def __init__(
        self, tok_type="bert-base-uncased", train_num: int = 2000, test_num: int = 100
    ):
        self._tokenizer = None
        self.__tok_type = tok_type
        self.train_num = train_num
        self.test_num = test_num

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.__tok_type)
        return self._tokenizer

    def _pad_and_resize(self, image: torch.Tensor, target_size: tuple[int, int]):
        # Calculate the aspect ratio of the target size
        target_aspect_ratio = target_size[0] / target_size[1]

        # Calculate the aspect ratio of the current image
        current_aspect_ratio = image.shape[2] / image.shape[1]

        # Determine the padding dimensions
        if current_aspect_ratio > target_aspect_ratio:
            # If the current image has a greater aspect ratio than the target,
            # we need to increase the height of the image (add padding to top and bottom)
            new_height = int(image.shape[2] / target_aspect_ratio)
            pad_top = (new_height - image.shape[1]) // 2
            pad_bottom = new_height - image.shape[1] - pad_top
            padding = (
                0,
                0,
                pad_top,
                pad_bottom,
            )  # padding for left, right, top, bottom
        else:
            # If the current image has a smaller aspect ratio than the target,
            # we need to increase the width of the image (add padding to left and right)
            new_width = int(image.shape[1] * target_aspect_ratio)
            pad_left = (new_width - image.shape[2]) // 2
            pad_right = new_width - image.shape[2] - pad_left
            padding = (
                pad_left,
                pad_right,
                0,
                0,
            )  # padding for left, right, top, bottom

        # Add padding to the image
        image = F.pad(image, padding, "constant", 0)

        # Resize the image tensor to the target size
        image = F.interpolate(
            image.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
        )

        return image.squeeze(0)

    def _load_image(self, url: str):
        try:
            # load the iamge
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img = torch.tensor(np.array(img))

            # convert to 3D float tensor
            if img.ndim == 2:
                img = img[:, :, None]
            if img.shape[2] == 1:
                img = img.repeat(1, 1, 3)
            img = img.permute(2, 0, 1).float()
            img = self._pad_and_resize(img, UNIT_SIZE)

            return img
        except Exception as e:
            print(f"Error loading image: {e}")
            return torch.zeros(3, UNIT_SIZE[0], UNIT_SIZE[1])

    def load(self, use_cache: bool = True):
        # check if cache exists
        if use_cache and os.path.exists(f"{CACHE_LOCATION}/ds_train-{self.train_num}"):
            # load the datasets from disk
            ds_train = load_from_disk(f"{CACHE_LOCATION}/ds_train-{self.train_num}")
            ds_test = load_from_disk(f"{CACHE_LOCATION}/ds_test-{self.test_num}")
            return ds_train, ds_test

        # load the dataset
        dataset = load_dataset("laion/laion-art")

        # split the dataset into train and test
        dataset_split = dataset["train"].train_test_split(test_size=0.1, seed=0)
        ds_train = dataset_split["train"].select(range(self.train_num))
        ds_test = dataset_split["test"].select(range(self.test_num))

        def preprocess_function(examples):
            # tokenize the text
            toks = self.tokenizer(
                examples["TEXT"], padding="max_length", truncation=True
            )

            # load the image
            toks["image"] = [self._load_image(url) for url in examples["URL"]]

            # filter out examples where image is empty tensor
            # toks = {k: [v[i] for i, img in enumerate(toks["image"]) if img.sum() > 0] for k, v in toks.items()}
            return toks

        # tokenize the data
        ds_train = ds_train.map(
            preprocess_function, batched=True, batch_size=100, num_proc=8
        )
        ds_test = ds_test.map(
            preprocess_function, batched=True, batch_size=100, num_proc=8
        )

        # filter all empty images
        ds_train = ds_train.filter(lambda x: torch.tensor(x["image"]).sum() > 0)
        ds_test = ds_test.filter(lambda x: torch.tensor(x["image"]).sum() > 0)

        # store the datasets to disk
        if use_cache:
            ds_train.save_to_disk(f"{CACHE_LOCATION}/ds_train-{self.train_num}")
            ds_test.save_to_disk(f"{CACHE_LOCATION}/ds_test-{self.test_num}")

        return ds_train, ds_test

    def visualize(self, ds, num: int):
        # visualize the images
        for i in range(num):
            print(f"Example {i+1}:")
            toks = ds[i]["input_ids"]
            img = torch.tensor(ds[i]["image"])
            img = img.permute(1, 2, 0).byte()

            # print the tokens and image
            print(self.tokenizer.decode(toks))
            plt.imshow(img)
            plt.show()
