"""Bunch of utilities for the library."""

from dataclasses import dataclass

import torch


def check_device() -> torch.device:
    torch_device = torch.device(
        "mps" if torch.has_mps else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {torch_device}")

    return torch_device


@dataclass
class Stats:
    pass
