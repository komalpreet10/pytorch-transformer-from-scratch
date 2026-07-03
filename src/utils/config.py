"""
Central config for the project.

Every script imports PROJECT_ROOT / DEVICE / paths from here instead of
hardcoding them. This is what makes the repo behave identically whether
run locally or cloned fresh into a Kaggle notebook.
"""

from dataclasses import dataclass
from pathlib import Path
import torch

# Computed relative to this file, not hardcoded -> works on any machine.
# config.py is at src/utils/config.py, so project root is 2 levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

CHECKPOINTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class EncoderConfig:
    # model
    vocab_size: int = 30000       # set after tokenizer/vocab is built
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 128
    num_classes: int = 4          # AG News: World, Sports, Business, Sci/Tech
    dropout: float = 0.1

    # training
    batch_size: int = 64
    lr: float = 3e-4
    epochs: int = 5
    warmup_steps: int = 500


@dataclass
class DecoderConfig:
    # model
    vocab_size: int = 65          # set after char-level vocab is built (Tiny Shakespeare)
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 256        # context window (block size)
    dropout: float = 0.1

    # training
    batch_size: int = 64
    lr: float = 3e-4
    epochs: int = 20
    samples_per_epoch: int = 3000   # random windows per epoch, not exhaustive
    eval_interval: int = 500

    # generation
    gen_max_new_tokens: int = 500
    gen_temperature: float = 1.0


if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("CHECKPOINTS_DIR:", CHECKPOINTS_DIR)
    print("RESULTS_DIR:", RESULTS_DIR)
    print("DEVICE:", DEVICE)