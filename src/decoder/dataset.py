import random
import torch
from torch.utils.data import Dataset, DataLoader


class CharTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}

    def build_vocab(self, text):
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for ch, i in self.char_to_id.items()}
        return self

    def encode(self, text):
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.id_to_char[i] for i in ids)

    def __len__(self):
        return len(self.char_to_id)


class ShakespeareDataset(Dataset):
    """Randomly samples fixed-length (input, target) windows from the character
    stream, nanoGPT-style, instead of exhaustively iterating every possible
    window. This makes epoch length controllable and training time predictable,
    rather than scaling with the full length of the text."""

    def __init__(self, token_ids, block_size, samples_per_epoch=2000):
        self.token_ids = token_ids
        self.block_size = block_size
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # ignore idx -- draw a fresh random window every call
        max_start = len(self.token_ids) - self.block_size - 1
        start = random.randint(0, max_start)
        chunk = self.token_ids[start: start + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_tiny_shakespeare(block_size=256, batch_size=64, train_split=0.9,
                           samples_per_epoch=2000):
    """Downloads the raw Tiny Shakespeare text directly (the HF dataset
    'karpathy/tiny_shakespeare' uses a deprecated loading-script format that
    the datasets library no longer supports, so we bypass it entirely)."""
    import urllib.request

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = urllib.request.urlopen(url).read().decode("utf-8")

    tokenizer = CharTokenizer().build_vocab(text)
    token_ids = tokenizer.encode(text)

    split_idx = int(len(token_ids) * train_split)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    train_ds = ShakespeareDataset(train_ids, block_size, samples_per_epoch)
    val_ds = ShakespeareDataset(val_ids, block_size, samples_per_epoch // 5)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    dummy_text = "ROMEO: But soft, what light through yonder window breaks? " * 20

    tokenizer = CharTokenizer().build_vocab(dummy_text)
    token_ids = tokenizer.encode(dummy_text)

    ds = ShakespeareDataset(token_ids, block_size=16, samples_per_epoch=40)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    print("dataset length (samples_per_epoch):", len(ds))
    print("num batches per epoch:", len(loader))

    x, y = next(iter(loader))
    print("x shape:", x.shape, "y shape:", y.shape)
    print("x[0] decoded:", repr(tokenizer.decode(x[0].tolist())))
    print("y[0] decoded:", repr(tokenizer.decode(y[0].tolist())))