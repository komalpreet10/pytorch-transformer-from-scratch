import torch
from torch.utils.data import Dataset, DataLoader


class CharTokenizer:
    """Character-level tokenizer -- every unique character gets an id."""

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
    """Chunks a long character stream into fixed-length (input, target) pairs
    for next-token prediction. target is input shifted by one character."""

    def __init__(self, token_ids, block_size):
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx):
        chunk = self.token_ids[idx: idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_tiny_shakespeare(block_size=256, batch_size=64, train_split=0.9):
    """Downloads Tiny Shakespeare from HuggingFace, builds char vocab, returns DataLoaders."""
    from datasets import load_dataset

    raw = load_dataset("karpathy/tiny_shakespeare")
    text = raw["train"]["text"][0]

    tokenizer = CharTokenizer().build_vocab(text)
    token_ids = tokenizer.encode(text)

    split_idx = int(len(token_ids) * train_split)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    train_ds = ShakespeareDataset(train_ids, block_size)
    val_ds = ShakespeareDataset(val_ids, block_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    # self-test with a small dummy text -- no network call
    dummy_text = "ROMEO: But soft, what light through yonder window breaks? " * 20

    tokenizer = CharTokenizer().build_vocab(dummy_text)
    token_ids = tokenizer.encode(dummy_text)

    ds = ShakespeareDataset(token_ids, block_size=16)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    x, y = next(iter(loader))
    print("vocab size:", len(tokenizer))
    print("x shape:", x.shape, "y shape:", y.shape)  # (4, 16) each
    print("x[0] decoded:", repr(tokenizer.decode(x[0].tolist())))
    print("y[0] decoded:", repr(tokenizer.decode(y[0].tolist())))
    print("(y is x shifted by one character -- next-token prediction target)")