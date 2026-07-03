import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.tokenizer import WordTokenizer


class AGNewsDataset(Dataset):
    """Wraps HuggingFace AG News with our own word-level tokenizer."""

    def __init__(self, texts, labels, tokenizer, max_seq_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.texts[idx], max_len=self.max_seq_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def load_ag_news(max_seq_len=128, max_vocab_size=30000, batch_size=64):
    """Downloads AG News from HuggingFace, builds vocab from train split, returns DataLoaders."""
    from datasets import load_dataset

    raw = load_dataset("fancyzhx/ag_news")

    train_texts = raw["train"]["text"]
    train_labels = raw["train"]["label"]
    test_texts = raw["test"]["text"]
    test_labels = raw["test"]["label"]

    tokenizer = WordTokenizer(max_vocab_size).build_vocab(train_texts)

    train_ds = AGNewsDataset(train_texts, train_labels, tokenizer, max_seq_len)
    test_ds = AGNewsDataset(test_texts, test_labels, tokenizer, max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, tokenizer


if __name__ == "__main__":
    # self-test with dummy in-memory data -- no network call, just checks the
    # Dataset/DataLoader/tokenizer wiring is correct
    texts = [
        "Stocks rallied today after strong earnings reports from tech firms.",
        "The championship game ended in a dramatic overtime victory.",
        "Scientists announced a breakthrough in quantum computing research.",
        "The central bank raised interest rates for the third time this year.",
    ]
    labels = [2, 1, 3, 2]  # business, sports, sci/tech, business

    tokenizer = WordTokenizer(max_vocab_size=200).build_vocab(texts)
    ds = AGNewsDataset(texts, labels, tokenizer, max_seq_len=15)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    batch_ids, batch_labels = next(iter(loader))
    print("batch token ids shape:", batch_ids.shape)   # (2, 15)
    print("batch labels shape:  ", batch_labels.shape) # (2,)
    print("vocab size:", len(tokenizer))