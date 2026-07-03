import re
from collections import Counter


class WordTokenizer:
    """Simple whitespace/punctuation word-level tokenizer with a built vocab."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, max_vocab_size=30000):
        self.max_vocab_size = max_vocab_size
        self.token_to_id = {}
        self.id_to_token = {}

    @staticmethod
    def _split(text):
        text = text.lower()
        return re.findall(r"\b\w+\b", text)

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(self._split(text))

        vocab = [self.PAD_TOKEN, self.UNK_TOKEN] + [
            tok for tok, _ in counter.most_common(self.max_vocab_size - 2)
        ]

        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        return self

    def encode(self, text, max_len=None):
        tokens = self._split(text)
        unk_id = self.token_to_id[self.UNK_TOKEN]
        ids = [self.token_to_id.get(tok, unk_id) for tok in tokens]

        if max_len is not None:
            pad_id = self.token_to_id[self.PAD_TOKEN]
            ids = ids[:max_len]
            ids = ids + [pad_id] * (max_len - len(ids))

        return ids

    def __len__(self):
        return len(self.token_to_id)


if __name__ == "__main__":
    texts = [
        "The stock market rallied today after strong earnings reports.",
        "The team won the championship game last night.",
        "New AI model achieves state of the art results.",
    ]

    tok = WordTokenizer(max_vocab_size=1000).build_vocab(texts)
    print("vocab size:", len(tok))

    encoded = tok.encode(texts[0], max_len=10)
    print("encoded (padded to 10):", encoded)

    encoded_unk = tok.encode("completely unseen sentence about spacecraft", max_len=10)
    print("encoded with unknown words:", encoded_unk)