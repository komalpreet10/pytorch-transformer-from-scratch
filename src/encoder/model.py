import torch
import torch.nn as nn

from src.core.positional_encoding import PositionalEncoding
from src.core.transformer_block import TransformerBlock


class EncoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff,
                 max_seq_len, num_classes, dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids, mask=None):
        x = self.token_embedding(token_ids)
        x = self.positional_encoding(x)

        attn_weights_all = []
        for block in self.blocks:
            x, attn_weights = block(x, mask)
            attn_weights_all.append(attn_weights)

        # mean pooling over non-padded tokens -> one vector per sequence
        pooled = x.mean(dim=1)
        logits = self.classifier(self.dropout(pooled))

        return logits, attn_weights_all


if __name__ == "__main__":
    vocab_size, d_model, heads, layers, d_ff = 1000, 128, 4, 4, 512
    max_seq_len, num_classes = 128, 4
    batch, seq_len = 2, 20

    model = EncoderTransformer(vocab_size, d_model, heads, layers, d_ff, max_seq_len, num_classes)
    token_ids = torch.randint(1, vocab_size, (batch, seq_len))

    logits, attn_weights_all = model(token_ids)

    print("logits shape:", logits.shape)  # (batch, num_classes)
    print("num transformer blocks tracked:", len(attn_weights_all))
    print("attn weights per block shape:", attn_weights_all[0].shape)
    print("total params:", sum(p.numel() for p in model.parameters()))