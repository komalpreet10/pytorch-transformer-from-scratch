import torch
import torch.nn as nn

from src.core.positional_encoding import PositionalEncoding
from src.core.transformer_block import TransformerBlock
from src.core.attention import causal_mask


class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff,
                 max_seq_len, dropout=0.1):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        batch, seq_len = token_ids.shape
        mask = causal_mask(seq_len, device=token_ids.device)

        x = self.token_embedding(token_ids)
        x = self.positional_encoding(x)

        attn_weights_all = []
        for block in self.blocks:
            x, attn_weights = block(x, mask)
            attn_weights_all.append(attn_weights)

        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        return logits, attn_weights_all


if __name__ == "__main__":
    vocab_size, d_model, heads, layers, d_ff = 65, 128, 4, 4, 512
    max_seq_len = 256
    batch, seq_len = 2, 20

    model = DecoderTransformer(vocab_size, d_model, heads, layers, d_ff, max_seq_len)
    token_ids = torch.randint(0, vocab_size, (batch, seq_len))

    logits, attn_weights_all = model(token_ids)

    print("logits shape:", logits.shape)  # (batch, seq_len, vocab_size)
    print("num transformer blocks tracked:", len(attn_weights_all))
    print("total params:", sum(p.numel() for p in model.parameters()))