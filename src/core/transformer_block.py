import torch
import torch.nn as nn

from src.core.attention import MultiHeadAttention
from src.core.feed_forward import PositionwiseFeedForward


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x, attn_weights


if __name__ == "__main__":
    from src.core.attention import causal_mask

    batch, seq_len, d_model, heads, d_ff = 2, 5, 32, 4, 128
    x = torch.randn(batch, seq_len, d_model)

    block = TransformerBlock(d_model, heads, d_ff)

    out, attn = block(x, mask=None)
    print("encoder-style output:", out.shape, "attn:", attn.shape)

    mask = causal_mask(seq_len)
    out, attn = block(x, mask=mask)
    print("decoder-style output:", out.shape, "attn:", attn.shape)