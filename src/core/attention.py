import math
import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        q = self.split_heads(self.w_q(query))
        k = self.split_heads(self.w_k(key))
        v = self.split_heads(self.w_v(value))

        attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        attn_output = self.combine_heads(attn_output)
        output = self.dropout(self.w_o(attn_output))
        return output, attn_weights


def causal_mask(seq_len, device=None):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def padding_mask(token_ids, pad_idx=0):
    mask = (token_ids != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask.int()


if __name__ == "__main__":
    batch, seq_len, d_model, heads = 2, 5, 32, 4
    x = torch.randn(batch, seq_len, d_model)

    mha = MultiHeadAttention(d_model, heads)

    out, attn = mha(x, x, x, mask=None)
    print("encoder-style output:", out.shape, "attn:", attn.shape)

    mask = causal_mask(seq_len)
    out, attn = mha(x, x, x, mask=mask)
    print("decoder-style output:", out.shape, "attn:", attn.shape)