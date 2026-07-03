import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


if __name__ == "__main__":
    batch, seq_len, d_model = 2, 5, 32
    x = torch.randn(batch, seq_len, d_model)

    pos_enc = PositionalEncoding(d_model, max_seq_len=100)
    out = pos_enc(x)

    print("input shape: ", x.shape)
    print("output shape:", out.shape)
    print("pe buffer shape:", pos_enc.pe.shape)
    print("position 0 vs position 1 differ:", not torch.allclose(pos_enc.pe[0, 0], pos_enc.pe[0, 1]))