import torch

from src.core.transformer_block import TransformerBlock
from src.core.attention import causal_mask


def test_transformer_block_output_shape():
    batch, seq_len, d_model, heads, d_ff = 2, 8, 32, 4, 128
    x = torch.randn(batch, seq_len, d_model)

    block = TransformerBlock(d_model, heads, d_ff)
    out, attn = block(x, mask=None)

    assert out.shape == x.shape
    assert attn.shape == (batch, heads, seq_len, seq_len)


def test_transformer_block_works_with_causal_mask():
    batch, seq_len, d_model, heads, d_ff = 2, 8, 32, 4, 128
    x = torch.randn(batch, seq_len, d_model)
    mask = causal_mask(seq_len)

    block = TransformerBlock(d_model, heads, d_ff)
    out, attn = block(x, mask=mask)

    assert out.shape == x.shape
    for i in range(seq_len):
        future_weights = attn[:, :, i, i + 1:]
        assert torch.allclose(future_weights, torch.zeros_like(future_weights), atol=1e-6)


def test_residual_connection_preserves_signal():
    """If attention and feed-forward were zeroed out entirely, output should
    still resemble the (normalized) input -- proves the residual path exists."""
    d_model, heads, d_ff = 32, 4, 128
    block = TransformerBlock(d_model, heads, d_ff, dropout=0.0)

    # zero out all learnable weights so sub-layers contribute nothing
    for p in block.attention.parameters():
        p.data.zero_()
    for p in block.feed_forward.parameters():
        p.data.zero_()

    x = torch.randn(1, 5, d_model)
    out, _ = block(x, mask=None)

    # output should equal LayerNorm(x + 0) = LayerNorm(x), not something unrelated to x
    expected = block.norm1(x)
    expected = block.norm2(expected)
    assert torch.allclose(out, expected, atol=1e-5)


def test_stacking_multiple_blocks_preserves_shape():
    d_model, heads, d_ff, num_layers = 32, 4, 128, 4
    x = torch.randn(2, 8, d_model)

    blocks = [TransformerBlock(d_model, heads, d_ff) for _ in range(num_layers)]
    for block in blocks:
        x, _ = block(x, mask=None)

    assert x.shape == (2, 8, d_model)