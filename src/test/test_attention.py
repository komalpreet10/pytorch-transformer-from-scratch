import torch
import pytest

from src.core.attention import MultiHeadAttention, scaled_dot_product_attention, causal_mask, padding_mask


def test_multihead_attention_output_shape():
    batch, seq_len, d_model, heads = 2, 10, 64, 8
    x = torch.randn(batch, seq_len, d_model)

    mha = MultiHeadAttention(d_model, heads)
    out, attn = mha(x, x, x, mask=None)

    assert out.shape == (batch, seq_len, d_model)
    assert attn.shape == (batch, heads, seq_len, seq_len)


def test_multihead_attention_rejects_bad_num_heads():
    with pytest.raises(AssertionError):
        MultiHeadAttention(d_model=65, num_heads=8)  # not divisible


def test_attention_weights_sum_to_one():
    batch, seq_len, d_model, heads = 2, 6, 32, 4
    x = torch.randn(batch, seq_len, d_model)

    mha = MultiHeadAttention(d_model, heads)
    _, attn = mha(x, x, x, mask=None)

    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_causal_mask_is_lower_triangular():
    seq_len = 5
    mask = causal_mask(seq_len)

    expected = torch.tril(torch.ones(seq_len, seq_len))
    assert torch.equal(mask.squeeze(), expected)


def test_causal_mask_blocks_future_positions():
    batch, seq_len, d_model, heads = 1, 6, 32, 4
    x = torch.randn(batch, seq_len, d_model)
    mask = causal_mask(seq_len)

    mha = MultiHeadAttention(d_model, heads)
    _, attn = mha(x, x, x, mask=mask)

    for i in range(seq_len):
        future_weights = attn[:, :, i, i + 1:]
        assert torch.allclose(future_weights, torch.zeros_like(future_weights), atol=1e-6)


def test_padding_mask_shape_and_values():
    token_ids = torch.tensor([[5, 7, 0, 0], [3, 3, 3, 0]])
    mask = padding_mask(token_ids, pad_idx=0)

    assert mask.shape == (2, 1, 1, 4)
    assert torch.equal(mask[0, 0, 0], torch.tensor([1, 1, 0, 0]))
    assert torch.equal(mask[1, 0, 0], torch.tensor([1, 1, 1, 0]))


def test_scaled_dot_product_attention_matches_manual_case():
    q = torch.randn(1, 1, 1, 8)
    k = q.clone()
    v = torch.randn(1, 1, 1, 8)

    out, attn = scaled_dot_product_attention(q, k, v, mask=None)

    assert torch.allclose(out, v, atol=1e-6)
    assert torch.allclose(attn, torch.ones_like(attn), atol=1e-6)