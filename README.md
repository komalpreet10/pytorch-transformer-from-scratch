# pytorch-transformer-from-scratch

Transformer architecture ("Attention Is All You Need", Vaswani et al., 2017) implemented from scratch in raw PyTorch — no `nn.Transformer`, no HuggingFace `transformers`, no pretrained weights. Every component (multi-head attention, positional encoding, feed-forward layers, layer norm, causal masking) is implemented and unit-tested from first principles.

Two variants share one core implementation:
- **Encoder-only** (BERT-style, bidirectional) → text classification on AG News
- **Decoder-only** (GPT-style, causal) → character-level text generation on Tiny Shakespeare

## Why this project

Most of my other projects use PyTorch through high-level wrappers (HuggingFace `Trainer`, PEFT). This project demonstrates the ability to implement deep learning architectures at the tensor level.

## Architecture

```
        Token Embedding + Positional Encoding
                      |
                      v
        +---------------------------+
        |     Transformer Block      |   x N   (shared for both variants)
        |  Multi-Head Attention      |
        |  -> Add & Norm             |
        |  -> Feed Forward           |
        |  -> Add & Norm             |
        +---------------------------+
                      |
          mask=None  /   \  mask=causal
                    /       \
                   v         v
          +--------------+  +--------------+
          |  ENCODER     |  |  DECODER     |
          |  Mean Pool   |  |  LM Head     |
          |  -> Classify |  |  -> Next tok |
          +--------------+  +--------------+
                 |                  |
                 v                  v
           AG News class      Generated text
           (4 categories)     (char by char)
```

The only structural difference between the two variants is the `mask` passed into `MultiHeadAttention`:
- **Encoder:** `mask=None` -> every token attends to every other token
- **Decoder:** `mask=causal_mask(seq_len)` -> token *i* attends only to tokens `0..i`, never the future

## Results

### Encoder-only — AG News classification
- Dataset: [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) (120K train / 7.6K test, 4 classes)
- Tokenizer: word-level, built from scratch (`src/utils/tokenizer.py`), vocab size 30,000
- Validation accuracy: **91.33%**
- Trained 5 epochs, train/val loss and accuracy tracked closely — no overfitting observed

### Decoder-only — Tiny Shakespeare generation
- Dataset: [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt), character-level
- Tokenizer: character-level, built from scratch (`src/decoder/dataset.py`), vocab size 65
- Validation loss: **2.1279** | Perplexity: **8.40**
- Trained 20 epochs, 3000 random 256-char windows/epoch
- Sample generation (prompt "ROMEO:"):
  ```
  ROMEO:
  Shat me le abe gom wereralit, chateren deats,
  Anding solld un pecketan mart byst buckers
  But tho whoy ber goa cire und den, Oumang y my bromes.

  PELUTER:
  Lor te thou the not my o but the wourst the our hat he mere ast
  he thal he blot wive ditanale her, ho Gows.
  ```
  Model correctly learned dialogue formatting (CHARACTER: line structure), word-length segmentation, and partial vocabulary; full sentence coherence would need a larger model or longer training run.

## Repo structure

```
src/
├── core/
│   ├── attention.py            # scaled dot-product + multi-head attention, causal_mask, padding_mask
│   ├── positional_encoding.py  # sinusoidal positional encoding
│   ├── feed_forward.py         # position-wise feed-forward network
│   └── transformer_block.py    # attention + FFN + residuals + layer norm
├── encoder/
│   ├── model.py                 # EncoderTransformer (mean pooling + classification head)
│   ├── dataset.py                # AG News loading + word-level tokenization
│   └── train.py
├── decoder/
│   ├── model.py                 # DecoderTransformer (causal, LM head)
│   ├── dataset.py                # Tiny Shakespeare + char-level tokenization
│   ├── train.py
│   └── generate.py               # autoregressive sampling
└── utils/
    ├── config.py                 # PROJECT_ROOT, DEVICE, EncoderConfig, DecoderConfig
    └── tokenizer.py               # word-level tokenizer (encoder)

tests/                 # unit tests: attention shapes, causal masking, residual correctness
checkpoints/           # saved model weights (gitignored)
```

## Running

Developed locally, trained on Kaggle Notebooks (free T4/P100 GPU).

```bash
pip install -r requirements.txt

# Encoder-only: AG News classification
python -m src.encoder.train

# Decoder-only: Tiny Shakespeare generation
python -m src.decoder.train
python -m src.decoder.generate --prompt "ROMEO:" --max_new_tokens 300
```

On Kaggle, after cloning:
```python
!git clone https://github.com/komalpreet10/pytorch-transformer-from-scratch.git
%cd pytorch-transformer-from-scratch
!pip install -q datasets
!python -m src.encoder.train
```

## Tests

```bash
pytest tests/ -v
```

11 tests covering: attention output shapes, multi-head split/combine correctness, attention weights summing to 1, causal mask correctness (zero leakage into future positions), padding mask correctness, and residual connections actually carrying signal through a block.

## References
- Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (2017)
- [AG News dataset](https://huggingface.co/datasets/fancyzhx/ag_news)
- [Tiny Shakespeare dataset (raw text)](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
