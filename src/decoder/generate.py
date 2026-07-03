import argparse
import torch

from src.decoder.model import DecoderTransformer
from src.decoder.dataset import CharTokenizer
from src.utils.config import DEVICE, CHECKPOINTS_DIR, DecoderConfig


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, max_seq_len, temperature=1.0, device="cpu"):
    model.eval()
    token_ids = tokenizer.encode(prompt)
    token_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq_len)

    for _ in range(max_new_tokens):
        context = token_ids[:, -max_seq_len:]  # truncate to model's context window

        logits, _ = model(context)
        next_logits = logits[:, -1, :] / temperature  # (1, vocab_size), last position only

        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

        token_ids = torch.cat([token_ids, next_id], dim=1)

    return tokenizer.decode(token_ids[0].tolist())


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = DecoderConfig(**checkpoint["config"])

    tokenizer = CharTokenizer()
    tokenizer.char_to_id = checkpoint["tokenizer_char_to_id"]
    tokenizer.id_to_char = {i: ch for ch, i in tokenizer.char_to_id.items()}

    model = DecoderTransformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, tokenizer, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="ROMEO:")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    ckpt_path = CHECKPOINTS_DIR / "decoder_best.pt"
    model, tokenizer, cfg = load_checkpoint(ckpt_path, DEVICE)

    output = generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_seq_len=cfg.max_seq_len,
        temperature=args.temperature,
        device=DEVICE,
    )

    print(output)


if __name__ == "__main__":
    main()