import math
import torch
import torch.nn as nn
from tqdm import tqdm

from src.decoder.model import DecoderTransformer
from src.decoder.dataset import load_tiny_shakespeare
from src.utils.config import DecoderConfig, DEVICE, CHECKPOINTS_DIR


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_tokens = 0.0, 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits, _ = model(x)  # (batch, seq_len, vocab_size)

        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()

        n_tokens = y.numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / total_tokens


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device), y.to(device)

        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        n_tokens = y.numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / total_tokens


def main():
    cfg = DecoderConfig()

    print("loading Tiny Shakespeare...")
    train_loader, val_loader, tokenizer = load_tiny_shakespeare(
        block_size=cfg.max_seq_len,
        batch_size=cfg.batch_size,
        samples_per_epoch=cfg.samples_per_epoch,
    )
    cfg.vocab_size = len(tokenizer)
    print(f"vocab size: {cfg.vocab_size}, device: {DEVICE}")

    model = DecoderTransformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = evaluate(model, val_loader, criterion, DEVICE)
        val_ppl = math.exp(val_loss)

        print(f"epoch {epoch}/{cfg.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "val_loss": val_loss,
                "tokenizer_char_to_id": tokenizer.char_to_id,
            }, CHECKPOINTS_DIR / "decoder_best.pt")
            print(f"  saved new best checkpoint (val_loss={val_loss:.4f})")

    print(f"training done. best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()