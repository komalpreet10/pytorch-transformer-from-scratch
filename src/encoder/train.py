import torch
import torch.nn as nn
from tqdm import tqdm

from src.encoder.model import EncoderTransformer
from src.encoder.dataset import load_ag_news
from src.utils.config import EncoderConfig, DEVICE, CHECKPOINTS_DIR


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for token_ids, labels in tqdm(loader, desc="train", leave=False):
        token_ids, labels = token_ids.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(token_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for token_ids, labels in tqdm(loader, desc="eval", leave=False):
        token_ids, labels = token_ids.to(device), labels.to(device)

        logits, _ = model(token_ids)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    cfg = EncoderConfig()

    print("loading AG News...")
    train_loader, test_loader, tokenizer = load_ag_news(
        max_seq_len=cfg.max_seq_len,
        batch_size=cfg.batch_size,
    )
    cfg.vocab_size = len(tokenizer)
    print(f"vocab size: {cfg.vocab_size}, device: {DEVICE}")

    model = EncoderTransformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        num_classes=cfg.num_classes,
        dropout=cfg.dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = evaluate(model, test_loader, criterion, DEVICE)

        print(f"epoch {epoch}/{cfg.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "val_acc": val_acc,
            }, CHECKPOINTS_DIR / "encoder_best.pt")
            print(f"  saved new best checkpoint (val_acc={val_acc:.4f})")

    print(f"training done. best val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()