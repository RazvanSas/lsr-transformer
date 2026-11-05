"""Train the encoder on isolated sign clips and save the lowest-validation-loss model."""
from pathlib import Path

import torch
from torch import nn

from sign_dataset import LABEL_MAP, make_loaders
from sign_encoder import SignEncoder


PROJECT_DIR = Path(__file__).resolve().parent
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 0.001
SEED = 42


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            features = batch["features"].to(device)
            present = batch["present"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            labels = batch["labels"].to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(features, present, padding_mask)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            count += labels.numel()
            total_loss += loss.item() * labels.numel()
            correct += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / count, correct, count


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_loaders(PROJECT_DIR / "data" / "raw",
                                            batch_size=BATCH_SIZE, seed=SEED)
    model = SignEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_epoch = None
    checkpoint_path = PROJECT_DIR / "best_sign_encoder.pt"
    print(f"Training on {device}. Validation clips:")
    for path, label in val_loader.dataset.samples:
        print(f"  {path.name} (label {label})")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_correct, train_count = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_correct, val_count = run_epoch(
            model, val_loader, criterion, device
        )
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train loss {train_loss:.4f}, correct {train_correct}/{train_count} | "
              f"val loss {val_loss:.4f}, correct {val_correct}/{val_count}")

        if val_loss < best_loss:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "label_map": LABEL_MAP,
                "epoch": epoch,
                "val_loss": val_loss,
                "split_seed": SEED,
            }
            temp_path = checkpoint_path.with_suffix(".tmp")
            torch.save(checkpoint, temp_path)
            temp_path.replace(checkpoint_path)
            best_loss = val_loss
            best_epoch = epoch

    if best_epoch is None:
        raise RuntimeError("No checkpoint saved. Check the printed validation losses.")
    print(f"Saved {checkpoint_path.name} from epoch {best_epoch} "
          f"(validation loss {best_loss:.4f}).")


if __name__ == "__main__":
    main()
