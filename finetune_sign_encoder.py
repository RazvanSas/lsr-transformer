"""Fine-tune the saved sign encoder on every existing clip."""
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from train_sign_encoder import (
    BATCH_SIZE, LABEL_MAP, PROJECT_DIR, SEED, SignEncoder, make_loaders, run_epoch,
)

EXTRA_EPOCHS = 20
LEARNING_RATE = 0.0003


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = torch.load(PROJECT_DIR / "best_sign_encoder.pt", map_location=device)
    if source["label_map"] != LABEL_MAP:
        raise ValueError("The checkpoint and dataset use different label mappings.")

    train, val = make_loaders(PROJECT_DIR / "data" / "raw",
                              batch_size=BATCH_SIZE, seed=SEED)
    clips = ConcatDataset([train.dataset, val.dataset])
    loader = DataLoader(clips, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=train.collate_fn)

    model = SignEncoder().to(device)
    model.load_state_dict(source["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    print(f"Starting from epoch {source['epoch']} with {len(clips)} clips.")
    for epoch in range(1, EXTRA_EPOCHS + 1):
        loss, correct, count = run_epoch(model, loader, criterion, device, optimizer)
        print(f"Extra epoch {epoch:02}/{EXTRA_EPOCHS} | loss {loss:.4f}, correct {correct}/{count}")

    output = PROJECT_DIR / "sign_encoder_all_clips.pt"
    temp = output.with_suffix(".tmp")
    torch.save({"model_state_dict": model.state_dict(), "label_map": LABEL_MAP,
                "epoch": source["epoch"] + EXTRA_EPOCHS}, temp)
    temp.replace(output)
    print(f"Saved {output.name}")


if __name__ == "__main__":
    main()
