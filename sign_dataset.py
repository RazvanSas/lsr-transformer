"""Load saved sign landmarks, split whole clips, and pad training batches."""
import argparse
import math
import random
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

LABEL_MAP = {"a_merge": 0, "a_pleca": 1, "a_privi": 2}


class SignDataset(Dataset):
    def __init__(self, samples):
        # Each sample contains the saved file path and its numeric label.
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        clip = torch.load(path, map_location="cpu", weights_only=True)
        coords = clip["coords"]
        present = clip["present"]

        if coords.ndim != 4 or tuple(coords.shape[1:]) != (2, 21, 3):
            raise ValueError(f"{path}: expected coordinates with shape (T, 2, 21, 3)")
        if len(coords) == 0 or not torch.isfinite(coords).all():
            raise ValueError(f"{path}: coordinates are empty or contain nonfinite values")
        if present.shape != coords.shape[:2] or present.dtype != torch.bool:
            raise ValueError(f"{path}: expected boolean presence with shape (T, 2)")

        return {
            "features": coords.float().flatten(start_dim=1),
            "present": present,
            "label": label,
        }


def collate_signs(samples):
    """Pad only to the longest clip in this batch."""
    lengths = torch.tensor([len(sample["features"]) for sample in samples])
    features = pad_sequence(
        [sample["features"] for sample in samples],
        batch_first=True,
        padding_value=0.0,
    )
    present = pad_sequence(
        [sample["present"] for sample in samples],
        batch_first=True,
        padding_value=False,
    )
    # True means padding. A real frame with no detected hands remains valid.
    positions = torch.arange(features.shape[1])
    padding_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)

    return {
        "features": features,
        "present": present,
        "padding_mask": padding_mask,
        "lengths": lengths,
        "labels": torch.tensor([sample["label"] for sample in samples],
                               dtype=torch.long),
    }


def make_loaders(data_dir, batch_size=4, seed=42):
    rng = random.Random(seed)
    training_samples, validation_samples = [], []
    frame_rates = []

    for sign_name, label in LABEL_MAP.items():
        folder = Path(data_dir) / sign_name
        files = sorted(folder.glob("*_landmarks.pt"))
        if len(files) < 2:
            raise ValueError(f"{folder}: need at least 2 landmark files, found {len(files)}")

        for path in files:
            clip = torch.load(path, map_location="cpu", weights_only=True)
            fps = float(clip["fps"])
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"{path}: invalid FPS {fps}")
            frame_rates.append(fps)

        rng.shuffle(files)
        validation_samples.append((files[0], label))
        training_samples.extend((path, label) for path in files[1:])

    # Allow small metadata differences such as 29.97 versus 30 FPS.
    if max(frame_rates) / min(frame_rates) > 1.05:
        raise ValueError(
            f"Different recording rates detected: {sorted(set(frame_rates))}. "
            "Resample clips to a common FPS before training."
        )

    training_loader = DataLoader(
        SignDataset(training_samples), batch_size=batch_size, shuffle=True,
        collate_fn=collate_signs, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        SignDataset(validation_samples), batch_size=batch_size, shuffle=False,
        collate_fn=collate_signs, num_workers=0,
    )
    return training_loader, validation_loader