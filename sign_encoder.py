"""Encoder-only Transformer for isolated signs. Run to check one real batch."""
import math
from pathlib import Path

import torch
from torch import nn

from sign_dataset import LABEL_MAP, make_loaders


class SignEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 32
        self.frame_projection = nn.Linear(126 + 2, d_model)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("frequencies", frequencies)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=64,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.classifier = nn.Linear(d_model, len(LABEL_MAP))

    def forward(self, features, present, padding_mask):
        # Give each frame one token containing both hands' coordinates and flags.
        frames = torch.cat([features, present.to(features.dtype)], dim=-1)
        tokens = self.frame_projection(frames)

        # Interleave sine and cosine values to encode each frame's position.
        positions = torch.arange(tokens.size(1), device=tokens.device)
        angles = positions[:, None] * self.frequencies
        position_encoding = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)
        tokens = tokens + position_encoding.to(tokens.dtype)

        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)

        # Average only real frames, even when a clip has missing hand detections.
        encoded = encoded.masked_fill(padding_mask.unsqueeze(-1), 0)
        lengths = (~padding_mask).sum(dim=1, keepdim=True)
        clip_vectors = encoded.sum(dim=1) / lengths
        return self.classifier(clip_vectors)


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "data" / "raw"
    train_loader, _ = make_loaders(data_dir)
    batch = next(iter(train_loader))
    model = SignEncoder()
    model.eval()
    with torch.no_grad():
        logits = model(batch["features"], batch["present"], batch["padding_mask"])
        loss = nn.functional.cross_entropy(logits, batch["labels"])
    print("Frame coordinates:", tuple(batch["features"].shape))
    print("Class scores:", tuple(logits.shape))
    print("Class order:", LABEL_MAP)
    print("First clip scores:", logits[0].tolist())
    print("First clip label:", batch["labels"][0].item())
    print("Loss before training:", loss.item())
