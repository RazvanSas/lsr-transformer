from pathlib import Path
import torch

data_folder = Path("data/raw")

label_map = {
    "a_merge": 0,
    "a_pleca": 1,
    "a_privi": 2,
}

videos = []
labels = []
presence_masks = []

for sign_name, label in label_map.items():
    sign_folder = data_folder / sign_name
    files = sorted(sign_folder.glob("*_landmarks.pt"))

    if not files:
        raise FileNotFoundError(
            f"No landmark files found in {sign_folder.resolve()}"
        )

    for file_path in files:
        clip = torch.load(
            file_path,
            map_location="cpu",
            weights_only=True,
        )

        coords = clip["coords"]       # (T, 2, 21, 3)
        present = clip["present"]     # (T, 2)

        features = coords.flatten(start_dim=1)  # (T, 126)

        videos.append(features)
        labels.append(label)
        presence_masks.append(present)

        print(f"Loaded {file_path.name}: {tuple(features.shape)}")

labels = torch.tensor(labels, dtype=torch.long)

print(f"\nTotal videos loaded: {len(videos)}")
print("Labels:", labels)