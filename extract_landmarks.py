"""Extract a tensor from one sign video and replay its landmarks over the clip."""
import argparse
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
# Landmark indices along the palm and each finger.
CHAINS = [(0, 1, 2, 3, 4), (0, 5, 9, 13, 17, 0), (5, 6, 7, 8),
          (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
COLORS = [(255, 120, 0), (0, 150, 255)]  # Left, right (OpenCV uses BGR).


def extract(video, model):
    cap = cv2.VideoCapture(str(video))
    frames, masks = [], []
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no usable frame rate. Export it at 30 FPS.")
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2)
        with vision.HandLandmarker.create_from_options(options) as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = tracker.detect_for_video(image, round(len(frames) * 1000 / fps))
                coords = torch.zeros(2, 21, 3)
                present = torch.zeros(2, dtype=torch.bool)
                scores = [-1., -1.]
                for labels, hand in zip(result.handedness, result.hand_landmarks):
                    label = labels[0]
                    slot = {"Left": 0, "Right": 1}[label.category_name]
                    if label.score > scores[slot]:
                        coords[slot] = torch.tensor([(p.x, p.y, p.z) for p in hand])
                        present[slot] = True
                        scores[slot] = label.score
                frames.append(coords)
                masks.append(present)
    finally:
        cap.release()
    if not frames:
        raise ValueError("No video frames could be decoded.")
    return {"coords": torch.stack(frames), "present": torch.stack(masks), "fps": fps}


def draw_overlay(video, data, output):
    cap = cv2.VideoCapture(str(video))
    writer = None
    try:
        for coords, present in zip(data["coords"], data["present"]):
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Could not reread the original video.")
            height, width = frame.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                                         data["fps"], (width, height))
                if not writer.isOpened():
                    raise OSError(f"Could not write {output}")
            for slot, color in enumerate(COLORS):
                if not present[slot]:
                    continue
                pixels = (coords[slot, :, :2].numpy() * [width, height]).round().astype(int)
                for chain in CHAINS:
                    for a, b in zip(chain, chain[1:]):
                        cv2.line(frame, tuple(pixels[a]), tuple(pixels[b]), color, 2)
                for point in pixels:
                    cv2.circle(frame, tuple(point), 4, color, -1)
            writer.write(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    video = parser.parse_args().video
    if not video.is_file():
        parser.error(f"Video file does not exist: {video}")
    model = Path(__file__).with_name("hand_landmarker.task")
    if not model.exists():
        print("Downloading Google's Hand Landmarker model...")
        temp = model.with_suffix(".download")
        urlretrieve(MODEL_URL, temp)
        temp.replace(model)
    tensor_path = video.with_name(video.stem + "_landmarks.pt")
    overlay_path = video.with_name(video.stem + "_overlay.mp4")
    torch.save(extract(video, model), tensor_path)
    data = torch.load(tensor_path, weights_only=True)
    draw_overlay(video, data, overlay_path)
    print("Coordinates:", tuple(data["coords"].shape))
    print("Frames with a detected hand:", data["present"].any(dim=1).sum().item(),
          "of", len(data["coords"]))
    print("Saved:", tensor_path, "and", overlay_path)
