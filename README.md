# Romanian Sign Language Transformer

A PyTorch classifier for short videos containing one of three isolated Romanian Sign Language (LSR) signs: **a merge**, **a pleca**, and **a privi**. The Streamlit app accepts a video and ranks these three signs.

## How it works

MediaPipe tracks 21 landmarks per hand. Each frame contributes 126 coordinates and two hand-presence flags. A linear layer projects each frame into a 32-dimensional token. Sinusoidal positions preserve frame order, and a single Transformer encoder layer uses four attention heads. The classifier averages the encoded frames while excluding padding and produces three class scores.

## Try a video

Run these commands from the project folder in a Python environment:

```bash
python -m pip install -r requirements.txt
```

Place a trained checkpoint named `sign_encoder_all_clips.pt` or `best_sign_encoder.pt` beside the scripts. The app prefers the all-clips checkpoint. Put [Google's hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) in the same folder, then run:

```bash
python -m streamlit run predict_sign_app.py
```

Upload a short clip recorded at about 30 FPS with your hands visible and select **Predict sign**. The scores compare the three classes. An unfamiliar gesture can receive a high score.

## Train on your own recordings

1. Create folders named `data/raw/a_merge`, `data/raw/a_pleca`, and `data/raw/a_privi`. Put at least two original recordings of the corresponding sign in each folder. Use similar frame rates across all recordings.
2. Extract landmarks for each original video. For example, run `python extract_landmarks.py data/raw/a_merge/clip01.mp4`. The script downloads the hand tracker if needed and writes a landmark tensor and a preview video beside the input.
3. Run `python train_sign_encoder.py`. It holds out one clip per sign and saves the checkpoint with the lowest validation loss to `best_sign_encoder.pt`.
4. Optionally run `python finetune_sign_encoder.py` to continue from that checkpoint using all clips. It saves `sign_encoder_all_clips.pt`. Test this version using fresh recordings.

## Pilot run

The original experiment used 15 clips across the three signs. The selected checkpoint classified all three validation clips correctly at epoch 40 (validation loss 0.8794). Those clips were used for checkpoint selection. Accuracy on new signers has yet to be measured.

