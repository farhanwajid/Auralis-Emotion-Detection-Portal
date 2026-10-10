# Auralis: Emotion Detection Portal

Auralis is a web portal that detects emotion from speech. Upload an audio file or
record your voice, and it returns the predicted emotion, probabilities for all
8 classes, and a timeline showing how the emotion changes over longer clips.

## How it works
- **Model:** Wav2Vec2-base fine-tuned on RAVDESS (8 emotions: neutral, calm, happy,
  sad, angry, fearful, disgust, surprised), with a learnable layer-weighted
  representation and attentive pooling.
- **Training:** speaker-independent split (actors never overlap between train,
  validation and test), with noise, reverb, band-limiting and clipping
  augmentation for robustness.
- **Backend:** FastAPI service that converts uploads to 16 kHz mono audio and runs inference.
- **Frontend:** React + Tailwind interface with waveform playback and charts.

## Results
| Split | Accuracy |
|---|---|
| Validation (3 unseen actors) | 82% |
| Test (5 unseen actors) | 74% |

Robustness results under noise, reverb and telephone-quality audio are in
`results/robustness_test.csv`.

## Privacy
Uploaded audio is processed in a temporary file and deleted right after
analysis. Nothing is stored or shared.

## Limitations
Trained on acted speech from 24 actors, so accuracy on spontaneous, real-world
speech will be lower. This is a research prototype, not a clinical or
diagnostic tool.
