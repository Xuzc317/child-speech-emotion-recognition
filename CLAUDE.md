# CLAUDE.md

Project constraints and conventions for AI assistants working in this repo.

## ONNX compatibility

`torch.nn.AdaptiveAvgPool1d` is **not ONNX-compatible** — it causes graph export failures in most ONNX runtimes. Do NOT use it. Use `nn.MaxPool1d` (or a fixed-kernel `nn.AvgPool1d`) instead.

When a model layer needs fixed output length regardless of input dimension, compute the post-convolution flattened dimension dynamically at `__init__` time with a dummy forward pass:

```python
with torch.no_grad():
    dummy = torch.zeros(1, 1, feat_dim)
    dummy_out = self.conv_stack(dummy)
    flat_dim = dummy_out.view(1, -1).shape[1]
self.classifier = nn.Linear(flat_dim, ...)
```

This keeps all ops ONNX-compatible while remaining dimension-agnostic.

## Feature dimension

`extract_features()` in `src/data/dataset.py` returns **162**-dim vectors: 1 ZCR + 12 Chroma + 20 MFCC + 1 RMS + 128 Mel (librosa defaults). The old comment that said 94 was wrong.

## Data split

Speaker-independent split is in `src/data/preprocess.py`. Key properties:
- 237 unique speakers, 4,179 WAVs (MY dataset = ENGLISH + TELUGU)
- Profile-based stratified split: speakers grouped by emotion-class profile, then 70/30 within each group
- Both train and test get 3x processing (original + noise + stretch+pitch)
- Output: `train_data.npy`, `test_data.npy` (162-dim features), plus label files

## Legacy checkpoints

All checkpoints in `checkpoints/legacy/` were trained with data leakage (same speaker in train and test). Their accuracies are invalid. Do not use them for evaluation.
