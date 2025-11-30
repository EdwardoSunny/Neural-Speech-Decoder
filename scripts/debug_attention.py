#!/usr/bin/env python
"""Debug script to see what the attention model is predicting"""
import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
from neural_decoder.dataset import SpeechDataset
from torch.utils.data import DataLoader
from neural_decoder.transformer_ctc import MultiScaleAttentionNeuralDecoder

# Load a small sample
trainDs = SpeechDataset('/home/edward/neural_seq_decoder/ptDecoder_ctc', partition='train')
trainLoader = DataLoader(trainDs, batch_size=4, shuffle=False, num_workers=0)

# Create model
model = MultiScaleAttentionNeuralDecoder(
    n_classes=41,
    input_dim=256,
    d_model=768,
    encoder_layers=6,
    decoder_layers=4,
    n_heads=8,
    dim_ff=2048,
    dropout=0.2,
    conv_kernel=31,
    n_days=1,
    gaussian_smooth_width=2.0,
    use_contrastive_pretraining=False,
    device='cuda',
).cuda()

model.eval()

# Get a batch
X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
X, y = X.cuda(), y.cuda()
dayIdx = dayIdx.cuda()

print("=" * 80)
print("DEBUGGING ATTENTION MODEL PREDICTIONS")
print("=" * 80)
print(f"Input shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target lengths: {y_len}")
print()

# Greedy decode
with torch.no_grad():
    decoded = model(X, dayIdx, target=None)

print(f"Decoded shape: {decoded.shape}")
print()

for i in range(min(4, decoded.shape[0])):
    dec_seq = decoded[i].cpu().numpy()
    true_seq = y[i, :y_len[i]].cpu().numpy()

    print(f"Sample {i}:")
    print(f"  True sequence ({len(true_seq)}): {true_seq[:20]}...")
    print(f"  Decoded sequence ({len(dec_seq)}): {dec_seq[:30]}...")

    # Count unique tokens
    unique_decoded = np.unique(dec_seq)
    print(f"  Unique tokens in decoded: {unique_decoded[:10]}")
    print(f"  Number of zeros: {(dec_seq == 0).sum()}")
    print()

# Test teacher forcing
print("=" * 80)
print("TESTING TEACHER FORCING")
print("=" * 80)

with torch.no_grad():
    max_len = y_len.max().item()
    y_padded = torch.zeros(y.shape[0], max_len, dtype=torch.long, device='cuda')
    for i in range(y.shape[0]):
        y_padded[i, :y_len[i]] = y[i, :y_len[i]]

    logits = model(X, dayIdx, target=y_padded)

print(f"Logits shape: {logits.shape}")
print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")

# Check what the model predicts
preds = logits.argmax(dim=-1)
print(f"Predictions shape: {preds.shape}")

for i in range(min(2, preds.shape[0])):
    pred_seq = preds[i].cpu().numpy()
    true_seq = y_padded[i].cpu().numpy()

    print(f"\nSample {i} (with teacher forcing):")
    print(f"  True:      {true_seq[:20]}")
    print(f"  Predicted: {pred_seq[:20]}")
    print(f"  Match: {(pred_seq[:y_len[i]] == true_seq[:y_len[i]]).sum()}/{y_len[i]}")
