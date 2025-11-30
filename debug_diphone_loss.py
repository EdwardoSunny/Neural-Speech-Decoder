"""
Debug script to check if diphone joint loss is working correctly
"""

import torch
import pickle
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.neural_decoder.transformer_ctc import MultiScaleCTCDecoder
from src.neural_decoder.diphone_utils import DiphoneVocabulary
from src.neural_decoder.dataset import SpeechDataset

# Load data
print("Loading data...")
with open('/home/edward/neural_seq_decoder/ptDecoder_ctc', 'rb') as f:
    data = pickle.load(f)

# Load diphone vocab
print("Loading diphone vocab...")
diphone_vocab = DiphoneVocabulary.load('diphone_vocab.pkl')
print(f"Diphone vocab size: {diphone_vocab.get_vocab_size()}")

# Create dataset with diphone labels
print("Creating dataset...")
dataset = SpeechDataset(data['train'][:1], diphone_vocab=diphone_vocab)  # Just first day

def _padding(batch):
    X, y, X_lens, y_lens, days, y_diphone, y_diphone_lens = zip(*batch)
    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)
    y_diphone_padded = pad_sequence(y_diphone, batch_first=True, padding_value=0)

    return (
        X_padded,
        y_padded,
        torch.stack(X_lens),
        torch.stack(y_lens),
        torch.stack(days),
        y_diphone_padded,
        torch.stack(y_diphone_lens),
    )

loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=_padding)

# Create model
print("Creating model...")
model = MultiScaleCTCDecoder(
    n_classes=41,
    input_dim=256,
    d_model=1024,
    encoder_layers=6,
    n_heads=8,
    dim_ff=2048,
    dropout=0.3,
    conv_kernel=31,
    n_days=1,
    gaussian_smooth_width=2.0,
    use_diphone_head=True,
    num_diphones=diphone_vocab.get_vocab_size(),
    device='cuda',
).cuda()

# Get one batch
print("\nGetting batch...")
X, y, X_len, y_len, day_idx, y_diphone, y_diphone_len = next(iter(loader))
X = X.cuda()
y = y.cuda()
day_idx = day_idx.cuda()
y_diphone = y_diphone.cuda()
y_diphone_len = y_diphone_len.cuda()
y_len = y_len.cuda()

print(f"Batch shapes:")
print(f"  X: {X.shape}")
print(f"  y (phonemes): {y.shape}, lens: {y_len}")
print(f"  y_diphone: {y_diphone.shape}, lens: {y_diphone_len}")

# Forward pass
print("\nForward pass...")
model.eval()
with torch.no_grad():
    phone_log_probs, out_lens, diphone_log_probs = model(X, day_idx)

print(f"Model outputs:")
print(f"  phone_log_probs: {phone_log_probs.shape}")
print(f"  diphone_log_probs: {diphone_log_probs.shape}")
print(f"  out_lens: {out_lens}")

# Compute losses
print("\nComputing losses...")
loss_ctc_phone = torch.nn.CTCLoss(blank=40, reduction="mean", zero_infinity=True)
loss_ctc_diphone = torch.nn.CTCLoss(blank=diphone_vocab.blank_id, reduction="mean", zero_infinity=True)

phone_loss = loss_ctc_phone(phone_log_probs, y, out_lens, y_len)
diphone_loss = loss_ctc_diphone(diphone_log_probs, y_diphone, out_lens, y_diphone_len)

print(f"\nLoss values:")
print(f"  Phoneme CTC loss: {phone_loss.item():.4f}")
print(f"  Diphone CTC loss: {diphone_loss.item():.4f}")
print(f"  Loss ratio (diphone/phone): {diphone_loss.item() / phone_loss.item():.2f}")

# Joint loss
alpha = 0.5
joint_loss = alpha * phone_loss + (1.0 - alpha) * diphone_loss
print(f"\nJoint loss (alpha=0.5): {joint_loss.item():.4f}")
print(f"  Contribution from phonemes: {(alpha * phone_loss).item():.4f}")
print(f"  Contribution from diphones: {((1-alpha) * diphone_loss).item():.4f}")

# Check if diphone vocab is reasonable
print(f"\nDiphone vocabulary stats:")
print(f"  Num phonemes: 41")
print(f"  Num diphones: {diphone_vocab.num_diphones}")
print(f"  Blank ID (phone): 40")
print(f"  Blank ID (diphone): {diphone_vocab.blank_id}")

# Check first sample's labels
print(f"\nFirst sample labels:")
print(f"  Phoneme seq length: {y_len[0].item()}")
print(f"  Diphone seq length: {y_diphone_len[0].item()}")
print(f"  Phonemes: {y[0, :y_len[0]].tolist()}")
print(f"  Diphones: {y_diphone[0, :y_diphone_len[0]].tolist()}")

print("\n✓ Debug complete!")
