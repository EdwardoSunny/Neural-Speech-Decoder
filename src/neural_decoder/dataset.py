import torch
from torch.utils.data import Dataset


class SpeechDataset(Dataset):
    def __init__(self, data, transform=None, diphone_vocab=None):
        self.data = data
        self.transform = transform
        self.diphone_vocab = diphone_vocab
        self.n_days = len(data)
        self.n_trials = sum([len(d["sentenceDat"]) for d in data])

        self.neural_feats = []
        self.phone_seqs = []
        self.diphone_seqs = []  # NEW: diphone labels
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.diphone_seq_lens = []  # NEW: diphone sequence lengths
        self.days = []

        for day in range(self.n_days):
            for trial in range(len(data[day]["sentenceDat"])):
                self.neural_feats.append(data[day]["sentenceDat"][trial])
                phone_seq = data[day]["phonemes"][trial]
                phone_len = data[day]["phoneLens"][trial]
                self.phone_seqs.append(phone_seq)
                self.neural_time_bins.append(data[day]["sentenceDat"][trial].shape[0])
                self.phone_seq_lens.append(phone_len)
                self.days.append(day)

                # Generate diphone labels if vocabulary provided
                if diphone_vocab is not None:
                    # CRITICAL: Use only the actual phoneme sequence, not the padding!
                    actual_phone_seq = phone_seq[:phone_len]
                    diphone_seq = diphone_vocab.phoneme_to_diphone_labels(actual_phone_seq)
                    self.diphone_seqs.append(diphone_seq)
                    self.diphone_seq_lens.append(len(diphone_seq))
                else:
                    self.diphone_seqs.append(None)
                    self.diphone_seq_lens.append(0)

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        neural_feats = torch.tensor(self.neural_feats[idx], dtype=torch.float32)

        if self.transform:
            neural_feats = self.transform(neural_feats)

        # Get diphone sequence if available
        if self.diphone_seqs[idx] is not None:
            diphone_seq = torch.tensor(self.diphone_seqs[idx], dtype=torch.int32)
            diphone_len = torch.tensor(self.diphone_seq_lens[idx], dtype=torch.int32)
        else:
            # Return empty tensors if diphone vocab not provided
            diphone_seq = torch.tensor([], dtype=torch.int32)
            diphone_len = torch.tensor(0, dtype=torch.int32)

        return (
            neural_feats,
            torch.tensor(self.phone_seqs[idx], dtype=torch.int32),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
            torch.tensor(self.phone_seq_lens[idx], dtype=torch.int32),
            torch.tensor(self.days[idx], dtype=torch.int64),
            diphone_seq,    # NEW
            diphone_len,    # NEW
        )
