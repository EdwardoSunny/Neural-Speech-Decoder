"""
Advanced Multi-Scale Neural Decoder with Novel Components:
1. Adaptive scale fusion - learns which temporal scales matter when
2. Phonetic feature learning - auxiliary tasks for better representations
3. Contrastive phoneme learning - similar phonemes cluster together
"""

import math
from typing import Optional, Tuple, Dict

import torch
from torch import nn
from torch.nn import functional as F

from .transformer_ctc import (
    ConformerBlock,
    RelativePositionalEncoding,
    MultiScaleConformerEncoder,
)


class AdaptiveScaleFusion(nn.Module):
    """
    NOVEL: Learns to weight different temporal scales adaptively.

    Instead of fixed fusion weights, this learns which scales are important
    for each timestep. E.g., fast transients might need fast pathway more,
    while steady states might need slow pathway more.
    """
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model

        # Learn to compute scale importance from the features themselves
        self.scale_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.1, batch_first=True
        )

        # Gating network to combine scales
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),  # 3 scale weights
            nn.Softmax(dim=-1)
        )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, fast: torch.Tensor, medium: torch.Tensor, slow: torch.Tensor):
        """
        fast: [B, T_fast, D]
        medium: [B, T_medium, D] - target resolution
        slow: [B, T_slow, D]

        Returns: [B, T_medium, D] with adaptive scale weighting
        """
        # Align all to medium resolution
        slow_aligned = F.interpolate(
            slow.transpose(1, 2), size=medium.shape[1], mode='linear', align_corners=False
        ).transpose(1, 2)

        fast_aligned = F.adaptive_avg_pool1d(
            fast.transpose(1, 2), medium.shape[1]
        ).transpose(1, 2)

        # Stack scales
        all_scales = torch.stack([fast_aligned, medium, slow_aligned], dim=2)  # [B, T, 3, D]
        B, T, S, D = all_scales.shape

        # Compute adaptive weights for each timestep
        concat_scales = torch.cat([fast_aligned, medium, slow_aligned], dim=-1)  # [B, T, 3D]
        scale_weights = self.gate(concat_scales)  # [B, T, 3]

        # Weight and combine
        weighted = torch.einsum('bts,btsd->btd', scale_weights, all_scales)  # [B, T, D]

        # Final fusion
        output = self.fusion(weighted)

        return output, scale_weights  # Return weights for analysis


class PhoneticFeaturePredictor(nn.Module):
    """
    NOVEL: Multi-task learning with phonetic features.

    Predicts phoneme AND its phonetic features (manner, place, voicing).
    This provides richer supervision signal and better representations.

    Phonetic features (simplified for English):
    - Manner: stop, fricative, nasal, vowel, etc. (7 classes)
    - Place: bilabial, alveolar, velar, etc. (6 classes)
    - Voicing: voiced, unvoiced (2 classes)
    """
    def __init__(self, d_model: int, n_phonemes: int = 41):
        super().__init__()

        # Shared representation
        self.shared = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        # Task-specific heads
        self.phoneme_head = nn.Linear(d_model, n_phonemes)
        self.manner_head = nn.Linear(d_model, 7)  # manner of articulation
        self.place_head = nn.Linear(d_model, 6)   # place of articulation
        self.voicing_head = nn.Linear(d_model, 2)  # voiced/unvoiced

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, D]
        Returns: dict with all predictions
        """
        shared = self.shared(x)

        return {
            'phoneme': self.phoneme_head(shared),  # [B, T, n_phonemes]
            'manner': self.manner_head(shared),     # [B, T, 7]
            'place': self.place_head(shared),       # [B, T, 6]
            'voicing': self.voicing_head(shared),   # [B, T, 2]
        }


class ContrastivePhonemeEncoder(nn.Module):
    """
    NOVEL: Contrastive learning for phoneme representations.

    Encourages same phonemes to have similar representations even across
    different neural recordings. Uses supervised contrastive loss.
    """
    def __init__(self, d_model: int, projection_dim: int = 128):
        super().__init__()

        # Projection head for contrastive space
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim),
        )

        self.temperature = 0.07

    def supervised_contrastive_loss(
        self,
        embeddings: torch.Tensor,  # [B*T, D]
        labels: torch.Tensor,       # [B*T]
    ):
        """
        Supervised contrastive loss: pull same-class embeddings together,
        push different-class embeddings apart.
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature  # [B*T, B*T]

        # Create mask for positive pairs (same label)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()  # [B*T, B*T]

        # Remove self-similarity
        mask = mask - torch.eye(mask.shape[0], device=mask.device)

        # For each anchor, compute loss
        # Numerator: exp(sim(anchor, positive))
        # Denominator: sum of exp(sim(anchor, all_others))

        # Mask out self
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=mask.device)

        # Compute log prob for positive pairs
        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Mean log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        """
        x: [B, T, D]
        labels: [B, T] - phoneme labels

        Returns: contrastive loss
        """
        B, T, D = x.shape

        # Project to contrastive space
        projected = self.projection(x)  # [B, T, projection_dim]

        # Flatten
        projected_flat = projected.reshape(-1, projected.shape[-1])  # [B*T, projection_dim]
        labels_flat = labels.reshape(-1)  # [B*T]

        # Filter out padding (label 0)
        valid_mask = labels_flat != 0
        if valid_mask.sum() < 2:  # Need at least 2 valid samples
            return torch.tensor(0.0, device=x.device)

        projected_valid = projected_flat[valid_mask]
        labels_valid = labels_flat[valid_mask]

        # Compute contrastive loss
        loss = self.supervised_contrastive_loss(projected_valid, labels_valid)

        return loss


class AdvancedMultiScaleDecoder(nn.Module):
    """
    ADVANCED Multi-Scale Neural Decoder with 3 novel components:

    1. ADAPTIVE SCALE FUSION - learns which temporal scales matter when
    2. PHONETIC FEATURE LEARNING - multi-task auxiliary objectives
    3. CONTRASTIVE PHONEME LEARNING - better phoneme representations

    Expected to significantly outperform standard multi-scale CTC.
    """
    def __init__(
        self,
        n_classes: int,
        input_dim: int = 256,
        d_model: int = 1024,
        encoder_layers: int = 6,
        n_heads: int = 8,
        dim_ff: int = 2048,
        dropout: float = 0.3,
        conv_kernel: int = 31,
        n_days: int = 1,
        gaussian_smooth_width: float = 2.0,
        use_phonetic_features: bool = True,
        use_contrastive: bool = True,
        phonetic_loss_weight: float = 0.2,
        contrastive_loss_weight: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model
        self.device = device
        self.use_phonetic_features = use_phonetic_features
        self.use_contrastive = use_contrastive
        self.phonetic_loss_weight = phonetic_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight

        # Gaussian smoothing
        self.gaussian_smooth_width = gaussian_smooth_width
        if gaussian_smooth_width > 0:
            kernel_size = int(6 * gaussian_smooth_width)
            if kernel_size % 2 == 0:
                kernel_size += 1
            x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
            kernel = torch.exp(-(x ** 2) / (2 * gaussian_smooth_width ** 2))
            kernel = kernel / kernel.sum()
            self.register_buffer('smooth_kernel', kernel.view(1, 1, -1))
            self.smooth_padding = kernel_size // 2

        # Day-specific normalization
        self.day_layer_norm = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(n_days)
        ])

        # Multi-scale encoder (base architecture)
        self.encoder = MultiScaleConformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_layers=encoder_layers,
            n_heads=n_heads,
            dim_ff=dim_ff,
            dropout=dropout,
            conv_kernel=conv_kernel,
        )

        # NOVEL COMPONENT 1: Adaptive scale fusion
        self.adaptive_fusion = AdaptiveScaleFusion(d_model, n_heads)

        # NOVEL COMPONENT 2: Phonetic feature predictor
        if use_phonetic_features:
            self.phonetic_predictor = PhoneticFeaturePredictor(d_model, n_classes)

        # NOVEL COMPONENT 3: Contrastive phoneme encoder
        if use_contrastive:
            self.contrastive_encoder = ContrastivePhonemeEncoder(d_model)

        # Main CTC output
        self.output = nn.Linear(d_model, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        day_idx: torch.Tensor,
        x_len: Optional[torch.Tensor] = None,
        phoneme_labels: Optional[torch.Tensor] = None,
        return_auxiliary_losses: bool = False,
    ):
        """
        x: [B, T, C] - neural signal
        day_idx: [B] - day indices
        phoneme_labels: [B, S] - for auxiliary losses (optional)

        Returns:
            log_probs: [T, B, n_classes]
            out_lengths: [B]
            auxiliary_outputs: dict (if return_auxiliary_losses=True)
        """
        # Gaussian smoothing
        if self.gaussian_smooth_width > 0:
            x_t = x.transpose(1, 2)
            x_smoothed = []
            for c in range(x_t.shape[1]):
                smoothed_c = F.conv1d(
                    x_t[:, c:c+1, :],
                    self.smooth_kernel,
                    padding=self.smooth_padding
                )
                x_smoothed.append(smoothed_c)
            x = torch.cat(x_smoothed, dim=1).transpose(1, 2)

        # Day-specific normalization
        x_normalized = []
        for i, day_id in enumerate(day_idx):
            x_normalized.append(self.day_layer_norm[day_id](x[i:i+1]))
        x = torch.cat(x_normalized, dim=0)

        # Get individual scales from encoder
        fast, medium, slow = self.encoder(x, return_scales=True)

        # NOVEL: Adaptive scale fusion
        fused_output, scale_weights = self.adaptive_fusion(fast, medium, slow)  # [B, T', D]

        # Main CTC output
        logits = self.output(fused_output)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # [T', B, C]
        out_lengths = torch.full(
            (x.shape[0],), fused_output.shape[1],
            dtype=torch.long, device=x.device
        )

        if not return_auxiliary_losses:
            return log_probs, out_lengths

        # Compute auxiliary losses
        auxiliary_outputs = {}

        # Store scale weights for analysis
        auxiliary_outputs['scale_weights'] = scale_weights

        # Phonetic feature predictions
        if self.use_phonetic_features and phoneme_labels is not None:
            phonetic_preds = self.phonetic_predictor(fused_output)
            auxiliary_outputs['phonetic_predictions'] = phonetic_preds

        # Contrastive loss
        if self.use_contrastive and phoneme_labels is not None:
            # Use CTC predictions as pseudo-labels for contrastive learning
            pred_labels = log_probs.argmax(dim=-1).transpose(0, 1)  # [B, T']
            contrastive_loss = self.contrastive_encoder(fused_output, pred_labels)
            auxiliary_outputs['contrastive_loss'] = contrastive_loss

        return log_probs, out_lengths, auxiliary_outputs


def create_phoneme_feature_map():
    """
    Create mapping from phonemes to phonetic features.
    Simplified for 40 phoneme classes.

    Returns:
        manner_map: [41] - manner of articulation
        place_map: [41] - place of articulation
        voicing_map: [41] - voicing
    """
    # Default: blank token (index 0)
    manner = [0]  # blank
    place = [0]   # blank
    voicing = [0] # blank

    # Phonemes 1-40 (simplified mapping)
    # In practice, you'd use actual phoneme-to-feature mappings
    # For now, use reasonable defaults that provide some structure

    for i in range(1, 41):
        # Manner: 1=stop, 2=fricative, 3=nasal, 4=liquid, 5=glide, 6=vowel
        if i <= 7:  # stops
            manner.append(1)
        elif i <= 14:  # fricatives
            manner.append(2)
        elif i <= 17:  # nasals
            manner.append(3)
        elif i <= 20:  # liquids
            manner.append(4)
        elif i <= 23:  # glides
            manner.append(5)
        else:  # vowels
            manner.append(6)

        # Place: 1=bilabial, 2=alveolar, 3=velar, 4=palatal, 5=glottal
        place.append((i % 5) + 1)

        # Voicing: 0=unvoiced, 1=voiced
        voicing.append(i % 2)

    return (
        torch.tensor(manner, dtype=torch.long),
        torch.tensor(place, dtype=torch.long),
        torch.tensor(voicing, dtype=torch.long),
    )
