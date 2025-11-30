import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pywt


class DaySpecificLinear(nn.Module):
    """
    Per-day affine transform. Mirrors the GRU baseline's dayWeights/dayBias block.
    """

    def __init__(self, n_days: int, dim: int, init_identity: bool = True):
        super().__init__()
        self.dim = dim
        self.day_weights = nn.Parameter(torch.randn(n_days, dim, dim))
        self.day_bias = nn.Parameter(torch.zeros(n_days, 1, dim))
        if init_identity:
            with torch.no_grad():
                for d in range(n_days):
                    self.day_weights[d].copy_(torch.eye(dim))

    def forward(self, x: torch.Tensor, day_ids: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], day_ids: [B]
        day_w = torch.index_select(self.day_weights, 0, day_ids)
        day_b = torch.index_select(self.day_bias, 0, day_ids)
        # einsum keeps batch/time alignment: [B, T, D]
        return torch.einsum("btd,bdk->btk", x, day_w) + day_b


class WaveletFeatureExtractor(nn.Module):
    """
    Fast GPU-based wavelet-inspired feature extractor using learned filter banks.
    Approximates wavelet decomposition with trainable convolutional filters at multiple scales.
    Much faster than pywt.wavedec while maintaining the spirit of MiSTR's approach.
    """

    def __init__(
        self,
        n_channels: int,
        n_scales: int = 4,
        wavelet_name: str = "db4",
        window_size: int = 10,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_scales = n_scales
        self.window_size = window_size

        # Initialize filter banks based on wavelet
        wave = pywt.Wavelet(wavelet_name)

        # Create multi-scale filters inspired by wavelet decomposition
        # Each scale has both high-pass (detail) and low-pass components
        self.filters = nn.ModuleList()

        for scale in range(n_scales):
            # Dilation increases with scale (2^scale)
            dilation = 2 ** scale

            # High-pass filter (detail coefficients)
            hi_filter = torch.tensor(wave.dec_hi, dtype=torch.float32)
            kernel_size = len(hi_filter)

            # Create 1D conv with groups for efficiency
            conv = nn.Conv1d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=(kernel_size - 1) * dilation // 2,
                groups=n_channels,  # Depthwise conv
                bias=False
            )

            # Initialize with wavelet filter (learnable - gradient clipping provides stability)
            with torch.no_grad():
                for i in range(n_channels):
                    conv.weight[i, 0, :] = hi_filter

            self.filters.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]
        returns: [B, T, C * n_scales] - wavelet-inspired features per scale
        """
        b, t, c = x.shape

        # Transpose for conv1d: [B, C, T]
        x_t = x.transpose(1, 2)

        scale_features = []
        for scale_conv in self.filters:
            # Apply filter at this scale
            filtered = scale_conv(x_t)  # [B, C, T]

            # Compute energy (square of coefficients)
            energy = filtered.pow(2)

            # Optional: smooth with small window
            if self.window_size > 1:
                energy = F.avg_pool1d(
                    energy,
                    kernel_size=min(self.window_size, energy.size(-1)),
                    stride=1,
                    padding=self.window_size // 2
                )

            # Ensure same length as input
            if energy.size(-1) != t:
                energy = F.interpolate(energy, size=t, mode='linear', align_corners=False)

            scale_features.append(energy)

        # Concatenate all scales: [B, C * n_scales, T]
        features = torch.cat(scale_features, dim=1)

        # Transpose back: [B, T, C * n_scales]
        features = features.transpose(1, 2)

        return features


class PACFeatureExtractor(nn.Module):
    """
    Optional cross-frequency coupling features using FIR bandpass filters and Hilbert envelope/phase.
    """

    def __init__(
        self,
        n_channels: int,
        low_band: Tuple[float, float] = (4.0, 8.0),
        high_band: Tuple[float, float] = (70.0, 170.0),
        sample_rate: float = 50.0,  # bins per second (20 ms bins)
        filter_len: int = 129,
        smoothing_bins: int = 5,
    ):
        super().__init__()
        import scipy.signal  # local import to avoid hard dependency at module import

        nyq = sample_rate / 2.0
        self.low_kernel = torch.tensor(
            scipy.signal.firwin(filter_len, [low_band[0] / nyq, low_band[1] / nyq], pass_zero=False),
            dtype=torch.float32,
        ).view(1, 1, -1)
        self.high_kernel = torch.tensor(
            scipy.signal.firwin(filter_len, [high_band[0] / nyq, high_band[1] / nyq], pass_zero=False),
            dtype=torch.float32,
        ).view(1, 1, -1)
        self.n_channels = n_channels
        self.filter_len = filter_len
        self.smoothing_bins = smoothing_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]
        returns: [B, T, C] PAC magnitude per channel.
        """
        b, t, c = x.shape
        x_t = x.transpose(1, 2)  # [B, C, T]
        device = x.device
        # Prepare grouped conv weights on the right device/dtype
        low_k = self.low_kernel.to(device=device, dtype=x.dtype).repeat(c, 1, 1)
        high_k = self.high_kernel.to(device=device, dtype=x.dtype).repeat(c, 1, 1)
        pad = self.filter_len // 2
        low = F.conv1d(x_t, low_k, padding=pad, groups=c)
        high = F.conv1d(x_t, high_k, padding=pad, groups=c)

        # Hilbert via FFT for analytic signals
        def analytic(sig: torch.Tensor) -> torch.Tensor:
            # sig: [B, C, T]
            fft = torch.fft.fft(sig, dim=-1)
            h = torch.zeros_like(fft)
            n = sig.shape[-1]
            if n % 2 == 0:
                h[..., 0] = 1
                h[..., n // 2] = 1
                h[..., 1 : n // 2] = 2
            else:
                h[..., 0] = 1
                h[..., 1 : (n + 1) // 2] = 2
            return torch.fft.ifft(fft * h, dim=-1)

        low_analytic = analytic(low)
        high_analytic = analytic(high)
        low_phase = torch.angle(low_analytic)
        high_amp = torch.abs(high_analytic)

        pac_complex = high_amp * torch.exp(1j * low_phase)
        # Smooth over time to stabilize PAC
        if self.smoothing_bins > 1:
            pad_s = self.smoothing_bins // 2
            real = F.avg_pool1d(pac_complex.real, kernel_size=self.smoothing_bins, stride=1, padding=pad_s)
            imag = F.avg_pool1d(pac_complex.imag, kernel_size=self.smoothing_bins, stride=1, padding=pad_s)
            pac_complex = torch.complex(real, imag)
        pac_mag = torch.abs(pac_complex)  # [B, C, T]
        return pac_mag.transpose(1, 2)  # [B, T, C]


class NeuralFrontend(nn.Module):
    """
    Combines raw day-normalized features with wavelet (and optional PAC) features, then projects.
    """

    def __init__(
        self,
        n_channels: int,
        use_wavelets: bool = True,
        n_scales: int = 4,
        wavelet_window_size: int = 10,
        use_pac_features: bool = False,
        pac_low: Tuple[float, float] = (4.0, 8.0),
        pac_high: Tuple[float, float] = (70.0, 170.0),
        pac_filter_len: int = 129,
        pac_smoothing_bins: int = 5,
        frontend_dim: int = 512,
        dropout: float = 0.1,
        temporal_kernel: int = 0,
        temporal_stride: int = 1,  # Add striding like GRU (stride=4)
        gaussian_smooth_width: float = 0.0,  # GRU uses 2.0
    ):
        super().__init__()
        self.use_wavelets = use_wavelets
        self.use_pac_features = use_pac_features
        self.n_channels = n_channels
        self.temporal_kernel = temporal_kernel
        self.temporal_stride = temporal_stride

        # Gaussian smoothing (like GRU preprocessing)
        if gaussian_smooth_width > 0:
            kernel_size = int(gaussian_smooth_width * 4) + 1  # 4 sigma window
            gaussian_kernel = self._make_gaussian_kernel(kernel_size, gaussian_smooth_width)
            self.register_buffer('gaussian_kernel', gaussian_kernel.view(1, 1, -1))
            self.gaussian_padding = kernel_size // 2
        else:
            self.gaussian_kernel = None

        # Strided temporal convolution (like GRU's kernel + stride)
        if temporal_kernel > 0:
            self.temporal_conv = nn.Conv1d(
                n_channels, n_channels,
                kernel_size=temporal_kernel,
                stride=temporal_stride,  # Add stride!
                padding=temporal_kernel // 2,
                groups=n_channels,
                bias=False
            )
            nn.init.constant_(self.temporal_conv.weight, 1.0 / temporal_kernel)
        else:
            self.temporal_conv = None

        if use_wavelets:
            self.wavelet_extractor = WaveletFeatureExtractor(
                n_channels=n_channels,
                n_scales=n_scales,
                wavelet_name="db4",
                window_size=wavelet_window_size,
            )
            wavelet_dim = n_channels * n_scales
        else:
            self.wavelet_extractor = None
            wavelet_dim = 0

        if use_pac_features:
            self.pac_extractor = PACFeatureExtractor(
                n_channels=n_channels,
                low_band=pac_low,
                high_band=pac_high,
                sample_rate=50.0,
                filter_len=pac_filter_len,
                smoothing_bins=pac_smoothing_bins,
            )
            pac_dim = n_channels
        else:
            self.pac_extractor = None
            pac_dim = 0

        input_dim = n_channels + wavelet_dim + pac_dim
        self.proj = nn.Linear(input_dim, frontend_dim)
        self.ln = nn.LayerNorm(frontend_dim)
        self.dropout = nn.Dropout(dropout)

    def _make_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create 1D Gaussian kernel for smoothing"""
        x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply Gaussian smoothing first (like GRU preprocessing)
        if self.gaussian_kernel is not None:
            x_t = x.transpose(1, 2)  # [B, C, T]
            # Apply same Gaussian kernel to all channels
            kernel = self.gaussian_kernel.repeat(self.n_channels, 1, 1)  # [C, 1, K]
            x_smooth = F.conv1d(x_t, kernel, padding=self.gaussian_padding, groups=self.n_channels)
            x = x_smooth.transpose(1, 2)  # [B, T, C]

        # Apply temporal smoothing/striding (like GRU's kernel + stride processing)
        if self.temporal_conv is not None:
            # x: [B, T, C] -> [B, C, T] for conv1d
            x_t = x.transpose(1, 2)
            x_smooth = self.temporal_conv(x_t).transpose(1, 2)  # [B, T', C] where T' = T/stride
            # Compute additional features on the strided output
            x_for_features = x_smooth
        else:
            x_for_features = x

        feats = [x_for_features]

        if self.wavelet_extractor is not None:
            wavelet_feats = self.wavelet_extractor(x_for_features)  # [B, T', C * n_scales]
            feats.append(wavelet_feats)

        if self.pac_extractor is not None:
            pac_feats = self.pac_extractor(x_for_features)  # [B, T', C]
            feats.append(pac_feats)

        # All features now have same temporal dimension [B, T', ...]
        x_cat = torch.cat(feats, dim=-1)
        x_proj = self.proj(x_cat)
        x_proj = self.ln(x_proj)
        x_proj = self.dropout(x_proj)
        return x_proj


class AutoEncoderEncoder(nn.Module):
    """
    Simple MLP encoder (MiSTR-style) used as a bottleneck projection.
    Uses fixed 128 hidden units as in original MiSTR implementation.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerConvModule(nn.Module):
    """
    Convolution module from Conformer architecture.
    Provides local pattern modeling to complement global attention.
    """
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        # Layernorm
        self.ln = nn.LayerNorm(d_model)
        # Pointwise conv (expansion)
        self.pw_conv1 = nn.Linear(d_model, d_model * 2)
        # GLU activation
        self.glu = nn.GLU(dim=-1)
        # Depthwise conv
        self.dw_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.ln_conv = nn.LayerNorm(d_model)  # Use LayerNorm instead of BatchNorm for stability
        self.activation = nn.SiLU()
        # Pointwise conv (projection)
        self.pw_conv2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        residual = x
        x = self.ln(x)
        x = self.pw_conv1(x)  # [B, T, 2D]
        x = self.glu(x)  # [B, T, D]

        # Depthwise conv requires [B, D, T]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.dw_conv(x)  # [B, D, T]
        x = x.transpose(1, 2)  # [B, T, D]
        x = self.ln_conv(x)  # LayerNorm for stability
        x = self.activation(x)

        x = self.pw_conv2(x)
        x = self.dropout(x)
        return residual + x


class ConformerBlock(nn.Module):
    """
    Conformer block: Feed-forward + Multi-head Attention + Convolution + Feed-forward
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        conv_kernel_size: int = 31,
    ):
        super().__init__()
        # First feed-forward module (half-step residual)
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Multi-head self-attention
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout)

        # Convolution module
        self.conv_module = ConformerConvModule(d_model, conv_kernel_size, dropout)

        # Second feed-forward module (half-step residual)
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.ln_final = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, D]
        # First FF module (half-step)
        x = x + 0.5 * self.ff1(x)

        # Multi-head attention
        x_attn = self.ln_attn(x)
        attn_out, _ = self.attn(x_attn, x_attn, x_attn, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout_attn(attn_out)

        # Convolution module
        x = self.conv_module(x)

        # Second FF module (half-step)
        x = x + 0.5 * self.ff2(x)

        x = self.ln_final(x)
        return x


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (proven to work well).
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.pe[:, : x.size(1)]


class NeuralTransformerCTCModel(nn.Module):
    """
    MiSTR-inspired Transformer encoder for phoneme CTC.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        n_days: int,
        use_wavelets: bool = True,
        n_scales: int = 4,
        wavelet_window_size: int = 10,
        use_pac_features: bool = False,
        frontend_dim: int = 512,
        latent_dim: int = 256,
        autoencoder_hidden_dim: int = 128,
        transformer_layers: int = 6,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 1024,
        transformer_dropout: float = 0.1,
        temporal_kernel: int = 0,  # Temporal convolution kernel size
        temporal_stride: int = 1,  # Temporal convolution stride
        gaussian_smooth_width: float = 0.0,  # GRU uses 2.0
        use_conformer: bool = False,  # Use Conformer blocks instead of standard transformer
        conformer_conv_kernel: int = 31,  # Conformer convolution kernel size
        device: str = "cuda",
    ):
        super().__init__()
        self.device_name = device
        self.use_conformer = use_conformer
        self.day_linear = DaySpecificLinear(n_days=n_days, dim=n_channels, init_identity=True)
        self.frontend = NeuralFrontend(
            n_channels=n_channels,
            use_wavelets=use_wavelets,
            n_scales=n_scales,
            wavelet_window_size=wavelet_window_size,
            use_pac_features=use_pac_features,
            frontend_dim=frontend_dim,
            temporal_kernel=temporal_kernel,
            temporal_stride=temporal_stride,
            gaussian_smooth_width=gaussian_smooth_width,
        )
        self.encoder = AutoEncoderEncoder(
            input_dim=frontend_dim, latent_dim=latent_dim, hidden_dim=autoencoder_hidden_dim
        )
        self.pos_enc = PositionalEncoding(d_model=latent_dim)

        if use_conformer:
            # Use Conformer blocks
            self.conformer_layers = nn.ModuleList([
                ConformerBlock(
                    d_model=latent_dim,
                    nhead=transformer_heads,
                    dim_feedforward=transformer_ff_dim,
                    dropout=transformer_dropout,
                    conv_kernel_size=conformer_conv_kernel,
                )
                for _ in range(transformer_layers)
            ])
            self.transformer = None
        else:
            # Use standard transformer
            enc_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_ff_dim,
                dropout=transformer_dropout,
                batch_first=True,
                activation="relu",
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
            self.conformer_layers = None

        self.output = nn.Linear(latent_dim, n_classes)

    def compute_output_lengths(self, input_lengths: torch.Tensor, actual_seq_len: int) -> torch.Tensor:
        # Account for temporal striding in frontend (if used)
        if hasattr(self.frontend, 'temporal_conv') and self.frontend.temporal_conv is not None:
            # Apply same formula as GRU: (length - kernel) / stride
            kernel = self.frontend.temporal_kernel
            stride = self.frontend.temporal_stride
            output_lengths = ((input_lengths - kernel) / stride).to(torch.int32)
        else:
            # No striding, temporal dimension preserved
            output_lengths = input_lengths

        # Clamp to actual sequence length
        return torch.clamp(output_lengths, max=actual_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        day_ids: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, C]
        x = self.day_linear(x, day_ids)
        feats = self.frontend(x)
        z = self.encoder(feats)
        z = self.pos_enc(z)

        # Get actual sequence length after frontend processing
        actual_seq_len = z.size(1)

        padding_mask = None
        if input_lengths is not None:
            # Update input_lengths to reflect actual output size
            out_lengths = self.compute_output_lengths(input_lengths, actual_seq_len)
            max_len = z.size(1)
            mask = torch.arange(max_len, device=z.device).expand(len(out_lengths), max_len) >= out_lengths.unsqueeze(1)
            padding_mask = mask  # [B, T]
        else:
            out_lengths = None

        if self.use_conformer:
            # Apply Conformer blocks
            for layer in self.conformer_layers:
                z = layer(z, src_key_padding_mask=padding_mask)
            z_enc = z
        else:
            # Use standard transformer
            z_enc = self.transformer(z, src_key_padding_mask=padding_mask)

        logits = self.output(z_enc)  # [B, T, C]
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # [T, B, C]
        return log_probs, out_lengths
