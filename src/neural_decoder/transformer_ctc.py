import math
from typing import Optional, Tuple

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


def _build_wavelet_filters(
    wavelet_name: str, n_scales: int, n_channels: int, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, list]:
    wave = pywt.Wavelet(wavelet_name)
    base_filter = torch.tensor(wave.dec_hi[::-1], device=device, dtype=dtype)  # high-pass reconstruction kernel
    filters = []
    dilations = []
    for s in range(n_scales):
        dilation = 2**s
        dilations.append(dilation)
        filt = base_filter.view(1, 1, -1).repeat(n_channels, 1, 1)  # [C, 1, K]
        filters.append(filt)
    stacked = torch.stack(filters, dim=0)  # [S, C, 1, K]
    return stacked, dilations


class WaveletFeatureExtractor(nn.Module):
    """
    Computes per-scale wavelet energy features with db4 filters and optional temporal smoothing.
    """

    def __init__(
        self,
        n_channels: int,
        n_scales: int = 4,
        window_bins: int = 5,
        stride_bins: int = 1,
        wavelet_name: str = "db4",
    ):
        super().__init__()
        if stride_bins != 1:
            raise ValueError("wavelet_stride_bins must be 1 to keep sequence lengths aligned with raw features.")
        self.n_channels = n_channels
        self.n_scales = n_scales
        self.window_bins = window_bins
        self.stride_bins = stride_bins
        # buffers initialized lazily because device/dtype are unknown here
        self.register_buffer("filters", None, persistent=False)
        self.dilations = None
        self.wavelet_name = wavelet_name

    def _lazy_init_filters(self, x: torch.Tensor):
        if self.filters is None:
            filt, dilations = _build_wavelet_filters(
                self.wavelet_name, self.n_scales, self.n_channels, x.device, x.dtype
            )
            self.register_buffer("filters", filt, persistent=False)
            self.dilations = dilations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]
        returns: [B, T_out, C * n_scales]
        """
        self._lazy_init_filters(x)
        b, t, c = x.shape
        x_t = x.transpose(1, 2)  # [B, C, T]
        feats = []
        for s in range(self.n_scales):
            filt = self.filters[s]
            dilation = self.dilations[s]
            padding = ((filt.shape[-1] - 1) * dilation) // 2
            conv = F.conv1d(x_t, filt, padding=padding, stride=self.stride_bins, groups=c, dilation=dilation)
            energy = conv.pow(2)
            if self.window_bins > 1:
                # smooth energy to reduce noise, keep stride at 1 in time dimension of conv output
                pad = self.window_bins // 2
                energy = F.avg_pool1d(energy, kernel_size=self.window_bins, stride=1, padding=pad)
            feats.append(energy)

        # feats: list of [B, C, T_out]; concat on channel dim -> [B, C*S, T_out] then transpose back
        feat_cat = torch.cat(feats, dim=1).transpose(1, 2)
        return feat_cat


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
        wavelet_window_bins: int = 5,
        wavelet_stride_bins: int = 1,
        use_pac_features: bool = False,
        pac_low: Tuple[float, float] = (4.0, 8.0),
        pac_high: Tuple[float, float] = (70.0, 170.0),
        pac_filter_len: int = 129,
        pac_smoothing_bins: int = 5,
        frontend_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_wavelets = use_wavelets
        self.use_pac_features = use_pac_features
        self.n_channels = n_channels

        if use_wavelets:
            self.wavelet_extractor = WaveletFeatureExtractor(
                n_channels=n_channels,
                n_scales=n_scales,
                window_bins=wavelet_window_bins,
                stride_bins=wavelet_stride_bins,
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
                sample_rate=50.0,  # 20 ms bins
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        if self.wavelet_extractor is not None:
            feats.append(self.wavelet_extractor(x))
        if self.pac_extractor is not None:
            feats.append(self.pac_extractor(x))
        x_cat = torch.cat(feats, dim=-1)
        x_proj = self.proj(x_cat)
        x_proj = self.ln(x_proj)
        x_proj = self.dropout(x_proj)
        return x_proj


class AutoEncoderEncoder(nn.Module):
    """
    Simple MLP encoder (MiSTR-style) used as a bottleneck projection.
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        hidden_dim = max(latent_dim * 2, input_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionalEncoding(nn.Module):
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
        wavelet_window_bins: int = 5,
        wavelet_stride_bins: int = 1,
        use_pac_features: bool = False,
        frontend_dim: int = 512,
        latent_dim: int = 256,
        transformer_layers: int = 6,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 1024,
        transformer_dropout: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.device_name = device
        self.day_linear = DaySpecificLinear(n_days=n_days, dim=n_channels, init_identity=True)
        self.frontend = NeuralFrontend(
            n_channels=n_channels,
            use_wavelets=use_wavelets,
            n_scales=n_scales,
            wavelet_window_bins=wavelet_window_bins,
            wavelet_stride_bins=wavelet_stride_bins,
            use_pac_features=use_pac_features,
            frontend_dim=frontend_dim,
        )
        self.encoder = AutoEncoderEncoder(input_dim=frontend_dim, latent_dim=latent_dim)
        self.pos_enc = PositionalEncoding(d_model=latent_dim)

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
        self.output = nn.Linear(latent_dim, n_classes)

    def compute_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        # Frontend uses stride 1 by default; lengths unchanged.
        return input_lengths

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

        padding_mask = None
        if input_lengths is not None:
            max_len = z.size(1)
            mask = torch.arange(max_len, device=z.device).expand(len(input_lengths), max_len) >= input_lengths.unsqueeze(1)
            padding_mask = mask  # [B, T]

        z_enc = self.transformer(z, src_key_padding_mask=padding_mask)
        logits = self.output(z_enc)  # [B, T, C]
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # [T, B, C]
        out_lengths = self.compute_output_lengths(input_lengths) if input_lengths is not None else None
        return log_probs, out_lengths
