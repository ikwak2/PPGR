from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from torchaudio.functional import lfilter

FS = 128
EMBED_DIM = 192
DROPOUT = 0.45
GATE_RANK = 32
GATE_CONTEXT_DIM = 16
CONTEXT_NORM_CHOICES = ("layernorm", "batchnorm")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    modalities: Tuple[str, ...]
    context_features: Tuple[str, ...]
    gated: bool
    encoder_type: str = "ecapa"
    embedding_dim: int = EMBED_DIM
    fusion_type: str = "late"
    training_objective: str = "aam_softmax"
    temperature_correction: str = "none"
    result_subdir: str | None = None


MODEL_SPECS: Dict[str, ModelSpec] = {
    "baseline": ModelSpec(
        name="baseline",
        modalities=("ppg", "temp", "acc"),
        context_features=(),
        gated=False,
    ),
    "ppg": ModelSpec(
        name="ppg",
        modalities=("ppg",),
        context_features=(),
        gated=False,
    ),
    "ppg_acc": ModelSpec(
        name="ppg_acc",
        modalities=("ppg", "acc"),
        context_features=(),
        gated=False,
    ),
    "cornet_ppg_acc": ModelSpec(
        name="cornet_ppg_acc",
        modalities=("ppg", "acc"),
        context_features=(),
        gated=False,
        encoder_type="cornet",
        result_subdir="encoder_comparison/cornet/ppg_acc",
    ),
    "cornet_ppg_acc_temp_mlp": ModelSpec(
        name="cornet_ppg_acc_temp_mlp",
        modalities=("ppg", "acc"),
        context_features=(),
        gated=False,
        encoder_type="cornet",
        temperature_correction="compact_mlp_residual",
        result_subdir="encoder_comparison/cornet/ppg_acc_temp_mlp",
    ),
    "cornet_ppg_acc_gate_temp_mlp": ModelSpec(
        name="cornet_ppg_acc_gate_temp_mlp",
        modalities=("ppg", "acc"),
        context_features=(
            "ppg_low_high_log_power_ratio",
            "acc_low_power",
            "acc_high_power",
            "ppg_acc_corr",
        ),
        gated=True,
        encoder_type="cornet",
        temperature_correction="compact_mlp_residual",
        result_subdir=(
            "encoder_comparison/cornet/"
            "ppg_acc_temp_mlp_residual_verification_gate"
        ),
    ),
    "ndss_bilstm_attention_ppg_acc": ModelSpec(
        name="ndss_bilstm_attention_ppg_acc",
        modalities=("ppg", "acc"),
        context_features=(),
        gated=False,
        encoder_type="ndss_bilstm_attention",
        result_subdir="encoder_comparison/ndss_bilstm_attention/ppg_acc",
    ),
    "ndss_bilstm_attention_ppg_acc_temp_mlp": ModelSpec(
        name="ndss_bilstm_attention_ppg_acc_temp_mlp",
        modalities=("ppg", "acc"),
        context_features=(),
        gated=False,
        encoder_type="ndss_bilstm_attention",
        temperature_correction="compact_mlp_residual",
        result_subdir=(
            "encoder_comparison/ndss_bilstm_attention/ppg_acc_temp_mlp"
        ),
    ),
    "ndss_bilstm_attention_ppg_acc_gate_temp_mlp": ModelSpec(
        name="ndss_bilstm_attention_ppg_acc_gate_temp_mlp",
        modalities=("ppg", "acc"),
        context_features=(
            "ppg_low_high_log_power_ratio",
            "acc_low_power",
            "acc_high_power",
            "ppg_acc_corr",
        ),
        gated=True,
        encoder_type="ndss_bilstm_attention",
        temperature_correction="compact_mlp_residual",
        result_subdir=(
            "encoder_comparison/ndss_bilstm_attention/"
            "ppg_acc_temp_mlp_residual_verification_gate"
        ),
    ),
    "ppg_acc_gate": ModelSpec(
        name="ppg_acc_gate",
        modalities=("ppg", "acc"),
        context_features=(
            "ppg_low_high_log_power_ratio",
            "acc_low_power",
            "acc_high_power",
            "ppg_acc_corr",
        ),
        gated=True,
    ),
    "ppg_acc_temp_mlp": ModelSpec(
        name="ppg_acc_temp_mlp",
        modalities=("ppg", "acc"),
        context_features=(),
        gated=False,
        temperature_correction="compact_mlp_residual",
    ),
    "ppg_acc_temp_ecapa_gate": ModelSpec(
        name="ppg_acc_temp_ecapa_gate",
        modalities=("ppg", "temp", "acc"),
        context_features=(
            "ppg_low_high_log_power_ratio",
            "acc_low_power",
            "acc_high_power",
            "ppg_acc_corr",
        ),
        gated=True,
    ),
    "ppg_acc_gate_temp_mlp": ModelSpec(
        name="ppg_acc_gate_temp_mlp",
        modalities=("ppg", "acc"),
        context_features=(
            "ppg_low_high_log_power_ratio",
            "acc_low_power",
            "acc_high_power",
            "ppg_acc_corr",
        ),
        gated=True,
        temperature_correction="compact_mlp_residual",
    ),
    "ppg_temp": ModelSpec(
        name="ppg_temp",
        modalities=("ppg", "temp"),
        context_features=(),
        gated=False,
    ),
}


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = x.shape
        weights = self.avg_pool(x).view(batch, channels)
        weights = self.fc(weights).view(batch, channels, 1)
        return x * weights


class TDNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        padding: int | None = None,
    ):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(F.relu(self.conv(x)))


class Res2NetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int = 8,
        dilation: int = 1,
    ):
        super().__init__()
        del in_channels
        self.width = out_channels // scale
        self.scale = scale
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.width,
                    self.width,
                    3,
                    dilation=dilation,
                    padding=dilation,
                )
                for _ in range(scale - 1)
            ]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(self.width) for _ in range(scale - 1)]
        )
        self.se = SEBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(x, self.width, dim=1)
        running = chunks[0]
        outputs = [chunks[0]]
        for index in range(self.scale - 1):
            residual = running if index > 0 else 0
            running = F.relu(
                self.bns[index](self.convs[index](chunks[index + 1] + residual))
            )
            outputs.append(running)
        return self.se(torch.cat(outputs, dim=1)) + x


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels: int, attention_channels: int = 128):
        super().__init__()
        self.tdnn = nn.Conv1d(channels, attention_channels, 1)
        self.conv = nn.Conv1d(attention_channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.conv(torch.tanh(self.tdnn(x))), dim=2)
        mean = torch.sum(x * weights, dim=2)
        variance = torch.clamp(
            torch.sum(x.square() * weights, dim=2) - mean.square(),
            min=1e-7,
        )
        return torch.cat((mean, torch.sqrt(variance)), dim=1)


class ECAPATDNN1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int] = (256, 256, 256, 256, 768),
        lin_neurons: int = EMBED_DIM,
        dropout_p: float = DROPOUT,
    ):
        super().__init__()
        self.layer1 = TDNNBlock(in_channels, channels[0], 5, 1)
        self.layer2 = Res2NetBlock(channels[0], channels[1], dilation=2)
        self.layer3 = Res2NetBlock(channels[1], channels[2], dilation=3)
        self.layer4 = Res2NetBlock(channels[2], channels[3], dilation=4)
        self.layer5 = TDNNBlock(channels[1] * 3, channels[4], 1, 1)
        self.asp = AttentiveStatisticsPooling(channels[4])
        self.bn_asp = nn.BatchNorm1d(channels[4] * 2)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(channels[4] * 2, lin_neurons)
        self.bn_final = nn.BatchNorm1d(lin_neurons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(torch.cat((x2, x3, x4), dim=1))
        pooled = self.bn_asp(self.asp(x5).unsqueeze(2)).squeeze(2)
        return self.bn_final(self.fc(self.dropout(pooled)))


def build_encoder(
    encoder_type: str,
    in_channels: int,
    embed_dim: int,
    dropout_p: float,
) -> nn.Module:
    """Load optional comparison encoders only when explicitly selected."""
    if encoder_type == "ecapa":
        return ECAPATDNN1D(
            in_channels,
            lin_neurons=embed_dim,
            dropout_p=dropout_p,
        )
    if encoder_type == "cornet":
        from cornet_backbone import CorNETEncoder

        return CorNETEncoder(
            in_channels=in_channels,
            embedding_dim=embed_dim,
        )
    if encoder_type == "ndss_bilstm_attention":
        from ndss_bilstm_attention_backbone import NDSSBiLSTMAttentionEncoder

        return NDSSBiLSTMAttentionEncoder(
            in_channels=in_channels,
            embedding_dim=embed_dim,
        )
    raise ValueError(f"Unknown encoder_type={encoder_type!r}")


def model_result_subdir(variant: str) -> Path:
    if variant not in MODEL_SPECS:
        raise ValueError(f"Unknown variant {variant!r}")
    return Path(MODEL_SPECS[variant].result_subdir or variant)


def encoder_config_payload(encoder_type: str) -> dict:
    """Keep ECAPA metadata independent of optional comparison modules."""
    if encoder_type == "cornet":
        from cornet_backbone import cornet_config_payload

        return cornet_config_payload()
    if encoder_type == "ndss_bilstm_attention":
        from ndss_bilstm_attention_backbone import (
            ndss_bilstm_attention_config_payload,
        )

        return ndss_bilstm_attention_config_payload()
    return {"encoder_type": encoder_type}


class ContextFeatureGate(nn.Module):
    """Identity-initialized signed Gate driven by fixed context features."""

    def __init__(
        self,
        spec: ModelSpec,
        embed_dim: int = EMBED_DIM,
        context_dim: int = GATE_CONTEXT_DIM,
        gate_rank: int = GATE_RANK,
        max_delta: float = 0.5,
        fs: int = FS,
        context_norm: str = "layernorm",
    ):
        super().__init__()
        if not spec.gated:
            raise ValueError(f"{spec.name} is not a gated model")
        if context_norm not in CONTEXT_NORM_CHOICES:
            raise ValueError(
                f"Unknown context norm {context_norm!r}; "
                f"choose from {CONTEXT_NORM_CHOICES}"
            )
        self.spec = spec
        self.embed_dim = embed_dim
        self.gate_rank = gate_rank
        self.max_delta = float(max_delta)
        self.fs = fs

        n_context = len(spec.context_features)
        n_modalities = len(spec.modalities)
        if n_context == 1:
            # LayerNorm(1) maps every scalar context to a constant. Running-stat
            # BatchNorm keeps PPG LH-LPR sample variation at train and evaluation.
            self.context_norm: nn.Module = nn.BatchNorm1d(
                1, affine=True, track_running_stats=True
            )
        elif context_norm == "batchnorm":
            # Normalize each physical feature independently across the batch.
            # This preserves the identities of differently scaled features,
            # unlike per-sample LayerNorm across the context-feature axis.
            self.context_norm = nn.BatchNorm1d(
                n_context, affine=True, track_running_stats=True
            )
        else:
            self.context_norm = nn.LayerNorm(n_context)

        self.gate_code = nn.Sequential(
            nn.Linear(n_context, context_dim),
            nn.ReLU(inplace=True),
            nn.Linear(context_dim, gate_rank * n_modalities),
        )
        self.gate_expand = nn.ModuleDict(
            {
                modality: nn.Linear(gate_rank, embed_dim)
                for modality in spec.modalities
            }
        )
        for expansion in self.gate_expand.values():
            nn.init.zeros_(expansion.weight)
            nn.init.zeros_(expansion.bias)

        if "acc" in spec.modalities:
            low_sos = signal.butter(
                4,
                2.0 / (0.5 * fs),
                btype="lowpass",
                output="sos",
            )
            high_sos = signal.butter(
                4,
                [2.0 / (0.5 * fs), 5.0 / (0.5 * fs)],
                btype="bandpass",
                output="sos",
            )
            self.register_buffer(
                "acc_low_sos", torch.tensor(low_sos, dtype=torch.float32)
            )
            self.register_buffer(
                "acc_high_sos", torch.tensor(high_sos, dtype=torch.float32)
            )

    @staticmethod
    def _safe_band_power(
        x: torch.Tensor,
        frequencies: torch.Tensor,
        low: float,
        high: float,
    ) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=-1)
        power = spectrum.real.square() + spectrum.imag.square()
        mask = (frequencies >= low) & (frequencies < high)
        if not bool(mask.any()):
            return torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        return power[..., mask].mean(dim=-1).mean(dim=1, keepdim=True)

    @staticmethod
    def _sos_filter(x: torch.Tensor, sos: torch.Tensor) -> torch.Tensor:
        """Apply an SOS cascade without a Python loop over signal samples."""
        output = x
        for section in sos:
            b0, b1, b2, a0, a1, a2 = section
            b_coeffs = torch.stack((b0, b1, b2)) / a0
            a_coeffs = torch.stack(
                (torch.ones_like(a0), a1 / a0, a2 / a0)
            )
            output = lfilter(
                output,
                a_coeffs,
                b_coeffs,
                clamp=False,
                batching=False,
            )
        return output

    def _butter_power(self, x: torch.Tensor, sos: torch.Tensor) -> torch.Tensor:
        filtered = self._sos_filter(sos=sos.to(x), x=x)
        return filtered.square().mean(dim=-1).mean(dim=1, keepdim=True)

    @staticmethod
    def _pearson_ppg_acc(
        ppg: torch.Tensor, acc: torch.Tensor
    ) -> torch.Tensor:
        ppg_signal = ppg.squeeze(1)
        acc_magnitude = torch.sqrt(acc.square().sum(dim=1) + 1e-8)
        ppg_centered = ppg_signal - ppg_signal.mean(dim=1, keepdim=True)
        acc_centered = acc_magnitude - acc_magnitude.mean(dim=1, keepdim=True)
        denominator = torch.sqrt(
            ppg_centered.square().sum(dim=1, keepdim=True)
            * acc_centered.square().sum(dim=1, keepdim=True)
            + 1e-8
        )
        correlation = (
            (ppg_centered * acc_centered).sum(dim=1, keepdim=True)
            / denominator
        )
        return torch.clamp(correlation, -1.0, 1.0)

    def _context(
        self,
        ppg: torch.Tensor,
        temp: torch.Tensor,
        acc: torch.Tensor,
    ) -> torch.Tensor:
        _, _, n_samples = ppg.shape
        with torch.no_grad():
            frequencies = torch.fft.rfftfreq(
                n_samples, d=1.0 / self.fs, device=ppg.device
            ).to(ppg.dtype)
            values: Dict[str, torch.Tensor] = {}
            low_band_power = self._safe_band_power(
                ppg, frequencies, 0.5, 2.5
            )
            high_band_power = self._safe_band_power(
                ppg, frequencies, 2.5, 8.0
            )
            values["ppg_low_high_log_power_ratio"] = torch.log(
                (low_band_power + 1e-8) / (high_band_power + 1e-8)
            )

            if "acc" in self.spec.modalities:
                acc_low_power = self._butter_power(
                    acc, self.acc_low_sos
                )
                acc_high_power = self._butter_power(
                    acc, self.acc_high_sos
                )
                correlation = self._pearson_ppg_acc(ppg, acc)
                values["acc_low_power"] = torch.log1p(acc_low_power)
                values["acc_high_power"] = torch.log1p(acc_high_power)
                values["ppg_acc_corr"] = correlation

            context = torch.cat(
                [values[name] for name in self.spec.context_features],
                dim=1,
            )
            return torch.nan_to_num(
                context, nan=0.0, posinf=10.0, neginf=-10.0
            )

    def forward(
        self,
        ppg: torch.Tensor,
        temp: torch.Tensor,
        acc: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch = ppg.size(0)
        context = self.context_norm(self._context(ppg, temp, acc))
        codes = self.gate_code(context).view(
            batch, len(self.spec.modalities), self.gate_rank
        )
        return {
            modality: 1.0
            + self.max_delta
            * torch.tanh(self.gate_expand[modality](codes[:, index]))
            for index, modality in enumerate(self.spec.modalities)
        }


class CompactTemperatureResidual(nn.Module):
    """Bounded temperature correction for a PPG+ACC embedding.

    The public temperature stream carries only about three source values in a
    four-second window.  Consequently, this module samples only the start,
    middle, and end anchors rather than treating all 512 interpolated values as
    independent observations.  Five descriptors are normalized using running
    training-fold statistics and passed through a small MLP.  The zero-initialized
    output makes the initial model exactly equal to the PPG+ACC baseline.
    """

    statistic_names = (
        "anchor_mean",
        "anchor_std",
        "anchor_drift",
        "anchor_curvature",
        "anchor_range",
    )

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = 32,
        max_residual: float = 0.1,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.max_residual = float(max_residual)
        self.statistics_norm = nn.BatchNorm1d(
            len(self.statistic_names),
            affine=True,
            track_running_stats=True,
        )
        self.hidden = nn.Sequential(
            nn.Linear(len(self.statistic_names), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )
        self.residual = nn.Linear(hidden_dim, embed_dim)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    @staticmethod
    def temperature_statistics(temp: torch.Tensor) -> torch.Tensor:
        if temp.ndim != 3 or temp.size(1) != 1:
            raise ValueError(
                "temperature must have shape [batch, 1, time]"
            )
        if temp.size(2) < 3:
            raise ValueError("temperature window needs at least three samples")
        anchors = temp[:, 0, [0, temp.size(2) // 2, temp.size(2) - 1]]
        start, middle, end = anchors.unbind(dim=1)
        statistics = torch.stack(
            (
                anchors.mean(dim=1),
                anchors.std(dim=1, unbiased=False),
                end - start,
                middle - 0.5 * (start + end),
                anchors.amax(dim=1) - anchors.amin(dim=1),
            ),
            dim=1,
        )
        return torch.nan_to_num(
            statistics, nan=0.0, posinf=10.0, neginf=-10.0
        )

    def forward(
        self,
        embedding: torch.Tensor,
        temp: torch.Tensor,
    ) -> torch.Tensor:
        statistics = self.temperature_statistics(temp).detach()
        hidden = self.hidden(self.statistics_norm(statistics))
        correction = self.max_residual * torch.tanh(self.residual(hidden))
        return embedding + correction


class FinalECAPAModel(nn.Module):
    def __init__(
        self,
        variant: str,
        embed_dim: int = EMBED_DIM,
        dropout_p: float = DROPOUT,
        context_norm: str = "layernorm",
        gate_max_delta: float = 0.5,
    ):
        super().__init__()
        if variant not in MODEL_SPECS:
            raise ValueError(
                f"Unknown variant {variant!r}; choose from {sorted(MODEL_SPECS)}"
            )
        self.variant = variant
        self.spec = MODEL_SPECS[variant]
        if self.spec.fusion_type not in {"early", "late"}:
            raise ValueError(
                f"Unknown fusion_type={self.spec.fusion_type!r}"
            )
        if self.spec.fusion_type == "early" and self.spec.gated:
            raise ValueError("Early-fusion gated models are not supported")
        self.encoders = nn.ModuleDict()
        if self.spec.fusion_type == "early":
            in_channels = sum(
                3 if modality == "acc" else 1
                for modality in self.spec.modalities
            )
            self.encoders["fusion"] = build_encoder(
                self.spec.encoder_type,
                in_channels,
                embed_dim,
                dropout_p,
            )
        else:
            for modality in self.spec.modalities:
                in_channels = 3 if modality == "acc" else 1
                self.encoders[modality] = build_encoder(
                    self.spec.encoder_type,
                    in_channels,
                    embed_dim,
                    dropout_p,
                )
        self.gate = (
            ContextFeatureGate(
                self.spec,
                embed_dim=embed_dim,
                max_delta=gate_max_delta,
                context_norm=context_norm,
            )
            if self.spec.gated
            else None
        )
        if self.spec.temperature_correction == "compact_mlp_residual":
            self.temperature_corrector: nn.Module | None = (
                CompactTemperatureResidual(embed_dim=embed_dim)
            )
        elif self.spec.temperature_correction == "none":
            self.temperature_corrector = None
        else:
            raise ValueError(
                "Unknown temperature_correction="
                f"{self.spec.temperature_correction!r}"
            )
        self.projector = nn.Sequential(
            nn.Linear(
                embed_dim
                * (
                    1
                    if self.spec.fusion_type == "early"
                    else len(self.spec.modalities)
                ),
                embed_dim,
            ),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(p=dropout_p),
        )

    def forward(
        self,
        ppg: torch.Tensor,
        temp: torch.Tensor,
        acc: torch.Tensor,
    ) -> torch.Tensor:
        inputs = {"ppg": ppg, "temp": temp, "acc": acc}
        if self.spec.fusion_type == "early":
            fused_input = torch.cat(
                [inputs[name] for name in self.spec.modalities], dim=1
            )
            fused = self.encoders["fusion"](fused_input)
            embedding = self.projector(fused)
            if self.temperature_corrector is not None:
                embedding = self.temperature_corrector(embedding, temp)
            return embedding
        embeddings = {
            name: self.encoders[name](inputs[name])
            for name in self.spec.modalities
        }
        if self.gate is not None:
            gates = self.gate(ppg, temp, acc)
            embeddings = {
                name: embedding * gates[name]
                for name, embedding in embeddings.items()
            }
        fused = torch.cat(
            [embeddings[name] for name in self.spec.modalities], dim=1
        )
        embedding = self.projector(fused)
        if self.temperature_corrector is not None:
            embedding = self.temperature_corrector(embedding, temp)
        return embedding


class CompactResidualGateWarmStartModel(FinalECAPAModel):
    """Compact-temperature model with an identity residual quality gate.

    The module names outside ``gate`` intentionally match
    ``ppg_acc_temp_mlp``.  Its E50 state can therefore be loaded unchanged;
    only the newly introduced gate parameters are absent from that source
    checkpoint.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        dropout_p: float = DROPOUT,
        context_norm: str = "layernorm",
        max_delta: float = 0.5,
        gate_variant: str = "ppg_acc_gate_temp_mlp",
        output_variant: str = (
            "ppg_acc_temp_mlp_residual_verification_gate"
        ),
    ):
        super().__init__(
            variant=gate_variant,
            embed_dim=embed_dim,
            dropout_p=dropout_p,
            context_norm=context_norm,
            gate_max_delta=max_delta,
        )
        if self.spec.temperature_correction != "compact_mlp_residual":
            raise ValueError("Gate warm start requires compact temperature MLP")
        self.variant = output_variant


class PPGACCResidualGateWarmStartModel(FinalECAPAModel):
    """PPG+ACC E50 with an identity-initialized quality Gate.

    All modules outside ``gate`` match the ungated ``ppg_acc`` backbone, so
    its E50 state is an exact C00 warm start.  The Gate uses only the four
    PPG/ACC quality descriptors and does not consume temperature.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        dropout_p: float = DROPOUT,
        context_norm: str = "layernorm",
        max_delta: float = 0.5,
    ):
        super().__init__(
            variant="ppg_acc_gate",
            embed_dim=embed_dim,
            dropout_p=dropout_p,
            context_norm=context_norm,
            gate_max_delta=max_delta,
        )
        self.variant = "ppg_acc_residual_verification_gate"


class ThreeECAPAGateWarmStartModel(FinalECAPAModel):
    """Three-ECAPA baseline with an identity-initialized channel-wise Gate.

    PPG, temperature, and ACC retain independent ECAPA encoders.  The Gate
    uses only the four PPG/ACC quality descriptors (no temperature mean),
    predicts a separate rank-32 code for every modality, and expands each code
    through its own 32-to-192 layer.  All non-Gate module names intentionally
    match ``baseline`` so an E50 checkpoint can be loaded exactly.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        dropout_p: float = DROPOUT,
        context_norm: str = "layernorm",
        max_delta: float = 0.5,
    ):
        super().__init__(
            variant="ppg_acc_temp_ecapa_gate",
            embed_dim=embed_dim,
            dropout_p=dropout_p,
            context_norm=context_norm,
            gate_max_delta=max_delta,
        )
        self.variant = "ppg_acc_temp_ecapa_verification_gate"


class AAMSoftmax(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_classes: int,
        margin: float = 0.2,
        scale: float = 30.0,
    ):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self, embedding: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        cosine = F.linear(
            F.normalize(embedding), F.normalize(self.weight)
        )
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt(1.0 - cosine.square())
        phi = (
            cosine * math.cos(self.margin)
            - sine * math.sin(self.margin)
        )
        phi = torch.where(
            cosine > math.cos(math.pi - self.margin),
            phi,
            cosine - math.sin(self.margin) * self.margin,
        )
        one_hot = torch.zeros_like(cosine).scatter_(
            1, labels.view(-1, 1), 1
        )
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return F.cross_entropy(logits * self.scale, labels)


def build_model(
    variant: str,
    context_norm: str = "layernorm",
) -> FinalECAPAModel:
    if variant not in MODEL_SPECS:
        raise ValueError(
            f"Unknown variant {variant!r}; choose from {sorted(MODEL_SPECS)}"
        )
    return FinalECAPAModel(
        variant=variant,
        embed_dim=MODEL_SPECS[variant].embedding_dim,
        dropout_p=DROPOUT,
        context_norm=context_norm,
    )


def build_compact_residual_gate_warmstart_model(
    context_norm: str = "layernorm",
    max_delta: float = 0.5,
    gate_variant: str = "ppg_acc_gate_temp_mlp",
    output_variant: str = (
        "ppg_acc_temp_mlp_residual_verification_gate"
    ),
) -> CompactResidualGateWarmStartModel:
    return CompactResidualGateWarmStartModel(
        embed_dim=EMBED_DIM,
        dropout_p=DROPOUT,
        context_norm=context_norm,
        max_delta=max_delta,
        gate_variant=gate_variant,
        output_variant=output_variant,
    )


def build_ppg_acc_residual_gate_warmstart_model(
    context_norm: str = "layernorm",
    max_delta: float = 0.5,
) -> PPGACCResidualGateWarmStartModel:
    return PPGACCResidualGateWarmStartModel(
        embed_dim=EMBED_DIM,
        dropout_p=DROPOUT,
        context_norm=context_norm,
        max_delta=max_delta,
    )


def build_three_ecapa_gate_warmstart_model(
    context_norm: str = "layernorm",
    max_delta: float = 0.5,
) -> ThreeECAPAGateWarmStartModel:
    return ThreeECAPAGateWarmStartModel(
        embed_dim=EMBED_DIM,
        dropout_p=DROPOUT,
        context_norm=context_norm,
        max_delta=max_delta,
    )
