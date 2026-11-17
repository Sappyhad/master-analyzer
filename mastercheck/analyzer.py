"""Runs every analysis module on one AudioData and collects the results."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audio_loader import AudioData
from .codec_simulation import CodecResult, simulate_codecs
from .dynamics import DynamicsResult, analyze_dynamics
from .loudness import LoudnessResult, analyze_loudness
from .normalization import NormalizationResult, simulate_normalization
from .spectrum import SpectrumResult, analyze_spectrum
from .stereo import StereoResult, analyze_stereo
from .transients import TransientResult, analyze_transients


@dataclass
class TrackInfo:
    name: str
    path: Path
    sample_rate: int
    channels: int
    duration_s: float
    offset_s: float
    format_name: str
    subtype: str
    is_lossy_source: bool


@dataclass
class TrackAnalysis:
    info: TrackInfo
    loudness: LoudnessResult
    dynamics: DynamicsResult
    spectrum: SpectrumResult
    stereo: StereoResult
    transients: TransientResult
    normalization: NormalizationResult | None
    codecs: list[CodecResult] | None = None
    codec_note: str = ""
    loudest_fragment: "TrackAnalysis | None" = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MetricDef:
    key: str
    label: str
    unit: str
    level_dependent: bool  # changes with a pure gain change (relevant for loudness matching)
    digits: int = 1


HEADLINE_METRICS: tuple[MetricDef, ...] = (
    MetricDef("integrated_lufs", "Integrated loudness", "LUFS", True),
    MetricDef("true_peak_dbtp", "True peak", "dBTP", True, 2),
    MetricDef("sample_peak_dbfs", "Sample peak", "dBFS", True, 2),
    MetricDef("lra_lu", "Loudness range (LRA)", "LU", False),
    MetricDef("short_term_max_lufs", "Max short-term loudness", "LUFS", True),
    MetricDef("rms_dbfs", "RMS", "dBFS", True),
    MetricDef("crest_factor_db", "Crest factor (whole range)", "dB", False),
    MetricDef("windowed_crest_db", "Crest factor (median, 400 ms)", "dB", False),
    MetricDef("plr_db", "PLR", "dB", False),
    MetricDef("peak_density_pct", "Peak density", "%", False, 3),
    MetricDef("transient_p95_db", "Transient prominence p95", "dB", False),
    MetricDef("bass_pct", "Bass group energy", "%", False),
    MetricDef("mid_pct", "Mid group energy", "%", False),
    MetricDef("high_pct", "High group energy", "%", False, 2),
    MetricDef("spectral_tilt", "Spectral tilt", "dB/oct", False, 2),
    MetricDef("correlation", "Stereo correlation", "r", False, 2),
    MetricDef("low_band_correlation", "Low-band correlation", "r", False, 2),
    MetricDef("side_to_mid_db", "Side/Mid ratio", "dB", False),
    MetricDef("mono_change_lu", "Mono fold-down loudness change", "LU", False),
)
METRICS_BY_KEY = {m.key: m for m in HEADLINE_METRICS}


def headline_metrics(analysis: TrackAnalysis) -> dict[str, float | None]:
    """Flat key -> value map used by comparisons, CSV export and diagnosis."""
    lo, dyn, sp, st, tr = analysis.loudness, analysis.dynamics, analysis.spectrum, analysis.stereo, analysis.transients
    return {
        "integrated_lufs": lo.integrated.value,
        "true_peak_dbtp": lo.true_peak.value,
        "sample_peak_dbfs": lo.sample_peak.value,
        "lra_lu": lo.lra.value,
        "short_term_max_lufs": lo.short_term_max.value,
        "rms_dbfs": lo.rms.value,
        "crest_factor_db": lo.crest_factor.value,
        "windowed_crest_db": dyn.windowed_crest_median.value,
        "plr_db": lo.plr.value,
        "peak_density_pct": dyn.peak_density.value,
        "transient_p95_db": tr.transient_prominence_p95.value,
        "bass_pct": sp.group_pct.get("bass"),
        "mid_pct": sp.group_pct.get("mid"),
        "high_pct": sp.group_pct.get("high"),
        "spectral_tilt": sp.spectral_tilt.value,
        "correlation": st.correlation.value,
        "low_band_correlation": st.correlation_low_band.value,
        "side_to_mid_db": st.side_to_mid.value,
        "mono_change_lu": st.mono_loudness_change.value,
    }


def analyze_track(audio: AudioData, config: dict[str, Any], *, ffmpeg: str | None = None,
                  run_codecs: bool = False, include_fragment: bool = True) -> TrackAnalysis:
    """Full analysis. Codec simulation runs only when `run_codecs` and an FFmpeg path are given."""
    loudness = analyze_loudness(audio, config)
    norm_cfg = config["normalization"]
    analysis = TrackAnalysis(
        info=TrackInfo(name=audio.path.name, path=audio.path, sample_rate=audio.sample_rate,
                       channels=audio.channels, duration_s=audio.duration_s, offset_s=audio.offset_s,
                       format_name=audio.format_name, subtype=audio.subtype, is_lossy_source=audio.is_lossy_source),
        loudness=loudness,
        dynamics=analyze_dynamics(audio, loudness, config),
        spectrum=analyze_spectrum(audio, config),
        stereo=analyze_stereo(audio, config),
        transients=analyze_transients(audio, config),
        normalization=simulate_normalization(loudness.integrated.value, loudness.true_peak.value,
                                             norm_cfg["target_lufs"], norm_cfg["true_peak_ceiling_dbtp"]),
    )
    if audio.duration_s < 3.0:
        analysis.warnings.append("Analyzed range shorter than 3 s: short-term loudness and LRA are unavailable.")

    if run_codecs:
        if ffmpeg is None:
            analysis.codec_note = "Skipped: FFmpeg not found (install FFmpeg or `pip install imageio-ffmpeg`)."
        else:
            analysis.codecs = simulate_codecs(audio, loudness.true_peak.value, loudness.integrated.value, config, ffmpeg)
    else:
        analysis.codec_note = "Skipped (disabled)."

    dyn = analysis.dynamics
    if include_fragment and dyn.loudest_fragment_end_s - dyn.loudest_fragment_start_s < audio.duration_s - 0.5:
        fragment = audio.segment(dyn.loudest_fragment_start_s, dyn.loudest_fragment_end_s)
        analysis.loudest_fragment = analyze_track(fragment, config, include_fragment=False)
    return analysis
