"""Loudness and level measurements.

* LUFS-I / LUFS-M / LUFS-S — ITU-R BS.1770-4 K-weighting and gating, EBU R128 / Tech 3341 windows.
* LRA — EBU Tech 3342.
* True peak — BS.1770-4 Annex 2 style oversampling (polyphase FIR). An approximation by nature.
* Sample peak, RMS, crest factor, PLR — simple definitions stated in each Measurement.

The K-weighting filter is derived analytically for any sample rate (same design as libebur128 /
pyloudnorm); at 48 kHz it reproduces the coefficients tabulated in BS.1770-4.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyloudnorm
from scipy import signal

from .audio_loader import AudioData
from .models import Compliance, Measurement, amp_to_db, power_to_db

LOUDNESS_OFFSET_DB = -0.691
ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_INTEGRATED_LU = -10.0
RELATIVE_GATE_LRA_LU = -20.0
MOMENTARY_WINDOW_S = 0.4
SHORT_TERM_WINDOW_S = 3.0
BLOCK_HOP_S = 0.1  # 75 % overlap of 400 ms gating blocks; 10 Hz meter refresh
LRA_LOW_PERCENTILE = 10
LRA_HIGH_PERCENTILE = 95
TRUE_PEAK_TAPS_PER_PHASE = 48
TRUE_PEAK_CHUNK_FRAMES = 1 << 18
TRUE_PEAK_REGION_S = 0.01
MAX_REPORTED_OVER_REGIONS = 200


@dataclass
class TruePeakResult:
    per_channel_dbtp: list[float]
    oversampling: int
    over_region_times_s: list[float]  # starts of 10 ms regions exceeding the ceiling (capped)
    over_region_count: int

    @property
    def max_dbtp(self) -> float:
        return max(self.per_channel_dbtp)


@dataclass
class LoudnessResult:
    integrated: Measurement
    short_term_max: Measurement
    momentary_max: Measurement
    lra: Measurement
    sample_peak: Measurement
    true_peak: Measurement
    rms: Measurement
    crest_factor: Measurement
    plr: Measurement
    sample_peak_per_channel_dbfs: list[float]
    true_peak_per_channel_dbtp: list[float]
    rms_per_channel_dbfs: list[float]
    true_peak_over_region_count: int
    true_peak_over_times_s: list[float]
    momentary_times_s: np.ndarray = field(repr=False)
    momentary_lufs: np.ndarray = field(repr=False)
    short_term_times_s: np.ndarray = field(repr=False)
    short_term_lufs: np.ndarray = field(repr=False)


# ---------------------------------------------------------------------------------------------
# BS.1770 building blocks
# ---------------------------------------------------------------------------------------------

def k_weighting_sos(sample_rate: int) -> np.ndarray:
    """Second-order sections of the BS.1770 K-weighting (pre-filter shelf + RLB high-pass)."""
    gain_db, q, fc = 3.999843853973347, 0.7071752369554196, 1681.974450955533
    k = math.tan(math.pi * fc / sample_rate)
    vh = 10 ** (gain_db / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = [(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0,
             1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]

    q, fc = 0.5003270373238773, 38.13547087602444
    k = math.tan(math.pi * fc / sample_rate)
    a0 = 1 + k / q + k * k
    highpass = [1.0, -2.0, 1.0, 1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    return np.array([shelf, highpass])


def k_weight(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply K-weighting along axis 0 of a (frames, channels) array."""
    return signal.sosfilt(k_weighting_sos(sample_rate), samples, axis=0)


def block_powers(weighted: np.ndarray, sample_rate: int, window_s: float,
                 hop_s: float = BLOCK_HOP_S) -> tuple[np.ndarray, np.ndarray]:
    """Channel-summed mean-square power of sliding windows.

    Returns (end_times_s, power). Power is Σ_channels G_i·z_i with G_i = 1 (mono/stereo, BS.1770).
    """
    window = int(round(window_s * sample_rate))
    hop = int(round(hop_s * sample_rate))
    frames = weighted.shape[0]
    if frames < window:
        return np.empty(0), np.empty(0)
    starts = np.arange(0, frames - window + 1, hop)
    cumulative = np.concatenate([np.zeros(1), np.cumsum(np.sum(weighted ** 2, axis=1))])
    power = (cumulative[starts + window] - cumulative[starts]) / window
    return (starts + window) / sample_rate, np.maximum(power, 0.0)


def power_to_lufs(power: float | np.ndarray) -> float | np.ndarray:
    return LOUDNESS_OFFSET_DB + power_to_db(power)


def gated_integrated_loudness(block_power: np.ndarray) -> float:
    """BS.1770-4 two-stage gating on 400 ms block powers."""
    if block_power.size == 0:
        return -math.inf
    loudness = power_to_lufs(block_power)
    above_absolute = loudness > ABSOLUTE_GATE_LUFS
    if not above_absolute.any():
        return -math.inf
    relative_gate = power_to_lufs(block_power[above_absolute].mean()) + RELATIVE_GATE_INTEGRATED_LU
    gated = above_absolute & (loudness > relative_gate)
    return float(power_to_lufs(block_power[gated].mean()))


def loudness_range(short_term_power: np.ndarray) -> float | None:
    """EBU Tech 3342 LRA from 3 s short-term block powers; None if too few blocks survive gating."""
    if short_term_power.size == 0:
        return None
    loudness = power_to_lufs(short_term_power)
    above_absolute = loudness > ABSOLUTE_GATE_LUFS
    if above_absolute.sum() < 2:
        return None
    relative_gate = power_to_lufs(short_term_power[above_absolute].mean()) + RELATIVE_GATE_LRA_LU
    values = loudness[above_absolute & (loudness > relative_gate)]
    if values.size < 2:
        return None
    return float(np.percentile(values, LRA_HIGH_PERCENTILE) - np.percentile(values, LRA_LOW_PERCENTILE))


def integrated_lufs(samples: np.ndarray, sample_rate: int) -> float:
    """Convenience: BS.1770-4 integrated loudness of a (frames, channels) array."""
    _, power = block_powers(k_weight(samples, sample_rate), sample_rate, MOMENTARY_WINDOW_S)
    return gated_integrated_loudness(power)


# ---------------------------------------------------------------------------------------------
# True peak
# ---------------------------------------------------------------------------------------------

def true_peak_oversampling_factor(sample_rate: int, base_factor: int) -> int:
    """Keep the oversampled rate around >= 176.4 kHz, as BS.1770-4 does for 44.1/48 kHz input."""
    if sample_rate < 88200:
        return base_factor
    if sample_rate < 176400:
        return max(1, base_factor // 2)
    return max(1, base_factor // 4)


def _interpolation_filter(factor: int) -> np.ndarray:
    taps = factor * TRUE_PEAK_TAPS_PER_PHASE + 1
    return signal.firwin(taps, 1.0 / factor, window=("kaiser", 9.0))


def true_peak(samples: np.ndarray, sample_rate: int, base_factor: int = 4,
              ceiling_dbtp: float | None = None) -> TruePeakResult:
    """Oversampled peak per channel, processed in overlapping chunks to bound memory."""
    factor = true_peak_oversampling_factor(sample_rate, base_factor)
    frames, channels = samples.shape
    fir = _interpolation_filter(factor) if factor > 1 else None
    pad = (len(fir) // factor + 8) if fir is not None else 0
    peaks = np.zeros(channels)
    ceiling = None if ceiling_dbtp is None else 10 ** (ceiling_dbtp / 20)
    over_regions: set[int] = set()
    region_len = TRUE_PEAK_REGION_S * sample_rate * factor
    for start in range(0, frames, TRUE_PEAK_CHUNK_FRAMES):
        stop = min(frames, start + TRUE_PEAK_CHUNK_FRAMES)
        if fir is None:
            core = np.abs(samples[start:stop])
        else:
            lo, hi = max(0, start - pad), min(frames, stop + pad)
            upsampled = signal.resample_poly(samples[lo:hi], factor, 1, axis=0, window=fir)
            core = np.abs(upsampled[(start - lo) * factor:(stop - lo) * factor])
        peaks = np.maximum(peaks, core.max(axis=0))
        if ceiling is not None:
            over_idx = np.nonzero(core.max(axis=1) > ceiling)[0]
            over_regions.update(((over_idx + start * factor) // region_len).astype(int).tolist())

    ordered = sorted(over_regions)
    return TruePeakResult(
        per_channel_dbtp=[float(amp_to_db(p)) for p in peaks],
        oversampling=factor,
        over_region_times_s=[r * TRUE_PEAK_REGION_S for r in ordered[:MAX_REPORTED_OVER_REGIONS]],
        over_region_count=len(ordered),
    )


# ---------------------------------------------------------------------------------------------
# Full loudness analysis
# ---------------------------------------------------------------------------------------------

def _pyloudnorm_integrated(audio: AudioData) -> float | None:
    try:
        meter = pyloudnorm.Meter(audio.sample_rate)
        data = audio.samples[:, 0] if audio.channels == 1 else audio.samples
        return float(meter.integrated_loudness(data))
    except Exception:  # pyloudnorm raises ValueError for too-short input; cross-check is optional
        return None


def analyze_loudness(audio: AudioData, config: dict[str, Any]) -> LoudnessResult:
    """All loudness / level metrics for one AudioData."""
    fs = audio.sample_rate
    tp_ceiling = config["thresholds"]["true_peak_max_dbtp"]
    weighted = k_weight(audio.samples, fs)
    m_times, m_power = block_powers(weighted, fs, MOMENTARY_WINDOW_S)
    s_times, s_power = block_powers(weighted, fs, SHORT_TERM_WINDOW_S)
    del weighted
    momentary = power_to_lufs(m_power)
    short_term = power_to_lufs(s_power)

    own_integrated = gated_integrated_loudness(m_power)
    reference_integrated = _pyloudnorm_integrated(audio)
    mono_note = (" Mono file: measured as a single channel (G=1). When played on two speakers it will be "
                 "perceived ~3 LU louder than this value." if audio.channels == 1 else "")
    if reference_integrated is not None and math.isfinite(reference_integrated):
        diff = abs(reference_integrated - own_integrated)
        integrated_value = reference_integrated
        cross_check = (f"Value from pyloudnorm; independent MasterCheck implementation gives "
                       f"{own_integrated:.2f} LUFS (difference {diff:.2f} LU).")
    else:
        integrated_value = own_integrated
        cross_check = "pyloudnorm cross-check unavailable; MasterCheck BS.1770-4 implementation used."
    integrated = Measurement(integrated_value, "LUFS",
                             "ITU-R BS.1770-4: K-weighting, 400 ms blocks with 75 % overlap, "
                             "absolute gate -70 LUFS, relative gate -10 LU.",
                             Compliance.STANDARD, cross_check + mono_note)

    lra_value = loudness_range(s_power)
    lra = Measurement(lra_value, "LU",
                      "EBU Tech 3342: 3 s short-term blocks every 100 ms, absolute gate -70 LUFS, "
                      "relative gate -20 LU, 95th minus 10th percentile.",
                      Compliance.STANDARD if lra_value is not None else Compliance.NOT_AVAILABLE,
                      "" if lra_value is not None else "Too few short-term blocks (needs > 3 s of non-silent audio).")

    st_max = float(short_term.max()) if short_term.size else None
    short_term_max = Measurement(st_max, "LUFS", "Maximum of 3 s sliding window (EBU Tech 3341), 10 Hz rate.",
                                 Compliance.STANDARD if st_max is not None else Compliance.NOT_AVAILABLE,
                                 "" if st_max is not None else "File shorter than 3 s.")
    momentary_max = Measurement(float(momentary.max()), "LUFS",
                                "Maximum of 400 ms sliding window (EBU Tech 3341), 10 Hz rate.",
                                Compliance.STANDARD)

    sample_peaks = np.max(np.abs(audio.samples), axis=0)
    sp_db = [float(amp_to_db(p)) for p in sample_peaks]
    sample_peak = Measurement(max(sp_db), "dBFS", "Maximum absolute sample value over all channels.",
                              Compliance.STANDARD,
                              "Not a true peak: inter-sample peaks between samples are not seen.")

    tp = true_peak(audio.samples, fs, config["analysis"]["true_peak_oversampling"], tp_ceiling)
    true_peak_m = Measurement(
        tp.max_dbtp, "dBTP",
        f"{tp.oversampling}x oversampling with a {TRUE_PEAK_TAPS_PER_PHASE}-tap-per-phase Kaiser FIR "
        f"interpolator, maximum absolute value (BS.1770-4 Annex 2 approach).",
        Compliance.APPROXIMATION,
        "Oversampled peak meters can under-read real inter-sample peaks (BS.1770-4 cites up to "
        "~0.7 dB for 4x on worst-case signals; typically much less on music). Treat as an estimate, "
        "keep a safety margin. Increase analysis.true_peak_oversampling for a tighter estimate.")

    channel_ms = np.mean(audio.samples ** 2, axis=0)
    rms_ch = [float(power_to_db(ms)) for ms in channel_ms]
    rms_all = float(power_to_db(channel_ms.mean()))
    rms = Measurement(rms_all, "dBFS",
                      "Unweighted RMS over the whole range, all channels pooled. Full-scale sine = -3.01 dBFS "
                      "(no AES17 +3 dB offset).",
                      Compliance.DERIVED, "Includes silence and fades, which lower the value.")
    crest = Measurement(sample_peak.value - rms_all, "dB",
                        "Crest factor = sample peak (dBFS) - RMS (dBFS), whole range.",
                        Compliance.DERIVED,
                        "Whole-file figure; silence/fades raise it. See windowed crest factor in dynamics.")
    plr = Measurement(tp.max_dbtp - integrated_value if math.isfinite(integrated_value) else None, "dB",
                      "PLR = true peak (dBTP) - integrated loudness (LUFS).",
                      Compliance.DERIVED, "Inherits the true-peak approximation.")

    return LoudnessResult(
        integrated=integrated, short_term_max=short_term_max, momentary_max=momentary_max, lra=lra,
        sample_peak=sample_peak, true_peak=true_peak_m, rms=rms, crest_factor=crest, plr=plr,
        sample_peak_per_channel_dbfs=sp_db, true_peak_per_channel_dbtp=tp.per_channel_dbtp,
        rms_per_channel_dbfs=rms_ch, true_peak_over_region_count=tp.over_region_count,
        true_peak_over_times_s=[t + audio.offset_s for t in tp.over_region_times_s],
        momentary_times_s=m_times + audio.offset_s, momentary_lufs=momentary,
        short_term_times_s=s_times + audio.offset_s, short_term_lufs=short_term,
    )
