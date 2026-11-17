"""Stereo image and phase analysis.

Definitions used throughout:
  * Correlation r = Σ L·R / sqrt(Σ L² · Σ R²)  (phase-meter style, no mean removal). +1 = identical,
    0 = uncorrelated, -1 = polarity-inverted.
  * Mid M = (L+R)/2, Side S = (L-R)/2. Side-to-mid ratio = 20·log10(RMS(S)/RMS(M)).
  * Side energy share = ΣS² / (ΣM² + ΣS²).
  * Mono fold-down loudness change = LUFS-I of [M, M] minus LUFS-I of [L, R] (BS.1770-4 both times).
Neither narrow nor wide is good or bad per se; these numbers only describe the image.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import signal

from .audio_loader import AudioData
from .loudness import integrated_lufs
from .models import Compliance, Measurement, amp_to_db, power_to_db

EPS = 1e-20


@dataclass
class StereoResult:
    applicable: bool
    correlation: Measurement
    correlation_low_band: Measurement
    side_to_mid: Measurement
    side_energy_pct: Measurement
    left_rms_dbfs: float | None
    right_rms_dbfs: float | None
    balance: Measurement
    mono_loudness_change: Measurement
    negative_correlation_pct: float | None
    problem_regions: list[dict[str, float]]
    times_s: np.ndarray = field(repr=False)
    correlation_series: np.ndarray = field(repr=False)
    side_to_mid_series_db: np.ndarray = field(repr=False)


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    denom = np.sqrt(np.sum(left ** 2) * np.sum(right ** 2))
    return float(np.sum(left * right) / denom) if denom > 0 else None


def side_to_mid_db(left: np.ndarray, right: np.ndarray) -> float:
    mid, side = (left + right) / 2, (left - right) / 2
    return float(power_to_db((np.mean(side ** 2) + EPS) / (np.mean(mid ** 2) + EPS)))


def windowed_stereo(left: np.ndarray, right: np.ndarray, sample_rate: int, window_s: float,
                    silence_floor_dbfs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Correlation and side-to-mid per window (hop = window/2). Silent windows -> NaN."""
    window = int(round(window_s * sample_rate))
    hop = max(1, window // 2)
    if left.size < window:
        return np.empty(0), np.empty(0), np.empty(0)
    starts = np.arange(0, left.size - window + 1, hop)

    def sums(x: np.ndarray) -> np.ndarray:
        cumulative = np.concatenate([[0.0], np.cumsum(x)])
        return cumulative[starts + window] - cumulative[starts]

    ll, rr, lr = sums(left ** 2), sums(right ** 2), sums(left * right)
    mm = (ll + rr + 2 * lr) / 4
    ss = (ll + rr - 2 * lr) / 4
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = lr / np.sqrt(ll * rr)
        width = power_to_db((ss + EPS) / (mm + EPS))
    silent = power_to_db((ll + rr) / (2 * window) + EPS) < silence_floor_dbfs
    corr[silent] = np.nan
    width[silent] = np.nan
    return (starts + window / 2) / sample_rate, corr, width


def _regions_below(times: np.ndarray, values: np.ndarray, threshold: float, window_s: float) -> list[dict[str, float]]:
    below = np.nan_to_num(values, nan=1.0) < threshold
    regions = []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], below.astype(int), [0]])))
    for start, end in zip(edges[::2], edges[1::2]):
        regions.append({"start_s": float(times[start] - window_s / 2), "end_s": float(times[end - 1] + window_s / 2),
                        "min_correlation": float(np.nanmin(values[start:end]))})
    return regions


def _not_applicable(reason: str) -> StereoResult:
    na = Measurement(None, "", "", Compliance.NOT_AVAILABLE, reason)
    empty = np.empty(0)
    return StereoResult(False, na, na, na, na, None, None, na, na, None, [], empty, empty, empty)


def analyze_stereo(audio: AudioData, config: dict[str, Any]) -> StereoResult:
    if audio.channels == 1:
        return _not_applicable("Mono file — stereo metrics do not apply. Mono compatibility is inherent.")
    cfg = config["analysis"]
    thresholds = config["thresholds"]
    fs = audio.sample_rate
    left, right = audio.samples[:, 0], audio.samples[:, 1]

    r = correlation(left, right)
    sos = signal.butter(4, cfg["low_band_cutoff_hz"], "lowpass", fs=fs, output="sos")
    r_low = correlation(signal.sosfilt(sos, left), signal.sosfilt(sos, right))
    sm = side_to_mid_db(left, right)
    mid, side = (left + right) / 2, (left - right) / 2
    side_pct = float(100 * np.sum(side ** 2) / (np.sum(mid ** 2) + np.sum(side ** 2) + EPS))
    l_rms = float(amp_to_db(np.sqrt(np.mean(left ** 2))))
    r_rms = float(amp_to_db(np.sqrt(np.mean(right ** 2))))

    stereo_lufs = integrated_lufs(audio.samples, fs)
    mono_lufs = integrated_lufs(np.column_stack([mid, mid]), fs)
    mono_change = mono_lufs - stereo_lufs if np.isfinite(stereo_lufs) else None
    if mono_change is not None and not np.isfinite(mono_change):
        mono_change = None  # fully cancelling mono sum

    times, corr_series, width_series = windowed_stereo(left, right, fs, cfg["stereo_window_s"],
                                                       cfg["silence_floor_dbfs"])
    valid = corr_series[np.isfinite(corr_series)]
    negative_pct = float(100 * np.mean(valid < thresholds["correlation_problem"])) if valid.size else None
    regions = _regions_below(times, corr_series, thresholds["correlation_problem"], cfg["stereo_window_s"])
    for region in regions:
        region["start_s"] += audio.offset_s
        region["end_s"] += audio.offset_s

    def metric(value: float | None, unit: str, method: str, notes: str = "") -> Measurement:
        return Measurement(value, unit, method, Compliance.DERIVED if value is not None else Compliance.NOT_AVAILABLE,
                           notes)

    return StereoResult(
        applicable=True,
        correlation=metric(r, "r", "Σ L·R / sqrt(Σ L² · Σ R²), whole range.",
                           "+1 is not a defect (mono-like image); negative values mean phase cancellation in mono."),
        correlation_low_band=metric(r_low, "r", f"Same correlation after 4th-order low-pass at "
                                    f"{cfg['low_band_cutoff_hz']:.0f} Hz.",
                                    "Low end is usually kept near mono for vinyl, clubs and mono playback."),
        side_to_mid=metric(sm, "dB", "20·log10(RMS(S)/RMS(M)), M=(L+R)/2, S=(L-R)/2.",
                           "-inf = mono; 0 dB = side as loud as mid; > 0 dB = more side than mid (phase issues likely)."),
        side_energy_pct=metric(side_pct, "%", "ΣS² / (ΣM² + ΣS²)."),
        left_rms_dbfs=l_rms, right_rms_dbfs=r_rms,
        balance=metric(l_rms - r_rms, "dB", "RMS(L) - RMS(R) in dB. Positive = left louder."),
        mono_loudness_change=metric(mono_change, "LU",
                                    "LUFS-I of mono fold-down [M, M] minus LUFS-I of the stereo signal.",
                                    "0 LU = no loss in mono; -3 LU = fully uncorrelated; more negative = cancellation."),
        negative_correlation_pct=negative_pct, problem_regions=regions,
        times_s=times + audio.offset_s, correlation_series=corr_series, side_to_mid_series_db=width_series,
    )
