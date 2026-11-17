"""Streaming-style loudness normalization SIMULATION.

This is arithmetic on measured LUFS-I and true peak, not a model of any specific service.
Real platforms differ in reference level, whether they turn quiet tracks up, whether they use a
limiter, album vs track mode, and they change their behaviour over time.

Two modes:
  * gain_only     — apply (target - LUFS-I) dB regardless of peaks; report the resulting true peak.
  * peak_limited  — apply the same gain but never let true peak exceed the ceiling (positive gain is
                    capped at ceiling - TP). Attenuation is never limited.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

SIMULATION_DISCLAIMER = ("Simulation only: gain arithmetic on measured LUFS-I and true peak. It is not an exact "
                         "model of Spotify, Apple Music, YouTube, Tidal or any other service.")


@dataclass
class NormalizationResult:
    target_lufs: float
    ceiling_dbtp: float
    input_lufs: float
    input_true_peak_dbtp: float
    gain_only_db: float
    gain_only_true_peak_dbtp: float
    gain_only_exceeds_ceiling: bool
    peak_limited_gain_db: float
    peak_limited_lufs: float
    peak_limited_true_peak_dbtp: float
    limited_by_ceiling: bool
    shortfall_lu: float  # how far below target the peak-limited result stays
    disclaimer: str = SIMULATION_DISCLAIMER


def simulate_normalization(integrated_lufs: float, true_peak_dbtp: float, target_lufs: float,
                           ceiling_dbtp: float) -> NormalizationResult | None:
    """None when loudness is not finite (e.g. silence) — nothing meaningful to simulate."""
    if not (math.isfinite(integrated_lufs) and math.isfinite(true_peak_dbtp)):
        return None
    gain = target_lufs - integrated_lufs
    gain_only_tp = true_peak_dbtp + gain
    headroom = ceiling_dbtp - true_peak_dbtp
    limited_gain = gain if gain <= 0 else min(gain, max(headroom, 0.0))
    limited = limited_gain < gain
    return NormalizationResult(
        target_lufs=target_lufs, ceiling_dbtp=ceiling_dbtp, input_lufs=integrated_lufs,
        input_true_peak_dbtp=true_peak_dbtp, gain_only_db=gain, gain_only_true_peak_dbtp=gain_only_tp,
        gain_only_exceeds_ceiling=gain_only_tp > ceiling_dbtp, peak_limited_gain_db=limited_gain,
        peak_limited_lufs=integrated_lufs + limited_gain,
        peak_limited_true_peak_dbtp=true_peak_dbtp + limited_gain, limited_by_ceiling=limited,
        shortfall_lu=gain - limited_gain,
    )
