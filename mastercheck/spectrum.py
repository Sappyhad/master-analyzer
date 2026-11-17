"""Long-term spectrum, band energy distribution and spectral indicators.

The average spectrum is a Welch PSD (Hann window, 50 % overlap) of the mid signal (L+R)/2.
Band energies integrate that PSD; percentages are relative to total 20 Hz-20 kHz energy
(or up to Nyquist for low sample rates). Nothing here judges a spectrum as good or bad by itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import signal

from .audio_loader import AudioData
from .models import Compliance, Measurement, power_to_db

ANALYSIS_LOW_HZ = 20.0
ANALYSIS_HIGH_HZ = 20000.0
TILT_RANGE_HZ = (100.0, 10000.0)
RESONANCE_RANGE_HZ = (30.0, 16000.0)
RESONANCE_NARROW_OCTAVE = 24
RESONANCE_WIDE_OCTAVE = 1
RESONANCE_MERGE_OCTAVES = 1 / 3
MAX_RESONANCES = 8
LOWPASS_SEARCH_FROM_HZ = 11000.0
LOWPASS_MAX_REPORTED_HZ = 19500.0  # normal anti-alias roll-off sits above this
LOWPASS_REFERENCE_BAND_HZ = (2000.0, 8000.0)
LOG_GRID_POINTS_PER_OCTAVE = 12


@dataclass
class BandEnergy:
    name: str
    label: str
    low_hz: float
    high_hz: float
    energy_pct: float | None        # share of total 20 Hz-20 kHz energy
    level_db_rel_total: float | None  # 10·log10(band / total)
    level_dbfs: float | None        # absolute band mean-square in dB re full scale


@dataclass
class SpectrumResult:
    bands: list[BandEnergy]
    group_pct: dict[str, float | None]
    group_dbfs: dict[str, float | None]
    spectral_tilt: Measurement
    spectral_centroid: Measurement
    bass_to_mid_db: Measurement
    resonances: list[dict[str, float]]
    lowpass_cutoff_hz: float | None
    freqs_hz: np.ndarray = field(repr=False)
    psd_db: np.ndarray = field(repr=False)


def welch_psd(mid: np.ndarray, sample_rate: int, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """One-sided power spectral density (full-scale units²/Hz)."""
    nperseg = min(nperseg, mid.size)
    return signal.welch(mid, fs=sample_rate, window="hann", nperseg=nperseg, noverlap=nperseg // 2,
                        scaling="density", detrend=False)


def band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    """Integrate the PSD over [low, high) Hz (rectangle rule on the FFT grid)."""
    df = freqs[1] - freqs[0]
    mask = (freqs >= low) & (freqs < high)
    return float(np.sum(psd[mask]) * df)


def fractional_octave_smooth(freqs: np.ndarray, psd: np.ndarray, fraction: int) -> np.ndarray:
    """Power average over a 1/`fraction`-octave window centred on each bin (bin 0 left as-is)."""
    cumulative = np.concatenate([[0.0], np.cumsum(psd)])
    half = 2 ** (1 / (2 * fraction))
    df = freqs[1] - freqs[0]
    lo_idx = np.clip(np.floor(freqs / half / df).astype(int), 0, psd.size - 1)
    hi_idx = np.clip(np.ceil(freqs * half / df).astype(int), 0, psd.size - 1)
    hi_idx = np.maximum(hi_idx, lo_idx)
    smoothed = (cumulative[hi_idx + 1] - cumulative[lo_idx]) / (hi_idx - lo_idx + 1)
    smoothed[0] = psd[0]
    return smoothed


def _log_grid(low: float, high: float) -> np.ndarray:
    octaves = math.log2(high / low)
    return low * 2 ** np.linspace(0, octaves, int(octaves * LOG_GRID_POINTS_PER_OCTAVE) + 1)


def spectral_tilt(freqs: np.ndarray, psd: np.ndarray) -> float | None:
    """Slope (dB/octave) of a least-squares line through the 1/3-octave-smoothed PSD on a log grid."""
    low, high = TILT_RANGE_HZ
    if freqs[-1] < high:
        return None
    grid = _log_grid(low, high)
    level = power_to_db(np.interp(grid, freqs, fractional_octave_smooth(freqs, psd, 3)) + 1e-30)
    slope, _ = np.polyfit(np.log2(grid), level, 1)
    return float(slope)


def find_resonances(freqs: np.ndarray, psd: np.ndarray, prominence_db: float) -> list[dict[str, float]]:
    """Narrow peaks: 1/24-octave smoothed level exceeding the 1-octave smoothed level by > prominence."""
    low, high = RESONANCE_RANGE_HZ
    high = min(high, freqs[-1] / 2 ** (1 / 2))
    narrow = power_to_db(fractional_octave_smooth(freqs, psd, RESONANCE_NARROW_OCTAVE) + 1e-30)
    wide = power_to_db(fractional_octave_smooth(freqs, psd, RESONANCE_WIDE_OCTAVE) + 1e-30)
    excess = narrow - wide
    mask = (freqs >= low) & (freqs <= high)
    peaks, _ = signal.find_peaks(np.where(mask, excess, -np.inf), height=prominence_db)
    candidates = sorted(((float(excess[i]), float(freqs[i])) for i in peaks), reverse=True)
    selected: list[dict[str, float]] = []
    for prominence, freq in candidates:
        if all(abs(math.log2(freq / s["freq_hz"])) > RESONANCE_MERGE_OCTAVES for s in selected):
            selected.append({"freq_hz": freq, "prominence_db": prominence})
        if len(selected) == MAX_RESONANCES:
            break
    return sorted(selected, key=lambda s: s["freq_hz"])


def detect_lowpass(freqs: np.ndarray, psd: np.ndarray, drop_db: float) -> float | None:
    """Highest frequency above 11 kHz after which the spectrum stays `drop_db` below the 2-8 kHz level.

    A steep shelf like this is typical of lossy encoders (≈16 kHz for 128 kbps MP3) but can also come
    from deliberate low-pass filtering or band-limited source material.
    """
    if freqs[-1] <= LOWPASS_SEARCH_FROM_HZ:
        return None
    smooth = power_to_db(fractional_octave_smooth(freqs, psd, 12) + 1e-30)
    ref_mask = (freqs >= LOWPASS_REFERENCE_BAND_HZ[0]) & (freqs < LOWPASS_REFERENCE_BAND_HZ[1])
    reference = float(np.median(smooth[ref_mask]))
    search = freqs >= LOWPASS_SEARCH_FROM_HZ
    below = smooth < reference - drop_db
    idx = np.flatnonzero(search)
    # first bin from which everything up to Nyquist (minus guard) stays below the threshold
    tail = below[idx][::-1]
    run = int(np.argmin(tail)) if not tail.all() else tail.size
    if run == 0 or run == tail.size:
        return None
    cutoff = float(freqs[idx[-run]])
    return cutoff if cutoff < LOWPASS_MAX_REPORTED_HZ else None


def analyze_spectrum(audio: AudioData, config: dict[str, Any]) -> SpectrumResult:
    fs = audio.sample_rate
    freqs, psd = welch_psd(audio.mid(), fs, config["analysis"]["welch_nperseg"])
    nyquist = fs / 2
    high_limit = min(ANALYSIS_HIGH_HZ, nyquist)
    total = band_power(freqs, psd, ANALYSIS_LOW_HZ, high_limit)

    bands = []
    for band in config["bands"]:
        low, high = float(band["low_hz"]), min(float(band["high_hz"]), nyquist)
        if low >= nyquist or total <= 0:
            bands.append(BandEnergy(band["name"], band["label"], low, float(band["high_hz"]), None, None, None))
            continue
        p = band_power(freqs, psd, low, high)
        bands.append(BandEnergy(band["name"], band["label"], low, float(band["high_hz"]), 100 * p / total,
                                float(power_to_db(p / total)), float(power_to_db(p))))

    by_name = {b.name: b for b in bands}
    group_pct: dict[str, float | None] = {}
    group_dbfs: dict[str, float | None] = {}
    for group, members in config["band_groups"].items():
        shares = [by_name[m].energy_pct for m in members if by_name[m].energy_pct is not None]
        group_pct[group] = float(sum(shares)) if shares else None
        group_dbfs[group] = (float(power_to_db(sum(shares) / 100 * total)) if shares else None)

    tilt = spectral_tilt(freqs, psd)
    analysis_mask = (freqs >= ANALYSIS_LOW_HZ) & (freqs < high_limit)
    centroid = float(np.sum(freqs[analysis_mask] * psd[analysis_mask]) / np.sum(psd[analysis_mask]))
    bass_mid = (float(power_to_db(group_pct["bass"] / group_pct["mid"]))
                if group_pct.get("bass") and group_pct.get("mid") else None)

    thresholds = config["thresholds"]
    return SpectrumResult(
        bands=bands, group_pct=group_pct, group_dbfs=group_dbfs,
        spectral_tilt=Measurement(tilt, "dB/oct",
                                  "Least-squares slope of the 1/3-oct smoothed PSD, 100 Hz-10 kHz, log grid.",
                                  Compliance.DERIVED if tilt is not None else Compliance.NOT_AVAILABLE,
                                  "Pink noise = -3 dB/oct. Brighter masters have flatter (less negative) slopes."),
        spectral_centroid=Measurement(centroid, "Hz", "Power-weighted mean frequency of the PSD, 20 Hz-20 kHz.",
                                      Compliance.DERIVED, "Strongly influenced by low-frequency energy."),
        bass_to_mid_db=Measurement(bass_mid, "dB",
                                   "10·log10(energy of 'bass' group / energy of 'mid' group).",
                                   Compliance.DERIVED if bass_mid is not None else Compliance.NOT_AVAILABLE),
        resonances=find_resonances(freqs, psd, thresholds["resonance_prominence_db"]),
        lowpass_cutoff_hz=detect_lowpass(freqs, psd, thresholds["lossy_lowpass_drop_db"]),
        freqs_hz=freqs, psd_db=power_to_db(psd + 1e-30),
    )
