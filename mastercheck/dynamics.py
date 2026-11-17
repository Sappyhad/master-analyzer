"""Dynamics and limiting indicators.

Everything here is computed from the finished file only. None of it measures limiter gain reduction:
without the pre-limiter mix that quantity cannot be recovered. The indicators describe *how
compressed the result looks*, and they are labelled heuristic where appropriate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter1d

from .audio_loader import AudioData
from .loudness import ABSOLUTE_GATE_LUFS, BLOCK_HOP_S, MOMENTARY_WINDOW_S, SHORT_TERM_WINDOW_S, LoudnessResult
from .models import Compliance, Measurement, amp_to_db, power_to_db

PEAK_DENSITY_WINDOW_DB = 1.0
SECTION_SMOOTHING_S = 3.0
PUMPING_FRAME_S = 0.05
PUMPING_DETREND_S = 1.5
PUMPING_LOW_CUTOFF_HZ = 120.0
PUMPING_HIGH_BAND_HZ = (2000.0, 8000.0)
KMEANS_ITERATIONS = 50


@dataclass
class Section:
    start_s: float
    end_s: float
    level: str  # "higher" | "lower" — relative energy level, NOT verse/chorus labels
    mean_lufs: float


@dataclass
class DynamicsResult:
    short_term_median_lufs: float | None
    short_term_p10_lufs: float | None
    short_term_p90_lufs: float | None
    short_term_std_lu: float | None
    windowed_crest_median: Measurement
    windowed_crest_p10_db: float | None
    windowed_crest_p90_db: float | None
    low_crest_fraction: Measurement
    peak_density: Measurement
    sections: list[Section]
    section_spread: Measurement
    loudest_fragment_start_s: float
    loudest_fragment_end_s: float
    loudest_fragment_lufs: float | None
    pumping_indicator: Measurement
    crest_times_s: np.ndarray = field(repr=False)
    crest_db: np.ndarray = field(repr=False)


def _power_mean_lufs(values: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(10 ** (values / 10))))


def windowed_crest(samples: np.ndarray, sample_rate: int, window_s: float,
                   silence_floor_dbfs: float) -> tuple[np.ndarray, np.ndarray]:
    """Crest factor (peak dBFS - RMS dBFS) of non-overlapping windows; silent windows dropped."""
    window = int(round(window_s * sample_rate))
    count = samples.shape[0] // window
    if count == 0:
        return np.empty(0), np.empty(0)
    blocks = samples[:count * window].reshape(count, window, samples.shape[1])
    peak_db = amp_to_db(np.max(np.abs(blocks), axis=(1, 2)))
    rms_db = power_to_db(np.mean(blocks ** 2, axis=(1, 2)))
    keep = rms_db > silence_floor_dbfs
    times = (np.arange(count) + 0.5) * window_s
    return times[keep], (peak_db - rms_db)[keep]


def _two_level_split(values: np.ndarray) -> tuple[float, float, np.ndarray]:
    """1-D k-means with k=2. Returns (low centroid, high centroid, boolean 'is high' labels)."""
    low, high = np.percentile(values, 25), np.percentile(values, 75)
    labels = values > (low + high) / 2
    for _ in range(KMEANS_ITERATIONS):
        if labels.all() or not labels.any():
            break
        low, high = values[~labels].mean(), values[labels].mean()
        new_labels = values > (low + high) / 2
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return float(low), float(high), labels


def _merge_short_runs(labels: np.ndarray, min_blocks: int) -> np.ndarray:
    """Relabel runs shorter than `min_blocks` to their longer neighbour, shortest first."""
    labels = labels.copy()
    while True:
        edges = np.flatnonzero(np.diff(labels.astype(int))) + 1
        starts = np.concatenate([[0], edges])
        ends = np.concatenate([edges, [labels.size]])
        lengths = ends - starts
        if lengths.size <= 1 or lengths.min() >= min_blocks:
            return labels
        i = int(np.argmin(lengths))
        left = lengths[i - 1] if i > 0 else -1
        right = lengths[i + 1] if i + 1 < lengths.size else -1
        neighbour = i - 1 if left >= right else i + 1
        labels[starts[i]:ends[i]] = labels[starts[neighbour]]


def detect_sections(times: np.ndarray, short_term: np.ndarray, integrated: float,
                    min_duration_s: float, min_difference_lu: float) -> list[Section]:
    """Split the track into contiguous higher/lower-energy sections from LUFS-S.

    Blocks below the LRA-style relative gate (integrated - 20 LU) are ignored when finding levels.
    Returns [] when the track has no clearly distinct energy levels.
    """
    if short_term.size < 2 or not math.isfinite(integrated):
        return []
    smooth_blocks = max(1, int(round(SECTION_SMOOTHING_S / BLOCK_HOP_S)))
    smoothed = median_filter(short_term, size=smooth_blocks, mode="nearest")
    active = (smoothed > ABSOLUTE_GATE_LUFS) & (smoothed > integrated - 20)
    if active.sum() < 2:
        return []
    low, high, _ = _two_level_split(smoothed[active])
    if high - low < min_difference_lu:
        return []
    labels = smoothed > (low + high) / 2
    labels = _merge_short_runs(labels, max(1, int(round(min_duration_s / BLOCK_HOP_S))))
    edges = np.flatnonzero(np.diff(labels.astype(int))) + 1
    centers = times - SHORT_TERM_WINDOW_S / 2
    sections = []
    for s, e in zip(np.concatenate([[0], edges]), np.concatenate([edges, [labels.size]])):
        segment = short_term[s:e]
        segment = segment[np.isfinite(segment)]
        sections.append(Section(
            start_s=float(times[0] - SHORT_TERM_WINDOW_S if s == 0 else centers[s]),
            end_s=float(times[-1] if e == labels.size else centers[e]),
            level="higher" if labels[s] else "lower",
            mean_lufs=_power_mean_lufs(segment) if segment.size else -math.inf,
        ))
    return sections


def loudest_fragment(times: np.ndarray, momentary: np.ndarray, duration_s: float,
                     total_s: float, offset_s: float) -> tuple[float, float, float | None]:
    """Window of `duration_s` with the highest energy-mean momentary loudness.
    Returns (start, end, LUFS) in seconds relative to the AudioData start."""
    if total_s <= duration_s or momentary.size == 0:
        return 0.0, total_s, None
    blocks = max(1, int(round(duration_s / BLOCK_HOP_S)))
    power = 10 ** (np.nan_to_num(momentary, neginf=-200.0) / 10)
    if power.size <= blocks:
        return 0.0, total_s, None
    cumulative = np.concatenate([[0.0], np.cumsum(power)])
    running = (cumulative[blocks:] - cumulative[:-blocks]) / blocks  # running[i] = mean(power[i:i+blocks])
    best = int(np.argmax(running))
    start = max(0.0, float(times[best] - MOMENTARY_WINDOW_S - offset_s))
    end = min(total_s, start + duration_s)
    return end - duration_s if end - start < duration_s else start, end, float(10 * np.log10(running[best]))


def pumping_indicator(mid: np.ndarray, sample_rate: int, silence_floor_dbfs: float) -> Measurement:
    """Correlation between short-term fluctuations of the low band and the 2-8 kHz band envelopes.

    A broadband compressor/limiter driven by kick/bass pulls the upper band down on each low-end hit,
    producing a negative correlation. Musical content usually correlates positively (kick and hats
    land together), so a clearly negative value is only a *hint* of audible pumping.
    """
    method = ("Pearson correlation of detrended 50 ms RMS envelopes: <120 Hz band vs 2-8 kHz band "
              "(1.5 s moving-average detrending).")
    frame = int(PUMPING_FRAME_S * sample_rate)
    nyquist = sample_rate / 2
    if mid.size < frame * 40 or PUMPING_HIGH_BAND_HZ[1] >= nyquist:
        return Measurement(None, "r", method, Compliance.NOT_AVAILABLE, "Signal too short or sample rate too low.")
    low = signal.sosfilt(signal.butter(4, PUMPING_LOW_CUTOFF_HZ, "lowpass", fs=sample_rate, output="sos"), mid)
    high = signal.sosfilt(signal.butter(4, PUMPING_HIGH_BAND_HZ, "bandpass", fs=sample_rate, output="sos"), mid)
    count = mid.size // frame
    low_db = power_to_db(np.mean(low[:count * frame].reshape(count, frame) ** 2, axis=1) + 1e-20)
    high_db = power_to_db(np.mean(high[:count * frame].reshape(count, frame) ** 2, axis=1) + 1e-20)
    keep = (low_db > silence_floor_dbfs) & (high_db > silence_floor_dbfs)
    if keep.sum() < 40:
        return Measurement(None, "r", method, Compliance.NOT_AVAILABLE,
                           "Not enough frames with energy in both bands.")
    detrend = max(3, int(PUMPING_DETREND_S / PUMPING_FRAME_S))
    low_fluct = (low_db - uniform_filter1d(low_db, detrend))[keep]
    high_fluct = (high_db - uniform_filter1d(high_db, detrend))[keep]
    if np.std(low_fluct) == 0 or np.std(high_fluct) == 0:
        return Measurement(None, "r", method, Compliance.NOT_AVAILABLE, "Flat envelope.")
    r = float(np.corrcoef(low_fluct, high_fluct)[0, 1])
    return Measurement(r, "r", method, Compliance.HEURISTIC,
                       "Negative values hint at low-end-triggered gain modulation; confirm by listening "
                       "(sustained pads/cymbals ducking on kicks). Arrangement can cause the same pattern.")


def analyze_dynamics(audio: AudioData, loudness: LoudnessResult, config: dict[str, Any]) -> DynamicsResult:
    cfg = config["analysis"]
    thresholds = config["thresholds"]
    silence = cfg["silence_floor_dbfs"]
    integrated = loudness.integrated.value if loudness.integrated.ok else -math.inf

    st = loudness.short_term_lufs
    active_st = st[(st > ABSOLUTE_GATE_LUFS) & (st > integrated - 20)] if st.size else st
    st_stats = ((float(np.median(active_st)), float(np.percentile(active_st, 10)),
                 float(np.percentile(active_st, 90)), float(np.std(active_st)))
                if active_st.size >= 2 else (None, None, None, None))

    crest_times, crest = windowed_crest(audio.samples, audio.sample_rate, cfg["crest_window_s"], silence)
    has_crest = crest.size > 0
    crest_median = Measurement(
        float(np.median(crest)) if has_crest else None, "dB",
        f"Median over non-silent {cfg['crest_window_s'] * 1000:.0f} ms windows of (peak dBFS - RMS dBFS).",
        Compliance.DERIVED if has_crest else Compliance.NOT_AVAILABLE,
        "Less sensitive to silence/fades than the whole-file crest factor.")
    low_fraction = float(np.mean(crest < thresholds["windowed_crest_low_db"])) if has_crest else None
    low_crest = Measurement(
        low_fraction, "fraction",
        f"Share of windows with crest factor below {thresholds['windowed_crest_low_db']} dB.",
        Compliance.HEURISTIC if has_crest else Compliance.NOT_AVAILABLE,
        "High values indicate sustained dense/compressed material. Says nothing about how it was achieved.")

    abs_samples = np.max(np.abs(audio.samples), axis=1)
    peak = abs_samples.max()
    density = float(np.mean(abs_samples >= peak * 10 ** (-PEAK_DENSITY_WINDOW_DB / 20)) * 100)
    peak_density = Measurement(
        density, "%", f"Percentage of sample frames within {PEAK_DENSITY_WINDOW_DB} dB of the sample peak.",
        Compliance.HEURISTIC,
        "Unprocessed material is typically far below 0.1 %; brick-wall limited material piles up at the "
        "ceiling. An indicator of limiting density, not of gain reduction.")

    sections = detect_sections(loudness.short_term_times_s, st, integrated, cfg["section_min_duration_s"],
                               cfg["section_min_level_difference_lu"])
    finite_levels = [s.mean_lufs for s in sections if math.isfinite(s.mean_lufs)]
    spread = max(finite_levels) - min(finite_levels) if len(finite_levels) >= 2 else None
    section_spread = Measurement(
        spread, "LU",
        "Difference between the loudest and quietest detected section (energy mean of LUFS-S). Sections "
        "come from 2-level clustering of 3 s-median-smoothed LUFS-S.",
        Compliance.HEURISTIC if spread is not None else Compliance.NOT_AVAILABLE,
        "Sections are energy levels, not verse/chorus labels." if spread is not None
        else "No clearly distinct loudness levels detected.")

    frag_start, frag_end, frag_lufs = loudest_fragment(
        loudness.momentary_times_s, loudness.momentary_lufs, cfg["loudest_fragment_s"], audio.duration_s,
        audio.offset_s)

    return DynamicsResult(
        short_term_median_lufs=st_stats[0], short_term_p10_lufs=st_stats[1], short_term_p90_lufs=st_stats[2],
        short_term_std_lu=st_stats[3], windowed_crest_median=crest_median,
        windowed_crest_p10_db=float(np.percentile(crest, 10)) if has_crest else None,
        windowed_crest_p90_db=float(np.percentile(crest, 90)) if has_crest else None,
        low_crest_fraction=low_crest, peak_density=peak_density, sections=sections,
        section_spread=section_spread, loudest_fragment_start_s=frag_start, loudest_fragment_end_s=frag_end,
        loudest_fragment_lufs=frag_lufs, pumping_indicator=pumping_indicator(audio.mid(), audio.sample_rate, silence),
        crest_times_s=crest_times + audio.offset_s, crest_db=crest,
    )
