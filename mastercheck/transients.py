"""Clipping, flat-topping and transient prominence.

Directly detected (facts about the file):
  * samples above 0 dBFS (possible only in float files),
  * samples at/near full scale and runs of consecutive full-scale samples.
Suspected (indicators, not proof):
  * flat-top runs below full scale — consecutive samples pinned at the file's peak level with steep
    entry/exit edges (a smooth peak approaches its maximum in ~1 LSB steps), typical of
    clipping followed by a gain change, or of a hard clipper / zero-lookahead limiter;
  * low transient prominence — peaks barely above the surrounding RMS.
A finished file cannot prove at which production stage clipping happened.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter1d

from .audio_loader import AudioData
from .models import Compliance, Measurement, amp_to_db, power_to_db

TRANSIENT_PEAK_WINDOW_S = 0.01
TRANSIENT_CONTEXT_WINDOW_S = 0.3
MAX_REPORTED_RUN_TIMES = 100
PLATEAU_MIN_EDGE_STEP = 4 / 32768  # four 16-bit LSBs (~ -78 dBFS)


@dataclass
class RunStats:
    count: int
    longest: int
    times_s: list[float]


@dataclass
class TransientResult:
    samples_over_full_scale: int
    samples_at_full_scale: int
    full_scale_runs: RunStats
    flat_top_runs: RunStats
    flat_top_level_dbfs: float | None
    clipping_status: str  # "detected" | "suspected" | "not_detected"
    transient_prominence_median: Measurement
    transient_prominence_p95: Measurement


def find_runs(mask: np.ndarray, min_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Start indices and lengths of True-runs of at least `min_length`."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    starts, ends = edges[::2], edges[1::2]
    lengths = ends - starts
    keep = lengths >= min_length
    return starts[keep], lengths[keep]


def sharp_plateaus(magnitude: np.ndarray, mask: np.ndarray, min_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Runs of `mask` whose entry and exit steps are steep.

    A smooth peak (e.g. low-frequency sine, quantized to 16 bit) can stay at its maximum for a few
    samples, but it approaches that level in steps of about one LSB. A clipped waveform hits the
    plateau with the slope the unclipped signal still had. Runs touching the file edges are ignored.
    """
    starts, lengths = find_runs(mask, min_length)
    ends = starts + lengths
    inside = (starts > 0) & (ends < magnitude.size)
    starts, lengths, ends = starts[inside], lengths[inside], ends[inside]
    rise = magnitude[starts] - magnitude[starts - 1]
    fall = magnitude[ends - 1] - magnitude[ends]
    keep = np.minimum(rise, fall) > PLATEAU_MIN_EDGE_STEP
    return starts[keep], lengths[keep]


def _run_stats(magnitude: np.ndarray, mask: np.ndarray, min_length: int, sample_rate: int,
               offset_s: float) -> RunStats:
    """Sharp plateaus over all channels (columns of `magnitude` / `mask`)."""
    found = [sharp_plateaus(magnitude[:, c], mask[:, c], min_length) for c in range(magnitude.shape[1])]
    starts = np.concatenate([f[0] for f in found])
    lengths = np.concatenate([f[1] for f in found])
    times = sorted(set(np.round(starts / sample_rate + offset_s, 3).tolist()))
    return RunStats(int(starts.size), int(lengths.max()) if lengths.size else 0, times[:MAX_REPORTED_RUN_TIMES])


def transient_prominence(samples: np.ndarray, sample_rate: int, silence_floor_dbfs: float) -> np.ndarray:
    """Per 10 ms frame: peak (dBFS) minus RMS of the surrounding 300 ms (dBFS). Silent frames dropped."""
    frame = max(1, int(TRANSIENT_PEAK_WINDOW_S * sample_rate))
    count = samples.shape[0] // frame
    if count < 3:
        return np.empty(0)
    trimmed = samples[:count * frame]
    peak = np.max(np.abs(trimmed).reshape(count, frame, -1), axis=(1, 2))
    frame_ms = np.mean((trimmed ** 2).reshape(count, frame, -1), axis=(1, 2))
    context = uniform_filter1d(frame_ms, size=max(1, int(TRANSIENT_CONTEXT_WINDOW_S / TRANSIENT_PEAK_WINDOW_S)),
                               mode="nearest")
    context_db = power_to_db(context + 1e-20)
    keep = context_db > silence_floor_dbfs
    return (amp_to_db(peak + 1e-20) - context_db)[keep]


def analyze_transients(audio: AudioData, config: dict[str, Any]) -> TransientResult:
    thresholds = config["thresholds"]
    min_run = int(thresholds["clip_run_min_samples"])
    fs = audio.sample_rate
    magnitude = np.abs(audio.samples)

    over = int(np.sum(magnitude > 1.0))
    full_scale_level = 10 ** (thresholds["full_scale_dbfs"] / 20)
    at_full_scale = magnitude >= full_scale_level
    fs_runs = _run_stats(magnitude, at_full_scale, min_run, fs, audio.offset_s)

    peak = float(magnitude.max())
    flat_level = None
    flat_runs = RunStats(0, 0, [])
    if peak < full_scale_level:
        tolerance = 10 ** (-thresholds["flat_top_tolerance_db"] / 20)
        near_peak = magnitude >= peak * tolerance
        flat_runs = _run_stats(magnitude, near_peak, min_run, fs, audio.offset_s)
        flat_level = float(amp_to_db(peak))

    # A smooth low-frequency peak can legitimately sit at the maximum for a few samples once; clipping
    # pins the waveform to the same level again and again. Hence minimum run counts, not just run lengths.
    if over > 0 or fs_runs.count >= thresholds["clip_min_runs"]:
        status = "detected"
    elif fs_runs.count > 0 or flat_runs.count >= thresholds["flat_top_min_runs"]:
        status = "suspected"
    else:
        status = "not_detected"

    prominence = transient_prominence(audio.samples, fs, config["analysis"]["silence_floor_dbfs"])
    has = prominence.size > 0
    method = (f"Per {TRANSIENT_PEAK_WINDOW_S * 1000:.0f} ms frame: peak dBFS minus RMS dBFS of the surrounding "
              f"{TRANSIENT_CONTEXT_WINDOW_S * 1000:.0f} ms.")
    return TransientResult(
        samples_over_full_scale=over,
        samples_at_full_scale=int(at_full_scale.sum()),
        full_scale_runs=fs_runs, flat_top_runs=flat_runs, flat_top_level_dbfs=flat_level,
        clipping_status=status,
        transient_prominence_median=Measurement(float(np.median(prominence)) if has else None, "dB",
                                                method + " Median.",
                                                Compliance.HEURISTIC if has else Compliance.NOT_AVAILABLE),
        transient_prominence_p95=Measurement(float(np.percentile(prominence, 95)) if has else None, "dB",
                                             method + " 95th percentile (how far the strongest hits stick out).",
                                             Compliance.HEURISTIC if has else Compliance.NOT_AVAILABLE,
                                             "Low values mean transients are flattened relative to the body."),
    )
