"""Track vs reference comparison.

Three views per reference:
  1. raw          — metrics as measured (playback level included),
  2. matched      — the reference virtually gained by (track LUFS-I - reference LUFS-I) dB, so that
                    level-dependent metrics compare at equal integrated loudness (same BS.1770 meter
                    for both files). Audio files are never modified; this is arithmetic on the metrics,
                    exact because a gain change shifts every dB level by the same amount.
  3. fragment     — loudest-fragment analysis of both files (same window length).
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from .analyzer import HEADLINE_METRICS, METRICS_BY_KEY, TrackAnalysis, headline_metrics


@dataclass
class ComparisonRow:
    key: str
    label: str
    unit: str
    track: float | None
    reference: float | None
    difference: float | None  # track - reference


@dataclass
class ReferenceComparison:
    reference_name: str
    rows: list[ComparisonRow]
    matched_gain_db: float | None  # gain applied (virtually) to the reference
    matched_rows: list[ComparisonRow]
    fragment_rows: list[ComparisonRow]
    notes: list[str] = field(default_factory=list)


@dataclass
class ReferenceStat:
    key: str
    label: str
    unit: str
    track: float | None
    values: list[float]
    median: float | None
    minimum: float | None
    maximum: float | None
    diff_to_median: float | None
    outside_range: bool


def _diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
        return None
    return a - b


def _rows(track: dict[str, float | None], ref: dict[str, float | None]) -> list[ComparisonRow]:
    return [ComparisonRow(m.key, m.label, m.unit, track.get(m.key), ref.get(m.key), _diff(track.get(m.key), ref.get(m.key)))
            for m in HEADLINE_METRICS]


def matched_rows(track: TrackAnalysis, reference: TrackAnalysis) -> tuple[float | None, list[ComparisonRow]]:
    """Level-dependent metrics and band levels with the reference gained to the track's LUFS-I."""
    gain = _diff(track.loudness.integrated.value, reference.loudness.integrated.value)
    if gain is None:
        return None, []
    t, r = headline_metrics(track), headline_metrics(reference)
    rows = []
    for metric in HEADLINE_METRICS:
        if metric.level_dependent and metric.key != "integrated_lufs":
            ref_value = None if r[metric.key] is None else r[metric.key] + gain
            rows.append(ComparisonRow(metric.key, metric.label + " (matched)", metric.unit, t[metric.key], ref_value,
                                      _diff(t[metric.key], ref_value)))
    for group, track_level in track.spectrum.group_dbfs.items():
        ref_level = reference.spectrum.group_dbfs.get(group)
        ref_matched = None if ref_level is None else ref_level + gain
        rows.append(ComparisonRow(f"{group}_level_db", f"{group.capitalize()} band level (matched)", "dBFS",
                                  track_level, ref_matched, _diff(track_level, ref_matched)))
    for band_t, band_r in zip(track.spectrum.bands, reference.spectrum.bands):
        ref_matched = None if band_r.level_dbfs is None else band_r.level_dbfs + gain
        rows.append(ComparisonRow(f"band_{band_t.name}_db", f"{band_t.label} {band_t.low_hz:.0f}-{band_t.high_hz:.0f} Hz "
                                  "(matched)", "dBFS", band_t.level_dbfs, ref_matched, _diff(band_t.level_dbfs, ref_matched)))
    return gain, rows


def compare_tracks(track: TrackAnalysis, reference: TrackAnalysis) -> ReferenceComparison:
    gain, matched = matched_rows(track, reference)
    notes = []
    fragment_rows: list[ComparisonRow] = []
    if track.loudest_fragment and reference.loudest_fragment:
        fragment_rows = _rows(headline_metrics(track.loudest_fragment), headline_metrics(reference.loudest_fragment))
    else:
        notes.append("Loudest-fragment comparison unavailable (a file is not longer than the fragment window).")
    if track.info.sample_rate != reference.info.sample_rate:
        notes.append(f"Sample rates differ ({track.info.sample_rate} vs {reference.info.sample_rate} Hz); "
                     "metrics are rate-independent but true-peak oversampling factors may differ.")
    if reference.info.is_lossy_source:
        notes.append("Reference is a lossy file: its true peak and top-octave content reflect the codec.")
    return ReferenceComparison(reference.info.name, _rows(headline_metrics(track), headline_metrics(reference)),
                               gain, matched, fragment_rows, notes)


def summarize_references(track: TrackAnalysis, references: list[TrackAnalysis]) -> dict[str, ReferenceStat]:
    """Track value vs min / median / max over all references, per headline metric."""
    track_metrics = headline_metrics(track)
    ref_metrics = [headline_metrics(r) for r in references]
    summary = {}
    for key, value in track_metrics.items():
        values = [m[key] for m in ref_metrics if m[key] is not None and math.isfinite(m[key])]
        median = statistics.median(values) if values else None
        lo, hi = (min(values), max(values)) if values else (None, None)
        outside = bool(values and value is not None and math.isfinite(value) and not lo <= value <= hi)
        metric = METRICS_BY_KEY[key]
        summary[key] = ReferenceStat(key, metric.label, metric.unit, value, values, median, lo, hi,
                                     _diff(value, median), outside)
    return summary


def matched_band_differences(comparisons: list[ReferenceComparison]) -> dict[str, list[float]]:
    """Per band group: loudness-matched level differences (track - reference) across references."""
    result: dict[str, list[float]] = {}
    for comparison in comparisons:
        for row in comparison.matched_rows:
            if row.key.endswith("_level_db") and row.difference is not None:
                result.setdefault(row.key.removesuffix("_level_db"), []).append(row.difference)
    return result
