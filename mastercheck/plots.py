"""PNG charts for the report (matplotlib, non-interactive Agg backend).

Colour roles: the analyzed track is always series 1 (blue); references take the next fixed slots,
and references beyond the third are drawn in neutral grey so that colours stay distinguishable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected first)
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

from .analyzer import METRICS_BY_KEY, TrackAnalysis, headline_metrics  # noqa: E402
from .audio_loader import AudioData  # noqa: E402
from .comparison import ReferenceComparison  # noqa: E402
from .spectrum import fractional_octave_smooth  # noqa: E402

log = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
TRACK_COLOR = "#2a78d6"
REFERENCE_COLORS = ("#eb6834", "#1baf7a", "#eda100")
OVERFLOW_REFERENCE_COLOR = "#a3a29c"
WARNING_COLOR = "#e34948"
SECTION_FILL = "#2a78d6"
DPI = 120
MAX_SPECTROGRAM_FRAMES = 2000
MAX_WAVEFORM_POINTS = 4000
FREQ_LIMITS = (20, 20000)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY, "axes.titlecolor": TEXT,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.4,
})


def reference_color(index: int) -> str:
    return REFERENCE_COLORS[index] if index < len(REFERENCE_COLORS) else OVERFLOW_REFERENCE_COLOR


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _freq_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_xlim(*FREQ_LIMITS)
    ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"])
    ax.set_xlabel("Frequency (Hz)")


def plot_waveform(audio: AudioData, path: Path) -> Path:
    channels = audio.channels
    fig, axes = plt.subplots(channels, 1, figsize=(11, 2.2 * channels), sharex=True, squeeze=False)
    bucket = max(1, audio.n_frames // MAX_WAVEFORM_POINTS)
    count = audio.n_frames // bucket
    times = audio.offset_s + (np.arange(count) + 0.5) * bucket / audio.sample_rate
    names = ["Mono"] if channels == 1 else ["Left", "Right"]
    for c, ax in enumerate(axes[:, 0]):
        blocks = audio.samples[:count * bucket, c].reshape(count, bucket)
        ax.fill_between(times, blocks.min(axis=1), blocks.max(axis=1), color=TRACK_COLOR, linewidth=0)
        ax.axhline(1.0, color=WARNING_COLOR, linewidth=0.8, linestyle="--")
        ax.axhline(-1.0, color=WARNING_COLOR, linewidth=0.8, linestyle="--")
        ax.set_ylim(-1.1, 1.1)
        ax.set_ylabel(f"{names[c]} (linear)")
    axes[0, 0].set_title("Waveform (min/max envelope; dashed = 0 dBFS)")
    axes[-1, 0].set_xlabel("Time (s)")
    return _save(fig, path)


def plot_loudness(track: TrackAnalysis, path: Path) -> Path:
    lo, dyn = track.loudness, track.dynamics
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for section in dyn.sections:
        if section.level == "higher":
            ax.axvspan(section.start_s, section.end_s, color=SECTION_FILL, alpha=0.07, linewidth=0)
    ax.plot(lo.momentary_times_s, lo.momentary_lufs, color=OVERFLOW_REFERENCE_COLOR, linewidth=0.8,
            label="Momentary (400 ms)")
    if lo.short_term_lufs.size:
        ax.plot(lo.short_term_times_s, lo.short_term_lufs, color=TRACK_COLOR, linewidth=1.8, label="Short-term (3 s)")
    if lo.integrated.ok:
        ax.axhline(lo.integrated.value, color=TEXT, linewidth=1, linestyle="--",
                   label=f"Integrated {lo.integrated.value:.1f} LUFS")
    finite = lo.momentary_lufs[np.isfinite(lo.momentary_lufs)]
    if finite.size:
        ax.set_ylim(max(finite.min(), lo.integrated.value - 30 if lo.integrated.ok else -60) - 2, finite.max() + 2)
    ax.set_title("Loudness over time (shaded = higher-energy sections)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("LUFS")
    ax.legend(loc="lower right", ncol=3)
    return _save(fig, path)


def plot_spectrum(track: TrackAnalysis, config: dict[str, Any], path: Path) -> Path:
    sp = track.spectrum
    fig, ax = plt.subplots(figsize=(11, 4))
    smoothed = 10 * np.log10(fractional_octave_smooth(sp.freqs_hz, 10 ** (sp.psd_db / 10),
                                                      config["analysis"]["plot_spectrum_smoothing_octave"]) + 1e-30)
    ax.plot(sp.freqs_hz[1:], smoothed[1:], color=TRACK_COLOR, linewidth=1.6)
    for band in config["bands"][1:]:
        ax.axvline(band["low_hz"], color=GRID, linewidth=1)
    for band in sp.bands:
        if band.energy_pct is not None:
            center = np.sqrt(band.low_hz * min(band.high_hz, FREQ_LIMITS[1]))
            ax.text(center, 0.97, f"{band.label}\n{band.energy_pct:.1f}%", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7, color=TEXT_SECONDARY)
    for res in sp.resonances:
        ax.axvline(res["freq_hz"], color=WARNING_COLOR, linewidth=0.8, linestyle=":")
    _freq_axis(ax)
    ax.set_ylabel("PSD (dB re FS²/Hz)")
    ax.set_title(f"Average spectrum (mid, Welch, 1/{config['analysis']['plot_spectrum_smoothing_octave']}-oct smoothed; "
                 "dotted = narrow peaks)")
    return _save(fig, path)


def plot_spectrogram(audio: AudioData, config: dict[str, Any], path: Path) -> Path:
    nperseg = min(config["analysis"]["spectrogram_nperseg"], audio.n_frames)
    hop = max(nperseg // 4, audio.n_frames // MAX_SPECTROGRAM_FRAMES)
    noverlap = max(0, nperseg - hop)
    freqs, times, power = signal.spectrogram(audio.mid(), fs=audio.sample_rate, window="hann", nperseg=nperseg,
                                             noverlap=noverlap, scaling="density")
    level = 10 * np.log10(power + 1e-20)
    top = np.percentile(level, 99.5)
    fig, ax = plt.subplots(figsize=(11, 4))
    mesh = ax.pcolormesh(times + audio.offset_s, freqs[1:], level[1:], shading="auto", cmap="magma",
                         vmin=top - 90, vmax=top, rasterized=True)
    ax.set_yscale("log")
    ax.set_ylim(FREQ_LIMITS[0], min(FREQ_LIMITS[1], audio.sample_rate / 2))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram (mid)")
    ax.grid(False)
    fig.colorbar(mesh, ax=ax, label="dB re FS²/Hz", pad=0.01)
    return _save(fig, path)


def plot_stereo(track: TrackAnalysis, path: Path) -> Path:
    st = track.stereo
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True)
    ax1.plot(st.times_s, st.correlation_series, color=TRACK_COLOR)
    ax1.axhline(0, color=WARNING_COLOR, linewidth=0.8, linestyle="--")
    for region in st.problem_regions:
        ax1.axvspan(region["start_s"], region["end_s"], color=WARNING_COLOR, alpha=0.12, linewidth=0)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_ylabel("Correlation r")
    ax1.set_title("Stereo correlation over time (red = below threshold)")
    ax2.plot(st.times_s, st.side_to_mid_series_db, color=TRACK_COLOR)
    ax2.set_ylabel("Side/Mid (dB)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Stereo width over time")
    return _save(fig, path)


def plot_dynamics(track: TrackAnalysis, path: Path) -> Path:
    dyn = track.dynamics
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(dyn.crest_times_s, dyn.crest_db, color=TRACK_COLOR, linewidth=1)
    if dyn.windowed_crest_median.ok:
        ax.axhline(dyn.windowed_crest_median.value, color=TEXT, linestyle="--", linewidth=1,
                   label=f"Median {dyn.windowed_crest_median.value:.1f} dB")
        ax.legend(loc="upper right")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Crest factor (dB)")
    ax.set_title("Crest factor over time (400 ms windows)")
    return _save(fig, path)


def _files_with_gain(track: TrackAnalysis, references: list[TrackAnalysis],
                     comparisons: list[ReferenceComparison]) -> list[tuple[str, TrackAnalysis, float, str]]:
    items = [(f"{track.info.name} (track)", track, 0.0, TRACK_COLOR)]
    for i, (ref, comp) in enumerate(zip(references, comparisons)):
        items.append((ref.info.name, ref, comp.matched_gain_db or 0.0, reference_color(i)))
    return items


def plot_comparison_spectrum(track: TrackAnalysis, references: list[TrackAnalysis],
                             comparisons: list[ReferenceComparison], config: dict[str, Any], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    fraction = config["analysis"]["plot_spectrum_smoothing_octave"]
    for label, analysis, gain, color in reversed(_files_with_gain(track, references, comparisons)):
        sp = analysis.spectrum
        smoothed = 10 * np.log10(fractional_octave_smooth(sp.freqs_hz, 10 ** (sp.psd_db / 10), fraction) + 1e-30)
        is_track = analysis is track
        ax.plot(sp.freqs_hz[1:], smoothed[1:] + gain, color=color, linewidth=2.0 if is_track else 1.2,
                label=label if is_track else f"{label} ({gain:+.1f} dB)")
    _freq_axis(ax)
    ax.set_ylabel("PSD (dB re FS²/Hz)")
    ax.set_title("Spectrum comparison at matched integrated loudness (gain applied to references shown)")
    ax.legend(loc="lower left")
    return _save(fig, path)


def plot_comparison_loudness(track: TrackAnalysis, references: list[TrackAnalysis], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 3.8))
    items = [(f"{track.info.name} (track)", track, TRACK_COLOR)] + \
            [(r.info.name, r, reference_color(i)) for i, r in enumerate(references)]
    for label, analysis, color in reversed(items):
        lo = analysis.loudness
        if lo.short_term_lufs.size:
            ax.plot(lo.short_term_times_s - analysis.info.offset_s, lo.short_term_lufs, color=color,
                    linewidth=1.8 if color == TRACK_COLOR else 1.1, label=label)
    ax.set_xlabel("Time from start of analyzed range (s)")
    ax.set_ylabel("LUFS-S")
    ax.set_title("Short-term loudness over time (no gain matching)")
    ax.legend(loc="lower right")
    finite = np.concatenate([a.loudness.short_term_lufs[np.isfinite(a.loudness.short_term_lufs)] for _, a, _ in items])
    if finite.size:
        ax.set_ylim(max(finite.min(), finite.max() - 35), finite.max() + 2)
    return _save(fig, path)


def plot_comparison_bands(track: TrackAnalysis, references: list[TrackAnalysis],
                          comparisons: list[ReferenceComparison], path: Path) -> Path:
    labels = [b.label for b in track.spectrum.bands]
    fig, ax = plt.subplots(figsize=(11, 4))
    width = 0.8 / max(1, len(comparisons))
    x = np.arange(len(labels))
    for i, comp in enumerate(comparisons):
        diffs = [next((r.difference for r in comp.matched_rows if r.key == f"band_{b.name}_db"), None)
                 for b in track.spectrum.bands]
        values = np.array([np.nan if d is None else d for d in diffs])
        ax.bar(x + (i - (len(comparisons) - 1) / 2) * width, values, width=width * 0.9,
               color=reference_color(i), label=f"vs {comp.reference_name}")
    ax.axhline(0, color=TEXT, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Track - reference (dB)")
    ax.set_title("Band level difference at matched loudness (positive = track has more)")
    ax.legend(loc="best")
    return _save(fig, path)


def plot_comparison_metrics(track: TrackAnalysis, references: list[TrackAnalysis], path: Path) -> Path:
    keys = ["integrated_lufs", "true_peak_dbtp", "lra_lu", "windowed_crest_db", "plr_db", "side_to_mid_db"]
    items = [("Track", track, TRACK_COLOR)] + [(r.info.name, r, reference_color(i)) for i, r in enumerate(references)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4))
    for ax, key in zip(axes.flat, keys):
        metric = METRICS_BY_KEY[key]
        values = [headline_metrics(a)[key] for _, a, _ in items]
        values = [np.nan if v is None or not np.isfinite(v) else v for v in values]
        ax.barh(range(len(items)), values, color=[c for _, _, c in items], height=0.6)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([n[:22] for n, _, _ in items], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"{metric.label} ({metric.unit})", fontsize=9)
        for i, v in enumerate(values):
            if np.isfinite(v):
                ax.text(v, i, f" {v:.1f}", va="center", fontsize=7, color=TEXT)
    fig.suptitle("Dynamics, peak and width comparison (raw values)", fontsize=11, fontweight="bold", color=TEXT)
    return _save(fig, path)


def generate_plots(audio: AudioData, track: TrackAnalysis, references: list[TrackAnalysis],
                   comparisons: list[ReferenceComparison], config: dict[str, Any], out_dir: Path) -> dict[str, str]:
    """Create every chart; failures are logged and skipped. Returns {chart key: file name}."""
    jobs: dict[str, Callable[[Path], Path]] = {
        "waveform": lambda p: plot_waveform(audio, p),
        "loudness": lambda p: plot_loudness(track, p),
        "spectrum": lambda p: plot_spectrum(track, config, p),
        "spectrogram": lambda p: plot_spectrogram(audio, config, p),
        "dynamics": lambda p: plot_dynamics(track, p),
    }
    if track.stereo.applicable:
        jobs["stereo"] = lambda p: plot_stereo(track, p)
    if references:
        jobs["comparison_spectrum"] = lambda p: plot_comparison_spectrum(track, references, comparisons, config, p)
        jobs["comparison_loudness"] = lambda p: plot_comparison_loudness(track, references, p)
        jobs["comparison_bands"] = lambda p: plot_comparison_bands(track, references, comparisons, p)
        jobs["comparison_metrics"] = lambda p: plot_comparison_metrics(track, references, p)

    created = {}
    for key, job in jobs.items():
        try:
            created[key] = job(out_dir / f"{key}.png").name
        except Exception as exc:  # a broken chart must never abort the report
            log.warning("Plot '%s' failed: %s", key, exc)
            plt.close("all")
    return created
