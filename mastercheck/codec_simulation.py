"""Lossy codec round-trip test with FFmpeg.

The analyzed range is written to a temporary 32-bit float WAV, encoded with each configured encoder,
decoded back to 32-bit float (so overs above 0 dBFS survive) and re-measured. The original file is
never touched. Results depend on the encoder implementation and settings; streaming platforms use
their own encoders, so this is an indication of codec sensitivity, not a prediction.
"""
from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

from .audio_loader import AudioData
from .loudness import integrated_lufs, true_peak
from .models import amp_to_db, power_to_db
from .spectrum import band_power, welch_psd

MAX_ENCODER_INPUT_RATE = 48000
ALIGNMENT_ANALYSIS_S = 10.0
MAX_ALIGNMENT_LAG = 8192

CODEC_DISCLAIMER = ("Encoded with the local FFmpeg encoders. Streaming services use their own encoders, bitrates "
                    "and pre-processing, so results indicate codec sensitivity, not what a platform will deliver.")


@dataclass
class CodecResult:
    name: str
    encoder: str
    bitrate: str
    ok: bool
    error: str | None = None
    sample_rate: int | None = None
    sample_peak_dbfs: float | None = None
    true_peak_dbtp: float | None = None
    integrated_lufs: float | None = None
    delta_sample_peak_db: float | None = None
    delta_true_peak_db: float | None = None
    delta_lufs: float | None = None
    true_peak_over_regions: int | None = None
    new_true_peak_overs: bool | None = None
    band_delta_db: dict[str, float | None] = field(default_factory=dict)
    null_residual_db: float | None = None
    alignment_offset_samples: int | None = None


def _run_ffmpeg(ffmpeg: str, args: list[str], timeout_s: float) -> None:
    result = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y", *args], capture_output=True, text=True,
                            timeout=timeout_s)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[:400] or f"FFmpeg exited with {result.returncode}")


def _align(reference: np.ndarray, test: np.ndarray, sample_rate: int) -> int:
    """Lag (samples) by which `test` is delayed relative to `reference` (encoder delay/padding)."""
    n = min(reference.size, test.size, int(ALIGNMENT_ANALYSIS_S * sample_rate))
    corr = signal.correlate(test[:n], reference[:n], mode="full", method="fft")
    lags = signal.correlation_lags(n, n, mode="full")
    window = np.abs(lags) <= MAX_ALIGNMENT_LAG
    return int(lags[window][np.argmax(corr[window])])


def _null_residual(original: np.ndarray, decoded: np.ndarray, sample_rate: int) -> tuple[float | None, int]:
    """RMS of (decoded - original) relative to RMS of original, after time alignment, in dB."""
    ref_mid, dec_mid = original.mean(axis=1), decoded.mean(axis=1)
    lag = _align(ref_mid, dec_mid, sample_rate)
    if lag >= 0:
        dec_mid = dec_mid[lag:]
    else:
        ref_mid = ref_mid[-lag:]
    n = min(ref_mid.size, dec_mid.size)
    ref_power = np.mean(ref_mid[:n] ** 2)
    if n == 0 or ref_power == 0:
        return None, lag
    return float(power_to_db(np.mean((dec_mid[:n] - ref_mid[:n]) ** 2) / ref_power)), lag


def _band_levels(samples: np.ndarray, sample_rate: int, bands: list[dict[str, Any]], nperseg: int) -> dict[str, float]:
    freqs, psd = welch_psd(samples.mean(axis=1), sample_rate, nperseg)
    levels = {}
    for band in bands:
        if band["low_hz"] < sample_rate / 2:
            levels[band["name"]] = float(power_to_db(band_power(freqs, psd, band["low_hz"],
                                                                min(band["high_hz"], sample_rate / 2)) + 1e-30))
    return levels


def simulate_codecs(audio: AudioData, original_tp_dbtp: float, original_lufs: float, config: dict[str, Any],
                    ffmpeg: str) -> list[CodecResult]:
    """Encode/decode the analyzed range with every configured codec and compare to the original."""
    codec_cfg = config["codec_simulation"]
    ceiling = config["thresholds"]["true_peak_max_dbtp"]
    oversampling = config["analysis"]["true_peak_oversampling"]
    nperseg = config["analysis"]["welch_nperseg"]
    original_sp = float(amp_to_db(np.max(np.abs(audio.samples))))
    original_bands = _band_levels(audio.samples, audio.sample_rate, config["bands"], nperseg)
    original_overs = original_tp_dbtp > ceiling
    results: list[CodecResult] = []

    with tempfile.TemporaryDirectory(prefix="mastercheck_codec_") as tmp:
        source = Path(tmp) / "source.wav"
        sf.write(str(source), audio.samples.astype(np.float32), audio.sample_rate, subtype="FLOAT")
        rate_args = ["-ar", str(MAX_ENCODER_INPUT_RATE)] if audio.sample_rate > MAX_ENCODER_INPUT_RATE else []

        for index, codec in enumerate(codec_cfg["codecs"]):
            result = CodecResult(codec["name"], codec["encoder"], codec["bitrate"], ok=False)
            encoded = Path(tmp) / f"encoded_{index}.{codec['extension']}"
            decoded = Path(tmp) / f"decoded_{index}.wav"
            try:
                _run_ffmpeg(ffmpeg, ["-i", str(source), *rate_args, "-c:a", codec["encoder"], "-b:a", codec["bitrate"],
                                     str(encoded)], codec_cfg["timeout_s"])
                _run_ffmpeg(ffmpeg, ["-i", str(encoded), "-c:a", "pcm_f32le", str(decoded)], codec_cfg["timeout_s"])
                samples, rate = sf.read(str(decoded), dtype="float64", always_2d=True)
            except (RuntimeError, subprocess.TimeoutExpired, sf.SoundFileError) as exc:
                result.error = str(exc)
                results.append(result)
                continue

            tp = true_peak(samples, rate, oversampling, ceiling)
            lufs = integrated_lufs(samples, rate)
            result.ok = True
            result.sample_rate = int(rate)
            result.sample_peak_dbfs = float(amp_to_db(np.max(np.abs(samples))))
            result.true_peak_dbtp = tp.max_dbtp
            result.integrated_lufs = lufs if math.isfinite(lufs) else None
            result.delta_sample_peak_db = result.sample_peak_dbfs - original_sp
            result.delta_true_peak_db = tp.max_dbtp - original_tp_dbtp
            result.delta_lufs = (lufs - original_lufs) if math.isfinite(lufs) and math.isfinite(original_lufs) else None
            result.true_peak_over_regions = tp.over_region_count
            result.new_true_peak_overs = tp.max_dbtp > ceiling and not original_overs
            decoded_bands = _band_levels(samples, rate, config["bands"], nperseg)
            result.band_delta_db = {name: (decoded_bands[name] - level if name in decoded_bands else None)
                                    for name, level in original_bands.items()}
            if rate == audio.sample_rate:
                result.null_residual_db, result.alignment_offset_samples = _null_residual(audio.samples, samples, rate)
            results.append(result)
    return results
