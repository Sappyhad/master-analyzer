"""Audio file discovery, decoding and validation.

Decoding strategy:
  * WAV / FLAC / AIFF / OGG / MP3 -> libsndfile via `soundfile` (MP3 needs libsndfile >= 1.1).
  * M4A / AAC / MP4 (and MP3 if libsndfile cannot read it) -> FFmpeg, decoded to a temporary
    32-bit float WAV so that inter-sample overs above 0 dBFS are preserved.

Original files are only ever opened for reading.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import soundfile as sf

from .errors import AudioLoadError, AudioTooLargeError, DependencyError, UnsupportedFormatError

SOUNDFILE_EXTENSIONS = {".wav", ".wave", ".flac", ".aif", ".aiff", ".ogg", ".mp3"}
FFMPEG_ONLY_EXTENSIONS = {".m4a", ".aac", ".mp4", ".alac"}
SUPPORTED_EXTENSIONS = SOUNDFILE_EXTENSIONS | FFMPEG_ONLY_EXTENSIONS
LOSSY_EXTENSIONS = {".mp3", ".m4a", ".aac", ".mp4", ".ogg"}
MIN_DURATION_S = 0.4  # one BS.1770 gating block
BYTES_PER_SAMPLE = 8  # analysis runs in float64


@dataclass(frozen=True)
class AudioData:
    """Decoded audio as float64 array of shape (frames, channels), full scale = ±1.0."""

    samples: np.ndarray
    sample_rate: int
    path: Path
    format_name: str
    subtype: str
    offset_s: float = 0.0  # position of samples[0] in the original file

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def n_frames(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.sample_rate

    @property
    def is_float_source(self) -> bool:
        return self.subtype.upper() in {"FLOAT", "DOUBLE"} or self.path.suffix.lower() in FFMPEG_ONLY_EXTENSIONS

    @property
    def is_lossy_source(self) -> bool:
        return self.path.suffix.lower() in LOSSY_EXTENSIONS

    def mid(self) -> np.ndarray:
        """Mono mid signal (L+R)/2, or the single channel for mono files."""
        return self.samples.mean(axis=1)

    def segment(self, start_s: float, end_s: float) -> "AudioData":
        """Sub-range [start_s, end_s) relative to this object's own start."""
        start = max(0, int(round(start_s * self.sample_rate)))
        end = min(self.n_frames, int(round(end_s * self.sample_rate)))
        if end - start < MIN_DURATION_S * self.sample_rate:
            raise AudioLoadError(f"Segment {start_s:.2f}-{end_s:.2f} s is shorter than {MIN_DURATION_S} s.")
        return replace(self, samples=self.samples[start:end], offset_s=self.offset_s + start / self.sample_rate)


def parse_time(text: str | float | None) -> float | None:
    """Parse '75', '75.5', '01:15', '1:02:03.5' into seconds."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    if not re.fullmatch(r"\d+(\.\d+)?(:\d+(\.\d+)?){0,2}", text.strip()):
        raise ValueError(f"Invalid time '{text}'. Use seconds (75) or mm:ss (01:15) or hh:mm:ss.")
    seconds = 0.0
    for part in text.strip().split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def find_ffmpeg() -> str | None:
    """FFmpeg from PATH, or the binary bundled with the optional `imageio-ffmpeg` package."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg  # optional dependency
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # not installed or binary missing — FFmpeg simply unavailable
        return None


def list_audio_files(directory: Path) -> list[Path]:
    """Supported audio files directly inside `directory`, sorted by name."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def _check_memory(frames: int, channels: int, max_memory_mb: float, path: Path) -> None:
    # Analysis keeps several working copies (filtered, oversampled chunks); 4x is a conservative factor.
    needed_mb = frames * channels * BYTES_PER_SAMPLE * 4 / 1e6
    if needed_mb > max_memory_mb:
        raise AudioTooLargeError(
            f"{path.name}: analysis needs ~{needed_mb:.0f} MB, limit is {max_memory_mb:.0f} MB "
            f"(analysis.max_memory_mb). Analyze a shorter range with --start/--end or raise the limit.")


def _read_with_soundfile(path: Path, max_memory_mb: float, start_s: float | None,
                         end_s: float | None) -> AudioData:
    info = sf.info(str(path))
    if info.frames <= 0:
        raise AudioLoadError(f"{path.name}: file contains no audio frames.")
    start = int(round((start_s or 0.0) * info.samplerate))
    stop = info.frames if end_s is None else min(info.frames, int(round(end_s * info.samplerate)))
    if start >= info.frames:
        raise AudioLoadError(f"{path.name}: start {start_s:.2f} s is beyond file end ({info.duration:.2f} s).")
    if stop <= start:
        raise AudioLoadError(f"{path.name}: end time must be greater than start time.")
    _check_memory(stop - start, info.channels, max_memory_mb, path)
    try:
        samples, rate = sf.read(str(path), start=start, stop=stop, dtype="float64", always_2d=True)
    except MemoryError as exc:
        raise AudioTooLargeError(f"{path.name}: not enough memory to decode the file.") from exc
    return AudioData(samples=samples, sample_rate=int(rate), path=path, format_name=info.format,
                     subtype=info.subtype, offset_s=start / info.samplerate)


def _read_with_ffmpeg(path: Path, max_memory_mb: float, start_s: float | None, end_s: float | None) -> AudioData:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise DependencyError(
            f"{path.name}: decoding {path.suffix} requires FFmpeg. Install FFmpeg and add it to PATH, "
            f"or run: pip install imageio-ffmpeg")
    with tempfile.TemporaryDirectory(prefix="mastercheck_") as tmp:
        wav_path = Path(tmp) / "decoded.wav"
        command = [ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(path), "-vn", "-c:a", "pcm_f32le", str(wav_path)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired as exc:
            raise AudioLoadError(f"{path.name}: FFmpeg decoding timed out.") from exc
        if result.returncode != 0 or not wav_path.exists():
            raise AudioLoadError(f"{path.name}: FFmpeg could not decode the file: {result.stderr.strip()[:500]}")
        audio = _read_with_soundfile(wav_path, max_memory_mb, start_s, end_s)
    return replace(audio, path=path, format_name=path.suffix.lstrip(".").upper() + " (via FFmpeg)")


def _validate(audio: AudioData) -> AudioData:
    name = audio.path.name
    if audio.channels > 2:
        raise UnsupportedFormatError(f"{name}: {audio.channels} channels. Only mono and stereo are supported.")
    if not np.all(np.isfinite(audio.samples)):
        raise AudioLoadError(f"{name}: contains NaN/Inf samples — file is corrupted.")
    if audio.duration_s < MIN_DURATION_S:
        raise AudioLoadError(f"{name}: {audio.duration_s:.2f} s is too short (minimum {MIN_DURATION_S} s).")
    if not np.any(audio.samples):
        raise AudioLoadError(f"{name}: the analyzed range is digital silence.")
    return audio


def load_audio(path: Path, max_memory_mb: float = 2048, start_s: float | None = None,
               end_s: float | None = None) -> AudioData:
    """Decode `path` (optionally only [start_s, end_s)) into a validated AudioData."""
    path = Path(path)
    if not path.is_file():
        raise AudioLoadError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"{path.name}: unsupported extension '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    if start_s is not None and start_s < 0:
        raise AudioLoadError("Start time cannot be negative.")
    if start_s is not None and end_s is not None and end_s <= start_s:
        raise AudioLoadError("End time must be greater than start time.")

    if suffix in FFMPEG_ONLY_EXTENSIONS:
        return _validate(_read_with_ffmpeg(path, max_memory_mb, start_s, end_s))
    try:
        return _validate(_read_with_soundfile(path, max_memory_mb, start_s, end_s))
    except (sf.SoundFileError, RuntimeError) as exc:  # LibsndfileError derives from these
        if suffix == ".mp3" and find_ffmpeg():
            return _validate(_read_with_ffmpeg(path, max_memory_mb, start_s, end_s))
        raise AudioLoadError(f"{path.name}: cannot decode — file corrupted or unsupported codec ({exc}).") from exc
