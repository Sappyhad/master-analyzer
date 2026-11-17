import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mastercheck.audio_loader import AudioData  # noqa: E402
from mastercheck.config import load_config  # noqa: E402

FS = 48000


def sine(freq: float, amplitude: float, seconds: float, fs: int = FS, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(seconds * fs)) / fs
    return amplitude * np.sin(2 * np.pi * freq * t + phase)


def make_audio(samples: np.ndarray, fs: int = FS, subtype: str = "FLOAT", name: str = "test.wav") -> AudioData:
    if samples.ndim == 1:
        samples = samples[:, None]
    return AudioData(samples=samples.astype(np.float64), sample_rate=fs, path=Path(name), format_name="WAV",
                     subtype=subtype)


@pytest.fixture(scope="session")
def config() -> dict:
    return load_config()
