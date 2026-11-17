import numpy as np
import pytest
import soundfile as sf

from conftest import FS, sine
from mastercheck.audio_loader import find_ffmpeg, list_audio_files, load_audio, parse_time
from mastercheck.errors import AudioLoadError, AudioTooLargeError, UnsupportedFormatError


@pytest.mark.parametrize("subtype,tolerance", [("PCM_16", 1e-4), ("PCM_24", 1e-6), ("PCM_32", 1e-8), ("FLOAT", 1e-7)])
def test_wav_subtypes_roundtrip(tmp_path, subtype, tolerance):
    data = np.column_stack([sine(440, 0.5, 1), sine(660, 0.25, 1)])
    path = tmp_path / f"t_{subtype}.wav"
    sf.write(path, data, FS, subtype=subtype)
    audio = load_audio(path)
    assert audio.sample_rate == FS and audio.channels == 2
    np.testing.assert_allclose(audio.samples, data, atol=tolerance)


def test_float_wav_keeps_overs(tmp_path):
    path = tmp_path / "over.wav"
    sf.write(path, sine(100, 1.5, 1).astype(np.float32), FS, subtype="FLOAT")
    audio = load_audio(path)
    assert audio.samples.max() > 1.4
    assert audio.is_float_source


def test_flac_mono_and_rate(tmp_path):
    path = tmp_path / "mono.flac"
    sf.write(path, sine(440, 0.5, 1, 44100), 44100, subtype="PCM_16")
    audio = load_audio(path)
    assert audio.channels == 1 and audio.sample_rate == 44100
    assert audio.mid().shape == (44100,)


def test_range_loading(tmp_path):
    path = tmp_path / "long.wav"
    sf.write(path, sine(440, 0.5, 10), FS)
    audio = load_audio(path, start_s=2, end_s=5)
    assert audio.duration_s == pytest.approx(3.0)
    assert audio.offset_s == pytest.approx(2.0)


def test_segment_of_loaded_audio(tmp_path):
    path = tmp_path / "seg.wav"
    sf.write(path, sine(440, 0.5, 10), FS)
    seg = load_audio(path).segment(1.0, 3.0)
    assert seg.duration_s == pytest.approx(2.0) and seg.offset_s == pytest.approx(1.0)


def test_errors(tmp_path):
    with pytest.raises(AudioLoadError, match="not found"):
        load_audio(tmp_path / "missing.wav")
    bad_ext = tmp_path / "x.xyz"
    bad_ext.write_bytes(b"123")
    with pytest.raises(UnsupportedFormatError):
        load_audio(bad_ext)
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"RIFF\x00\x00\x00\x00WAVEgarbage")
    with pytest.raises(AudioLoadError):
        load_audio(corrupt)
    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(FS), FS)
    with pytest.raises(AudioLoadError, match="silence"):
        load_audio(silent)
    short = tmp_path / "short.wav"
    sf.write(short, sine(440, 0.5, 0.1), FS)
    with pytest.raises(AudioLoadError, match="too short"):
        load_audio(short)
    multi = tmp_path / "surround.wav"
    sf.write(multi, np.random.default_rng(0).standard_normal((FS, 6)) * 0.1, FS)
    with pytest.raises(UnsupportedFormatError, match="channels"):
        load_audio(multi)
    with pytest.raises(AudioTooLargeError):
        load_audio(silent, max_memory_mb=0.001)
    with pytest.raises(AudioLoadError, match="greater"):
        load_audio(silent, start_s=5, end_s=2)


def test_list_audio_files(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "b.txt").write_text("x")
    (tmp_path / "c.FLAC").write_bytes(b"")
    assert [p.name for p in list_audio_files(tmp_path)] == ["a.wav", "c.FLAC"]
    assert list_audio_files(tmp_path / "nope") == []


@pytest.mark.parametrize("text,expected", [("45", 45.0), ("00:45", 45.0), ("1:15.5", 75.5), ("1:00:00", 3600.0)])
def test_parse_time(text, expected):
    assert parse_time(text) == expected


def test_parse_time_rejects_garbage():
    with pytest.raises(ValueError):
        parse_time("1m15s")


@pytest.mark.skipif(find_ffmpeg() is None, reason="FFmpeg not available")
def test_m4a_through_ffmpeg(tmp_path):
    import subprocess
    wav = tmp_path / "src.wav"
    sf.write(wav, np.column_stack([sine(440, 0.5, 2)] * 2), FS)
    m4a = tmp_path / "src.m4a"
    subprocess.run([find_ffmpeg(), "-v", "error", "-y", "-i", str(wav), "-c:a", "aac", str(m4a)], check=True)
    audio = load_audio(m4a)
    assert audio.channels == 2 and audio.duration_s == pytest.approx(2.0, abs=0.1)
