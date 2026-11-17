"""Loudness tests against the reference behaviour defined in ITU-R BS.1770-4 and EBU Tech 3341/3342."""
import math

import numpy as np
import pyloudnorm
import pytest

from conftest import FS, make_audio, sine
from mastercheck.loudness import (analyze_loudness, block_powers, integrated_lufs, k_weight, k_weighting_sos,
                                  loudness_range, true_peak, SHORT_TERM_WINDOW_S)


def test_k_weighting_matches_bs1770_table_at_48k():
    sos = k_weighting_sos(48000)
    np.testing.assert_allclose(sos[0], [1.53512485958697, -2.69169618940638, 1.19839281085285,
                                        1.0, -1.69065929318241, 0.73248077421585], atol=1e-9)
    np.testing.assert_allclose(sos[1], [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621], atol=1e-9)


@pytest.mark.parametrize("fs", [44100, 48000, 96000])
def test_stereo_1khz_at_minus_23_dbfs_reads_minus_23_lufs(fs):
    """EBU Tech 3341 case 1: stereo 1 kHz sine at -23 dBFS -> -23.0 LUFS (±0.1)."""
    tone = sine(1000, 10 ** (-23 / 20), 20, fs)
    assert integrated_lufs(np.column_stack([tone, tone]), fs) == pytest.approx(-23.0, abs=0.1)


def test_matches_pyloudnorm_on_noise():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((FS * 10, 2)) * 0.1
    expected = pyloudnorm.Meter(FS).integrated_loudness(noise)
    assert integrated_lufs(noise, FS) == pytest.approx(expected, abs=0.05)


def test_relative_gate_ignores_quiet_part():
    """EBU Tech 3341 case 3-style: -36 / -23 / -36 dBFS segments; gating keeps the result near -23."""
    parts = [sine(1000, 10 ** (db / 20), sec) for db, sec in ((-36, 10), (-23, 60), (-36, 10))]
    tone = np.concatenate(parts)
    assert integrated_lufs(np.column_stack([tone, tone]), FS) == pytest.approx(-23.0, abs=0.1)


def test_silence_is_minus_infinity():
    assert integrated_lufs(np.zeros((FS * 2, 2)), FS) == -math.inf


def test_lra_of_two_level_signal():
    """EBU Tech 3342 case 1: 20 s at -20 dBFS then 20 s at -30 dBFS -> LRA 10 LU (±1)."""
    tone = np.concatenate([sine(1000, 10 ** (-20 / 20), 20), sine(1000, 10 ** (-30 / 20), 20)])
    weighted = k_weight(np.column_stack([tone, tone]), FS)
    _, power = block_powers(weighted, FS, SHORT_TERM_WINDOW_S)
    assert loudness_range(power) == pytest.approx(10.0, abs=1.0)


def test_lra_unavailable_for_short_signal():
    assert loudness_range(np.empty(0)) is None


def test_true_peak_finds_intersample_peak():
    """fs/4 sine at 45° phase: samples at ±0.707 (-3.01 dBFS), true peak 0 dBTP."""
    n = np.arange(FS)
    tone = np.sin(2 * np.pi * n / 4 + np.pi / 4)
    fade = np.ones_like(tone)
    ramp = np.hanning(9600)
    fade[:4800], fade[-4800:] = ramp[:4800], ramp[4800:]
    result = true_peak((tone * fade)[:, None], FS, 4)
    assert 20 * np.log10(np.max(np.abs(tone))) == pytest.approx(-3.01, abs=0.01)
    assert result.max_dbtp == pytest.approx(0.0, abs=0.1)


def test_true_peak_never_below_sample_peak_and_counts_overs():
    tone = sine(997, 0.99, 3)
    result = true_peak(tone[:, None], FS, 4, ceiling_dbtp=-1.0)
    assert result.max_dbtp >= 20 * np.log10(0.99) - 0.01
    assert result.over_region_count > 0


def test_analyze_loudness_reports_consistent_derived_metrics(config):
    tone = sine(1000, 0.5, 10)
    result = analyze_loudness(make_audio(np.column_stack([tone, tone])), config)
    assert result.sample_peak.value == pytest.approx(20 * np.log10(0.5), abs=0.01)
    assert result.rms.value == pytest.approx(20 * np.log10(0.5 / np.sqrt(2)), abs=0.01)
    assert result.crest_factor.value == pytest.approx(3.01, abs=0.02)
    assert result.plr.value == pytest.approx(result.true_peak.value - result.integrated.value)
    assert result.lra.value == pytest.approx(0.0, abs=0.1)
    assert result.short_term_lufs.size > 0 and result.momentary_lufs.size > 0
