"""Dynamics, spectrum, stereo, transients, normalization, comparison and diagnosis."""
import numpy as np
import pytest

from conftest import FS, make_audio, sine
from mastercheck.analyzer import analyze_track, headline_metrics
from mastercheck.comparison import compare_tracks, summarize_references
from mastercheck.config import genre_profile
from mastercheck.diagnosis import Kind, answer_questions, run_diagnosis
from mastercheck.dynamics import _merge_short_runs, loudest_fragment, windowed_crest
from mastercheck.models import to_jsonable
from mastercheck.normalization import simulate_normalization
from mastercheck.spectrum import analyze_spectrum, fractional_octave_smooth
from mastercheck.stereo import analyze_stereo, correlation
from mastercheck.transients import analyze_transients, find_runs


# --- spectrum ---------------------------------------------------------------------------------

def test_sine_energy_lands_in_its_band(config):
    result = analyze_spectrum(make_audio(sine(1000, 0.5, 5)), config)
    mid = next(b for b in result.bands if b.name == "mid")
    assert mid.energy_pct > 99
    assert sum(b.energy_pct for b in result.bands) == pytest.approx(100, abs=0.5)


def test_white_noise_tilt_is_flat_and_pink_is_minus_3(config):
    rng = np.random.default_rng(1)
    white = rng.standard_normal(FS * 10) * 0.1
    assert analyze_spectrum(make_audio(white), config).spectral_tilt.value == pytest.approx(0.0, abs=0.3)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(white.size, 1 / FS)
    spectrum[1:] /= np.sqrt(freqs[1:])
    pink = np.fft.irfft(spectrum, white.size)
    pink *= 0.1 / pink.std()
    assert analyze_spectrum(make_audio(pink), config).spectral_tilt.value == pytest.approx(-3.0, abs=0.3)


def test_bands_above_nyquist_are_unavailable(config):
    result = analyze_spectrum(make_audio(sine(1000, 0.5, 3, 22050), fs=22050), config)
    air = next(b for b in result.bands if b.name == "air")
    assert air.energy_pct is None


def test_fractional_octave_smoothing_preserves_flat_spectrum():
    freqs = np.linspace(0, 24000, 1001)
    np.testing.assert_allclose(fractional_octave_smooth(freqs, np.ones_like(freqs), 3), 1.0)


# --- stereo -----------------------------------------------------------------------------------

def test_correlation_extremes():
    x = sine(440, 0.5, 1)
    assert correlation(x, x) == pytest.approx(1.0)
    assert correlation(x, -x) == pytest.approx(-1.0)
    rng = np.random.default_rng(2)
    assert abs(correlation(rng.standard_normal(FS), rng.standard_normal(FS))) < 0.05


def test_stereo_metrics_for_identical_and_inverted_channels(config):
    x = sine(440, 0.5, 5)
    same = analyze_stereo(make_audio(np.column_stack([x, x])), config)
    assert same.mono_loudness_change.value == pytest.approx(0.0, abs=0.05)
    assert same.balance.value == pytest.approx(0.0, abs=1e-6)
    assert same.side_energy_pct.value == pytest.approx(0.0, abs=1e-6)
    inverted = analyze_stereo(make_audio(np.column_stack([x, -x])), config)
    assert inverted.correlation.value == pytest.approx(-1.0)
    assert inverted.mono_loudness_change.value is None  # full cancellation
    assert inverted.problem_regions


def test_uncorrelated_noise_loses_3_lu_in_mono(config):
    rng = np.random.default_rng(3)
    noise = rng.standard_normal((FS * 10, 2)) * 0.1
    result = analyze_stereo(make_audio(noise), config)
    assert result.mono_loudness_change.value == pytest.approx(-3.0, abs=0.2)


def test_mono_file_marks_stereo_not_applicable(config):
    assert analyze_stereo(make_audio(sine(440, 0.5, 2)), config).applicable is False


# --- transients / clipping ----------------------------------------------------------------------

def test_find_runs():
    starts, lengths = find_runs(np.array([0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], bool), 3)
    assert starts.tolist() == [1, 8] and lengths.tolist() == [3, 4]


def test_hard_clipped_sine_is_detected(config):
    clipped = np.clip(sine(100, 1.5, 2), -1.0, 1.0)
    result = analyze_transients(make_audio(clipped, subtype="PCM_24"), config)
    assert result.clipping_status == "detected"
    assert result.full_scale_runs.count > 10


def test_clip_then_attenuate_is_suspected(config):
    clipped = np.clip(sine(100, 1.5, 2), -1.0, 1.0) * 0.5
    result = analyze_transients(make_audio(clipped), config)
    assert result.clipping_status == "suspected"
    assert result.flat_top_level_dbfs == pytest.approx(-6.02, abs=0.01)


def test_clean_low_frequency_sine_is_not_flagged(config):
    """A smooth 55 Hz peak sits near its maximum for many samples; that must not count as clipping."""
    tone = np.round(sine(55, 0.3, 5, 44100) * 32767) / 32768  # 16-bit quantized
    assert analyze_transients(make_audio(tone, fs=44100, subtype="PCM_16"), config).clipping_status == "not_detected"


def test_mild_low_frequency_clipping_is_still_detected(config):
    """50 Hz sine driven 1 dB into a 0 dBFS clip, 16-bit at 44.1 kHz."""
    clipped = np.clip(sine(50, 10 ** (1 / 20), 5, 44100), -1.0, 32767 / 32768)
    result = analyze_transients(make_audio(clipped, fs=44100, subtype="PCM_16"), config)
    assert result.clipping_status == "detected"


def test_float_overs_are_detected(config):
    result = analyze_transients(make_audio(sine(100, 1.2, 1)), config)
    assert result.samples_over_full_scale > 0 and result.clipping_status == "detected"


# --- dynamics ---------------------------------------------------------------------------------

def test_windowed_crest_of_sine_is_3db():
    times, crest = windowed_crest(sine(1000, 0.5, 4)[:, None], FS, 0.4, -60)
    assert np.allclose(crest, 3.01, atol=0.05) and times.size == 10


def test_merge_short_runs():
    labels = np.array([1] * 10 + [0] * 2 + [1] * 10, bool)
    assert _merge_short_runs(labels, 5).all()


def test_loudest_fragment_finds_loud_part():
    times = np.arange(1, 601) * 0.1 + 0.3  # 60 s of momentary blocks
    momentary = np.full(600, -30.0)
    momentary[300:400] = -10.0  # loud between ~30 and 40 s
    start, end, lufs = loudest_fragment(times, momentary, 10.0, 60.4, 0.0)
    assert 29 <= start <= 31 and end - start == pytest.approx(10.0) and lufs == pytest.approx(-10.0, abs=0.5)


def test_sections_and_spread_detected(config):
    quiet, loud = sine(1000, 0.05, 20), sine(1000, 0.5, 20)
    tone = np.concatenate([quiet, loud, quiet])
    track = analyze_track(make_audio(np.column_stack([tone, tone])), config, include_fragment=False)
    assert [s.level for s in track.dynamics.sections] == ["lower", "higher", "lower"]
    assert track.dynamics.section_spread.value == pytest.approx(20.0, abs=1.0)


# --- normalization ----------------------------------------------------------------------------

def test_normalization_loud_master_is_attenuated():
    r = simulate_normalization(-8.0, -0.2, -14.0, -1.0)
    assert r.gain_only_db == -6.0 and r.peak_limited_gain_db == -6.0 and not r.limited_by_ceiling


def test_normalization_quiet_master_is_limited_by_peak():
    r = simulate_normalization(-20.0, -3.0, -14.0, -1.0)
    assert r.gain_only_db == 6.0 and r.gain_only_exceeds_ceiling
    assert r.peak_limited_gain_db == 2.0 and r.peak_limited_lufs == -18.0 and r.shortfall_lu == 4.0


def test_normalization_with_peak_already_over_ceiling_gets_no_boost():
    r = simulate_normalization(-16.0, 0.5, -14.0, -1.0)
    assert r.peak_limited_gain_db == 0.0


def test_normalization_of_silence_is_none():
    assert simulate_normalization(float("-inf"), float("-inf"), -14, -1) is None


# --- comparison & diagnosis -------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_tracks(config):
    rng = np.random.default_rng(4)
    noise = rng.standard_normal((FS * 30, 2)) * 0.05
    loud = np.clip(noise * 12, -1, 1)
    track = analyze_track(make_audio(loud, name="track.wav"), config)
    reference = analyze_track(make_audio(noise, name="ref.wav"), config)
    return track, reference


def test_matched_comparison_uses_lufs_difference(two_tracks):
    track, reference = two_tracks
    comparison = compare_tracks(track, reference)
    gain = track.loudness.integrated.value - reference.loudness.integrated.value
    assert comparison.matched_gain_db == pytest.approx(gain)
    tp_row = next(r for r in comparison.matched_rows if r.key == "true_peak_dbtp")
    assert tp_row.reference == pytest.approx(reference.loudness.true_peak.value + gain)
    assert comparison.fragment_rows  # 30 s > 15 s fragment


def test_reference_summary_and_diagnosis(two_tracks, config):
    track, reference = two_tracks
    summary = summarize_references(track, [reference])
    assert summary["integrated_lufs"].diff_to_median > 10
    findings = run_diagnosis(track, summary, [compare_tracks(track, reference)], config, genre_profile(config, None))
    ids = {f.id for f in findings}
    assert "clipping_detected" in ids and "loud_master" in ids
    assert findings == sorted(findings, key=lambda f: f.sort_key)
    assert findings[0].kind == Kind.MEASURED
    answers = answer_questions(track, findings, summary, [], config)
    assert len(answers) == 7


def test_diagnosis_without_references_runs(two_tracks, config):
    track, _ = two_tracks
    findings = run_diagnosis(track, None, [], config, genre_profile(config, "edm"))
    assert all(f.evidence for f in findings)


def test_everything_is_json_serializable(two_tracks):
    import json
    track, _ = two_tracks
    json.dumps(to_jsonable(track))
    assert set(headline_metrics(track)) >= {"integrated_lufs", "true_peak_dbtp", "lra_lu"}
