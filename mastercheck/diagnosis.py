"""Rule-based diagnosis.

Each rule turns measurements into a Finding that keeps three things apart:
  * evidence   — measured values (facts about the file),
  * causes     — hypotheses that could explain them,
  * actions    — things worth testing, and things that could make it worse.
Findings are classified as a measured problem, a potential issue or an observation, with a
confidence level. Thresholds come from config.json. This is decision support, not a replacement
for listening on a calibrated system.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .analyzer import TrackAnalysis, headline_metrics
from .comparison import ReferenceComparison, ReferenceStat, matched_band_differences
from .models import fmt


class Kind(str, Enum):
    MEASURED = "measured problem"
    POTENTIAL = "potential issue"
    OBSERVATION = "observation"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Finding:
    id: str
    title: str
    kind: Kind
    confidence: Confidence
    evidence: list[str]
    likely_causes: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)

    @property
    def sort_key(self) -> tuple[int, int]:
        kind_rank = {Kind.MEASURED: 0, Kind.POTENTIAL: 1, Kind.OBSERVATION: 2}[self.kind]
        conf_rank = {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2}[self.confidence]
        return kind_rank, conf_rank


GENERAL_LIMITATIONS = [
    "Limiter/compressor gain reduction cannot be measured from a finished stereo file; dynamics indicators "
    "describe the result, not the processing that produced it.",
    "No metric here measures perceived quality, emotion, groove or whether a choice is intentional.",
    "Section detection finds energy levels, not musical form (verse/chorus labels are not inferred).",
    "Clipping in earlier production stages that was later attenuated or limited may leave no detectable trace.",
    "Streaming normalization and codec tests are simulations; platforms use their own encoders and rules.",
    "Spectral balance norms depend on genre, arrangement and intent; references are a guide, not a target.",
    "Monitoring, room and listening level strongly affect how a master is perceived; none of that is visible here.",
]


@dataclass
class Context:
    track: TrackAnalysis
    metrics: dict[str, float | None]
    ref_summary: dict[str, ReferenceStat] | None
    comparisons: list[ReferenceComparison]
    thresholds: dict[str, float]
    profile: dict[str, float]

    @property
    def has_refs(self) -> bool:
        return bool(self.ref_summary)

    def ref(self, key: str) -> ReferenceStat | None:
        return self.ref_summary.get(key) if self.ref_summary else None


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _ref_text(ctx: Context, key: str, digits: int = 1) -> str:
    stat = ctx.ref(key)
    if not stat or stat.median is None:
        return ""
    return (f" (references: median {fmt(stat.median, digits)}, range {fmt(stat.minimum, digits)} to "
            f"{fmt(stat.maximum, digits)} {stat.unit})")


# ---------------------------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------------------------

def rule_true_peak(ctx: Context) -> list[Finding]:
    lo = ctx.track.loudness
    tp, ceiling = lo.true_peak.value, ctx.thresholds["true_peak_max_dbtp"]
    if not _finite(tp) or tp <= ceiling:
        return []
    evidence = [f"True peak {fmt(tp, 2)} dBTP > limit {ceiling} dBTP ({lo.true_peak_over_region_count} x 10 ms regions "
                f"over the limit).",
                f"Sample peak {fmt(lo.sample_peak.value, 2)} dBFS; inter-sample overshoot "
                f"{fmt(tp - lo.sample_peak.value, 2)} dB."]
    for codec in ctx.track.codecs or []:
        if codec.ok and codec.true_peak_dbtp is not None:
            evidence.append(f"After {codec.name}: true peak {fmt(codec.true_peak_dbtp, 2)} dBTP "
                            f"({fmt(codec.delta_true_peak_db, 2)} dB change).")
    return [Finding(
        "true_peak_high", "True peak above the configured limit", Kind.MEASURED,
        Confidence.HIGH if tp > ceiling + 0.5 else Confidence.MEDIUM, evidence,
        likely_causes=["Limiter ceiling set above the limit, or limiter running without true-peak / oversampling mode.",
                       "Processing after the limiter (clipper, EQ, sample-rate conversion, dither) adding overshoot.",
                       "A deliberate loudness-first choice."],
        tests=["Check the limiter's true-peak (ISP) mode and oversampling setting.",
               "Render a version with the ceiling at the limit and compare both after the codec test (--no-codec off).",
               "Listen to the lossy-encoded versions for crackle or harshness on peaks."],
        actions=["If the delivery spec or your platform target asks for a true-peak limit, adjust the limiter ceiling/"
                 "true-peak mode rather than adding gain reduction elsewhere."],
        avoid=["Do not treat a high true peak as proof of a bad master by itself.",
               "Do not just add another limiter/clipper after the existing one — that stacks distortion.",
               "Do not lower the gain after the limiter without re-checking loudness and the codec test."],
        missing_info=["Delivery specification / target platforms.",
                      "Oversampled meters under-read worst-case peaks — keep some safety margin."],
    )]


def rule_clipping(ctx: Context) -> list[Finding]:
    tr = ctx.track.transients
    findings = []
    if tr.clipping_status == "detected":
        evidence = []
        if tr.samples_over_full_scale:
            evidence.append(f"{tr.samples_over_full_scale} samples above 0 dBFS (float file) — will clip when "
                            "converted to fixed point.")
        if tr.full_scale_runs.count:
            times = ", ".join(f"{t:.2f}s" for t in tr.full_scale_runs.times_s[:8])
            evidence.append(f"{tr.full_scale_runs.count} runs of >= {int(ctx.thresholds['clip_run_min_samples'])} "
                            f"consecutive full-scale samples (longest {tr.full_scale_runs.longest}); first at {times}.")
        evidence.append(f"{tr.samples_at_full_scale} samples at or above {ctx.thresholds['full_scale_dbfs']} dBFS in total.")
        findings.append(Finding(
            "clipping_detected", "Clipping detected in the file", Kind.MEASURED, Confidence.HIGH, evidence,
            likely_causes=["Output exceeding full scale at render/export (no limiter, or ceiling at 0 dBFS).",
                           "A clipper used deliberately as part of the loudness chain.",
                           "Sample-rate conversion or format conversion after limiting."],
            tests=["Listen solo at the listed times, preferably at moderate level and on headphones.",
                   "Re-export with the limiter ceiling below 0 dBFS (and true-peak mode) and compare."],
            actions=["Find the stage that hits full scale (check meters on master bus, export settings)."],
            avoid=["Do not assume all clipping is audible — short deliberate clipping can be transparent; listen.",
                   "Do not normalize a clipped file upward; the flat tops stay."],
            missing_info=["Which stage clipped cannot be determined from the finished file."],
        ))
    elif tr.clipping_status == "suspected":
        if tr.full_scale_runs.count:
            times = ", ".join(f"{t:.2f}s" for t in tr.full_scale_runs.times_s[:8])
            evidence = [f"Only {tr.full_scale_runs.count} run(s) of consecutive full-scale samples at {times} — "
                        "could be a single normalized peak or isolated clipping."]
        else:
            times = ", ".join(f"{t:.2f}s" for t in tr.flat_top_runs.times_s[:8])
            evidence = [f"{tr.flat_top_runs.count} separate runs of consecutive samples pinned at the peak level "
                        f"{fmt(tr.flat_top_level_dbfs, 2)} dBFS (longest {tr.flat_top_runs.longest}); first at {times}."]
        findings.append(Finding(
            "clipping_suspected", "Flat-topped peaks (possible clipping)", Kind.POTENTIAL, Confidence.MEDIUM,
            evidence,
            likely_causes=["Clipping followed by a gain reduction.",
                           "Hard clipper or zero-lookahead limiter at the ceiling (may be intentional)."],
            tests=["Zoom into the waveform at the listed times.", "Listen for crunch on kick/snare hits."],
            actions=["If unintended, reduce input to the clipping stage and compare at matched loudness."],
            avoid=["Do not treat flat tops as proof of a fault — modern clippers produce them deliberately."],
            missing_info=["The pre-master mix would show whether clipping existed before mastering."],
        ))
    return findings


def rule_density(ctx: Context) -> list[Finding]:
    m, th = ctx.metrics, ctx.thresholds
    dyn, tr = ctx.track.dynamics, ctx.track.transients
    indicators = []
    if _finite(m["windowed_crest_db"]) and m["windowed_crest_db"] < th["windowed_crest_low_db"]:
        indicators.append(f"Median 400 ms crest factor {fmt(m['windowed_crest_db'])} dB < {th['windowed_crest_low_db']} dB"
                          + _ref_text(ctx, "windowed_crest_db"))
    if _finite(dyn.low_crest_fraction.value) and dyn.low_crest_fraction.value > th["low_crest_fraction_high"]:
        indicators.append(f"{dyn.low_crest_fraction.value * 100:.0f} % of windows have crest < {th['windowed_crest_low_db']} dB")
    if _finite(m["peak_density_pct"]) and m["peak_density_pct"] > th["peak_density_high_pct"]:
        indicators.append(f"Peak density {fmt(m['peak_density_pct'], 2)} % of samples within 1 dB of peak"
                          + _ref_text(ctx, "peak_density_pct", 2))
    if _finite(m["plr_db"]) and m["plr_db"] < th["plr_low_db"]:
        indicators.append(f"PLR {fmt(m['plr_db'])} dB < {th['plr_low_db']} dB" + _ref_text(ctx, "plr_db"))
    if _finite(m["lra_lu"]) and m["lra_lu"] < th["lra_low_lu"]:
        indicators.append(f"LRA {fmt(m['lra_lu'])} LU < {th['lra_low_lu']} LU" + _ref_text(ctx, "lra_lu"))
    if _finite(tr.transient_prominence_p95.value) and tr.transient_prominence_p95.value < th["transient_prominence_p95_low_db"]:
        indicators.append(f"Transient prominence p95 {fmt(tr.transient_prominence_p95.value)} dB < "
                          f"{th['transient_prominence_p95_low_db']} dB" + _ref_text(ctx, "transient_p95_db"))
    if len(indicators) < 2:
        return []

    kind, confidence = Kind.POTENTIAL, (Confidence.MEDIUM if len(indicators) >= 4 else Confidence.LOW)
    crest_ref = ctx.ref("windowed_crest_db")
    plr_ref = ctx.ref("plr_db")
    in_ref_range = all(s is not None and s.median is not None and s.track is not None
                       and s.track >= s.minimum - 1.0 for s in (crest_ref, plr_ref))
    if ctx.has_refs and in_ref_range:
        kind, confidence = Kind.OBSERVATION, Confidence.LOW
        indicators.append("Crest factor and PLR are within ~1 dB of the reference range — density looks genre-typical.")
    return [Finding(
        "heavy_limiting", "Signs of heavy limiting / sustained density", kind, confidence, indicators,
        likely_causes=["Strong limiting or clipping to reach a loudness target.",
                       "Dense arrangement or heavily compressed mix bus (density can come from the mix, not mastering).",
                       "Genre convention."],
        tests=["Render a version with 1-2 dB less limiter input; compare at matched loudness (gain-match by ear or "
               "LUFS) and check whether punch/clarity improves.",
               "Compare transients (kick/snare attack) against references at matched loudness.",
               "Check the normalization simulation: a louder master is turned down anyway on many platforms."],
        actions=["Decide loudness against references and the intended playback context, not against maximum possible LUFS."],
        avoid=["Do not add more limiting to win loudness if transients already sound soft.",
               "Do not conclude over-limiting from one number; listen at matched loudness."],
        missing_info=["Limiter gain reduction cannot be measured from the finished file.",
                      "The pre-master mix would show how much density came from mastering."],
    )]


def rule_low_end(ctx: Context) -> list[Finding]:
    sp, th = ctx.track.spectrum, ctx.thresholds
    sub = next((b for b in sp.bands if b.name == "sub_bass"), None)
    evidence = []
    matched = matched_band_differences(ctx.comparisons).get("bass", [])
    if matched:
        median_diff = statistics.median(matched)
        if median_diff > th["reference_band_diff_db"]:
            evidence.append(f"Bass group {fmt(median_diff)} dB above references at matched loudness "
                            f"(per reference: {', '.join(fmt(d) for d in matched)} dB).")
    elif sub and _finite(sub.energy_pct) and sub.energy_pct > ctx.profile["sub_bass_max_pct"]:
        evidence.append(f"Sub-bass holds {fmt(sub.energy_pct)} % of 20 Hz-20 kHz energy (profile bound "
                        f"{ctx.profile['sub_bass_max_pct']} %, rough heuristic without references).")
    if not evidence:
        return []
    if _finite(ctx.metrics["plr_db"]) and ctx.metrics["plr_db"] < th["plr_low_db"] + 2:
        evidence.append(f"PLR {fmt(ctx.metrics['plr_db'])} dB: low end may be using a large part of the peak headroom "
                        "(low frequencies carry most of the waveform amplitude but little K-weighted loudness).")
    if _finite(ctx.metrics["low_band_correlation"]):
        evidence.append(f"Low-band correlation {fmt(ctx.metrics['low_band_correlation'], 2)}.")
    return [Finding(
        "low_end_heavy", "Large share of low-frequency energy", Kind.POTENTIAL,
        Confidence.MEDIUM if matched else Confidence.LOW, evidence,
        likely_causes=["Bass/kick level or sub content high in the mix.", "Low-end EQ boost in mastering.",
                       "Genre and arrangement (e.g. 808-driven material)."],
        tests=["Compare the low end with references at matched loudness on full-range monitors or good headphones.",
               "Try a gentle, narrow test: -1 dB low shelf or dynamic EQ on the sub, then re-check LUFS-I vs true peak.",
               "Check low end in mono and on small speakers."],
        actions=["If headroom is the issue, consider controlling only the resonant/peaky low notes rather than "
                 "the whole band."],
        avoid=["Do not automatically high-pass or cut sub-bass; it may be the point of the track.",
               "Do not judge the low end on monitors that cannot reproduce it."],
        missing_info=["Genre/intent; references from the same genre make this much more reliable."],
    )]


def rule_quiet_vs_refs(ctx: Context) -> list[Finding]:
    stat = ctx.ref("integrated_lufs")
    if not stat or stat.diff_to_median is None or stat.diff_to_median >= -ctx.thresholds["reference_lufs_diff_lu"]:
        return []
    th = ctx.thresholds
    evidence = [f"LUFS-I {fmt(stat.track)} vs reference median {fmt(stat.median)} ({fmt(stat.diff_to_median)} LU)."]
    for key, label in (("plr_db", "PLR"), ("windowed_crest_db", "Windowed crest"), ("lra_lu", "LRA")):
        s = ctx.ref(key)
        if s and s.diff_to_median is not None and s.diff_to_median > th["reference_crest_diff_db"]:
            evidence.append(f"{label} {fmt(s.track)} vs reference median {fmt(s.median)} — more peak-to-average "
                            "headroom used for dynamics.")
    bands = matched_band_differences(ctx.comparisons)
    for group, diffs in bands.items():
        if diffs and abs(statistics.median(diffs)) > th["reference_band_diff_db"]:
            evidence.append(f"{group.capitalize()} group {fmt(statistics.median(diffs))} dB vs references at matched "
                            "loudness — tonal balance changes how loud it reads.")
    return [Finding(
        "quieter_than_references", "Lower integrated loudness than references", Kind.OBSERVATION,
        Confidence.MEDIUM if len(evidence) > 1 else Confidence.LOW, evidence,
        likely_causes=["Less limiting / more preserved dynamics (may be intentional).",
                       "Less energy in the 1-5 kHz region where BS.1770 and the ear are most sensitive.",
                       "More low-frequency energy consuming peak headroom.", "Transients with higher peaks."],
        tests=["Compare at matched loudness first: does it actually sound worse, or just quieter?",
               "Check the normalization simulation — on normalized platforms the difference may vanish.",
               "Inspect the LUFS-S plot: is the whole track quieter or only some sections?"],
        actions=["Address the specific factor above (tonal balance, transient peaks) rather than adding limiting blindly."],
        avoid=["Do not conclude anything from LUFS-I alone.",
               "Do not raise limiter input until it matches — check what it costs in transients first."],
        missing_info=["Target distribution (streaming with normalization vs. club/CD/broadcast)."],
    )]


def rule_loud_master(ctx: Context) -> list[Finding]:
    lufs = ctx.metrics["integrated_lufs"]
    th = ctx.thresholds
    stat = ctx.ref("integrated_lufs")
    louder_than_refs = bool(stat and stat.diff_to_median is not None and stat.diff_to_median > th["reference_lufs_diff_lu"])
    if not _finite(lufs) or not (lufs >= th["loud_master_lufs"] or louder_than_refs):
        return []
    tr = ctx.track.transients
    evidence = [f"LUFS-I {fmt(lufs)}" + _ref_text(ctx, "integrated_lufs")]
    if tr.clipping_status != "not_detected":
        evidence.append(f"Clipping status: {tr.clipping_status}.")
    if _finite(tr.transient_prominence_p95.value):
        evidence.append(f"Transient prominence p95 {fmt(tr.transient_prominence_p95.value)} dB"
                        + _ref_text(ctx, "transient_p95_db"))
    norm = ctx.track.normalization
    if norm:
        evidence.append(f"Simulated normalization to {norm.target_lufs} LUFS would apply {fmt(norm.gain_only_db)} dB.")
    return [Finding(
        "loud_master", "Very loud master — check the cost of loudness", Kind.OBSERVATION, Confidence.MEDIUM, evidence,
        likely_causes=["Deliberate loudness target.", "Aggressive limiting/clipping."],
        tests=["Listen to the master turned down to the reference level: does it hold up against quieter references?",
               "If it sounds weak/small despite high LUFS, check tonal balance (matched band table), distortion "
               "(clipping section) and transient prominence before anything else."],
        actions=["Compare against a version 1-2 LU quieter at matched playback level."],
        avoid=["Do not automatically add more limiting to fix a weak-sounding loud master — it usually makes it worse."],
        missing_info=["Whether loudness is a client requirement."],
    )]


def rule_sections(ctx: Context) -> list[Finding]:
    dyn = ctx.track.dynamics
    spread = dyn.section_spread.value
    if not _finite(spread) or spread <= ctx.thresholds["section_difference_lu"]:
        return []
    evidence = [f"Loudest vs quietest section: {fmt(spread)} LU."]
    evidence += [f"{s.start_s:.1f}-{s.end_s:.1f} s: {s.level} level, {fmt(s.mean_lufs)} LUFS" for s in dyn.sections[:10]]
    if _finite(dyn.short_term_std_lu):
        evidence.append(f"LUFS-S standard deviation {fmt(dyn.short_term_std_lu)} LU.")
    return [Finding(
        "section_contrast", "Large loudness differences between sections", Kind.POTENTIAL, Confidence.LOW, evidence,
        likely_causes=["Intentional arrangement contrast (quiet intro/verse vs full chorus).",
                       "Uneven mix balance between sections, or automation not matched.",
                       "Very dynamic material (orchestral, acoustic)."],
        tests=["Look at the loudness-over-time plot and listen across each boundary.",
               "Compare with references' LUFS-S curves: do they have similar contrast?"],
        actions=["If unintended, address it with automation in the mix rather than heavier compression on the master."],
        avoid=["Do not flatten intentional contrast with more bus compression."],
        missing_info=["Artistic intent; musical structure is not detected, only energy levels."],
    )]


def rule_stereo(ctx: Context) -> list[Finding]:
    st, th = ctx.track.stereo, ctx.thresholds
    if not st.applicable:
        return []
    findings = []
    if _finite(st.correlation.value) and st.correlation.value < th["correlation_problem"]:
        findings.append(Finding(
            "negative_correlation", "Negative overall stereo correlation", Kind.MEASURED, Confidence.HIGH,
            [f"Correlation {fmt(st.correlation.value, 2)}; mono fold-down change {fmt(st.mono_loudness_change.value)} LU."],
            likely_causes=["Polarity inverted on one channel.", "Extreme stereo widening.", "Mic phase problems."],
            tests=["Listen in mono.", "Flip polarity of one channel and compare."],
            actions=["Find the source of anti-phase content before mastering decisions."],
            avoid=["Do not add more widening."]))
    elif st.problem_regions:
        regions = ", ".join(f"{r['start_s']:.1f}-{r['end_s']:.1f}s (min {r['min_correlation']:.2f})"
                            for r in st.problem_regions[:8])
        findings.append(Finding(
            "phase_regions", "Fragments with negative correlation", Kind.POTENTIAL, Confidence.MEDIUM,
            [f"{len(st.problem_regions)} regions below {th['correlation_problem']}: {regions}.",
             f"{fmt(st.negative_correlation_pct)} % of analyzed windows."],
            likely_causes=["Wide stereo effects, chorus/Haas widening, reverbs.", "Phase issues on specific elements."],
            tests=["Listen to these fragments in mono — does something disappear or thin out?"],
            actions=["If an element vanishes in mono, fix at the source (mix) rather than on the master."],
            avoid=["Do not narrow the whole master to fix one element."],
            missing_info=["Whether mono playback matters for the target audience."]))
    if _finite(st.correlation_low_band.value) and st.correlation_low_band.value < th["low_band_correlation_min"]:
        findings.append(Finding(
            "wide_low_end", "Low end is not mono-compatible", Kind.POTENTIAL, Confidence.MEDIUM,
            [f"Low-band correlation {fmt(st.correlation_low_band.value, 2)} "
             f"(threshold {th['low_band_correlation_min']}). {st.correlation_low_band.method}"],
            likely_causes=["Stereo bass/synth patches, wide reverbs on low elements, widened kick."],
            tests=["Check bass in mono and on a club/PA mono sub setup.", "Solo the side channel below 150 Hz."],
            actions=["Consider a low-frequency mono-maker/elliptical EQ only after confirming the problem by ear."],
            avoid=["Do not mono the low end blindly; some stereo low end is intentional."]))
    if _finite(st.mono_loudness_change.value) and st.mono_loudness_change.value < th["mono_loss_warn_lu"]:
        findings.append(Finding(
            "mono_loss", "Significant loudness loss in mono", Kind.POTENTIAL, Confidence.MEDIUM,
            [f"Mono fold-down is {fmt(st.mono_loudness_change.value)} LU vs stereo."],
            tests=["Compare stereo and mono at the same monitor level."],
            avoid=["Do not judge only by the number: uncorrelated ambience naturally loses level in mono."]))
    if _finite(st.balance.value) and abs(st.balance.value) > th["lr_balance_warn_db"]:
        side = "left" if st.balance.value > 0 else "right"
        findings.append(Finding(
            "lr_imbalance", "Left/right level imbalance", Kind.MEASURED, Confidence.MEDIUM,
            [f"RMS L {fmt(st.left_rms_dbfs)} dBFS, R {fmt(st.right_rms_dbfs)} dBFS: {side} louder by "
             f"{fmt(abs(st.balance.value))} dB."],
            likely_causes=["Panning decisions (may be intentional for this arrangement).", "Gain mismatch on the bus."],
            tests=["Check the balance meter over time; is it constant or arrangement-driven?"],
            avoid=["Do not auto-balance if the asymmetry is artistic."]))
    if _finite(st.side_to_mid.value) and st.side_to_mid.value > th["side_to_mid_high_db"]:
        findings.append(Finding(
            "very_wide", "Very high side-to-mid ratio", Kind.POTENTIAL, Confidence.LOW,
            [f"Side/Mid {fmt(st.side_to_mid.value)} dB" + _ref_text(ctx, "side_to_mid_db")],
            tests=["Check mono compatibility and centre image (vocal/kick/bass solidity)."],
            avoid=["Do not consider wide automatically better."]))
    return findings


def rule_pumping(ctx: Context) -> list[Finding]:
    r = ctx.track.dynamics.pumping_indicator.value
    if not _finite(r) or r > ctx.thresholds["pumping_correlation"]:
        return []
    return [Finding(
        "pumping", "Possible pumping (low-end-driven gain modulation)", Kind.POTENTIAL, Confidence.LOW,
        [f"Envelope fluctuation correlation low band vs 2-8 kHz: {fmt(r, 2)} (threshold {ctx.thresholds['pumping_correlation']})."],
        likely_causes=["Broadband compressor/limiter reacting to kick/bass.", "Deliberate sidechain effect.",
                       "Arrangement (e.g. hats placed between kicks)."],
        tests=["Listen to sustained elements (pads, cymbals, reverb tails) on kick hits.",
               "Try longer release or multiband/sidechain HPF on the bus compressor/limiter and compare."],
        avoid=["Do not 'fix' an intentional sidechain groove."],
        missing_info=["This is a weak statistical hint; listening decides."])]


def rule_resonances(ctx: Context) -> list[Finding]:
    res = ctx.track.spectrum.resonances
    if not res:
        return []
    listing = ", ".join(f"{r['freq_hz']:.0f} Hz (+{r['prominence_db']:.1f} dB)" for r in res)
    return [Finding(
        "resonances", "Narrow spectral peaks", Kind.POTENTIAL, Confidence.LOW,
        [f"Long-term spectrum peaks above the 1-octave average by > {ctx.thresholds['resonance_prominence_db']} dB: {listing}."],
        likely_causes=["Sustained notes/tonic of the song (common and harmless).", "Room or instrument resonances.",
                       "Narrow EQ boosts."],
        tests=["Sweep a narrow EQ cut at those frequencies (1-2 dB) and listen whether the change is an improvement.",
               "Check whether the frequencies match the key/bass notes of the song."],
        avoid=["Do not notch every peak — many are musical notes."])]


def rule_lossy_source(ctx: Context) -> list[Finding]:
    info, sp = ctx.track.info, ctx.track.spectrum
    evidence = []
    if info.is_lossy_source:
        evidence.append(f"Input file is a lossy format ({info.path.suffix}).")
    if sp.lowpass_cutoff_hz:
        evidence.append(f"Spectrum drops > {ctx.thresholds['lossy_lowpass_drop_db']} dB above ~{sp.lowpass_cutoff_hz:.0f} Hz.")
    if not evidence:
        return []
    return [Finding(
        "lossy_source", "Signs of lossy encoding or band-limiting in the source", Kind.POTENTIAL,
        Confidence.MEDIUM if info.is_lossy_source else Confidence.LOW, evidence,
        likely_causes=["Analyzing an MP3/AAC instead of the lossless master.", "Lossy samples/stems in the production.",
                       "Deliberate low-pass filtering."],
        tests=["Re-run the analysis on the lossless master if available."],
        avoid=["Do not judge true peak or top-octave content of a lossy file as if it were the master."])]


def rule_codecs(ctx: Context) -> list[Finding]:
    threshold = ctx.thresholds["codec_true_peak_increase_db"]
    flagged = [c for c in ctx.track.codecs or [] if c.ok and
               ((c.delta_true_peak_db or 0) > threshold or c.new_true_peak_overs)]
    if not flagged:
        return []
    evidence = [f"{c.name}: true peak {fmt(c.true_peak_dbtp, 2)} dBTP ({'+' if c.delta_true_peak_db >= 0 else ''}"
                f"{fmt(c.delta_true_peak_db, 2)} dB), sample peak {fmt(c.sample_peak_dbfs, 2)} dBFS"
                + (", NEW overs above limit" if c.new_true_peak_overs else "") for c in flagged]
    return [Finding(
        "codec_overs", "Peaks grow after lossy encoding", Kind.POTENTIAL, Confidence.MEDIUM, evidence,
        likely_causes=["Codec filtering/quantization adds overshoot to dense, near-ceiling material."],
        tests=["Listen to the encoded versions (MP3/AAC at low bitrate are the most revealing).",
               "Compare a version with ~1 dB lower ceiling through the same test."],
        actions=["If overs matter for your delivery, leave more true-peak headroom."],
        avoid=["Do not assume a platform's encoder behaves like the local FFmpeg encoder."],
        missing_info=["Actual platform encoder and bitrate."])]


def rule_normalization(ctx: Context) -> list[Finding]:
    norm = ctx.track.normalization
    if norm is None:
        return []
    evidence = [f"Gain-only: {fmt(norm.gain_only_db)} dB -> true peak {fmt(norm.gain_only_true_peak_dbtp, 2)} dBTP.",
                f"Peak-limited: {fmt(norm.peak_limited_gain_db)} dB -> {fmt(norm.peak_limited_lufs)} LUFS."]
    if norm.gain_only_db < -3:
        title = "Master will be turned down substantially by loudness normalization"
        tests = ["Compare the master and references after normalization (table in the report): which has more punch "
                 "and clarity at equal loudness?"]
    elif norm.limited_by_ceiling and norm.shortfall_lu > 1:
        title = "Master may play below target on normalized platforms"
        tests = ["Decide whether that is acceptable: services that do not limit will leave it quieter than others."]
        evidence.append(f"Peak headroom stops gain {fmt(norm.shortfall_lu)} LU short of the target.")
    else:
        return []
    return [Finding("normalization", title, Kind.OBSERVATION, Confidence.MEDIUM, evidence, tests=tests,
                    avoid=["Do not tune a master to one platform's assumed behaviour; rules change."],
                    missing_info=[norm.disclaimer])]


def rule_tonal_vs_refs(ctx: Context) -> list[Finding]:
    bands = matched_band_differences(ctx.comparisons)
    th = ctx.thresholds["reference_band_diff_db"]
    evidence = []
    consistent = True
    for group, diffs in bands.items():
        if group == "bass" or not diffs:  # bass handled by rule_low_end
            continue
        med = statistics.median(diffs)
        if abs(med) > th:
            evidence.append(f"{group.capitalize()} group {fmt(med)} dB vs references at matched loudness "
                            f"(per reference: {', '.join(fmt(d) for d in diffs)}).")
            consistent &= all((d > 0) == (med > 0) for d in diffs)
    if not evidence:
        return []
    return [Finding(
        "tonal_balance", "Tonal balance differs from references", Kind.POTENTIAL,
        Confidence.MEDIUM if consistent and len(ctx.comparisons) > 1 else Confidence.LOW, evidence,
        likely_causes=["Mix/mastering EQ choices.", "Different arrangement/instrumentation than the references.",
                       "References from a different genre or era."],
        tests=["A/B with references at matched loudness focusing on the named region.",
               "Try a broad 1 dB shelf/bell move in that region and compare again."],
        avoid=["Do not EQ to match a reference curve exactly — arrangement differences make that misleading.",
               "Do not treat one reference as the target."],
        missing_info=["Whether the references share genre, arrangement and intent."])]


def rule_dynamics_vs_refs(ctx: Context) -> list[Finding]:
    if not ctx.has_refs:
        return []
    th = ctx.thresholds
    evidence = []
    lra = ctx.ref("lra_lu")
    if lra and lra.diff_to_median is not None and abs(lra.diff_to_median) > th["reference_lra_diff_lu"]:
        evidence.append(f"LRA {fmt(lra.track)} LU vs reference median {fmt(lra.median)} LU "
                        f"({'more' if lra.diff_to_median > 0 else 'less'} macro-dynamic variation).")
    for key, label in (("windowed_crest_db", "Windowed crest"), ("plr_db", "PLR")):
        s = ctx.ref(key)
        if s and s.diff_to_median is not None and abs(s.diff_to_median) > th["reference_crest_diff_db"]:
            evidence.append(f"{label} {fmt(s.track)} dB vs reference median {fmt(s.median)} dB "
                            f"({'more' if s.diff_to_median > 0 else 'less'} micro-dynamics/peak headroom).")
    if not evidence:
        return []
    return [Finding(
        "dynamics_vs_refs", "Dynamics differ from references", Kind.OBSERVATION, Confidence.MEDIUM, evidence,
        tests=["Compare the LUFS-S curves and crest plots of the track and references.",
               "Listen at matched loudness: is the difference musical or a problem?"],
        avoid=["Do not match reference dynamics numerically if the material is different."])]


def rule_width_vs_refs(ctx: Context) -> list[Finding]:
    if not ctx.has_refs or not ctx.track.stereo.applicable:
        return []
    th = ctx.thresholds
    evidence = []
    corr, sm = ctx.ref("correlation"), ctx.ref("side_to_mid_db")
    if corr and corr.diff_to_median is not None and abs(corr.diff_to_median) > th["reference_correlation_diff"]:
        evidence.append(f"Correlation {fmt(corr.track, 2)} vs reference median {fmt(corr.median, 2)}.")
    if sm and sm.diff_to_median is not None and abs(sm.diff_to_median) > th["reference_side_mid_diff_db"]:
        evidence.append(f"Side/Mid {fmt(sm.track)} dB vs reference median {fmt(sm.median)} dB "
                        f"({'wider' if sm.diff_to_median > 0 else 'narrower'}).")
    if not evidence:
        return []
    return [Finding("width_vs_refs", "Stereo width differs from references", Kind.OBSERVATION, Confidence.MEDIUM,
                    evidence, tests=["A/B stereo image and mono fold-down against references."],
                    avoid=["Do not treat wider as better or narrower as worse."])]


def rule_profile_balance(ctx: Context) -> list[Finding]:
    if ctx.has_refs:
        return []
    sp = ctx.track.spectrum
    evidence = []
    mid, high = sp.group_pct.get("mid"), sp.group_pct.get("high")
    if _finite(mid) and mid < ctx.profile.get("mid_group_min_pct", 0):
        evidence.append(f"Mid group holds {fmt(mid)} % of energy (rough bound {ctx.profile['mid_group_min_pct']} %).")
    if _finite(high) and high > ctx.profile.get("high_group_max_pct", 100):
        evidence.append(f"High group holds {fmt(high, 2)} % of energy (rough bound {ctx.profile['high_group_max_pct']} %).")
    if not evidence:
        return []
    return [Finding(
        "profile_balance", "Unusual energy distribution (no references to compare)", Kind.POTENTIAL, Confidence.LOW,
        evidence, tests=["Add 2-3 genre references to reference/ and re-run for a meaningful comparison."],
        avoid=["Do not EQ based on these rough bounds alone."],
        missing_info=["References from the same genre."])]


RULES: tuple[Callable[[Context], list[Finding]], ...] = (
    rule_true_peak, rule_clipping, rule_density, rule_low_end, rule_quiet_vs_refs, rule_loud_master, rule_sections,
    rule_stereo, rule_pumping, rule_resonances, rule_lossy_source, rule_codecs, rule_normalization,
    rule_tonal_vs_refs, rule_dynamics_vs_refs, rule_width_vs_refs, rule_profile_balance,
)


def run_diagnosis(track: TrackAnalysis, ref_summary: dict[str, ReferenceStat] | None,
                  comparisons: list[ReferenceComparison], config: dict[str, Any],
                  profile: dict[str, float]) -> list[Finding]:
    """Apply every rule; findings sorted measured -> potential -> observation, then by confidence."""
    ctx = Context(track, headline_metrics(track), ref_summary, comparisons, config["thresholds"], profile)
    findings = [f for rule in RULES for f in rule(ctx)]
    return sorted(findings, key=lambda f: f.sort_key)


# ---------------------------------------------------------------------------------------------
# Summary answers
# ---------------------------------------------------------------------------------------------

def answer_questions(track: TrackAnalysis, findings: list[Finding], ref_summary: dict[str, ReferenceStat] | None,
                     normalized_refs: list[tuple[str, float]], config: dict[str, Any]) -> list[tuple[str, str]]:
    """Short plain-language answers to the report's key questions."""
    ids = {f.id for f in findings}
    lo = track.loudness
    ceiling = config["thresholds"]["true_peak_max_dbtp"]
    answers = []

    tp = lo.true_peak.value
    if "true_peak_high" in ids:
        peak = f"No: true peak {fmt(tp, 2)} dBTP exceeds the {ceiling} dBTP limit."
    else:
        peak = (f"Yes, within the configured limit: true peak {fmt(tp, 2)} dBTP (limit {ceiling} dBTP). "
                "Oversampled meters can under-read slightly; keep a margin.")
    answers.append(("Is the peak level safe?", peak))

    if ref_summary:
        dyn = [k for k in ("lra_lu", "windowed_crest_db", "plr_db") if ref_summary[k].outside_range]
        answers.append(("Is the dynamics significantly different from references?",
                        ("Yes — outside the reference range for: " + ", ".join(ref_summary[k].label for k in dyn))
                        if dyn or "dynamics_vs_refs" in ids else "No major difference in LRA, crest factor or PLR."))
        answers.append(("Does the tonal balance differ from references?",
                        "Yes — see the matched band comparison." if {"tonal_balance", "low_end_heavy"} & ids
                        else "No band group differs by more than the configured threshold at matched loudness."))
    else:
        answers.append(("Is the dynamics significantly different from references?", "No references provided."))
        answers.append(("Does the tonal balance differ from references?", "No references provided."))

    status = track.transients.clipping_status
    clip = {"detected": "Clipping detected directly in the file.",
            "suspected": "Flat-topped peaks found — possible clipping (not proven).",
            "not_detected": "No direct clipping found."}[status]
    if "heavy_limiting" in ids:
        clip += " Several indicators of heavy limiting / flattened transients."
    answers.append(("Were potential transient or clipping problems found?", clip))

    norm = track.normalization
    if norm:
        text = (f"Simulated: {fmt(norm.gain_only_db)} dB gain to reach {norm.target_lufs} LUFS "
                f"(peak-limited result {fmt(norm.peak_limited_lufs)} LUFS).")
        if normalized_refs:
            diffs = [norm.peak_limited_lufs - v for _, v in normalized_refs]
            text += f" Difference to normalized references: {', '.join(fmt(d) for d in diffs)} LU."
            if any(abs(d) > 1 for d in diffs):
                text += " Playback loudness will differ; tonal/dynamic differences then dominate perception."
            else:
                text += " After normalization, loudness is close — differences will be tonal and dynamic."
        answers.append(("Could it sound different from references after streaming normalization?", text))

    first = [f.title for f in findings if f.kind != Kind.OBSERVATION][:3] or [f.title for f in findings][:3]
    answers.append(("What to check first?", "; ".join(first) if first else
                    "No rule fired. Verify by listening against references at matched loudness."))
    answers.append(("What cannot be determined from these measurements?", " ".join(GENERAL_LIMITATIONS[:4])))
    return answers
