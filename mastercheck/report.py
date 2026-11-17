"""Report writers: analysis.json, metrics.csv, report.html and the terminal summary."""
from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .analyzer import HEADLINE_METRICS, TrackAnalysis, headline_metrics
from .codec_simulation import CODEC_DISCLAIMER
from .comparison import ComparisonRow, ReferenceComparison, ReferenceStat
from .diagnosis import GENERAL_LIMITATIONS, Finding, Kind
from .models import Measurement, fmt, to_jsonable
from .normalization import SIMULATION_DISCLAIMER, NormalizationResult


@dataclass
class Session:
    track: TrackAnalysis
    references: list[TrackAnalysis]
    comparisons: list[ReferenceComparison]
    reference_summary: dict[str, ReferenceStat] | None
    findings: list[Finding]
    answers: list[tuple[str, str]]
    settings: dict[str, Any]
    plots: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    created: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------------------------
# JSON / CSV
# ---------------------------------------------------------------------------------------------

def write_json(session: Session, path: Path) -> None:
    payload = {
        "tool": "MasterCheck", "version": __version__, "created": session.created.isoformat(timespec="seconds"),
        "settings": session.settings,
        "track": session.track,
        "references": session.references,
        "comparisons": session.comparisons,
        "reference_summary": session.reference_summary,
        "findings": session.findings,
        "answers": [{"question": q, "answer": a} for q, a in session.answers],
        "limitations": GENERAL_LIMITATIONS,
        "warnings": session.warnings + session.track.warnings,
        "plots": session.plots,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False)


def write_csv(session: Session, path: Path) -> None:
    track_metrics = headline_metrics(session.track)
    ref_metrics = [(r.info.name, headline_metrics(r)) for r in session.references]
    rows = []
    for metric in HEADLINE_METRICS:
        row = {"metric": metric.label, "key": metric.key, "unit": metric.unit,
               session.track.info.name: track_metrics[metric.key]}
        for name, values in ref_metrics:
            row[name] = values[metric.key]
        if session.reference_summary:
            row["reference_median"] = session.reference_summary[metric.key].median
            row["track_minus_median"] = session.reference_summary[metric.key].diff_to_median
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame = frame.apply(lambda col: col.map(lambda v: None if isinstance(v, float) and not math.isfinite(v) else v))
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.4f")


# ---------------------------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------------------------

CSS = """
:root{--bg:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--accent:#2a78d6;
--measured:#c62f2e;--potential:#b86e00;--observation:#2a78d6;--ok:#1a7f37;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--panel:#232322;--ink:#f0efec;
--ink2:#c3c2b7;--line:#3a3a37;--accent:#3987e5;--measured:#e66767;--potential:#e0a030;--observation:#3987e5;
--ok:#4fb26b;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#1a1a19;--panel:#232322;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--accent:#3987e5;
--measured:#e66767;--potential:#e0a030;--observation:#3987e5;--ok:#4fb26b;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0}
main{max-width:1180px;margin:0 auto;padding:24px 20px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:15px;margin:20px 0 8px}.muted{color:var(--ink2)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.tile .v{font-size:22px;font-weight:650}.tile .l{color:var(--ink2);font-size:12px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--ink2);font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.badge{display:inline-block;font-size:11px;font-weight:650;padding:1px 8px;border-radius:999px;border:1px solid currentColor;margin-right:6px}
.measured{color:var(--measured)}.potential{color:var(--potential)}.observation{color:var(--observation)}
.finding{border-left:4px solid var(--line)}.finding.k-measured{border-left-color:var(--measured)}
.finding.k-potential{border-left-color:var(--potential)}.finding.k-observation{border-left-color:var(--observation)}
.finding h3{margin:0 0 6px}.finding ul{margin:4px 0 8px 18px;padding:0}.finding .lbl{font-weight:600;font-size:12px;color:var(--ink2);text-transform:uppercase;letter-spacing:.03em}
img{max-width:100%;border:1px solid var(--line);border-radius:8px;background:#fcfcfb;margin:6px 0}
.qa dt{font-weight:650;margin-top:8px}.qa dd{margin:2px 0 0 0}
details summary{cursor:pointer;color:var(--ink2)}
code{font-size:12px}
"""


def _e(value: Any) -> str:
    return html.escape(str(value))


def _num(value: float | None, digits: int = 1) -> str:
    return fmt(value, digits)


def _table(headers: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(
        f'<td class="num">{c}</td>' if i >= numeric_from else f"<td>{c}</td>" for i, c in enumerate(r)) + "</tr>"
        for r in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _img(session: Session, key: str, alt: str) -> str:
    name = session.plots.get(key)
    return f'<img src="{_e(name)}" alt="{_e(alt)}" loading="lazy">' if name else ""


def _measurement_rows(items: list[tuple[str, Measurement]]) -> list[list[str]]:
    rows = []
    for label, m in items:
        notes = _e(m.notes) if m.notes else ""
        rows.append([_e(label), f"{_num(m.value, 2)} {_e(m.unit)}", _e(m.compliance.value), _e(m.method), notes])
    return rows


def _comparison_table(rows: list[ComparisonRow], ref_name: str) -> str:
    body = [[_e(r.label), f"{_num(r.track, 2)} {_e(r.unit)}", f"{_num(r.reference, 2)} {_e(r.unit)}",
             _num(r.difference, 2)] for r in rows]
    return _table(["Metric", "Track", ref_name, "Difference (track - ref)"], body)


def _finding_html(f: Finding) -> str:
    kind_class = f.kind.name.lower()

    def block(label: str, items: list[str]) -> str:
        if not items:
            return ""
        return f'<div class="lbl">{_e(label)}</div><ul>' + "".join(f"<li>{_e(i)}</li>" for i in items) + "</ul>"

    return (f'<div class="panel finding k-{kind_class}"><h3>{_e(f.title)}</h3>'
            f'<span class="badge {kind_class}">{_e(f.kind.value)}</span>'
            f'<span class="badge">confidence: {_e(f.confidence.value)}</span>'
            + block("Evidence (measured)", f.evidence) + block("Possible causes (hypotheses)", f.likely_causes)
            + block("Suggested tests", f.tests) + block("Recommended action", f.actions)
            + block("May make it worse", f.avoid) + block("Missing information / caveats", f.missing_info) + "</div>")


def _normalization_rows(items: list[tuple[str, NormalizationResult | None]]) -> list[list[str]]:
    rows = []
    for name, n in items:
        if n is None:
            rows.append([_e(name)] + ["n/a"] * 6)
            continue
        rows.append([_e(name), _num(n.input_lufs), _num(n.input_true_peak_dbtp, 2), f"{n.gain_only_db:+.1f}",
                     _num(n.gain_only_true_peak_dbtp, 2) + (" ⚠" if n.gain_only_exceeds_ceiling else ""),
                     f"{n.peak_limited_gain_db:+.1f}", _num(n.peak_limited_lufs)])
    return rows


def write_html(session: Session, path: Path) -> None:
    t = session.track
    lo, dyn, sp, st, tr = t.loudness, t.dynamics, t.spectrum, t.stereo, t.transients
    info = t.info
    parts: list[str] = ["<main>"]

    seg = (f" · analyzed range {info.offset_s:.2f}-{info.offset_s + info.duration_s:.2f} s"
           if session.settings.get("start") is not None or session.settings.get("end") is not None else "")
    parts.append(f"<h1>MasterCheck report: {_e(info.name)}</h1><p class='muted'>{_e(info.format_name)} / "
                 f"{_e(info.subtype)} · {info.sample_rate} Hz · {info.channels} ch · {info.duration_s:.1f} s{seg} · "
                 f"{len(session.references)} reference(s) · generated {session.created:%Y-%m-%d %H:%M} · "
                 f"MasterCheck {__version__}</p>")
    parts.append("<div class='panel muted'>Automated measurements and heuristics to support mastering decisions. "
                 "They do not replace critical listening on a calibrated monitoring system.</div>")

    tiles = [("Integrated", lo.integrated.value, "LUFS", 1), ("True peak", lo.true_peak.value, "dBTP", 2),
             ("LRA", lo.lra.value, "LU", 1), ("PLR", lo.plr.value, "dB", 1),
             ("Crest (400 ms median)", dyn.windowed_crest_median.value, "dB", 1),
             ("Correlation", st.correlation.value, "", 2)]
    parts.append("<div class='tiles'>" + "".join(
        f"<div class='tile'><div class='v'>{_num(v, d)} <span class='muted' style='font-size:13px'>{u}</span></div>"
        f"<div class='l'>{_e(label)}</div></div>" for label, v, u, d in tiles) + "</div>")

    parts.append("<h2>Summary</h2><dl class='qa panel'>" + "".join(
        f"<dt>{_e(q)}</dt><dd>{_e(a)}</dd>" for q, a in session.answers) + "</dl>")

    parts.append("<h2>Diagnosis</h2>")
    if session.findings:
        counts = {k: sum(f.kind == k for f in session.findings) for k in Kind}
        parts.append("<p class='muted'>" + " · ".join(f"{n} {k.value}(s)" for k, n in counts.items()) + "</p>")
        parts += [_finding_html(f) for f in session.findings]
    else:
        parts.append("<div class='panel'>No diagnostic rule fired with the current thresholds. That is not a "
                     "quality verdict — compare against references by ear.</div>")

    parts.append("<h2>Loudness & level</h2>" + _img(session, "loudness", "Loudness over time"))
    parts.append(_table(["Metric", "Value", "Status", "Method", "Limitations / notes"], _measurement_rows([
        ("Integrated loudness", lo.integrated), ("Max short-term", lo.short_term_max),
        ("Max momentary", lo.momentary_max), ("Loudness range", lo.lra), ("True peak", lo.true_peak),
        ("Sample peak", lo.sample_peak), ("RMS", lo.rms), ("Crest factor", lo.crest_factor), ("PLR", lo.plr)])))
    ch_names = ["Mono"] if info.channels == 1 else ["Left", "Right"]
    parts.append("<h3>Per channel</h3>" + _table(["Channel", "Sample peak (dBFS)", "True peak (dBTP)", "RMS (dBFS)"], [
        [n, _num(a, 2), _num(b, 2), _num(c, 2)] for n, a, b, c in zip(
            ch_names, lo.sample_peak_per_channel_dbfs, lo.true_peak_per_channel_dbtp, lo.rms_per_channel_dbfs)]))
    if lo.true_peak_over_region_count:
        times = ", ".join(f"{x:.2f}" for x in lo.true_peak_over_times_s[:30])
        parts.append(f"<p>True-peak overs above the limit in {lo.true_peak_over_region_count} × 10 ms regions; first at "
                     f"(s): {times}</p>")
    parts.append(_img(session, "waveform", "Waveform"))

    parts.append("<h2>Dynamics</h2>" + _img(session, "dynamics", "Crest factor over time"))
    parts.append(_table(["Metric", "Value", "Status", "Method", "Limitations / notes"], _measurement_rows([
        ("Crest factor, 400 ms median", dyn.windowed_crest_median), ("Low-crest window share", dyn.low_crest_fraction),
        ("Peak density", dyn.peak_density), ("Section spread", dyn.section_spread),
        ("Pumping indicator", dyn.pumping_indicator)])))
    parts.append(f"<p>LUFS-S distribution (gated): median {_num(dyn.short_term_median_lufs)}, p10 "
                 f"{_num(dyn.short_term_p10_lufs)}, p90 {_num(dyn.short_term_p90_lufs)} LUFS, std "
                 f"{_num(dyn.short_term_std_lu)} LU. Windowed crest p10/p90: {_num(dyn.windowed_crest_p10_db)} / "
                 f"{_num(dyn.windowed_crest_p90_db)} dB. Loudest {dyn.loudest_fragment_end_s - dyn.loudest_fragment_start_s:.0f} s "
                 f"fragment: {info.offset_s + dyn.loudest_fragment_start_s:.1f}-{info.offset_s + dyn.loudest_fragment_end_s:.1f} s "
                 f"({_num(dyn.loudest_fragment_lufs)} LUFS energy mean of LUFS-M).</p>")
    if dyn.sections:
        parts.append("<h3>Detected energy sections</h3>" + _table(["Start (s)", "End (s)", "Level", "Mean LUFS-S"], [
            [f"{s.start_s:.1f}", f"{s.end_s:.1f}", s.level, _num(s.mean_lufs)] for s in dyn.sections], numeric_from=0))
    parts.append("<p class='muted'>Gain reduction of a limiter cannot be measured from the finished file; "
                 "these indicators describe the result only.</p>")

    parts.append("<h2>Spectrum & tonal balance</h2>" + _img(session, "spectrum", "Average spectrum")
                 + _img(session, "spectrogram", "Spectrogram"))
    parts.append(_table(["Band", "Range (Hz)", "Energy share (%)", "Level re total (dB)", "Level (dBFS)"], [
        [_e(b.label), f"{b.low_hz:.0f}-{b.high_hz:.0f}", _num(b.energy_pct, 2), _num(b.level_db_rel_total),
         _num(b.level_dbfs)] for b in sp.bands]))
    parts.append(_table(["Metric", "Value", "Status", "Method", "Limitations / notes"], _measurement_rows([
        ("Spectral tilt", sp.spectral_tilt), ("Spectral centroid", sp.spectral_centroid),
        ("Bass / mid energy", sp.bass_to_mid_db)])))
    groups = ", ".join(f"{g}: {_num(v)} %" for g, v in sp.group_pct.items())
    parts.append(f"<p>Band groups: {_e(groups)}.")
    if sp.resonances:
        parts.append(" Narrow peaks: " + ", ".join(f"{r['freq_hz']:.0f} Hz (+{r['prominence_db']:.1f} dB)"
                                                   for r in sp.resonances) + ".")
    if sp.lowpass_cutoff_hz:
        parts.append(f" Steep high-frequency drop above ~{sp.lowpass_cutoff_hz:.0f} Hz.")
    parts.append("</p>")

    parts.append("<h2>Stereo & phase</h2>")
    if st.applicable:
        parts.append(_img(session, "stereo", "Stereo correlation and width"))
        parts.append(_table(["Metric", "Value", "Status", "Method", "Limitations / notes"], _measurement_rows([
            ("Correlation", st.correlation), ("Low-band correlation", st.correlation_low_band),
            ("Side/Mid ratio", st.side_to_mid), ("Side energy share", st.side_energy_pct),
            ("L/R balance", st.balance), ("Mono fold-down change", st.mono_loudness_change)])))
        if st.problem_regions:
            parts.append("<h3>Regions with negative correlation</h3>" + _table(["Start (s)", "End (s)", "Min r"], [
                [f"{r['start_s']:.1f}", f"{r['end_s']:.1f}", f"{r['min_correlation']:.2f}"] for r in st.problem_regions[:50]],
                numeric_from=0))
    else:
        parts.append("<p>Mono file — stereo analysis not applicable.</p>")

    parts.append("<h2>Clipping & transients</h2>")
    status_text = {"detected": "Clipping detected directly", "suspected": "Clipping suspected (flat tops below full scale)",
                   "not_detected": "No direct clipping detected"}[tr.clipping_status]
    parts.append(_table(["Item", "Value"], [
        ["Status", status_text],
        ["Samples above 0 dBFS (float only)", str(tr.samples_over_full_scale)],
        ["Samples at/above full-scale threshold", str(tr.samples_at_full_scale)],
        ["Runs of consecutive full-scale samples", f"{tr.full_scale_runs.count} (longest {tr.full_scale_runs.longest})"],
        ["Flat-top runs below full scale", f"{tr.flat_top_runs.count} (longest {tr.flat_top_runs.longest}, level "
                                           f"{_num(tr.flat_top_level_dbfs, 2)} dBFS)"],
        ["True-peak over regions", str(lo.true_peak_over_region_count)],
    ]))
    parts.append(_table(["Metric", "Value", "Status", "Method", "Limitations / notes"], _measurement_rows([
        ("Transient prominence (median)", tr.transient_prominence_median),
        ("Transient prominence (p95)", tr.transient_prominence_p95)])))

    parts.append("<h2>Loudness normalization simulation</h2>")
    parts.append(f"<div class='panel muted'>{_e(SIMULATION_DISCLAIMER)}</div>")
    norm_items = [(f"{info.name} (track)", t.normalization)] + [(r.info.name, r.normalization) for r in session.references]
    target = t.normalization.target_lufs if t.normalization else session.settings.get("target_lufs")
    ceiling = t.normalization.ceiling_dbtp if t.normalization else None
    parts.append(f"<p>Target {target} LUFS, true-peak ceiling {ceiling} dBTP.</p>")
    parts.append(_table(["File", "LUFS-I", "TP (dBTP)", "Gain-only (dB)", "TP after gain-only",
                         "Peak-limited gain (dB)", "LUFS after peak-limited"], _normalization_rows(norm_items)))

    parts.append("<h2>Codec simulation</h2>")
    if t.codecs:
        parts.append(f"<div class='panel muted'>{_e(CODEC_DISCLAIMER)}</div>")
        band_names = [b["name"] for b in session.settings["bands"]]
        rows = []
        for c in t.codecs:
            if not c.ok:
                rows.append([_e(c.name), "failed: " + _e(c.error)] + [""] * 6)
                continue
            bands = ", ".join(f"{n}: {c.band_delta_db[n]:+.1f}" for n in band_names
                              if c.band_delta_db.get(n) is not None)
            rows.append([_e(c.name), _num(c.sample_peak_dbfs, 2), f"{_num(c.true_peak_dbtp, 2)} ({c.delta_true_peak_db:+.2f})",
                         f"{_num(c.integrated_lufs, 2)} ({_num(c.delta_lufs, 2)})",
                         "yes" if c.new_true_peak_overs else "no", _num(c.null_residual_db), _e(bands),
                         str(c.alignment_offset_samples) if c.alignment_offset_samples is not None else "n/a"])
        parts.append(_table(["Codec", "Sample peak", "True peak (Δ)", "LUFS-I (Δ)", "New TP overs",
                             "Null residual (dB re signal)", "Band level change (dB)", "Delay (samples)"], rows))
    else:
        parts.append(f"<p>{_e(t.codec_note or 'Not run.')}</p>")

    parts.append("<h2>Reference comparison</h2>")
    if session.references:
        parts.append(_img(session, "comparison_spectrum", "Spectrum comparison")
                     + _img(session, "comparison_bands", "Band differences")
                     + _img(session, "comparison_loudness", "Short-term loudness comparison")
                     + _img(session, "comparison_metrics", "Metric comparison"))
        if session.reference_summary and len(session.references) > 1:
            parts.append("<h3>Track vs all references</h3>" + _table(
                ["Metric", "Track", "Ref median", "Ref min", "Ref max", "Track - median", "Outside range"],
                [[_e(s.label), _num(s.track, 2), _num(s.median, 2), _num(s.minimum, 2), _num(s.maximum, 2),
                  _num(s.diff_to_median, 2), "yes" if s.outside_range else ""]
                 for s in session.reference_summary.values()]))
        for comp in session.comparisons:
            parts.append(f"<h3>vs {_e(comp.reference_name)}</h3>")
            parts.append(_comparison_table(comp.rows, comp.reference_name))
            if comp.matched_gain_db is not None:
                parts.append(f"<p><b>Loudness-matched view</b>: reference virtually gained by "
                             f"{comp.matched_gain_db:+.2f} dB (same BS.1770-4 meter for both, files unchanged).</p>")
                parts.append(_comparison_table(comp.matched_rows, comp.reference_name + " (matched)"))
            if comp.fragment_rows:
                parts.append("<details><summary>Loudest fragment comparison</summary>"
                             + _comparison_table(comp.fragment_rows, comp.reference_name) + "</details>")
            for note in comp.notes:
                parts.append(f"<p class='muted'>{_e(note)}</p>")
        parts.append("<p class='muted'>No single reference is a universal quality target.</p>")
    else:
        parts.append("<p>No references analyzed. Put 2-3 commercial masters of the same genre into reference/ "
                     "for comparative diagnosis.</p>")

    parts.append("<h2>What these measurements cannot tell</h2><ul>" + "".join(
        f"<li>{_e(x)}</li>" for x in GENERAL_LIMITATIONS) + "</ul>")
    warnings = session.warnings + t.warnings
    if warnings:
        parts.append("<h2>Warnings</h2><ul>" + "".join(f"<li>{_e(w)}</li>" for w in warnings) + "</ul>")
    parts.append("</main>")
    document = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                f"<title>MasterCheck — {_e(info.name)}</title><style>{CSS}</style></head><body>"
                + "\n".join(parts) + "</body></html>")
    path.write_text(document, encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------------------------

TERMINAL_ASCII = str.maketrans({"—": "-", "·": "|", "²": "2", "Σ": "sum ", "≥": ">=", "⚠": "!"})


def terminal_summary(session: Session, out_dir: Path) -> str:
    t = session.track
    lo, dyn, st = t.loudness, t.dynamics, t.stereo
    lines = [
        "", "=" * 78, f" MasterCheck  |  {t.info.name}", "=" * 78,
        f" Integrated {fmt(lo.integrated.value, 1, 'LUFS'):>12}   True peak {fmt(lo.true_peak.value, 2, 'dBTP'):>11}"
        f"   Sample peak {fmt(lo.sample_peak.value, 2, 'dBFS')}",
        f" LRA        {fmt(lo.lra.value, 1, 'LU'):>12}   PLR       {fmt(lo.plr.value, 1, 'dB'):>11}"
        f"   Crest 400ms {fmt(dyn.windowed_crest_median.value, 1, 'dB')}",
        f" RMS        {fmt(lo.rms.value, 1, 'dBFS'):>12}   Correl.   {fmt(st.correlation.value, 2):>11}"
        f"   Side/Mid    {fmt(st.side_to_mid.value, 1, 'dB')}",
        " Bands      " + "  ".join(f"{b.label} {fmt(b.energy_pct, 1)}%" for b in t.spectrum.bands),
    ]
    if session.reference_summary:
        lines.append(f" References {len(session.references)}: " + ", ".join(r.info.name for r in session.references))
        for key in ("integrated_lufs", "true_peak_dbtp", "lra_lu", "windowed_crest_db", "side_to_mid_db"):
            s = session.reference_summary[key]
            lines.append(f"   {s.label:<32} track {fmt(s.track, 2):>7}  ref median {fmt(s.median, 2):>7}  "
                         f"diff {fmt(s.diff_to_median, 2):>6}")
    lines += ["-" * 78, " Findings:"]
    if session.findings:
        lines += [f"  [{f.kind.value} | {f.confidence.value}] {f.title}" for f in session.findings]
    else:
        lines.append("  none fired with current thresholds")
    lines += ["-" * 78]
    for question, answer in session.answers:
        lines.append(f" {question}\n   {answer}")
    lines += ["-" * 78, f" Report: {out_dir / 'report.html'}", "=" * 78]
    return "\n".join(lines).translate(TERMINAL_ASCII)
