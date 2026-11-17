"""Command-line interface."""
from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .analyzer import TrackAnalysis, analyze_track
from .audio_loader import find_ffmpeg, list_audio_files, load_audio, parse_time
from .comparison import compare_tracks, summarize_references
from .config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, genre_profile, load_config
from .diagnosis import answer_questions, run_diagnosis
from .errors import MasterCheckError
from .report import Session, terminal_summary, write_csv, write_html, write_json

log = logging.getLogger("mastercheck")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mastercheck",
        description="MasterCheck — measure loudness, dynamics, spectrum, stereo and clipping of a master, "
                    "optionally against reference tracks.")
    parser.add_argument("--input", "-i", required=True, type=Path, help="Audio file to analyze.")
    parser.add_argument("--reference-dir", "-r", type=Path, default=PROJECT_ROOT / "reference",
                        help="Folder with reference tracks (default: ./reference next to main.py).")
    parser.add_argument("--no-reference", action="store_true", help="Skip reference analysis.")
    parser.add_argument("--start", type=str, help="Start of analyzed range: seconds (45) or mm:ss (00:45).")
    parser.add_argument("--end", type=str, help="End of analyzed range: seconds (75) or mm:ss (01:15).")
    parser.add_argument("--ref-start", type=str, help="Start of range analyzed in every reference.")
    parser.add_argument("--ref-end", type=str, help="End of range analyzed in every reference.")
    parser.add_argument("--target-lufs", type=float, help="Normalization simulation target (default from config).")
    parser.add_argument("--tp-ceiling", type=float, help="True-peak ceiling for normalization simulation (dBTP).")
    parser.add_argument("--tp-limit", type=float, help="True-peak warning limit for diagnosis (dBTP).")
    parser.add_argument("--genre", type=str, help="Genre profile from config.json (affects no-reference heuristics).")
    parser.add_argument("--no-codec", action="store_true", help="Skip lossy codec simulation.")
    parser.add_argument("--no-plots", action="store_true", help="Skip chart generation.")
    parser.add_argument("--output-dir", "-o", type=Path, default=PROJECT_ROOT / "output",
                        help="Parent folder for session output (default: ./output).")
    parser.add_argument("--config", type=Path, default=None,
                        help=f"Extra config JSON merged over {DEFAULT_CONFIG_PATH.name}.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    return parser


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.target_lufs is not None:
        config["normalization"]["target_lufs"] = args.target_lufs
    if args.tp_ceiling is not None:
        config["normalization"]["true_peak_ceiling_dbtp"] = args.tp_ceiling
    if args.tp_limit is not None:
        config["thresholds"]["true_peak_max_dbtp"] = args.tp_limit


def _session_dir(parent: Path, input_path: Path) -> Path:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", input_path.stem)[:60] or "track"
    base = parent / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}"
    candidate, n = base, 2
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{n}")
        n += 1
    candidate.mkdir(parents=True)
    return candidate


def _parse_range(start: str | None, end: str | None, what: str) -> tuple[float | None, float | None]:
    try:
        return parse_time(start), parse_time(end)
    except ValueError as exc:
        raise MasterCheckError(f"{what}: {exc}") from exc


def _analyze_references(args: argparse.Namespace, config: dict[str, Any], input_path: Path,
                        warnings: list[str]) -> list[TrackAnalysis]:
    if args.no_reference:
        return []
    ref_dir = args.reference_dir
    if not ref_dir.is_dir():
        warnings.append(f"Reference folder {ref_dir} not found — analyzing the master alone.")
        return []
    files = [p for p in list_audio_files(ref_dir) if p.resolve() != input_path.resolve()]
    if not files:
        log.info("No reference files in %s — analyzing the master alone.", ref_dir)
        return []
    start, end = _parse_range(args.ref_start, args.ref_end, "Reference range")
    references = []
    for path in files:
        log.info("Analyzing reference %s", path.name)
        try:
            audio = load_audio(path, config["analysis"]["max_memory_mb"], start, end)
            references.append(analyze_track(audio, config))
        except MasterCheckError as exc:
            warnings.append(f"Reference skipped: {exc}")
            log.warning("Reference skipped: %s", exc)
    return references


def run(args: argparse.Namespace) -> Path:
    """Execute a full analysis session; returns the output folder."""
    config = load_config(args.config)
    _apply_overrides(config, args)
    profile = genre_profile(config, args.genre)
    warnings: list[str] = []

    start, end = _parse_range(args.start, args.end, "Input range")
    log.info("Loading %s", args.input)
    audio = load_audio(args.input, config["analysis"]["max_memory_mb"], start, end)

    run_codecs = config["codec_simulation"]["enabled"] and not args.no_codec
    ffmpeg = find_ffmpeg() if run_codecs else None
    if run_codecs and ffmpeg is None:
        warnings.append("FFmpeg not found: codec simulation skipped. Install FFmpeg or `pip install imageio-ffmpeg`.")
    log.info("Analyzing %s (%.1f s, %d Hz, %d ch)", audio.path.name, audio.duration_s, audio.sample_rate, audio.channels)
    track = analyze_track(audio, config, ffmpeg=ffmpeg, run_codecs=run_codecs)

    references = _analyze_references(args, config, args.input, warnings)
    comparisons = [compare_tracks(track, ref) for ref in references]
    ref_summary = summarize_references(track, references) if references else None
    normalized_refs = [(r.info.name, r.normalization.peak_limited_lufs) for r in references if r.normalization]

    findings = run_diagnosis(track, ref_summary, comparisons, config, profile)
    answers = answer_questions(track, findings, ref_summary, normalized_refs, config)

    out_dir = _session_dir(args.output_dir, args.input)
    settings = {"input": str(args.input), "start": start, "end": end, "ref_start": args.ref_start,
                "ref_end": args.ref_end, "genre": args.genre or "default", "target_lufs": config["normalization"]["target_lufs"],
                "normalization": config["normalization"], "thresholds": config["thresholds"], "bands": config["bands"],
                "true_peak_oversampling": config["analysis"]["true_peak_oversampling"]}
    session = Session(track, references, comparisons, ref_summary, findings, answers, settings, warnings=warnings)

    if not args.no_plots:
        log.info("Rendering charts")
        from .plots import generate_plots  # imported lazily: matplotlib is slow to load
        session.plots = generate_plots(audio, track, references, comparisons, config, out_dir)

    write_json(session, out_dir / "analysis.json")
    write_csv(session, out_dir / "metrics.csv")
    write_html(session, out_dir / "report.html")
    print(terminal_summary(session, out_dir))
    for warning in warnings + track.warnings:
        print(f" ! {warning}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[mastercheck] %(message)s", stream=sys.stderr)
    try:
        run(args)
    except MasterCheckError as exc:
        log.error("Error: %s", exc)
        return 1
    except MemoryError:
        log.error("Error: out of memory. Analyze a shorter range with --start/--end.")
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted.")
        return 130
    return 0
