"""Configuration loading. config.json in the project root is the single source of defaults;
an optional user file is deep-merged on top of it."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"

REQUIRED_SECTIONS = ("analysis", "bands", "band_groups", "normalization", "codec_simulation",
                     "thresholds", "genre_profiles")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Config file {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a JSON object.")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate(config: dict[str, Any]) -> None:
    missing = [s for s in REQUIRED_SECTIONS if s not in config]
    if missing:
        raise ConfigError(f"Config is missing sections: {', '.join(missing)}")
    band_names = set()
    for band in config["bands"]:
        try:
            low, high = float(band["low_hz"]), float(band["high_hz"])
            band_names.add(band["name"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid band definition {band!r}: needs name, low_hz, high_hz") from exc
        if not 0 < low < high:
            raise ConfigError(f"Band {band['name']}: low_hz must be > 0 and < high_hz")
    for group, members in config["band_groups"].items():
        unknown = set(members) - band_names
        if unknown:
            raise ConfigError(f"Band group '{group}' references unknown bands: {sorted(unknown)}")
    if "default" not in config["genre_profiles"]:
        raise ConfigError("genre_profiles must contain a 'default' profile")
    oversampling = config["analysis"].get("true_peak_oversampling", 4)
    if not isinstance(oversampling, int) or oversampling < 1:
        raise ConfigError("analysis.true_peak_oversampling must be a positive integer")


def load_config(user_path: Path | None = None) -> dict[str, Any]:
    """Load default config.json, merge an optional user config over it, validate."""
    config = _read_json(DEFAULT_CONFIG_PATH)
    if user_path is not None and Path(user_path).resolve() != DEFAULT_CONFIG_PATH:
        config = _deep_merge(config, _read_json(Path(user_path)))
    _validate(config)
    return config


def genre_profile(config: dict[str, Any], genre: str | None) -> dict[str, Any]:
    """Default profile overlaid with the named genre profile (if any)."""
    profiles = {k: v for k, v in config["genre_profiles"].items() if not k.startswith("_")}
    if genre and genre not in profiles:
        raise ConfigError(f"Unknown genre '{genre}'. Available: {', '.join(sorted(profiles))}")
    profile = dict(profiles["default"])
    if genre:
        profile.update(profiles[genre])
    return profile
