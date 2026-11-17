"""Exception hierarchy. Every error raised on purpose by MasterCheck derives from MasterCheckError,
so the CLI can show a readable message instead of a traceback."""


class MasterCheckError(Exception):
    """Base class for all expected MasterCheck errors."""


class ConfigError(MasterCheckError):
    """config.json is missing, unreadable or invalid."""


class AudioLoadError(MasterCheckError):
    """File missing, corrupted, silent, too short or otherwise unreadable."""


class UnsupportedFormatError(AudioLoadError):
    """Extension, codec or channel layout that MasterCheck cannot analyze."""


class AudioTooLargeError(AudioLoadError):
    """Decoded audio would exceed the configured memory budget."""


class DependencyError(MasterCheckError):
    """A required external tool (e.g. FFmpeg) is not available."""
