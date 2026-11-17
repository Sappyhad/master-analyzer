"""MasterCheck entry point: python main.py --input "input/my_master.wav" """
import importlib
import sys

REQUIRED_MODULES = {"numpy": "numpy", "scipy": "scipy", "soundfile": "soundfile", "pyloudnorm": "pyloudnorm",
                    "matplotlib": "matplotlib", "pandas": "pandas"}


def _check_dependencies() -> list[str]:
    missing = []
    for module, package in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
        except OSError as exc:  # e.g. soundfile installed but libsndfile missing
            missing.append(f"{package} ({exc})")
    return missing


if __name__ == "__main__":
    if sys.version_info < (3, 10):
        sys.exit("MasterCheck requires Python 3.10 or newer.")
    missing = _check_dependencies()
    if missing:
        sys.exit("Missing dependencies: " + ", ".join(missing) + "\nInstall with: pip install -r requirements.txt")
    from mastercheck.cli import main
    sys.exit(main())
