import glob
import os
import platform
import re
import sys


def _matches_runtime(path):
    """Return True when a CARLA binary package matches this CPython runtime."""
    name = os.path.basename(path).lower()
    major, minor = sys.version_info[:2]
    tags = (f"cp{major}{minor}", f"py{major}.{minor}", f"py{major}{minor}")
    if os.path.isdir(path):
        return True
    tagged = bool(re.search(r"(?:cp|py)\d", name))
    return (not tagged) or any(tag in name for tag in tags)


def _normalise_root(carla_install):
    if not carla_install:
        return None
    path = os.path.abspath(os.path.expandvars(os.path.expanduser(str(carla_install))))
    base = os.path.basename(path).lower()
    if os.path.isfile(path) or base in {"carlaue4.sh", "carlaue4.exe"}:
        path = os.path.dirname(path)
    return path


def _candidate_paths(root):
    patterns = [
        os.path.join(root, "PythonAPI", "carla", "dist", "carla-*.egg"),
        os.path.join(root, "PythonAPI", "carla"),
    ]
    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern))
    return sorted(dict.fromkeys(found))


def _diagnostic_packages(root):
    if not root:
        return []
    patterns = [
        os.path.join(root, "PythonAPI", "carla", "dist", "carla-*.egg"),
        os.path.join(root, "PythonAPI", "carla", "dist", "carla-*.whl"),
    ]
    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern))
    return sorted(dict.fromkeys(found))


def load_carla(carla_install=None):
    """Import CARLA from the active environment or a matching CARLA egg/source tree.

    Binary .whl files are intentionally not appended directly to sys.path. Wheels that
    contain compiled extensions must be installed into the active Python environment.
    the platform setup script installs the matching CARLA Python package into the active environment.
    """
    first_error = None
    try:
        import carla  # type: ignore
        return carla
    except Exception as exc:
        first_error = exc

    root = _normalise_root(
        carla_install
        or os.environ.get("CARLA_ROOT")
        or os.environ.get("CARLA_LAUNCHER")
    )
    discovered = _candidate_paths(root) if root else []
    for path in discovered:
        if not _matches_runtime(path):
            continue
        if path not in sys.path:
            sys.path.insert(0, path)
        try:
            import carla  # type: ignore
            return carla
        except Exception:
            continue

    runtime = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    packages = _diagnostic_packages(root)
    found_text = ", ".join(os.path.basename(p) for p in packages) or "none"
    raise ImportError(
        "CARLA Python API could not be imported. "
        f"Platform={platform.system()} Python={runtime}; discovered CARLA packages: {found_text}. "
        "Install the CARLA Python API version that matches your simulator. "
        "Run the platform setup script so the matching package is installed into .venv_live. "
        f"Initial import error: {first_error!r}"
    )
