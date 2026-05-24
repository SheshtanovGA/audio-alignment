import os
from typing import Dict, Optional
import numpy as np
import pandas as pd

ALIGNMENT_ARTIFACT_NAMES = ("ts_ref", "ts_stream", "path_ref", "path_stream")


def session_artifact_dir(session: str, artifacts_dir: str = "artifacts") -> str:
    return os.path.join(artifacts_dir, session)


def session_artifact_paths(session: str, artifacts_dir: str = "artifacts") -> Dict[str, str]:
    base = session_artifact_dir(session, artifacts_dir)
    return {name: os.path.join(base, f"{name}.npy") for name in ALIGNMENT_ARTIFACT_NAMES}


def resolve_alignment_paths(
    session: Optional[str] = None,
    artifacts_dir: str = "artifacts",
    overrides: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, str]:
    """Resolve paths to alignment .npy artifacts from a session name and/or explicit paths."""
    paths: Dict[str, str] = {}
    if session:
        paths = session_artifact_paths(session, artifacts_dir)
    if overrides:
        for key, value in overrides.items():
            if value:
                paths[key] = value
    missing = [name for name in ALIGNMENT_ARTIFACT_NAMES if name not in paths]
    if missing:
        names = ", ".join(f"--{n}" for n in missing)
        raise ValueError(
            f"Missing alignment artifacts ({names}). "
            f"Provide --session NAME and/or explicit paths for: {', '.join(missing)}."
        )
    return paths

def safe_npy_load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Numpy file not found: {path}")
    return np.load(path)


def safe_npy_save(array, path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    np.save(path, array)


def safe_csv_read(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)

def safe_save_csv(df, path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    df.to_csv(path, index=False)
