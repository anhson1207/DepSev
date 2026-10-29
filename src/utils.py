"""Small file and directory helpers shared by DepSev-ML scripts."""

import json
from pathlib import Path

import pandas as pd

from config import ARTIFACTS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR, TABLES_DIR


def ensure_output_directories():
    """Create all pipeline output directories if they do not exist."""
    for directory in [PROCESSED_DATA_DIR, TABLES_DIR, FIGURES_DIR, ARTIFACTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def read_csv_with_fallback(path, **kwargs):
    """Read a CSV as UTF-8, falling back to cp1252 for medication text."""
    try:
        return pd.read_csv(path, encoding="utf-8", **kwargs), "utf-8"
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", **kwargs), "cp1252"


def save_json(data, path):
    """Save JSON with readable indentation."""
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path):
    """Load a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
