"""The repository root, path resolution, the RoboMME recordings and the `.env` file."""
from __future__ import annotations

import os
from pathlib import Path

#: The repository root. This file is <root>/memi/paths.py.
ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    """`path`, read against the repository root when it is relative."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


#: The RoboMME recordings, `record_dataset_<Task>.h5`. ROBOMME_H5_DIR, or data/robomme_data_h5.
H5_DIR = resolve(os.environ.get("ROBOMME_H5_DIR") or "data/robomme_data_h5")


def load_env() -> None:
    """Read `KEY=value` lines from the repository's `.env` into the environment.

    A variable that is already set keeps its value. Blank lines, comments and
    an `export ` prefix are allowed. Quotes around a value are removed.
    """
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
