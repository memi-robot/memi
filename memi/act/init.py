"""The Act initializer of one task, initialize(E) of Algorithm 1.

    python -m memi.act.init --task BinFill

One call writes the initial annotator prompt (phi) and policy prompt (theta) to <runs-dir>/<Task>/texts.iteration0.json.
"""
from __future__ import annotations

from .optimize import cli


def main(argv=None) -> None:
    cli("python -m memi.act.init", __doc__, argv, initialize_only=True)


if __name__ == "__main__":
    main()
