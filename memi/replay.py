"""The offline replay of every released optimization run, with no API key and no simulator.

    python -m memi.replay                    every family
    python -m memi.replay --family claude    one family, gpt, claude or act
    python -m memi.replay --tasks StopCube

Each family ends with ALL IDENTICAL when every file the code writes equals the released one. The act family reads
the RoboMME h5 files (ROBOMME_H5_DIR) and is skipped without them, unless --family act names it.
"""
from __future__ import annotations

import argparse

from .paths import H5_DIR
from .tasks import TASKS

FAMILIES = ("gpt", "claude", "act")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m memi.replay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=FAMILIES)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    a = ap.parse_args(argv)
    rc = 0
    for family in [a.family] if a.family else FAMILIES:
        print(f"== {family}", flush=True)
        if family != "act":
            from .plan import replay
            rc |= replay.main(["--family", family, "--tasks", *a.tasks])
        elif any(H5_DIR.glob("record_dataset_*.h5")):
            from .act import replay
            rc |= replay.main(["--tasks", *a.tasks])
        else:
            print(f"skipped: no RoboMME h5 files under {H5_DIR}. Set ROBOMME_H5_DIR.", flush=True)
            rc |= a.family == "act"
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
