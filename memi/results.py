"""The rollout logs of both action types, read the way every table and figure of the page counts them."""
from __future__ import annotations

import gzip
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Iterable

#: Endings a rerun replaces. They count only when an episode never finished.
UNFINISHED = ("harness_error", "process_crash")


def half_up(v: float, digits: int) -> Decimal:
    """`v` rounded with a 5 always rounded up, as the project page rounds 91.25 to 91.3."""
    return Decimal(repr(round(v, 9))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)


def records(d: Path) -> list[dict]:
    """The episode records of one condition directory in the order they were written, each with its `outcome`.

    A Plan rollout writes one row per episode run to `episodes.jsonl`, plain or
    gzipped. An Act rollout writes `<Task>_ep<e>/result.json`, whose `success`
    gives the outcome.
    """
    act = sorted(d.glob("*_ep*/result.json"))
    if act:
        return [_with_outcome(json.loads(p.read_text())) for p in act]
    for log in (d / "episodes.jsonl", d / "episodes.jsonl.gz"):
        if log.exists():
            with (gzip.open(log, "rt") if log.suffix == ".gz" else open(log)) as fh:
                return [json.loads(line) for line in fh if line.strip()]
    return []


def _with_outcome(result: dict) -> dict:
    ok = result.get("success")
    outcome = "success" if ok else "fail" if ok is not None else result.get("status") or "harness_error"
    return {**result, "outcome": outcome}


def first_finished(rows: Iterable[dict]) -> dict[tuple[str, int], dict]:
    """(task, episode) -> its first finished record, or its last record when it never finished."""
    out: dict[tuple[str, int], dict] = {}
    for r in rows:
        k = (r["task"], int(r["episode"]))
        if k not in out or out[k].get("outcome") in UNFINISHED:
            out[k] = r
    return out
