"""Offline replay of the released Act optimization, with no API key or GPU. `memi.replay --family act` runs it.

    python -m memi.act.replay [--tasks StopCube ...]

Each task's optimization runs again with every model call answered from artifacts/act/runs/opt_astra_act/<Task>/, and
every file it writes must equal the released one apart from cost and timing. It needs the RoboMME h5 files
(ROBOMME_H5_DIR), see README.md.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import tempfile
import threading
from pathlib import Path

from ..llm.parse import parse_object
from ..tasks import TASKS
from . import config, episodes, optimize, policy_turn, variables

#: Keys that record what a call cost or how long it took, which a replay does not have.
UNRECORDED = ("cost", "cost_total", "usage", "s")


def read(path: Path):
    return json.loads(path.read_text())


def digest(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def split_replies(joined: str, n: int) -> list[str]:
    """The n annotator.forward replies of an iteration, which the iteration file keeps joined by blank lines."""
    if n == 1:
        return [joined]
    parts = joined.split("\n\n")
    for i in range(1, len(parts)):
        head, tail = "\n\n".join(parts[:i]), "\n\n".join(parts[i:])
        if all(isinstance((parse_object(x) or {}).get("memory"), list) for x in (head, tail)):
            return [head, *split_replies(tail, n - 1)]
    raise ValueError("the annotator replies of an iteration could not be told apart")


class Recorded:
    """Every answer of one released run, by the call that asked for it."""

    def __init__(self, task: str, src: Path):
        self.lock = threading.Lock()
        self.queues: dict[str, list[str]] = collections.defaultdict(list)   # calls made one after another
        self.keyed: dict[tuple, list[str]] = collections.defaultdict(list)   # policy calls, made in parallel
        self.misses: list[str] = []
        self.queues[f"{task}.initialize"].append(read(src / "initializer.json")["text"])
        annotator_and_policy_prompts = read(src / "texts.iteration0.json")
        for k in range(1, config.ITERATIONS + 1):
            r = read(src / f"iteration{k}.json")
            self.queues[f"{task}.annotator.forward"] += split_replies(r["annotations"]["text"],
                                                                   r["annotations"]["calls"])
            names = variables.field_names(annotator_and_policy_prompts)
            prefix = policy_turn.policy_prefix(annotator_and_policy_prompts)
            eps = {e: episodes.load(task, e) for e in r["episodes"]}
            for s in r["steps"]:
                content = policy_turn.training_turn(eps[s["episode"]], eps[s["episode"]]["cycles"][s["k"] - 1],
                                                    s["carried"], names)
                self.keyed[(f"{task}.policy.forward", digest(prefix, content))].append(json.dumps(s["answer"]))
            for f in r["failures"]:
                content = policy_turn.training_turn(eps[f["episode"]], eps[f["episode"]]["cycles"][f["k"] - 1],
                                                    f["carried"], names)
                self.keyed[(f"{task}.policy.backward", digest(prefix, content))].append(json.dumps(f["critique"]))
            if r["proposal"].get("text"):
                self.queues[f"{task}.annotator.backward"].append(r["proposal"]["text"])
            if r["optimizer"].get("text"):
                self.queues[f"{task}.optimizer.step"].append(r["optimizer"]["text"])
            annotator_and_policy_prompts = read(src / f"texts.iteration{k}.json")

    def answer(self, purpose: str, content: list[dict], prefix: list[dict] | None) -> str:
        with self.lock:
            if purpose.endswith((".policy.forward", ".policy.backward")):
                # A critique is asked with the cycle as policy.forward showed it, then the comparison.
                shown = content if purpose.endswith(".forward") else content[:-1]
                got = self.keyed.get((purpose, digest(prefix or [], shown)))
            else:
                got = self.queues.get(purpose)
            if got:
                return got.pop(0)
            self.misses.append(purpose)
            return "{}"

    def unused(self) -> int:
        return sum(map(len, self.queues.values())) + sum(map(len, self.keyed.values()))


def replay_client(rec: Recorded):
    """The client `optimize` makes, answering every call from `rec`."""
    class Client:
        def __init__(self, model: str, tag: str = ""):
            self.model = model

        def ask(self, system: str, content: list[dict], purpose: str, prefix: list[dict] | None = None) -> dict:
            text = rec.answer(purpose, content, prefix)
            return {"text": text, "json": parse_object(text), "usage": {}, "cost": 0.0, "model": self.model, "s": 0}
    return Client


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in UNRECORDED}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def differing_files(got: Path, want: Path) -> list[str]:
    names = sorted({p.name for p in got.glob("*.json")} | {p.name for p in want.glob("*.json")})
    out = []
    for n in names:
        a, b = got / n, want / n
        if not (a.exists() and b.exists()) or strip(read(a)) != strip(read(b)):
            out.append(n)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m memi.act.replay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    a = ap.parse_args(argv)
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for task in a.tasks:
            src = config.RELEASED / task
            rec = Recorded(task, src)
            optimize.optimize(task, runs_dir=Path(tmp), client=replay_client(rec))
            diff = differing_files(Path(tmp) / task, src)
            ok = not diff and not rec.misses and not rec.unused()
            bad += not ok
            print(f"{task:18s} {'ok  ' if ok else 'FAIL'} misses {len(rec.misses)}  unused {rec.unused()}  "
                  f"differing files {len(diff)}" + (f" {diff[:3]}" if diff else ""), flush=True)
    print("ALL IDENTICAL" if not bad else f"{bad} MISMATCHES")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
