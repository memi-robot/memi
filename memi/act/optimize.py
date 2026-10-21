"""The MEMI optimization of one Act task, the K iterations of Algorithm 1 (blog Section 2.3). A rerun resumes after the
last iteration written.

    python -m memi.act.optimize --task BinFill [--iterations 5]

Iteration k writes its record to <runs-dir>/<Task>/iteration{k}.json and the revised prompts to
texts.iteration{k}.json. When texts.iteration0.json is missing, the initializer runs first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from ..llm import responses as llm
from ..paths import resolve
from ..tasks import TASKS
from . import config, episodes
from .loop.initializer import initialize
from .loop.iteration import iteration


def optimize(task: str, iterations: int = config.ITERATIONS, *, runs_dir: Path,
             client: Callable[..., llm.Client] = llm.Client) -> None:
    """The initializer and `iterations` iterations of one task, written under `runs_dir`/<task>.
    `client(model, tag=...)` makes the model client. Exits when the task's config.json records other settings."""
    d = runs_dir / task
    d.mkdir(parents=True, exist_ok=True)
    settings = config.settings()
    cfg = d / "config.json"                  # one task's iterations all run under the same settings
    if cfg.exists() and json.loads(cfg.read_text()) != settings:
        raise SystemExit(f"{cfg} records other settings than these: {settings}")
    cfg.write_text(json.dumps(settings, indent=1))
    if all((d / f).exists()
           for f in ["texts.iteration0.json"] + [f"iteration{k}.json" for k in range(1, iterations + 1)]):
        print(f"[{task}] the initializer and {iterations} iterations are already in {d}", flush=True)
        return
    offline = client(config.ASTRA, tag=f"loop.{task}")
    policy_client = client(config.ASTRA, tag=f"loop.{task}")
    batches = episodes.minibatches(task)
    p0 = d / "texts.iteration0.json"
    if p0.exists():
        initial = json.loads(p0.read_text())
    else:
        initial, meta = initialize(task, batches["init"], offline)
        p0.write_text(json.dumps(initial, indent=1))
        (d / "initializer.json").write_text(json.dumps({**meta, "episodes": batches["init"]}, indent=1))
        print(f"[{task}] initialized, {len(initial['fields'])} fields, ${meta['cost']:.2f}", flush=True)
    annotator_and_policy_prompts, history = initial, []
    for k in range(1, iterations + 1):
        pr = d / f"iteration{k}.json"
        if pr.exists():
            rec = json.loads(pr.read_text())
        else:
            rec = iteration(task, k, batches[f"iteration{k}"], annotator_and_policy_prompts, history, initial, offline,
                            policy_client)
            pr.write_text(json.dumps(rec, indent=1, default=str))
            (d / f"texts.iteration{k}.json").write_text(json.dumps(rec["texts"], indent=1))
            s = rec["score"]
            print(f"[{task}] iteration {k}: waypoint wrong {s['waypoint_wrong']}/{s['act_cycles']}, memory wrong "
                  f"{s['memory_wrong']}/{s['cycles']}, ${rec['cost_total']:.2f}, ledger ${llm.spent():.2f}", flush=True)
        if rec["applied"] is not None:     # a quiet iteration adds no history entry
            history.append({"score": rec["score"], "critique": rec["critique"], "applied": rec["applied"],
                            "not_applied": rec["not_applied"]})
        annotator_and_policy_prompts = rec["texts"]


def cli(prog: str, doc: str, argv=None, initialize_only: bool = False) -> None:
    """The command line `memi.act.init` and `memi.act.optimize` share."""
    ap = argparse.ArgumentParser(prog=prog, description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--runs-dir", default=config.RUNS_DIR, help="the run writes under <runs-dir>/<Task>/")
    if not initialize_only:
        ap.add_argument("--iterations", type=int, default=config.ITERATIONS, choices=range(1, config.ITERATIONS + 1),
                        help=f"1 to {config.ITERATIONS}, one per minibatch drawn")
    a = ap.parse_args(argv)
    optimize(a.task, 0 if initialize_only else a.iterations, runs_dir=resolve(a.runs_dir))


def main(argv=None) -> None:
    cli("python -m memi.act.optimize", __doc__, argv)


if __name__ == "__main__":
    main()
