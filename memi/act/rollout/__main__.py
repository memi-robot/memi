"""Act test episodes of one condition, each in its own process. `python -m memi.rollout astra_act/<method>` runs it.

    python -m memi.act.rollout --condition memi --out outputs/results/astra_act/memi \\
        --texts artifacts/act/runs/opt_astra_act/{task}/texts.iteration5.json --tasks BinFill --episodes 0 1 2

`init` and `memi` read each task's prompts from --texts, and `fullhist` needs none. Each episode goes to
<out>/<Task>_ep<e>/, where a finished one is skipped and a harness error is rerun.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
from pathlib import Path

from ...llm import responses as llm
from ...paths import ROOT, resolve
from ...tasks import TASKS
from .. import config
from . import fullhist
from .episode import read_result, run_episode, write_result


def one(task: str, episode: int, condition: str, texts: str, out: Path) -> None:
    """Runs one episode in this process and writes it to <out>/<task>_ep<episode>/."""
    out = out / f"{task}_ep{episode}"
    try:
        client = llm.Client(config.ASTRA, tag=f"rollout.{condition}.{task}")
        if condition == "fullhist":
            meta = {"condition": condition, "texts": None, "config": config.settings()}
            row = fullhist.run_episode(task, episode, client, out, meta)
        else:
            p = resolve(texts.format(task=task))
            meta = {"condition": condition, "texts": p.name, "config": config.settings()}
            row = run_episode(task, episode, json.loads(p.read_text()), client, out, meta)
        print(f"[{condition}] {task} ep{episode} -> {row['status']} ({row['end_reason']}) calls={row['model_calls']} "
              f"cost=${row['cost_usd']}", flush=True)
    except Exception as ex:                              # a harness failure is recorded unscored and rerun on resume
        out.mkdir(parents=True, exist_ok=True)
        write_result(out, {"task": task, "episode": episode, "condition": condition, "status": "harness_error",
                           "success": None, "error": repr(ex)})
        print(f"[{condition}] {task} ep{episode} -> harness error: {ex!r}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m memi.act.rollout", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", required=True, choices=("init", "memi", "fullhist"))
    ap.add_argument("--texts", default="", help="init and memi, the prompts file of each task with {task} in its path")
    ap.add_argument("--out", required=True, help="the condition's directory, <OUT>/results/astra_act/<method>")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--episodes", nargs="+", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--one", nargs=2, metavar=("TASK", "EPISODE"), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.condition != "fullhist" and not a.texts:
        ap.error(f"{a.condition} needs --texts")
    out = resolve(a.out)
    if a.one:
        one(a.one[0], int(a.one[1]), a.condition, a.texts, out)
        return 0
    if a.episodes is None:
        ap.error("--episodes is required")

    def pending(t, e):
        r = read_result(out / f"{t}_ep{e}" / "result.json")
        return r is None or r.get("status") == "harness_error"

    def launch(te):                 # SAPIEN cannot build a second environment in one process
        return subprocess.run([sys.executable, "-m", "memi.act.rollout", "--condition", a.condition,
                               "--texts", a.texts, "--out", str(out), "--one", te[0], str(te[1])],
                              cwd=ROOT).returncode

    before = llm.spent()
    with cf.ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(launch, [(t, e) for t in a.tasks for e in a.episodes if pending(t, e)]))
    print(f"Ledger ${llm.spent():.2f}, up ${llm.spent() - before:.2f} during this run (calls of every process)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
