"""Test rollouts of one condition of configs/conditions.json on the simulator with the LLM API, then its success table.

    python -m memi.rollout luna/memi
    python -m memi.rollout astra_act/memi --prompts-from outputs --workers 8
    python -m memi.rollout haiku/memi --tasks PickXtimes --episodes 0 1 2

The prompts are the released ones, or with --prompts-from outputs those your own run wrote under --out. The episodes
go to <out>/results/<policy>/<method>/, and a rerun resumes.
"""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

from . import conditions, summarize
from .paths import resolve
from .tasks import TASKS


def runner(name: str, cond: dict, prompts: str, dest: Path) -> tuple[str, list[str]]:
    """The runner module of a condition and its arguments, except the tasks, the episodes and the workers."""
    method = name.split("/")[1]
    if conditions.is_act(name):
        # An Act result records its condition as init, memi or fullhist.
        args = ["--condition", method.partition("_")[0]] + (["--texts", prompts] if prompts else [])
        return "memi.act.rollout.__main__", args + ["--out", str(dest)]
    policy = "fullhist" if method == "fullhist" else "memory"
    args = ["--policy", policy, "--model", cond["model"], "--effort", cond["effort"]]
    args += ["--policy-prompt", prompts] if prompts else []
    return "memi.plan.rollout.__main__", args + ["--split", "test", "--out", str(dest)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m memi.rollout", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("condition", help="<policy>/<method>, e.g. luna/memi")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--episodes", nargs="+", type=int, help="the condition's own count by default, from episode 0")
    ap.add_argument("--workers", type=int, default=1, help="episode processes run at once")
    ap.add_argument("--prompts-from", choices=["release", "outputs"], default="release")
    ap.add_argument("--out", default="outputs", help="the root the episodes and your own prompts are under")
    a = ap.parse_args(argv)
    known = conditions.load()
    if a.condition not in known:
        ap.error(f"no condition {a.condition!r}. Known: {', '.join(known)}")
    cond = known[a.condition]
    released = cond.get("tasks")
    tasks = [t for t in a.tasks if not released or t in released]
    if not tasks:
        ap.error(f"{a.condition} is released for {' '.join(released)} only")
    act = conditions.is_act(a.condition)
    if act:
        from .act import config
        if (cond["model"], cond["effort"]) != (config.ASTRA, config.EFFORT):
            ap.error(f"the Act rollouts run {config.ASTRA} at effort {config.EFFORT} only")
    prompts = str(Path("." if a.prompts_from == "release" else a.out) / cond["prompts"]) if cond["prompts"] else ""
    for t in tasks if prompts else []:
        if not resolve(prompts.format(task=t)).exists():
            ap.error(f"{a.condition}: no prompts at {prompts.format(task=t)}")
    dest = Path(a.out) / "results" / a.condition
    module, args = runner(a.condition, cond, prompts, dest)
    if act:
        os.environ.setdefault("NVIDIA_DRIVER_CAPABILITIES", "compute,graphics,utility,video")
    else:
        # The released rollouts rendered the scene on the CPU.
        os.environ.setdefault("SAPIEN_RENDER_DEVICE", "cpu")
    episodes = a.episodes if a.episodes is not None else list(range(cond["episodes"]))
    rc = importlib.import_module(module).main(args + ["--tasks", *tasks, "--episodes", *map(str, episodes),
                                                      "--workers", str(a.workers)])
    summarize.main([str(dest)])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
