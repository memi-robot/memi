"""Export the policy prompts theta of a runs directory into the one file the rollouts read, an entry per task.

    python -m memi.plan.export --runs-dir outputs/artifacts/gpt/runs/opt_luna \\
        --stage optimized --out outputs/artifacts/gpt/policy_prompts/optimized_luna.json

The rollouts read an entry's `policy_prompt_for_memory`, `policy_prompt_for_subgoal` and `fields`. `policy_prompt`,
`has_demo` and `menu` are there for readers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..paths import resolve
from ..tasks import TASKS
from . import policy_turn, variables
from .loop import episodes
from .loop.config import LoopConfig
from .loop.store import Store
from .prompts.context import POLICY_PROMPT_HEAD

MENUS = json.loads((Path(__file__).parent / "task_menus.json").read_text())


def entry(policy_prompt: dict, cfg: LoopConfig) -> dict:
    """One task's entry, from its policy prompt file."""
    subgoal = " ".join(str(policy_prompt.get(variables.SUBGOAL) or "").split())
    block = variables.fields_block(policy_prompt.get(variables.FIELDS) or []).rstrip()
    one = {"policy_prompt": f"{POLICY_PROMPT_HEAD}\n" + block
           + (f"\n\nTHE POLICY PROMPT FOR SUBGOAL\n    {subgoal}" if subgoal else ""),
           **policy_turn.entry(variables.policy_prompt(policy_prompt))}
    task = policy_prompt.get("task")
    if task:
        one["has_demo"] = episodes.has_demo(cfg, task)
        one["menu"] = MENUS[task]
    return one


def pick(st: Store, run: str, stage: str = "", after_iteration: int = 0) -> Path | None:
    """The policy prompt file to export, or None. The given run, else the newest one of `stage`.

    With `after_iteration` k, the given run's policy prompt as it stood after k iterations. An iteration's file is
    the one that iteration was scored with, so after iteration k is the file of iteration k + 1. After the last
    iteration it is the run's own file.
    """
    if after_iteration:
        q = st.policy_prompt(run, after_iteration + 1)
        if q.exists():
            return q
        q = st.policy_prompt(run)
        last = json.loads(q.read_text()) if q.exists() else {}
        return q if last.get("iterations") == after_iteration else None
    if str(run).isdigit():
        # A missing explicit run is reported missing. Another run never stands in for it.
        return st.policy_prompt(run) if st.policy_prompt(run).exists() else None
    best = None
    for r in st.runs_with_policy_prompt():
        q = st.policy_prompt(r)
        if stage and json.loads(q.read_text()).get("stage") != stage:
            continue
        best = q
    return best


def build(cfg: LoopConfig, tasks: list[str], run: str, stage: str = "",
          after_iteration: int = 0) -> tuple[dict, list[str]]:
    """{task: entry} for every task with a policy prompt, and the tasks without one."""
    out, missing = {}, []
    for task in tasks:
        path = pick(Store(cfg, task), run, stage, after_iteration)
        if path is None:
            missing.append(task)
            continue
        out[task] = entry(json.loads(path.read_text()), cfg)
        print(f"{task:18s} {len(out[task][variables.FIELDS]):2d} fields  <- {path}")
    return dict(sorted(out.items())), missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m memi.plan.export", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True,
                    help="the loop's --runs-dir, holding <Task>/policy_prompt/")
    ap.add_argument("--out", required=True, help="the file to write")
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--run", default="", help="the run to export, e.g. 1 or 2")
    which.add_argument("--stage", default="", choices=["init", "optimized"],
                       help="the newest policy prompt from this stage")
    ap.add_argument("--after-iteration", type=int, default=0,
                    help="with --run, the policy prompt as it stood after this "
                         "many iterations")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--decisions-dir", default=LoopConfig.decisions_dir)
    a = ap.parse_args(argv)
    cfg = LoopConfig(runs_dir=a.runs_dir, decisions_dir=a.decisions_dir)
    if a.after_iteration and not str(a.run).isdigit():
        ap.error("--after-iteration needs --run")
    exported, missing = build(cfg, a.tasks, a.run, a.stage, a.after_iteration)
    if missing and str(a.run).isdigit():
        print(f"no policy prompt for run {a.run}"
              + (f" after iteration {a.after_iteration}" if a.after_iteration else "")
              + f" of: {', '.join(missing)}. Nothing written.")
        return 1
    out = resolve(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(exported, indent=1))
    if missing:
        print(f"no policy prompt for: {', '.join(missing)}")
    print(f"{len(exported)} of {len(a.tasks)} tasks -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
