"""The initializer of blog Section 2.3. It writes the initial annotator prompt phi and policy prompt theta of a task.

    python -m memi.plan.init --task PickXtimes

One call reads `--num-episodes` training episodes. The prompts go under `<runs-dir>/<Task>/` as the next run.
"""
from __future__ import annotations

import argparse
import asyncio

from ..paths import load_env
from . import variables
from .loop import episodes, initializer
from .loop.config import LoopConfig, add_common_args, client
from .loop.store import Store, write_json


async def init(cfg: LoopConfig, task: str, n: int, offline, seed: int) -> int:
    """Initialize `task` from `n` training episodes. Returns the exit code, 1 when no answer parsed."""
    st = Store(cfg, task)
    run = st.next_run()
    eps = episodes.sampler(task, episodes.train(cfg, task), n, seed, salt="init")()
    print(f"[{task}] run {run}: initializing on {len(eps)} episodes {eps} "
          f"with {offline.provider}/{offline.model}", flush=True)
    annotator_and_policy_prompts, meta = await initializer.initialize(cfg, task, eps, offline)
    write_json(st.path("initializer", run), meta)
    if not annotator_and_policy_prompts:
        print(f"[{task}] no answer parsed. The reply is in {st.path('initializer', run)}", flush=True)
        return 1
    stamp = {"task": task, "run": run, "stage": "init", "initialized_on": eps, "seed": seed,
             "provider": offline.provider, "model": offline.model, "reasoning_effort": offline.effort}
    write_json(st.policy_prompt(run), {**stamp, **variables.policy_prompt(annotator_and_policy_prompts)})
    write_json(st.annotator_prompt(run), variables.stored_annotator_prompt(annotator_and_policy_prompts, **stamp))
    for bad in annotator_and_policy_prompts.get("name_mismatch") or []:
        print(f"[{task}] NAME MISMATCH: {bad}", flush=True)
    print(f"[{task}] run {run}: {variables.describe(annotator_and_policy_prompts)}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m memi.plan.init", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--num-episodes", type=int, default=5, help="training episodes the initializer reads")
    ap.add_argument("--seed", type=int, default=episodes.DEFAULT_SEED)
    a = ap.parse_args(argv)
    load_env()
    cfg = LoopConfig(runs_dir=a.runs_dir, decisions_dir=a.decisions_dir)
    offline = client(a.model, a.effort, a.max_tokens, Store(cfg, a.task).provider_trace())
    return asyncio.run(init(cfg, a.task, a.num_episodes, offline, a.seed))


if __name__ == "__main__":
    raise SystemExit(main())
