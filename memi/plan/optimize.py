"""The MEMI optimization of one Plan task, the K iterations of Algorithm 1 (blog Section 2.3).

    python -m memi.plan.optimize --task PickXtimes                                               MEMI-Astra-Plan
    python -m memi.plan.optimize --task PickXtimes --policy-model gpt-6-luna                     MEMI-Luna-Plan
    python -m memi.plan.optimize --task StopCube --policy-model gpt-6-luna --ablate sufficiency  Section 3.4 ablation

A run starts from the latest phi and theta that `memi.plan.init` wrote under `<runs-dir>/<Task>/` and keeps the last
iteration's prompts. `--resume` continues a run that a failed model call stopped, after its last finished iteration.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
from dataclasses import dataclass, field

from ..llm import LLMClient, budget
from ..llm.client import EFFORTS
from ..paths import load_env
from ..text import ANNOTATOR_PROMPT, MEMORY, memory_line
from . import ablation, variables
from .loop import episodes
from .loop.config import LoopConfig, add_common_args, client
from .loop.episodes import key_of
from .loop.iteration import Iteration, iteration, models
from .loop.store import Store, write_json, write_jsonl


class Stopped(RuntimeError):
    """A role gave no usable answer. The run stops and `--resume` picks it up."""


@dataclass
class Run:
    """The state of a run. Its state file holds it, so `--resume` can continue the run."""
    no: int
    start: int
    annotator_and_policy_prompts: dict
    initial: dict
    history: list = field(default_factory=list)
    revisions: int = 0
    quiet: int = 0
    measured: list = field(default_factory=list)


def _ablate(cfg: LoopConfig) -> dict:
    """The principle this run leaves out, as a record key. The full loop adds none."""
    return {"ablate": cfg.ablate} if cfg.ablate else {}


def _resume(cfg: LoopConfig, st: Store, task: str, latest: int) -> Run | None:
    """Run `latest` from its state file, or None when it has none."""
    path = st.path("state", latest)
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    # Resuming a run under another ablation would mix two conditions in one run.
    if state.get("ablate") != cfg.ablate:
        def arm(p):
            return f"--ablate {p}" if p else "the full loop"
        raise SystemExit(f"[{task}] run {state['run']} was written with {arm(state.get('ablate'))}, "
                         f"not {arm(cfg.ablate)}")
    print(f"[{task}] resuming run {state['run']} at iteration {state['iteration'] + 1}", flush=True)
    return Run(state["run"], state["iteration"] + 1, state["variables"],
               copy.deepcopy(state.get("initial") or state["variables"]), state["history"], state["revisions"],
               state["quiet"], state.get("measured") or [])


def _start(st: Store, task: str, latest: int) -> Run:
    """A new run that starts from the prompts of run `latest`."""
    annotator_and_policy_prompts = variables.merge(json.loads(st.policy_prompt(latest).read_text()),
                                json.loads(st.annotator_prompt(latest).read_text()))
    print(f"[{task}] starting from {st.policy_prompt(latest)}", flush=True)
    return Run(st.next_run(), 1, annotator_and_policy_prompts, copy.deepcopy(annotator_and_policy_prompts))


def _recorder(cfg: LoopConfig, st: Store, task: str, run: Run, k: int, offline: LLMClient, policy_client: LLMClient):
    """The callback that writes one iteration's step rows and summary."""
    names = variables.field_names(run.annotator_and_policy_prompts)

    def record(it: Iteration) -> None:
        for e in it.eps:
            write_jsonl(st.steps(run.no, k, e),
                        [{**x, "memory_carried": memory_line(names, it.carried.get(key_of(x["step"])) or {}),
                          "critique": it.critiques.get(key_of(x["step"]), {})}
                         for x in it.rows if x["step"][0] == e])
        write_json(st.path("iteration", run.no, k), {
            "run": run.no, "iteration": k, "fields": names, **_ablate(cfg),
            "models": models(offline, policy_client), **it.scored, "critique": it.critique,
            "proposal": it.proposal, "critiques": list(it.critiques.values())})
        s = it.scored
        print(f"[{task}] run {run.no} iteration {k}: eps {it.eps}  {s['scored']} scored, {s['failed']} failed "
              f"({s.get('format_wrong', 0)} format, {s['choice_wrong']} choice, {s['point_wrong']} point, "
              f"{s['memory_wrong']} memory), verified {s['verified_of_all']:.3f}", flush=True)
        if (it.proposal.get("refused") or "").strip():
            print(f"[{task}]   the annotator refused: {it.proposal['refused']}", flush=True)
    return record


def _revise(st: Store, task: str, run: Run, k: int, it: Iteration) -> None:
    """Take in what the optimizer committed, or count a quiet iteration. Raises Stopped on an unusable revision."""
    if it.revised is None:
        run.quiet += 1
        print(f"[{task}]   nothing to fix ({run.quiet} quiet in a row)", flush=True)
        return
    run.quiet = 0
    meta = it.meta
    write_json(st.path("optimizer", run.no, k), meta)
    if not it.revised.get(variables.FIELDS):
        raise Stopped(f"iteration {k}: the optimizer gave no usable revision (see {st.path('optimizer', run.no, k)})")
    for bad in meta.get("not_applied") or []:
        print(f"[{task}]   NOT APPLIED: {bad}", flush=True)
    if (meta.get("refused") or "").strip():
        print(f"[{task}]   REFUSED: {meta['refused']}", flush=True)
    applied = meta.get("applied") or {}
    if not any(applied.get(key) for key in (variables.SUBGOAL, MEMORY, ANNOTATOR_PROMPT)):
        # Every change refused is a quiet iteration, not a failed one.
        run.quiet += 1
        print(f"[{task}]   nothing landed ({run.quiet} quiet in a row)", flush=True)
    # The history keeps the critique and what landed. The optimizer is shown phi and theta once, as they stand.
    run.history = run.history + [{"scored": it.scored_for_optimizer, "critique": it.critique,
                                  "applied": meta.get("applied") or {}, "not_applied": meta.get("not_applied") or []}]
    run.annotator_and_policy_prompts, run.revisions = it.revised, run.revisions + 1
    print(f"[{task}]   revision {run.revisions}: {variables.describe(run.annotator_and_policy_prompts)}", flush=True)


def _finish(st: Store, task: str, run: Run, cfg: LoopConfig, last: int, max_iterations: int) -> None:
    """The run's final policy prompt and annotator prompt, which are the last iteration's."""
    why = f"ran the {max_iterations} iterations it was given"
    stamp = {"task": task, "run": run.no, **_ablate(cfg), "iterations": last, "revisions": run.revisions,
             "stopped_because": why,
             "measured": [{k: v for k, v in m.items() if k != "variables"} for m in run.measured]}
    # Each iteration draws a fresh minibatch, so its numbers are recorded and never used to choose.
    write_json(st.policy_prompt(run.no),
               {**stamp, **variables.policy_prompt(run.annotator_and_policy_prompts), "stage": "optimized"})
    write_json(st.annotator_prompt(run.no),
               variables.stored_annotator_prompt(run.annotator_and_policy_prompts, **stamp, stage="optimized"))
    print(f"[{task}] run {run.no}: {why}, {run.revisions} revisions -> {st.policy_prompt(run.no)}", flush=True)


async def optimize(cfg: LoopConfig, task: str, batch: int, offline: LLMClient, seed: int, resume: bool,
                   max_iterations: int, policy_client: LLMClient | None = None) -> int:
    """Run `max_iterations` iterations of one task from its latest run under `cfg.runs_dir`. Returns the exit code."""
    st = Store(cfg, task)
    policy_client = policy_client or offline
    sample = episodes.sampler(task, episodes.train(cfg, task), batch, seed)
    demo = episodes.has_demo(cfg, task)
    latest = st.latest_run()
    if latest is None:
        raise SystemExit(f"[{task}] no policy prompt under {st.root}. Run memi.plan.init first")
    run = (_resume(cfg, st, task, latest) if resume else None) or _start(st, task, latest)
    # A resumed run winds the minibatch stream past the iterations already run.
    for _ in range(run.start - 1):
        sample()
    last = run.start + max_iterations - 1
    for k in range(run.start, last + 1):
        write_json(st.policy_prompt(run.no, k),
                   {"task": task, **variables.policy_prompt(run.annotator_and_policy_prompts)})
        write_json(st.annotator_prompt(run.no, k),
                   variables.stored_annotator_prompt(run.annotator_and_policy_prompts, task=task))
        it = await iteration(cfg, task, sample, run.annotator_and_policy_prompts, run.history, run.initial, offline,
                             policy_client, demo, _recorder(cfg, st, task, run, k, offline, policy_client))
        run.measured.append({"iteration": k, "episodes": list(it.eps),
                             "verified_of_all": it.scored.get("verified_of_all"),
                             "subgoal_verified": it.scored.get("subgoal_verified"),
                             "variables": copy.deepcopy(run.annotator_and_policy_prompts)})
        _revise(st, task, run, k, it)
        write_json(st.path("state", run.no), {
            "run": run.no, "iteration": k, "variables": run.annotator_and_policy_prompts, "initial": run.initial,
            "history": run.history, "revisions": run.revisions, "quiet": run.quiet, "measured": run.measured,
            **_ablate(cfg)})
    _finish(st, task, run, cfg, last, max_iterations)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m memi.plan.optimize", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--batch-episodes", type=int, default=5, help="training episodes per iteration")
    ap.add_argument("--seed", type=int, default=episodes.DEFAULT_SEED)
    ap.add_argument("--max-iterations", type=int, default=5, help="the iterations to run, 1 or more")
    ap.add_argument("--concurrency", type=int, default=LoopConfig.concurrency, help="policy calls in flight at once")
    ap.add_argument("--policy-model", default="", help="the policy's model when it differs from --model")
    ap.add_argument("--policy-effort", choices=[""] + EFFORTS, default="",
                    help="the policy's effort when it differs from --effort")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ablate", choices=ablation.PRINCIPLES, default=None,
                    help="remove one principle from the loop, its wording in the offline roles' prompts and the "
                         "feedback it produces")
    a = ap.parse_args(argv)
    if a.max_iterations < 1:
        ap.error("--max-iterations must be 1 or more")
    load_env()
    if a.ablate:
        print(f"[{a.task}] ablation: {a.ablate} removed from the loop", flush=True)
    cfg = LoopConfig(runs_dir=a.runs_dir, decisions_dir=a.decisions_dir, concurrency=a.concurrency, ablate=a.ablate)
    trace = Store(cfg, a.task).provider_trace()
    offline = client(a.model, a.effort, a.max_tokens, trace)
    policy_client = None
    if a.policy_model or a.policy_effort:
        policy_client = client(a.policy_model or offline.model, a.policy_effort or offline.effort,
                               budget(a.policy_effort) if a.policy_effort else offline.max_tokens, trace)
        print(f"[{a.task}] the policy runs as {policy_client.model} at effort {policy_client.effort}. "
              f"The offline roles run at {offline.effort}", flush=True)
    return asyncio.run(optimize(cfg, a.task, a.batch_episodes, offline, a.seed, a.resume, a.max_iterations,
                                policy_client))


if __name__ == "__main__":
    raise SystemExit(main())
