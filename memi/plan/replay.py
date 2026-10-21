"""Offline replay of the released Plan optimization runs, with no API call.

    python -m memi.plan.replay                                  the GPT-6 runs, Astra-Plan and Luna-Plan
    python -m memi.plan.replay --family claude                  the Claude runs, Fable-Plan and Haiku-Plan
    python -m memi.plan.replay --tasks BinFill --arms opt_luna
    python -m memi.plan.replay --arms opt_luna_no_sufficiency   one principle ablation

Every model call of `init` and `optimize` gets the answer recorded for the same purpose in the released
`provider_trace.jsonl`, in call order. Every file the run writes must equal the released one under
`artifacts/<family>/runs/`, and its export must equal the file under `artifacts/<family>/policy_prompts/`.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import json
import shutil
import tempfile
from pathlib import Path

from ..llm import LLMClient, LLMResponse
from ..paths import ROOT
from ..tasks import TASKS
from . import export, optimize
from . import init as init_
from .loop.config import LoopConfig
from .loop.store import DIRS

#: Each released GPT-6 run, as (the policy model where it is not the offline one, the run it wrote, its export).
ARMS = {"init": ("", 1, "initial.json"),
        "opt_astra_plan": ("", 2, "optimized_astra_plan.json"),
        "opt_luna": ("gpt-6-luna", 2, "optimized_luna.json"),
        "opt_luna_no_sufficiency": ("gpt-6-luna", 2, "optimized_luna_no_sufficiency.json"),
        "opt_luna_no_predictability": ("gpt-6-luna", 2, "optimized_luna_no_predictability.json")}
#: Each model family, with its released files, the model of the offline roles and its runs.
FAMILIES = {
    "gpt": {"root": ROOT / "artifacts" / "gpt", "offline": "gpt-6-astra", "arms": ARMS},
    "claude": {"root": ROOT / "artifacts" / "claude", "offline": "claude-fable-5-1",
               "arms": {"init": ("", 1, "initial.json"),
                        "opt_fable": ("", 2, "optimized_fable.json"),
                        "opt_haiku": ("claude-haiku-5-5", 2, "optimized_haiku.json"),
                        "opt_haiku_no_necessity": ("claude-haiku-5-5", 2, "optimized_haiku_no_necessity.json")}},
}
#: The released ablation runs, with the principle each removes and the tasks it was released for.
ABLATIONS = {"opt_luna_no_sufficiency": ("sufficiency", ("StopCube",)),
             "opt_luna_no_predictability": ("predictability", ("RouteStick",)),
             "opt_haiku_no_necessity": ("necessity", ("VideoPlaceOrder",))}
#: Keys of a final policy prompt or annotator prompt that record how a run was invoked. They are not compared,
#: because four GPT-6 tasks (ButtonUnmask, PickHighlight, PickXtimes, VideoPlaceButton) ran their five iterations as
#: five resumed invocations.
BOOKKEEPING = ("iterations", "revisions", "stopped_because", "measured")
#: The directories a run writes whose files are compared. The initializer and
#: optimizer records also hold the call's usage, which a replay does not have.
COMPARED = tuple(DIRS[k][0] for k in ("policy_prompt", "annotator_prompt", "state", "iteration", "steps"))
#: Trace rows [start, end) that the replay skips. The Astra-Plan trace of PickHighlight holds iteration 5 twice, from
#: a repeated invocation, and the released files are the second copy's. Either copy gives the released policy prompt.
SUPERSEDED = {("opt_astra_plan", "PickHighlight"): (90, 106)}


def recorded(trace: Path, skip: tuple[int, int] | None = None) -> dict[str, list[str]]:
    """purpose -> the recorded answers, last first, so `pop()` gives them in order. Rows in `skip` are left out."""
    out: dict[str, list[str]] = collections.defaultdict(list)
    for i, line in enumerate(open(trace)):
        if skip and skip[0] <= i < skip[1]:
            continue
        r = json.loads(line)
        out[r["purpose"]].append(r["response"])
    for v in out.values():
        v.reverse()
    return out


@contextlib.contextmanager
def replaying(answers: dict[str, list[str]], calls: list[dict], misses: list[str]):
    """Every `LLMClient.call` answered from `answers`. Each call is kept in `calls`.

    A purpose with no recorded answer left gets "{}" and is listed in `misses`.
    """
    original = LLMClient.call

    async def call(self, *, system, messages, purpose):
        text = answers[purpose].pop() if answers.get(purpose) else None
        if text is None:
            misses.append(purpose)
            text = "{}"
        calls.append({"purpose": purpose, "model": self.model, "effort": self.effort,
                      "max_tokens": self.max_tokens, "system": system,
                      "messages": [dict(m) for m in messages], "response": text})
        return LLMResponse(text=text, usage={}, raw={"model": self.model})

    LLMClient.call = call
    try:
        yield
    finally:
        LLMClient.call = original


def run(arm: str, task: str, runs_dir: Path, family: str = "gpt") -> dict:
    """Replay one released run into `runs_dir`. Returns its exit code, its calls, the purposes it found no answer
    for, and how many recorded answers it left unused."""
    fam = FAMILIES[family]
    src = fam["root"] / "runs" / arm / task
    policy_model = fam["arms"][arm][0]
    answers = recorded(src / "provider_trace.jsonl", SUPERSEDED.get((arm, task)))
    calls: list[dict] = []
    misses: list[str] = []
    task_dir = runs_dir / task
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)
    with replaying(answers, calls, misses), contextlib.redirect_stdout(None):
        if arm == "init":
            rc = init_.main(["--task", task, "--runs-dir", str(runs_dir), "--model", fam["offline"]])
        else:
            for sub in (DIRS[k][0] for k in ("policy_prompt", "annotator_prompt", "initializer")):
                (task_dir / sub).mkdir()
                for f in (src / sub).glob("*run1.json"):
                    shutil.copyfile(f, task_dir / sub / f.name)
            argv = ["--task", task, "--runs-dir", str(runs_dir), "--model", fam["offline"]]
            if policy_model:
                argv += ["--policy-model", policy_model, "--policy-effort", "low"]
            if arm in ABLATIONS:
                argv += ["--ablate", ABLATIONS[arm][0]]
            rc = optimize.main(argv)
    return {"rc": rc, "calls": calls, "misses": misses,
            "unused": sum(len(v) for v in answers.values())}


def strip(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k not in BOOKKEEPING}


def differing_files(arm: str, task: str, runs_dir: Path, family: str = "gpt") -> list[str]:
    """The files of a replayed run that differ from the released ones. The final policy prompt and annotator prompt
    are compared without BOOKKEEPING."""
    fam = FAMILIES[family]
    run_no = fam["arms"][arm][1]
    finals = {"/".join(DIRS[k]) + f".run{run_no}.json" for k in ("policy_prompt", "annotator_prompt")}
    got_dir, want_dir = runs_dir / task, fam["root"] / "runs" / arm / task
    got = {str(p.relative_to(got_dir)) for d in COMPARED for p in (got_dir / d).glob("*")}
    want = {str(p.relative_to(want_dir)) for d in COMPARED for p in (want_dir / d).glob("*")}
    out = sorted(got ^ want)
    for rel in sorted(got & want):
        a, b = (got_dir / rel).read_text(), (want_dir / rel).read_text()
        if rel in finals:
            same = strip(json.loads(a)) == strip(json.loads(b))
        else:
            same = a == b
        if not same:
            out.append(rel)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m memi.plan.replay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=list(FAMILIES), default="gpt",
                    help="gpt for the Astra-Plan and Luna-Plan runs, claude for the Fable-Plan and Haiku-Plan runs")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="the released runs to replay. By default init, the two MEMI runs and the ablations of the "
                         "family")
    ap.add_argument("--runs-dir", default=None,
                    help="where the replayed runs are written. A temporary directory by default")
    a = ap.parse_args(argv)
    fam = FAMILIES[a.family]
    arms = fam["arms"]
    a.arms = a.arms or list(arms)
    for arm in a.arms:
        if arm not in arms:
            ap.error(f"arm {arm!r} is not a run of family {a.family}. Known: {', '.join(arms)}")
    tmp = None
    if a.runs_dir:
        root = Path(a.runs_dir).resolve()
    else:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
    bad = 0
    try:
        for arm in a.arms:
            runs_dir = root / arm
            tasks = [t for t in a.tasks if arm not in ABLATIONS or t in ABLATIONS[arm][1]]
            if not tasks:
                print(f"{arm:26s} skipped: released only for {', '.join(ABLATIONS[arm][1])}", flush=True)
                continue
            for task in tasks:
                got = run(arm, task, runs_dir, family=a.family)
                diff = differing_files(arm, task, runs_dir, a.family)
                ok = got["rc"] == 0 and not diff and not got["misses"] and not got["unused"]
                bad += not ok
                print(f"{arm:26s} {task:18s} {'ok  ' if ok else 'FAIL'} "
                      f"calls {len(got['calls']):4d}  misses {len(got['misses'])}  "
                      f"unused {got['unused']}  differing files {len(diff)}"
                      + (f" {diff[:3]}" if diff else ""), flush=True)
            entries, _ = export.build(LoopConfig(runs_dir=str(runs_dir)), tasks,
                                      str(arms[arm][1]))
            want_file = fam["root"] / "policy_prompts" / arms[arm][2]
            want = json.loads(want_file.read_text())
            # A GPT-6 ablation file is the exact file the rollouts read and also holds the other ablation's task,
            # which has no released run here. Only the replayed tasks are compared.
            diff = [t for t in tasks if entries.get(t) != want.get(t)]
            bad += bool(diff)
            print(f"{arm:26s} export vs {want_file.relative_to(ROOT)}: "
                  f"{'identical' if not diff else 'differs for ' + ', '.join(diff)}",
                  flush=True)
    finally:
        if tmp is not None:
            tmp.cleanup()
    print("ALL IDENTICAL" if not bad else f"{bad} MISMATCHES")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
