"""A task's run directory and the JSON files in it, numbered by run.

    policy_prompt/       policy_prompt.run2.json, policy_prompt.run2.iteration1.json
    annotator_prompt/    annotator_prompt.run2.json, annotator_prompt.run2.iteration1.json
    initializer/         initializer_output.run1.json
    optimizer/           optimizer_output.run2.iteration1.json
    iterations/          iteration_summary.run2.iteration1.json
    steps/               steps.run2.iteration1.ep37.jsonl, one row per decision step
    state/               run_state.run2.json
    provider_trace.jsonl every model call

A new run of `init` or `optimize` takes the next run number, so the released initializer files are run 1 and the
optimization's run 2. An iteration's policy prompt and annotator prompt are the ones that iteration was scored with.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .config import LoopConfig

#: Each kind of file a run writes, with its directory and its file name.
DIRS = {"policy_prompt": ("policy_prompt", "policy_prompt"),
        "annotator_prompt": ("annotator_prompt", "annotator_prompt"),
        "initializer": ("initializer", "initializer_output"),
        "optimizer": ("optimizer", "optimizer_output"),
        "iteration": ("iterations", "iteration_summary"),
        "steps": ("steps", "steps"),
        "state": ("state", "run_state")}
#: The kinds a run has started once it writes one, so the run's number is never handed out again.
_NUMBERED = ("policy_prompt", "annotator_prompt", "initializer", "state")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class Store:
    """The files of one task under `cfg.runs_dir`."""

    def __init__(self, cfg: LoopConfig, task: str):
        self.root = cfg.task_dir(task)

    def path(self, kind: str, run: int | str, iteration: int | str | None = None,
             episode: int | None = None, ext: str = ".json") -> Path:
        d, stem = DIRS[kind]
        suffix = f".run{run}" + (f".iteration{iteration}" if iteration is not None else "") + (
            f".ep{episode}" if episode is not None else "")
        return self.root / d / f"{stem}{suffix}{ext}"

    def provider_trace(self) -> Path:
        return self.root / "provider_trace.jsonl"

    def policy_prompt(self, run: int | str, iteration: int | str | None = None) -> Path:
        """theta of a run, the file `export` reads, or the theta an iteration was scored with."""
        return self.path("policy_prompt", run, iteration)

    def annotator_prompt(self, run: int | str, iteration: int | str | None = None) -> Path:
        """phi of a run or an iteration. Training time only."""
        return self.path("annotator_prompt", run, iteration)

    def steps(self, run: int | str, iteration: int | str, episode: int) -> Path:
        return self.path("steps", run, iteration, episode, ext=".jsonl")

    def _numbered(self, kinds) -> list[int]:
        out = set()
        for kind in kinds:
            d, stem = DIRS[kind]
            for p in (self.root / d).glob(f"{stem}.run*.json"):
                m = re.search(r"\.run(\d+)", p.stem)
                if m:
                    out.add(int(m.group(1)))
        return sorted(out)

    def next_run(self) -> int:
        """One past the highest run on disk, so no run overwrites another."""
        rs = self._numbered(_NUMBERED)
        return (rs[-1] + 1) if rs else 1

    def latest_run(self) -> int | None:
        """The last run with a policy prompt, and failing that, the last run."""
        return (self._numbered(_NUMBERED[:1]) or self._numbered(_NUMBERED) or [None])[-1]

    def runs_with_policy_prompt(self) -> list[int]:
        return self._numbered(_NUMBERED[:1])
