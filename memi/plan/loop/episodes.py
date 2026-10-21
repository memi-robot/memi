"""The recorded decision episodes and the sampler of step 1 of the loop, which draws the minibatch B^(k).

`memi.plan.data.build_decisions` writes one line per episode to `<decisions_dir>/<Task>.jsonl`. Each decision step
carries its segment already rendered, and on a task with a demonstration the first segment shows it first.
"""
from __future__ import annotations

import functools
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from ..policy_turn import command
from .config import LoopConfig

#: The seed of the released runs.
DEFAULT_SEED = 7
#: A row names its step [episode, AXIS, t].
AXIS = "exec"


@dataclass
class DecisionStep:
    """One decision point of a recorded episode, the n-th of its episode (from 1)."""
    task: str
    episode: int
    t: int
    n: int
    goal: str
    subgoal: str                               # the demonstrated subgoal a_t, as text
    segment: str = ""
    menu: list = field(default_factory=list)
    label: dict = field(default_factory=dict)  # the recorded choice, point and object
    demo: bool = False                         # the segment shows the demonstration
    candidates: dict | None = None             # label -> the option's candidate objects
    camera: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple[int, int]:
        return (self.episode, self.t)

    @property
    def step(self) -> list:
        """How a row names this step."""
        return [self.episode, AXIS, self.t]


def key_of(step: list) -> tuple[int, int]:
    """The key of a row's [episode, axis, t]."""
    return (int(step[0]), int(step[2]))


@functools.lru_cache(maxsize=None)
def _read(p: str) -> dict[int, dict]:
    if not Path(p).exists():
        raise SystemExit(f"no decision data at {p}. Build it with python -m memi.plan.data.build_decisions")
    out = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            out[int(r["episode"])] = r
    return out


def records(cfg: LoopConfig, task: str) -> dict[int, dict]:
    return _read(str(cfg.decisions(task)))


def train(cfg: LoopConfig, task: str) -> list[int]:
    """The training episodes. The initializer and every iteration draw from these only."""
    return sorted(e for e, r in records(cfg, task).items() if r.get("split") == "train")


def has_demo(cfg: LoopConfig, task: str) -> bool:
    """Whether the task's episodes open with a demonstration video."""
    return any(int(r.get("video_steps") or 0) > 0 for r in records(cfg, task).values())


def sampler(task: str, pool: list[int], n: int, seed: int = DEFAULT_SEED, salt: str = ""):
    """A function that draws a minibatch of n episodes of `pool` at each call, without replacement within a draw.

    Seeded on the task and the seed, so a run repeats and two tasks do not take the same episode numbers. Episodes
    may repeat across draws. The initializer draws with `salt="init"`, a stream apart from the iterations'.
    """
    pool = sorted(pool)
    rng = random.Random(f"{task}:{salt}:{seed}" if salt else f"{task}:{seed}")
    return lambda: sorted(rng.sample(pool, min(n, len(pool))))


def steps(cfg: LoopConfig, task: str, episode: int) -> list[DecisionStep]:
    """One episode's decision steps in the order they ran."""
    r = records(cfg, task)[episode]
    out = []
    for n, s in enumerate(sorted(r["steps"], key=lambda s: int(s["t"])), 1):
        label = s.get("label") or {}
        out.append(DecisionStep(
            task=task, episode=episode, t=int(s["t"]), n=n, goal=r["goal"],
            subgoal=command(r["menu"], label.get("choice"), label.get("point")),
            segment=s.get("segment", ""), menu=r["menu"], label=label, demo=bool(s.get("demo_shown")),
            candidates=s.get("candidates"), camera=s.get("camera") or {}))
    return out
