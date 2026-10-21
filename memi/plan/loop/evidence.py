"""Recorded decision episodes as the initializer and the annotator read them, step by step.

Each episode opens with its goal and its menu, and its steps are numbered from #1 within it.
"""
from __future__ import annotations

from ...text import heading
from ..policy_turn import menu_lines
from ..prompts import context
from . import episodes
from .config import LoopConfig
from .episodes import DecisionStep


def step_text(s: DecisionStep, note_head: str = "", note: str = "") -> str:
    """One step with its observed steps and its demonstrated subgoal, and `note` under `note_head` when given."""
    seg = "\n".join("        " + l for l in (s.segment or "(no observations)").splitlines())
    out = (f"STEP #{s.n}  t={s.t} [decision step]\n    OBSERVED STEPS\n{seg}\n    SUBGOAL\n        "
           + (s.subgoal or "(none)"))
    if note:
        out += f"\n    {note_head}\n        {note}"
    return out


def episodes_turn(cfg: LoopConfig, task: str, eps: list[int], opening: str = "", closing: str = "",
                  notes: dict | None = None, note_head: str = "") -> tuple[list[str], list[DecisionStep]]:
    """The user turn as text parts to join with newlines, and the steps they show. `notes` maps a step's key to a
    line shown under `note_head` at that step."""
    parts = [context.EPISODES_INTRO]
    if opening:
        parts.append("\n" + opening)
    shown: list[DecisionStep] = []
    for e in eps:
        seq = episodes.steps(cfg, task, e)
        if not seq:
            continue
        shown += seq
        parts.append("\n\n" + heading(f"EPISODE {e} ({len(seq)} steps)") + f"\nGOAL\n    {seq[0].goal}"
                     + "\nMENU\n" + "\n".join("  " + l for l in menu_lines(seq[0].menu).splitlines()))
        parts += ["\n" + step_text(s, note_head, (notes or {}).get(s.key, "")) for s in seq]
    if closing:
        parts.append("\n" + closing)
    return parts, shown
