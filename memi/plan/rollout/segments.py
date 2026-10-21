"""The segment a policy call is shown, the text observations of the env steps since the previous decision point. The
rollouts and the dataset builder share it, so an offline step reads like a rollout call.
"""
from __future__ import annotations

import re

STRIDE = 16
DEMO_HEAD = "DEMONSTRATION STEPS"
EXEC_HEAD = "EXECUTION STEPS"

_ANGLE = re.compile(r"<\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*>")

#: VideoRepick builds its cubes as `bin_<i>` except on hard, so the scene would call them containers.
_RENAMED = {"VideoRepick": (re.compile(r"^bin_"), "cube_")}


def as_seen(rows: list[dict], task: str) -> list[dict]:
    """`rows` with the actor names the scene text should use.

    Only the names change, and only on copies. Rows of other tasks come back
    as they are.
    """
    if task not in _RENAMED:
        return rows
    pat, new = _RENAMED[task]
    out = []
    for r in rows:
        recs = r.get("visible_object_records")
        if not recs:
            out.append(r)
            continue
        out.append(dict(r, visible_object_records=[
            dict(o, actor_name=pat.sub(new, o["actor_name"]))
            if isinstance(o.get("actor_name"), str) else o for o in recs]))
    return out


def bracket(text: str) -> str:
    """Every `<y, x>` in `text` written `[y, x]`."""
    return _ANGLE.sub(r"[\1, \2]", text)


def picked(start: int, end: int, stride: int = STRIDE) -> list[int]:
    """Timesteps in [start, end) that a call over that range shows. The multiples of `stride`, and always the decision
    step end - 1."""
    if end <= start:
        return []
    out = [t for t in range(start, end) if t % stride == 0]
    if not out or out[-1] != end - 1:
        out.append(end - 1)
    return out


def render(steps: list[tuple[int, str]]) -> str:
    """The segment text of `steps`, [(timestep, text observation)] in time order.

    The planner takes a point as [y, x], so every coordinate the model reads is written [y, x]. `scene.describe` writes
    <y, x>, and `bracket` rewrites it.
    """
    if not steps:
        return "(no observations)"
    return "\n".join(
        f"t={t}\n" + "\n".join(f"  {bracket(l.strip())}" for l in text.splitlines()
                               if l.strip())
        for t, text in steps)


def with_demo(demo: str, execution: str) -> str:
    """The first decision step's segment on a task with a demonstration. Every demonstration step, sampled the same way,
    comes before the scene at the start of execution."""
    return f"{DEMO_HEAD}\n{demo}\n{EXEC_HEAD}\n{execution}"
