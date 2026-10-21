"""Recorded episodes as the initializer and the annotator read them, cycle by cycle."""
from __future__ import annotations

import json

from ..llm import responses as llm
from ..text import memory_line
from . import episodes
from .policy_turn import STEPS_OF


def gripper_word(after: int) -> str:
    return "open" if after > 0 else "closed"


def recorded_episode(ep: dict, annotated: dict | None = None, names: list[str] | None = None,
                     show: set | None = None) -> list[dict]:
    """One episode as an offline role reads it, with what each cycle shows as the policy is shown it and the action
    chosen there. `annotated` (the memory after each cycle) adds the memory carried in at each cycle where it
    changes, written with the field `names`, which it then needs. `show`, when given, is the cycles whose images are
    sent. The others keep their text."""
    out = [llm.text(f"===== EPISODE {ep['episode']} =====\nInstruction: {ep['goal']}\n"
                    f"Camera calibration: {json.dumps(ep['calibration'])}")]
    carried_before = None
    for c in ep["cycles"]:
        shown = show is None or c["k"] in show
        head = f"#{c['k']} {c['kind'].upper()} cycle"
        out.append(llm.text(f"{head}: front-camera frames at steps {c['labels']}{STEPS_OF[c['kind']]}"
                            + (":" if shown else " (not shown here)")))
        labels = episodes.frame_labels(c)
        if shown:
            for t, lab in zip(c["frames"], labels):
                out += [llm.text(f"  {lab}:"), llm.image(episodes.front_png(ep, t))]
        lines = []
        if c["kind"] == "act":
            if shown:
                out += [llm.text(f"  current wrist camera ({labels[-1]}):"), llm.image(episodes.wrist_png(ep, c["t"]))]
            tg = c["target"]
            if tg.get("wait"):
                chosen = (f"  wait chosen: {tg['wait']} control steps in place, "
                          f"gripper {gripper_word(tg['gripper_after'])}")
            else:
                chosen = f"  waypoint chosen: {tg['waypoint']}, gripper after: {gripper_word(tg['gripper_after'])}"
            lines += [f"  robot state: {json.dumps(c['state'])}", chosen]
        if annotated is not None:
            carried = annotated.get(c["k"] - 1, {})
            if carried != carried_before:
                lines.insert(0, f"  MEMORY CARRIED IN: {memory_line(names, carried)}")
                carried_before = carried
        if lines:
            out.append(llm.text("\n".join(lines)))
    return out
