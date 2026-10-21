"""Step 4 of Figure 5, the loss of Eqs. (3) to (5). Sufficiency and predictability at each cycle, necessity over the
iteration.

`waypoint_wrong` counts the act cycles that fail sufficiency (D_a, Eq. 3), `memory_wrong` the cycles that fail
predictability (D_m, Eq. 4), and `unused_fields` holds the fields that fail necessity (Eq. 5).
"""
from __future__ import annotations

import collections

from .. import config, memory, variables
from ..policy_turn import parse_action

#: Eq. (5) counts only the fields no cycle used. This code also reports a field used at no more than this share of
#: the iteration's cycles.
NECESSITY_SHARE = 0.05


def gripper_after(command, c: dict) -> int:
    """The gripper state a command leaves. 0 keeps the state last commanded, which is the expert's before this
    cycle under teacher forcing, as at test."""
    return int(command) if command in (-1, 1) else c["gripper_before"]


def grade(task: str, c: dict, answer: dict, mem_after: dict, annotated_after: dict, fields: dict) -> dict:
    """D_a of Eq. (3) and D_m of Eq. (4) at one cycle. An act cycle's waypoint is right within config.POS_TOL_M and
    with the right gripper state after it, and its wait within config.WAIT_TOL steps. `memory_wrong` lists the
    fields whose written value does not match the annotation."""
    g = {"waypoint_wrong": False, "memory_wrong": [], "distance_m": None}
    target = c.get("target")
    kind, arg = parse_action(answer)
    if c["kind"] == "act" and target.get("wait"):
        g["waypoint_wrong"] = not (kind == "wait" and abs(arg - target["wait"]) <= config.WAIT_TOL)
    elif c["kind"] == "act":
        if kind != "move":
            g["waypoint_wrong"] = True
        else:
            d = sum((a - b) ** 2 for a, b in zip(answer["waypoint"][:3], target["waypoint"][:3])) ** 0.5
            g["distance_m"] = round(d, 4)
            grip_ok = (task in config.STICK_TASKS
                       or gripper_after(answer.get("gripper", 0), c) == target["gripper_after"])
            g["waypoint_wrong"] = d > config.POS_TOL_M or not grip_ok
            g["gripper_ok"] = grip_ok
    for n in annotated_after:              # a field the annotator left unset is not checked
        if not memory.matches(mem_after.get(n, ""), annotated_after[n], fields.get(n, {})):
            g["memory_wrong"].append(n)
    return g


def unused_fields(steps: list[dict], failed: list[dict], names: list[str]) -> dict[str, str]:
    """Necessity, Eq. (5). Each field used at NECESSITY_SHARE of the cycles or fewer, with where it was used and at
    how many failed cycles it was the only wrong field."""
    read_at = collections.defaultdict(list)
    for s in steps:
        for u in s["used"]:
            read_at[u].append(f"ep{s['episode']} #{s['k']}")
    sole = collections.Counter(s["grade"]["memory_wrong"][0] for s in failed
                               if len(s["grade"]["memory_wrong"]) == 1 and not s["grade"]["waypoint_wrong"])
    out = {}
    for n in names:
        if len(read_at[n]) > NECESSITY_SHARE * len(steps):
            continue
        line = (f"read at no cycle of {len(steps)}" if not read_at[n]
                else f"read at {len(read_at[n])} cycles of {len(steps)}: {', '.join(read_at[n])}")
        if sole[n]:
            line += f"; it is the only wrong field at {sole[n]} failed cycles"
        out[n] = line
    return out


def loss(task: str, eps: list[dict], steps: list[dict], annotator_and_policy_prompts: dict
         ) -> tuple[dict, list[dict], dict]:
    """Grades every step in place. Returns the score, the failed steps and the unused fields."""
    names = variables.field_names(annotator_and_policy_prompts)
    fields = {f["name"]: f for f in annotator_and_policy_prompts["fields"]}
    by_ep = {ep["episode"]: ep for ep in eps}
    for s in steps:
        s["grade"] = grade(task, by_ep[s["episode"]]["cycles"][s["k"] - 1], s["answer"], s["mem_after"],
                           s["ann_after"], fields)
    act = [s for s in steps if s["kind"] == "act"]
    score = {"cycles": len(steps), "act_cycles": len(act),
             "waypoint_wrong": sum(s["grade"]["waypoint_wrong"] for s in act),
             "memory_wrong": sum(bool(s["grade"]["memory_wrong"]) for s in steps),
             "fields": dict(collections.Counter(f for s in steps for f in s["grade"]["memory_wrong"])),
             "n_fields": len(names)}
    failed = [s for s in steps if s["grade"]["waypoint_wrong"] or s["grade"]["memory_wrong"]]
    return score, failed, unused_fields(steps, failed, names)
