"""The Plan policy pi_theta of Eq. (1) at one decision step, as the loop and the rollouts both run it.

Memory is a dict {field: value} without empty fields, and an answer is one JSON object with `choice`, `point`
([y, x] or null), `memory_updates` and `used_fields`.
"""
from __future__ import annotations

import json
import math
import re

from ..llm.parse import parse_object
from ..text import MEMORY, fill, memory_line
from . import prompts
from .variables import FIELDS, SUBGOAL, field_names, fields_block


def entry(policy_prompt: dict | None) -> dict:
    """The parts of the policy prompt (theta) the policy turn is filled from.

    The policy prompt for subgoal keeps its whitespace and paragraphs.
    """
    policy_prompt = policy_prompt or {}
    fields = [{"name": str(s.get("name", "")), "description": str(s.get("description", ""))}
              for s in policy_prompt.get(FIELDS) or []]
    return {MEMORY: fields_block(policy_prompt.get(FIELDS) or [], indent=2) if fields else "  (none on file)",
            SUBGOAL: str(policy_prompt.get(SUBGOAL) or ""), FIELDS: fields}


def memory_text(entry: dict | None, memory: dict | None) -> str:
    """The carried memory as the MEMORY CARRIED IN line of the turn."""
    return memory_line(field_names(entry), memory or {})


def updates(answer) -> dict:
    """The fields an answer changes, {field: value}. A later entry for a field wins.

    `memory_updates` is asked for as a list of {"field", "value"}. A dict, or `name=value` strings, are read too.
    """
    raw = (answer if isinstance(answer, dict) else {}).get("memory_updates") or []
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    out: dict[str, str] = {}
    for e in raw if isinstance(raw, list) else []:
        if isinstance(e, dict) and e.get("field") is not None:
            out[str(e["field"])] = str(e.get("value", ""))
        elif isinstance(e, str) and "=" in e:
            k, v = e.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def carry(names: list[str], memory: dict | None, changes: dict | None) -> dict:
    """`memory` with `changes` applied. Only a field in `names` takes a value. An empty value clears its field, and
    fields with no value are left out."""
    state = {k: str((memory or {}).get(k) or "") for k in names}
    for k, v in (changes or {}).items():
        if k in state:
            state[k] = str(v)
    return {k: v for k, v in state.items() if v}


def build_user(goal: str, entry: dict, menu: list, segment: str, memory: str, phase: str, saw_demo: bool) -> str:
    """The policy's user turn. `entry` is `entry(policy_prompt)` and `memory` the line `memory_text` renders. `phase` is
    "first" at the first decision step of an episode with a demonstration, and "decide" otherwise."""
    if phase not in ("first", "decide"):
        raise ValueError(f"unknown phase {phase!r}")
    video = "first" if phase == "first" else ("exec" if saw_demo else "none")
    entry = entry or {}
    return fill(
        prompts.POLICY_USER,
        policy_prompt_for_subgoal="  " + ((entry.get(SUBGOAL) or "").strip() or "(none on file)"),
        policy_prompt_for_memory=(entry.get(MEMORY) or "").rstrip() or "  (none on file)",
        goal=goal,
        video=prompts.policy.VIDEO_LINE[video],
        segment="\n".join("  " + l for l in (segment or "(no observations)").splitlines()),
        memory=f"  {memory}" if memory else "  (empty)",
        options=menu_lines(menu))


def label_of(choice) -> str:
    """A choice as one menu label. `a`, `A`, `(A)` and `A: pick up` are all `A`."""
    m = re.match(r"\s*\(?\s*([A-Za-z0-9]+)", str(choice or ""))
    return m.group(1).upper() if m else ""


def option(menu: list, choice) -> dict | None:
    want = label_of(choice)
    return next((o for o in menu if label_of(o.get("label")) == want), None) if want else None


def needs_point(option: dict | None) -> bool:
    """Whether the planner's wrapper needs a point for `option`. It does whenever the option has an `available` key,
    even an empty one (`rollout.episode.follow_wrapper_rule`)."""
    o = option or {}
    return bool(o.get("need_parameter")) or "available" in o


def menu_lines(menu: list) -> str:
    """The menu as the policy and the offline roles read it."""
    return "\n".join(
        f"  {label_of(o.get('label'))}  {o.get('action', '')}" + ("  (needs a point)" if needs_point(o) else "")
        for o in menu) or "  (none)"


def command(menu: list, choice, point) -> str:
    """One subgoal as text. The label, its action, and the point when it needs one."""
    label = label_of(choice)
    if not label:
        return ""
    o = option(menu, label)
    if o is None:
        return f"{label} (not on the menu)"
    text = f"{label}: {o.get('action', '')}"
    if needs_point(o):
        text += f" at [{point[0]}, {point[1]}]" if point else " (no point)"
    return text


def _label_ok(choice, menu: list):
    """The menu option the wrapper would select, or None. The wrapper strips and lowercases the choice and then needs
    an exact label match."""
    if not isinstance(choice, str) or not choice.strip():
        return None
    want = choice.strip().lower()
    return next((o for o in menu or [] if o.get("label") == want), None)


def _point_ok(point) -> bool:
    """The wrapper's own test. A list of two finite numbers, row first."""
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return False
    try:
        return all(math.isfinite(float(v)) for v in point[:2])
    except (TypeError, ValueError):
        return False


def parse_answer(text: str, menu: list) -> tuple[dict, str]:
    """(the command sent to the env, its status). Nothing is repaired.

    Status is one of
      valid           a label in the menu, with a usable point where one is needed
      empty           the model answered with no text
      unparseable     the text holds no JSON object
      not_in_menu     the choice is not a label of the current menu
      missing_point   the command needs a point and there is none
      bad_point       the command needs a point and it is not two numbers
    Only `valid` executes. The wrapper rejects every other status, and the episode ends as `api_error`. An answer
    without a JSON object goes on as {"choice": None}. The loop grades its policy with this rule too.
    """
    if not (text or "").strip():
        return {"choice": None, "point": None}, "empty"
    answer = parse_object(text)
    if not isinstance(answer, dict):
        return {"choice": None, "point": None}, "unparseable"
    cmd = {"choice": answer.get("choice"), "point": answer.get("point")}
    chosen = _label_ok(cmd["choice"], menu)
    if chosen is None:
        return cmd, "not_in_menu"
    if needs_point(chosen):
        if cmd["point"] is None:
            return cmd, "missing_point"
        if not _point_ok(cmd["point"]):
            return cmd, "bad_point"
    return cmd, "valid"


def why_invalid(cmd: dict, status: str) -> str:
    """One sentence on why the planner would not execute `cmd`, empty when valid. The loss records it as
    `format_reason` and in the answered subgoal."""
    choice, point = json.dumps(cmd.get("choice")), json.dumps(cmd.get("point"))
    return {
        "valid": "",
        "empty": "The answer had no text.",
        "unparseable": "The answer held no JSON object.",
        "not_in_menu": f"The choice {choice} is not a label of the menu. The choice "
                       "must be the label alone, copied verbatim.",
        "missing_point": f"The option {choice} needs a point and the answer gave none.",
        "bad_point": f"The point {point} is not a list of two numbers [y, x].",
    }.get(status, f"The answer was not executable ({status}).")
