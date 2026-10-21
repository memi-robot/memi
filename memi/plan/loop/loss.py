"""Step 4 of the loop, the loss. Sufficiency and predictability at each decision step, necessity over the iteration.

The record keys map to the three terms of Eq. (6). `subgoal_wrong` counts the steps that fail sufficiency (D_a,
Eq. 3), `memory_wrong` the steps that fail predictability (D_m, Eq. 4), and `unused_fields` holds the fields that fail
necessity (Eq. 5). `loss` is the share of the scored steps that fail either of the first two.
"""
from __future__ import annotations

import math
import re

from ..data import candidates
from ..policy_turn import carry, command, label_of, needs_point, option, parse_answer, updates, why_invalid
from .episodes import DecisionStep, key_of

#: Two coordinates within this many pixels are the same place.
COORD_TOL = 10.0
#: Necessity. Eq. (5) counts the fields no step used. A field read at no more than this share of the scored steps
#: is reported as unused too.
NECESSITY_SHARE = 0.05
_ANGLE = re.compile(r"<\s*(\d+)\s*,\s*(\d+)\s*>")
_SQUARE = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")


def _same_angle(a: str, b: str) -> bool:
    """Two values with points written <y, x> match. The words must be the same, ignoring case, and so must the
    number of points. Each point may move COORD_TOL px."""
    a, b = (a or "").strip(), (b or "").strip()
    if a.lower() == b.lower():
        return True
    if _ANGLE.sub("<y, x>", a).lower() != _ANGLE.sub("<y, x>", b).lower():
        return False
    pa = [(int(y), int(x)) for y, x in _ANGLE.findall(a)]
    pb = [(int(y), int(x)) for y, x in _ANGLE.findall(b)]
    if len(pa) != len(pb) or not pa:
        return False
    return all(math.dist(p, q) <= COORD_TOL for p, q in zip(pa, pb))


def same_value(a: str, b: str) -> bool:
    """D_m of Eq. (4) for one field. Two memory values match when equal ignoring case, or when they differ only in
    points within COORD_TOL px. Points written [y, x] are compared in the <y, x> form."""
    def angle(v: str) -> str:
        return _SQUARE.sub(lambda m: f"<{m.group(1)}, {m.group(2)}>", v or "")
    return _same_angle(angle(a), angle(b))


def same_target(step: DecisionStep, choice, point: list[int]) -> tuple[bool, dict | None]:
    """The object half of D_a of Eq. (3). Whether `point`, sent with `choice`, selects the recorded command's object.

    The selection is the wrapper's (`candidates.select`), the nearest projected candidate among the option's
    `available` actors. A step without candidates for the option, or whose recorded point matched none, raises.
    """
    label = str(choice or "").strip().lower()
    if step.candidates is None or label not in step.candidates:
        raise ValueError(
            f"{step.task} ep{step.episode} t{step.t}: no candidates for option {label!r}. "
            "Rebuild the decision data with python -m memi.plan.data.build_decisions.")
    want = step.label.get("object")
    if not want:
        raise ValueError(f"{step.task} ep{step.episode} t{step.t}: the recorded point "
                         "matched no candidate, so this step cannot be graded.")
    got = candidates.select(step.candidates[label], point, step.camera)
    return got is not None and candidates.key(got) == candidates.key(want), got


def _point(value) -> list[int]:
    """A point that `parse_answer` passed, rounded as the wrapper rounds it."""
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return []
    return [int(round(float(value[0]))), int(round(float(value[1])))]


def grade(answer: dict, step: DecisionStep, carried: dict, after: dict, names: list[str]) -> dict:
    """D_a and D_m at one step, written into `answer`.

        format_wrong   the planner would not execute the answer (`format_status` says why)
        choice_wrong   the label differs from the recorded one
        point_wrong    the option needs a point, the label is right, and the point selects another object
        subgoal_wrong  any of the three
        memory_wrong   the memory after this step disagrees with the annotation after it, at the `memory_diff` fields

    `answer` holds the parsed JSON and `raw`, the answer text. The memory written is `carry` over the policy prompt's
    fields and the carried ones, so an update to any other field is dropped as the rollouts drop it.
    """
    choice, point, reason = "", [], ""
    cmd, status = parse_answer(answer.get("raw") or "", step.menu)
    format_wrong = status != "valid"
    if format_wrong:
        reason = why_invalid(cmd, status)
        subgoal = (f"NOT EXECUTABLE ({status}). {reason} The answer was "
                   f"choice={cmd.get('choice')!r}, point={cmd.get('point')!r}.")
    else:
        choice, point = label_of(cmd["choice"]), _point(cmd["point"])
        subgoal = command(step.menu, choice, point)
    answer.update(step=step.step, recorded=step.subgoal, choice=choice, point=point, subgoal=subgoal,
                  format_status=status, format_wrong=format_wrong)
    if reason:
        answer["format_reason"] = reason
    choice_wrong = point_wrong = False
    if not format_wrong:
        choice_wrong = choice != label_of(step.label.get("choice"))
        if not choice_wrong and needs_point(option(step.menu, choice)):
            ok, got = same_target(step, choice, point)
            point_wrong = not ok
            answer["selected"] = got
            answer["recorded_object"] = step.label.get("object")
    answer["choice_wrong"], answer["point_wrong"] = choice_wrong, point_wrong
    answer["subgoal_wrong"] = format_wrong or choice_wrong or point_wrong
    answer["command_right"] = not answer["subgoal_wrong"]

    delta = updates(answer)
    merged = carry(list(names) + [k for k in carried if k not in names], carried, delta)
    answer["memory_updates_written"] = delta
    answer["memory_written"] = merged
    want = {k: str(v) for k, v in after.items()}
    diff = sorted(k for k, v in want.items() if not same_value(merged.get(k, ""), v))
    answer["memory_wrong"] = bool(want) and bool(diff)
    answer["memory_diff"] = diff
    answer["verified"] = not answer["subgoal_wrong"] and not answer["memory_wrong"]
    return answer


def failed(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not r.get("verified", True)]


def read_at(rows: list[dict], names: list[str]) -> dict[str, list[tuple]]:
    """Per field, the steps that said they used it."""
    out: dict[str, list[tuple]] = {f: [] for f in names}
    for r in rows:
        for f in r.get("used_fields") or []:
            if f in out:
                out[f].append(key_of(r["step"]))
    return out


def field_errors(rows: list[dict], names: list[str]) -> dict[str, dict]:
    """Per field, how many failed steps it is wrong at, and at how many it is the only wrong field."""
    out = {f: {"in": 0, "sole": 0} for f in names}
    for r in rows:
        diff = r.get("memory_diff") or []
        for f in diff:
            if f in out:
                out[f]["in"] += 1
                if len(diff) == 1:
                    out[f]["sole"] += 1
    return out


def unused_fields(rows: list[dict], names: list[str], number: dict, sole: bool = True) -> dict[str, str]:
    """L_nec of Eq. (5). The fields used at NECESSITY_SHARE of the scored steps or fewer, with how often and where.
    `number` maps a step's key to its number in its episode. With `sole`, how often each was the only wrong field."""
    scored = len(rows)
    if not scored:
        return {}
    errors, out = field_errors(rows, names), {}
    for name, keys in sorted(read_at(rows, names).items(), key=lambda kv: len(kv[1])):
        if len(keys) > scored * NECESSITY_SHARE:
            continue
        if not keys:
            out[name] = f"read at no step of {scored}"
            continue
        where = ", ".join(f"ep{e} #{number.get((e, t), '?')}" for e, t in sorted(keys))
        c = errors.get(name) or {}
        out[name] = (f"read at {len(keys)} step{'s' if len(keys) > 1 else ''} of {scored}: {where}"
                     + (f"; it is the only wrong field at {c['sole']} failed step{'s' if c['sole'] > 1 else ''}"
                        if sole and c.get("sole") else ""))
    return out


def _share(k: int, n: int) -> float:
    return k / n if n else float("nan")


def loss(answers: list[dict], steps: list[DecisionStep], carried: dict, after: dict, names: list[str],
         number: dict, eps: list[int], problems: list[str]) -> dict:
    """Step 4. Grades each answer at its step against the demonstrated subgoal and the annotated memory, and returns
    the iteration's record. `number` maps a step key to the step's number in its episode."""
    rows = [grade(a, s, carried.get(s.key) or {}, after.get(s.key) or {}, names) for a, s in zip(answers, steps)]
    n = len(rows)
    bad, sub = len(failed(rows)), sum(1 for r in rows if r.get("subgoal_wrong"))
    mem = sum(1 for r in rows if r.get("memory_wrong"))
    return {"episodes": eps, "scored": n, "failed": bad, "subgoal_wrong": sub, "memory_wrong": mem,
            "subgoal_verified": _share(n - sub, n), "memory_verified": _share(n - mem, n),
            "reads": {f: len(v) for f, v in read_at(rows, names).items()}, "field_errors": field_errors(rows, names),
            "unused_fields": unused_fields(rows, names, number), "loss": _share(bad, n),
            "verified_of_all": _share(n - bad, n), "other_problems": problems,
            # An answer the planner would not execute. Counted in subgoal_wrong.
            "format_wrong": sum(1 for r in rows if r.get("format_wrong")),
            "choice_wrong": sum(1 for r in rows if r.get("choice_wrong")),
            "point_wrong": sum(1 for r in rows if r.get("point_wrong")),
            "decisions": n}


def for_optimizer(scored: dict, rows: list[dict], critiques: dict, reported: tuple[str, ...]) -> dict:
    """This iteration's counts as the optimizer's history keeps them. `render_score` shows the first three and
    `fields`. `memory_only` counts the failed steps whose only fault is the memory, and `memory_only_unexplained`
    those of them whose critique names nothing in `reported`."""
    only = [r for r in failed(rows) if r.get("memory_wrong") and not r.get("subgoal_wrong")]
    blank = sum(1 for r in only if not any((critiques.get(key_of(r["step"])) or {}).get(k) for k in reported))
    return {"scored": scored.get("scored"), "subgoal_wrong": scored.get("subgoal_wrong"),
            "memory_wrong": scored.get("memory_wrong"), "memory_only": len(only), "memory_only_unexplained": blank,
            "fields": {f: c.get("in", 0) for f, c in (scored.get("field_errors") or {}).items()},
            "format_wrong": scored.get("format_wrong"), "choice_wrong": scored.get("choice_wrong"),
            "point_wrong": scored.get("point_wrong")}
