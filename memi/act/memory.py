"""The Act policy's memory m_t and its updates, and D_m of Eq. (4), how a written value is compared with the annotated
one."""
from __future__ import annotations

import json
import re

from . import config

VALUE_TYPES = ("world_coordinate", "count", "boolean", "text", "other", "keyframe")
_LABEL = re.compile(r"([VE])(-?\d+)", re.I)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def value_text(v) -> str:
    """A value as one line. A list or a number is written as JSON, whoever wrote it."""
    return v if isinstance(v, str) else json.dumps(v)


def apply_updates(memory: dict, updates, names: list[str]) -> dict:
    """The memory with an answer's `memory_updates` applied. Only declared fields, in the declared order."""
    new = dict(memory)
    for u in updates or []:
        if isinstance(u, dict) and u.get("field") in names:
            new[u["field"]] = value_text(u.get("value", ""))
    return {n: new[n] for n in names if n in new}


def keyframe_labels(v) -> list[str] | None:
    """The frame labels a keyframe value names, in order (`V80`, `[V80, E44]`), or None when it names none."""
    s = str(v)
    found = _LABEL.findall(s)
    if not found or not re.fullmatch(r"[\s\[\],'\"]*", _LABEL.sub("", s)):
        return None
    return [f"{a.upper()}{int(n)}" for a, n in found]


def _normalized(v) -> str:
    s = re.sub(r"\s+", " ", str(v).strip().lower())
    try:
        return repr(round(float(s), 3))
    except ValueError:
        return s.replace(" ,", ",").replace(", ", ",")


def matches(written, annotated, field: dict) -> bool:
    """D_m for one field. A value the policy wrote against the annotated one, by the field's value type. A field
    without a valid type is compared as text."""
    vt = field.get("value_type")
    if vt not in VALUE_TYPES:
        vt = "text"
    a, b = _normalized(written), _normalized(annotated)
    if a == b:
        return True
    if vt in ("count", "boolean", "text"):
        return False
    if vt == "keyframe":                     # the same frames in order, by letter and within one sampled frame
        la, lb = keyframe_labels(a), keyframe_labels(b)
        return bool(la and lb and len(la) == len(lb) and all(
            x[0] == y[0] and abs(int(x[1:]) - int(y[1:])) <= config.FRAME_STRIDE for x, y in zip(la, lb)))
    tol = config.POS_TOL_M if vt == "world_coordinate" else float(field.get("tolerance") or 0)
    xs, ys = _NUMBER.findall(a), _NUMBER.findall(b)
    return (_NUMBER.sub("#", a) == _NUMBER.sub("#", b) and len(xs) == len(ys)
            and all(abs(float(x) - float(y)) <= tol + 1e-9 for x, y in zip(xs, ys)))
