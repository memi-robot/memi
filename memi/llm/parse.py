"""The JSON object in a model's answer, fenced or not."""
from __future__ import annotations

import json
import re

JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def parse_object(text: str, close: bool = False) -> dict | None:
    """The JSON object in an answer, or None.

    The last fenced block is tried first, then the whole answer. With `close`, a
    long answer missing only its closing brackets is closed. Nothing else is
    repaired.
    """
    body = text.split("```")[-2] if "```" in text else text
    for candidate in (body.replace("json", "", 1), text):
        m = JSON_BLOCK.search(candidate)
        if not m:
            continue
        for attempt in (m.group(0), _closed(candidate[m.start():].rstrip()) if close else None):
            if not attempt:
                continue
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue
    return None


def _closed(s: str) -> str | None:
    """`s` with its unclosed brackets closed, or None if they are balanced.

    A long answer that is whole except for its final brace is otherwise lost
    for one character. Closing only ever adds to the end, so a reply cut off
    mid-sentence or mid-string still fails to parse.
    """
    depth, instr, esc = [], False, False
    for ch in s:
        if esc:
            esc = False
        elif instr:
            if ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
        elif ch == '"':
            instr = True
        elif ch in "{[":
            depth.append(ch)
        elif ch in "}]" and depth:
            depth.pop()
    if instr or not depth:
        return None
    return s + "".join("}" if c == "{" else "]" for c in reversed(depth))
