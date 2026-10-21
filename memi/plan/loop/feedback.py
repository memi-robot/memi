"""A diagnostic note on a proposal whose policy prompt edit shares almost no words with the policy's critiques.

The note is logged and kept in the iteration summary. No role reads it, so it is not part of the method.
"""
from __future__ import annotations

import re

from ...text import MEMORY
from ..variables import SUBGOAL

#: Below this share of the complaints answered, a policy prompt change is reported as
#: possibly unfounded.
TRACED = 0.35
#: Fewer complaints than this and the check is not run. A short list has a small
#: vocabulary, so the share is low whatever the answer says.
MIN_COMPLAINTS = 8
_STOP = frozenset(
    "the a an of to at in is was were and or that this it its for with on from "
    "be been as by not no one two into which when where what if then than so "
    "has have had you your are over under up down each any all more most only "
    "step steps frame frames policy memory instruction instructions prompt prompts field value".split())


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z_]{4,}", str(text).lower())
            if w not in _STOP}


def untraced(proposal: dict, complaints: str) -> str:
    """The note, or "" when there are few complaints or the policy prompt text of `proposal` shares enough words with
    them. The words of what the annotator refused do not count as unanswered."""
    lines = [l for l in complaints.splitlines() if l.strip()]
    heard = _words(complaints)
    if len(lines) < MIN_COMPLAINTS or not heard:
        return ""
    wrote = _words(" ".join([proposal.get(SUBGOAL) or ""]
                            + [f.get("description", "") for f in proposal.get(MEMORY) or []]))
    if not wrote:
        return ""
    heard -= _words(proposal.get("refused") or "")
    if not heard:
        return ""
    share = len(wrote & heard) / len(heard)
    if share >= TRACED:
        return ""
    return (f"the policy prompt edit answers {share:.0%} of what the policy reported "
            f"({len(heard - wrote)} of {len(heard)} reported words appear nowhere "
            f"in it): it may be an account of the episodes rather than of the "
            f"failures")
