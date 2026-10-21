"""The annotator prompt phi of Eq. (2) and the policy prompt theta of Eq. (1) in one object, and how turns print them.

    {"fields": [{"name", "annotation", "description"}, ...], "policy_prompt_for_subgoal": "..."}

A field's `annotation` is its part of phi, and its `description` is its part of the policy prompt for memory. The
policy's value is graded against the annotator's, so a field whose two descriptions disagree fails predictability.
`policy_prompt_for_subgoal` is the rest of theta, and only theta ships to the robot.
"""
from __future__ import annotations

from ..text import fill, heading
from .prompts import context

#: The key of the field list in every file that holds phi or theta.
FIELDS = "fields"
#: The key of the policy prompt for subgoal. memi.text holds the keys of the two field lists.
SUBGOAL = "policy_prompt_for_subgoal"


def fields(variables: dict | None) -> list[dict]:
    return list((variables or {}).get(FIELDS) or [])


def field_names(variables: dict | None) -> list[str]:
    return [str(s.get("name", "")) for s in fields(variables)]


def annotator_prompt(variables: dict | None) -> dict:
    """phi. Each field as the annotator is told to fill it. A field with no `annotation` falls back to its
    `description`."""
    return {FIELDS: [{"name": s.get("name"),
                      "description": str(s.get("annotation") or s.get("description") or "").strip()}
                     for s in fields(variables)]}


def policy_prompt(variables: dict | None) -> dict:
    """theta. The policy prompt for subgoal, and each memory field as the policy is told to keep it."""
    return {SUBGOAL: (variables or {}).get(SUBGOAL) or "",
            FIELDS: [{"name": s.get("name", ""), "description": str(s.get("description") or "").strip()}
                     for s in fields(variables)]}


def stored_annotator_prompt(variables: dict, **extra) -> dict:
    """phi as its file holds it, after the `extra` keys."""
    return {**extra, **annotator_prompt(variables)}


def merge(policy_prompt_file: dict, annotator_prompt_file: dict | None) -> dict:
    """A policy prompt file and an annotator prompt file as one object, the annotator prompt put back on the
    fields."""
    by_name = {s.get("name"): s.get("description", "") for s in (annotator_prompt_file or {}).get(FIELDS) or []}
    return {**policy_prompt_file, FIELDS: [{**s, "annotation": by_name.get(s.get("name")) or s.get("annotation") or ""}
                                           for s in fields(policy_prompt_file)]}


def describe(variables: dict) -> str:
    """One log line on the fields."""
    names = field_names(variables)
    split = sum(1 for s in fields(variables) if (s.get("annotation") or "") not in ("", s.get("description")))
    return f"{len(names)} fields ({split} written differently for the annotator): {', '.join(names)}"


def fields_block(fields_: list[dict], head: str | None = None, indent: int = 4) -> str:
    """Each field's name, and its description under it, after `head` when given."""
    pad = " " * indent
    out = [heading(head)] if head else []
    for s in fields_ or []:
        out += [f"{pad}{s.get('name')}", f"{pad}    {' '.join(str(s.get('description', '')).split())}"]
    return "\n".join(out)


def annotator_prompt_block(variables: dict) -> str:
    """phi under its heading, as the annotator and the optimizer read it."""
    return fields_block(annotator_prompt(variables)[FIELDS], context.ANNOTATOR_PROMPT_HEAD)


def policy_prompts_block(variables: dict) -> str:
    """theta under its two headings, as the offline roles read it. The policy prompt for memory comes first."""
    prompt = policy_prompt(variables)
    return fill(context.POLICY_PROMPTS,
                policy_prompt_for_subgoal="  " + (str(prompt[SUBGOAL]).strip() or "(none on file)"),
                policy_prompt_for_memory=fields_block(prompt[FIELDS], context.POLICY_PROMPT_HEAD))
