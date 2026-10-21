"""The optimization variables of Figure 4, the annotator prompt (phi) and the policy prompt (theta), held in one object,
and how a turn prints them.

    {"fields": [{"name", "annotation", "description", "value_type", "tolerance"}, ...], "action": "..."}

A field's `annotation` is its annotator prompt, what the annotator fills in. Its `description` is its policy prompt
for memory, how the policy keeps it. `value_type` and `tolerance` say how a written value is compared with the
annotated one. `action` is the policy prompt for action.
"""
from __future__ import annotations

from ..text import ANNOTATOR_PROMPT, MEMORY, fill, heading
from . import memory
from .prompts import context

#: The key under which the roles answer the policy prompt for action. memi.text holds the other two.
ACTION = "policy_prompt_for_action"


def field_names(annotator_and_policy_prompts: dict) -> list[str]:
    return [f["name"] for f in annotator_and_policy_prompts["fields"]]


def set_comparison(field: dict, entry: dict) -> bool:
    """Take `value_type` and `tolerance` from an answer's entry. True if either changed."""
    changed = False
    vt = str(entry.get("value_type") or "").strip().lower()
    if vt in memory.VALUE_TYPES and vt != field.get("value_type"):
        field["value_type"], changed = vt, True
    try:
        tol = float(entry["tolerance"])
        if tol >= 0 and tol != field.get("tolerance"):
            field["tolerance"], changed = tol, True
    except (KeyError, TypeError, ValueError):
        pass
    return changed


def compared_as(entry: dict) -> str:
    vt = entry.get("value_type")
    if vt == "other":
        return f"other, tolerance {entry.get('tolerance') or 0}"
    if vt in memory.VALUE_TYPES:
        return vt
    return "text (no value_type given)"


def from_answer(answer: dict) -> dict:
    """phi and theta from the initializer's answer, joined on the field name. The policy prompt's order comes first,
    then the names only the annotator prompt gives."""
    annotator = {s["name"]: s for s in answer.get(ANNOTATOR_PROMPT) or [] if isinstance(s, dict) and s.get("name")}
    policy = {s["name"]: s for s in answer.get(MEMORY) or [] if isinstance(s, dict) and s.get("name")}
    fields = []
    for name in list(policy) + [n for n in annotator if n not in policy]:
        r, p = annotator.get(name, {}), policy.get(name, {})
        field = {"name": name, "description": p.get("description") or r.get("description", ""),
                 "annotation": r.get("description") or p.get("description", "")}
        set_comparison(field, r)
        fields.append(field)
    return {"fields": fields, "action": str(answer.get(ACTION) or "")}


def action_text(annotator_and_policy_prompts: dict) -> str:
    return str(annotator_and_policy_prompts.get("action") or "").strip() or "(none on file)"


def annotator_prompt_block(annotator_and_policy_prompts: dict) -> str:
    """phi, one field to a heading, with how each field is compared."""
    out = [heading(context.ANNOTATOR_PROMPT_HEAD)]
    for f in annotator_and_policy_prompts["fields"]:
        out += [f"    {f['name']}  (value_type: {compared_as(f)})",
                f"        {' '.join(str(f.get('annotation') or f.get('description', '')).split())}"]
    return "\n".join(out)


def memory_prompt(annotator_and_policy_prompts: dict) -> str:
    out = []
    for f in annotator_and_policy_prompts["fields"]:
        out += [f"  {f['name']}", f"      {' '.join(str(f.get('description', '')).split())}"]
    return "\n".join(out) or "  (none on file)"


def policy_prompts_block(annotator_and_policy_prompts: dict) -> str:
    """theta, in the order a cycle is answered in. Keep the memory, then choose."""
    return fill(context.POLICY_PROMPTS,
                policy_prompt_for_memory=(heading(context.POLICY_PROMPT_HEAD) + "\n"
                                          + memory_prompt(annotator_and_policy_prompts)),
                policy_prompt_for_action="  " + action_text(annotator_and_policy_prompts))
