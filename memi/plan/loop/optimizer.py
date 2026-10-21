"""The optimizer, step 7 of the loop. One edit to phi and theta with every earlier iteration in view.

Its answer quotes passages with what each becomes, and adds or deletes fields. `apply` splices them in.
"""
from __future__ import annotations

from ...llm import LLMClient
from ...llm.parse import parse_object
from ...text import ANNOTATOR_PROMPT, MEMORY, fill, heading, prose, splice
from ..prompts import optimizer as blocks
from ..prompts import system_prompts
from ..variables import FIELDS, SUBGOAL, annotator_prompt_block, policy_prompts_block
from .config import LoopConfig, ask


def render_applied(applied: dict) -> str:
    """One revision as it landed, in full. The history shows no phi and theta between the initial ones and the
    current ones, so a revision named only by its field would leave its wording unknown."""
    out = []
    for side, key in (("annotator prompt", ANNOTATOR_PROMPT), ("policy prompt", MEMORY)):
        for f in applied.get(key) or []:
            kind, name = f.get("kind"), f.get("field")
            if kind == "delete":
                out.append(f"  field {name} was deleted")
            elif f.get("edits"):
                out.append(f"  field {name}, where the {side} read")
                for e in f["edits"]:
                    out += ["    " + (e.get("modify_from") or "(nothing: appended)"),
                            "  now reads",
                            "    " + (e.get("modify_to") or "(nothing: deleted)")]
            else:
                out += [f"  field {name} was added, {side}",
                        "    " + f.get("description", "")]
    for e in applied.get(SUBGOAL) or []:
        out += ["  the policy prompt for subgoal, where it read",
                "    " + (e.get("modify_from") or "(nothing: this was appended)"),
                "  now reads",
                "    " + (e.get("modify_to") or "(nothing: this was deleted)")]
    if (applied.get("refused") or "").strip():
        out += ["  refused", "    " + applied["refused"]]
    return "\n".join(out) or "  (nothing was committed)"


def render_score(s: dict, ablate: str | None) -> str:
    """What one iteration measured. The two ways a step was wrong, and by field."""
    n = s.get("scored") or 0
    # An ablation does not report the count its principle produces.
    if ablate == "sufficiency":
        lines = [f"  Among {n} steps, the policy wrote a different memory at "
                 f"{s.get('memory_wrong', 0)}."]
    elif ablate == "predictability":
        return (f"  Among {n} steps, the policy chose a different subgoal at "
                f"{s.get('subgoal_wrong', 0)}.")
    else:
        lines = [f"  Among {n} steps, the policy chose a different subgoal at "
                 f"{s.get('subgoal_wrong', 0)}, and wrote a different memory at "
                 f"{s.get('memory_wrong', 0)}."]
    fields = s.get("fields") or {}
    per = [(f, c) for f, c in sorted(fields.items(), key=lambda kv: -kv[1]) if c]
    if per:
        said = [f"{f} got wrong {c} time{'s' if c != 1 else ''}" for f, c in per]
        joined = said[0] if len(said) == 1 else ", ".join(said[:-1]) + ", and " + said[-1]
        lines.append(f"  Among {len(fields)} fields, {joined}.")
    return "\n".join(lines)


def history_block(history: list[dict], initial: dict, annotator_and_policy_prompts: dict, critique: str,
                  ablate: str | None) -> str:
    """phi and theta as initialized, every iteration since, and phi and theta as they stand, printed last because the
    answer quotes them. The last history entry is this iteration and carries only its score."""
    def both(x: dict) -> str:
        return annotator_prompt_block(x) + "\n\n" + policy_prompts_block(x).rstrip("\n")

    out = [heading("AS INITIALIZED") + "\n" + both(initial)]
    for i, h in enumerate(history or [], 1):
        if h.get("scored"):
            out.append(heading(f"ITERATION {i}: WHAT IT SCORED") + "\n" + render_score(h["scored"], ablate))
        if (h.get("critique") or "").strip():
            out.append(heading(f"ITERATION {i}: THE REPORT") + "\n" + h["critique"].strip())
        if h.get("applied") is not None:
            out.append(heading(f"ITERATION {i}: THE EDIT THAT LANDED") + "\n" + render_applied(h["applied"]))
        # Without what did not land, a missed quote reads as an applied one and is quoted again.
        if h.get("not_applied"):
            out.append(heading(f"ITERATION {i}: WHAT DID NOT APPLY") + "\n"
                       + "\n".join("  " + m for m in h["not_applied"])
                       + "\n  Quote the policy prompt printed below, not an earlier answer.")
    out.append(heading("AS THEY STAND THIS ITERATION") + "\n" + both(annotator_and_policy_prompts))
    out.append(fill(blocks.REPORT, critique=critique).rstrip("\n"))
    out.append(blocks.OUTPUT_FORMAT.rstrip("\n"))
    return fill(blocks.HISTORY, history="\n\n".join(out))


def parse(answer: dict | None) -> dict:
    """The revision. Passage edits of the policy prompt for subgoal, and field changes on each side."""
    answer = answer or {}
    subgoal = []
    for e in answer.get(SUBGOAL) or []:
        if not isinstance(e, dict):
            continue
        subgoal.append({"modify_from": str(e.get("modify_from", "")),
                        "modify_to": str(e.get("modify_to", ""))})

    def fields(key: str) -> list[dict]:
        out = []
        for f in answer.get(key) or []:
            if not isinstance(f, dict) or not str(f.get("field", "")).strip():
                continue
            out.append({"kind": str(f.get("kind", "")).strip().lower(),
                        "field": str(f["field"]).strip(),
                        "edits": [{"modify_from": str(e.get("modify_from", "")),
                                   "modify_to": str(e.get("modify_to", ""))}
                                  for e in (f.get("edits") or [])
                                  if isinstance(e, dict)],
                        "description": str(f.get("description", ""))})
        return out

    return {SUBGOAL: subgoal,
            MEMORY: fields(MEMORY),
            ANNOTATOR_PROMPT: fields(ANNOTATOR_PROMPT),
            "refused": prose(answer.get("refused"))}


def _text(field: dict, side: str) -> str:
    """One field as the optimizer saw it, on the side being revised."""
    key = "annotation" if side == ANNOTATOR_PROMPT else "description"
    return str(field.get(key) or field.get("description", ""))


def _put(field: dict, side: str, text: str) -> None:
    field["annotation" if side == ANNOTATOR_PROMPT else "description"] = text


def apply(annotator_and_policy_prompts: dict, revision: dict) -> tuple[dict, list[str], dict]:
    """phi and theta with the revision in them, what could not be applied (`not_applied`), and what was (`applied`).

    Every text is revised by quoted passage, and a field being added arrives in full. A field carries its annotation
    and its description, so adding and deleting reach both sides. Only `modify` is one-sided.
    """
    before = str(annotator_and_policy_prompts.get(SUBGOAL) or "")
    text, missed, landed = splice(before, revision.get(SUBGOAL) or [])
    problems = [f"the policy prompt for subgoal does not contain: {m}" for m in missed]
    applied: dict = {ANNOTATOR_PROMPT: [], MEMORY: [], SUBGOAL: landed}

    fields = [dict(s) for s in annotator_and_policy_prompts.get(FIELDS) or []]
    by_name = {s.get("name"): s for s in fields}
    for side in (ANNOTATOR_PROMPT, MEMORY):
        where = "policy prompt" if side == MEMORY else "annotator prompt"
        for f in revision.get(side) or []:
            kind, name = f["kind"], f["field"]
            if kind == "delete":
                # On both sides. A field the annotator no longer fills is one the policy cannot be checked on.
                if name in by_name:
                    fields = [s for s in fields if s.get("name") != name]
                    by_name.pop(name, None)
                    applied[side].append(dict(f))
                continue
            if name in by_name:
                field = by_name[name]
                if f["edits"]:
                    shown, missed_f, landed_f = splice(_text(field, side), f["edits"])
                    if landed_f:
                        applied[side].append({**f, "edits": landed_f})
                    _put(field, side, shown)
                    problems += [f"{name}: the {where} does not contain: {m}"
                                 for m in missed_f]
                elif f["description"]:
                    _put(field, side, f["description"])
                    applied[side].append(dict(f))
            elif kind in ("add", "modify"):
                # Added on both sides at once. The half that was not asked for gets the same sentence.
                new = {"name": name, "description": f["description"],
                       "annotation": f["description"]}
                fields.append(new)
                by_name[name] = new
                applied[side].append(dict(f))
            else:
                problems.append(f"unknown edit kind {kind!r} on field {name}")
    return {**annotator_and_policy_prompts, SUBGOAL: text, FIELDS: fields}, problems, applied


async def step(cfg: LoopConfig, task: str, eps: list[int], annotator_and_policy_prompts: dict, history: list[dict],
               initial: dict, critique: str, client: LLMClient) -> tuple[dict, dict]:
    """Step 7. phi and theta revised, and the record of the call. The turn shows `initial`, the prompts the run began
    with, and no episodes. `eps` goes into the record only."""
    user = history_block(history, initial, annotator_and_policy_prompts, critique, cfg.ablate)
    response = await ask(client, system_prompts(cfg.ablate).optimizer, user, f"{task}.optimize")
    revision = parse(parse_object(response.text, close=True))
    revised, problems, applied = apply(annotator_and_policy_prompts, revision)
    applied["refused"] = revision.get("refused") or ""
    meta = {"episodes": list(eps), "user_chars": len(user), "usage": dict(response.usage or {}),
            "reasoning": getattr(response, "reasoning", ""), "revision": revision, "applied": applied,
            "not_applied": problems, "refused": revision.get("refused") or "", "text": response.text}
    return revised, meta
