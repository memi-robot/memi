"""The optimizer, step 7 of Figure 5. It commits one edit to phi and theta with every earlier iteration in view."""
from __future__ import annotations

from ...llm import responses as llm
from ...text import ANNOTATOR_PROMPT, MEMORY, fill, heading, prose, splice
from .. import prompts, variables


def apply(annotator_and_policy_prompts: dict, rev: dict) -> tuple[dict, list[str], dict]:
    """phi and theta with the optimizer's revision applied, what could not be applied, and what landed."""
    action_edits = [e for e in rev.get(variables.ACTION) or [] if isinstance(e, dict)]
    action, missed, landed = splice(annotator_and_policy_prompts.get("action", ""), action_edits)
    problems = [f"the policy prompt for action does not contain: {m}" for m in missed]
    applied = {ANNOTATOR_PROMPT: [], MEMORY: [], variables.ACTION: landed, "refused": prose(rev.get("refused"))}
    said = {ANNOTATOR_PROMPT: "the annotator prompt", MEMORY: "the policy prompt"}
    fields = [dict(f) for f in annotator_and_policy_prompts["fields"]]
    by = {f["name"]: f for f in fields}
    for side, key in ((ANNOTATOR_PROMPT, "annotation"), (MEMORY, "description")):
        for f in rev.get(side) or []:
            if not isinstance(f, dict) or not str(f.get("field", "")).strip():
                continue
            kind, name = str(f.get("kind", "")).lower(), str(f["field"]).strip()
            if kind == "delete":
                if name in by:
                    fields = [s for s in fields if s["name"] != name]
                    by.pop(name)
                    applied[side].append(f)
                continue
            if name in by:
                # A new comparison is recorded even when the edits of its field miss.
                if side == ANNOTATOR_PROMPT and variables.set_comparison(by[name], f):
                    applied[side].append({"kind": "compare", "field": name, "value_type": by[name].get("value_type"),
                                          "tolerance": by[name].get("tolerance")})
                if f.get("edits"):
                    new, miss, land = splice(by[name].get(key, ""), [e for e in f["edits"] if isinstance(e, dict)])
                    by[name][key] = new
                    if land:
                        applied[side].append({**f, "edits": land})
                    problems += [f"{name}: {said[side]} does not contain: {m}" for m in miss]
                elif f.get("description"):
                    by[name][key] = f["description"]
                    applied[side].append(f)
            elif kind in ("add", "modify"):
                desc = prose(f.get("description"))
                new = {"name": name, "description": desc, "annotation": desc}
                variables.set_comparison(new, f)
                fields.append(new)
                by[name] = new
                applied[side].append(f)
            else:
                problems.append(f"{name}: unknown edit kind {kind!r}")
    return {"fields": fields, "action": action}, problems, applied


def render_applied(applied: dict) -> str:
    """One revision as it landed, in full."""
    out = []
    for side, key in (("annotator prompt", ANNOTATOR_PROMPT), ("policy prompt", MEMORY)):
        for f in applied.get(key) or []:
            kind, name = f.get("kind"), f.get("field")
            how = f" (value_type: {variables.compared_as(f)})" if f.get("value_type") else ""
            if kind == "delete":
                out.append(f"  field {name} was deleted")
            elif f.get("edits"):
                out.append(f"  field {name}{how}, where the {side} read")
                for e in f["edits"]:
                    out += ["    " + prose(e.get("modify_from") or "(nothing: appended)"), "  now reads",
                            "    " + prose(e.get("modify_to") or "(nothing: deleted)")]
            elif kind == "compare":
                out.append(f"  field {name} is now compared as {variables.compared_as(f)}")
            else:
                out += [f"  field {name} was added, {side}{how}", "    " + prose(f.get("description"))]
    for e in applied.get(variables.ACTION) or []:
        out += ["  the policy prompt for action, where it read",
                "    " + prose(e.get("modify_from") or "(nothing: this was appended)"), "  now reads",
                "    " + prose(e.get("modify_to") or "(nothing: this was deleted)")]
    if str(applied.get("refused") or "").strip():
        out += ["  refused", "    " + applied["refused"]]
    return "\n".join(out) or "  (nothing was committed)"


def render_score(s: dict) -> str:
    """What one iteration measured, the two ways a cycle was wrong and the count by field."""
    lines = [f"  Among {s['cycles']} cycles ({s['act_cycles']} act cycles), the policy chose a different waypoint at "
             f"{s['waypoint_wrong']}, and wrote a different memory at {s['memory_wrong']}."]
    per = [(f, c) for f, c in sorted((s.get("fields") or {}).items(), key=lambda kv: -kv[1]) if c]
    if per:
        said = [f"{f} got wrong {c} time{'s' if c != 1 else ''}" for f, c in per]
        joined = said[0] if len(said) == 1 else ", ".join(said[:-1]) + ", and " + said[-1]
        lines.append(f"  Among {s.get('n_fields', len(per))} fields, {joined}.")
    return "\n".join(lines)


def history_block(history: list[dict], initial: dict, annotator_and_policy_prompts: dict, critique: str) -> str:
    """phi and theta as initialized, every iteration since, and phi and theta as they stand, printed last because the
    answer quotes them."""
    def both(x: dict) -> str:
        return variables.annotator_prompt_block(x) + "\n\n" + variables.policy_prompts_block(x).rstrip("\n")

    out = [heading("AS INITIALIZED") + "\n" + both(initial)]
    for i, it in enumerate(history, 1):
        out.append(heading(f"ITERATION {i}: WHAT IT SCORED") + "\n" + render_score(it["score"]))
        if it.get("critique"):
            out.append(heading(f"ITERATION {i}: THE REPORT") + "\n" + it["critique"])
        if it.get("applied") is not None:
            out.append(heading(f"ITERATION {i}: THE EDIT THAT LANDED") + "\n" + render_applied(it["applied"]))
        if it.get("not_applied"):
            out.append(heading(f"ITERATION {i}: WHAT DID NOT APPLY") + "\n"
                       + "\n".join("  " + m for m in it["not_applied"])
                       + "\n  Quote the policy prompt printed below, not an earlier answer.")
    out.append(heading("AS THEY STAND THIS ITERATION") + "\n" + both(annotator_and_policy_prompts))
    out.append(fill(prompts.optimizer.REPORT, critique=critique or "(nothing was reported)").rstrip("\n"))
    out.append(prompts.optimizer.OUTPUT_FORMAT.rstrip("\n"))
    return fill(prompts.optimizer.HISTORY, history="\n\n".join(out))


def step(task: str, annotator_and_policy_prompts: dict, history: list[dict], initial: dict, critique: str,
         client: llm.Client) -> dict:
    """Step 7. phi and theta revised, with the record of the call."""
    user = history_block(history, initial, annotator_and_policy_prompts, critique)
    r = client.ask(prompts.optimizer_system(), [llm.text(user)], purpose=f"{task}.optimizer.step")
    rev = r["json"] if isinstance(r["json"], dict) else {}
    new, problems, applied = apply(annotator_and_policy_prompts, rev)
    return {"texts": new, "revision": rev, "applied": applied, "not_applied": problems, "text": r["text"],
            "cost": r["cost"]}
