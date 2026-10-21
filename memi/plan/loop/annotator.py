"""The annotator alpha_phi of Eq. (2), steps 2 and 6 of the loop.

`forward` fills the memory in over a minibatch from the annotator prompt (phi). `backward` merges the policy's
critiques of the failed steps into one proposal for phi and theta.
"""
from __future__ import annotations

import collections

from ...llm import LLMClient
from ...llm.parse import parse_object
from ...text import ANNOTATOR_PROMPT, MEMORY, fill, prose
from .. import policy_turn, variables
from ..prompts import annotator as blocks
from ..prompts import system_prompts
from ..variables import SUBGOAL
from . import evidence, feedback
from .config import LoopConfig, ask
from .episodes import DecisionStep

#: The parts of a policy critique, as (its label in the failure block, its key).
REPORTED = (("the memory lacked", "missing_from_memory"),
            ("the policy prompt for memory lacked", "missing_from_policy_prompt_for_memory"),
            ("the policy prompt for subgoal lacked", "missing_from_policy_prompt_for_subgoal"))
#: The two of those that are about the policy prompt rather than the memory.
PROMPT_KEYS = tuple(k for _, k in REPORTED[1:])


def empty() -> dict:
    """No proposal. An iteration that has none is a quiet iteration, not a failed one."""
    return {SUBGOAL: "", MEMORY: [], ANNOTATOR_PROMPT: [], "refused": ""}


def _placed(item: dict, index: dict, sole: int | None) -> DecisionStep | None:
    """The step an entry's `{episode, step}` names, or None. An entry without an episode is placed only when the
    turn shows one episode."""
    try:
        n = int(item["step"])
    except (KeyError, TypeError, ValueError):
        return None
    try:
        ep = int(item.get("episode"))
    except (TypeError, ValueError):
        ep = sole
    return index.get((ep, n))


def read_annotations(answer: dict, steps: list[DecisionStep]) -> tuple[list[dict], list[str]]:
    """The answer's entries as annotations {episode, t, field, value}, and the entries that name no step."""
    index = {(s.episode, s.n): s for s in steps}
    eps = {s.episode for s in steps}
    sole = next(iter(eps)) if len(eps) == 1 else None
    entries = list(answer.get("memory") or [])
    problems = [f"entry is not an object: {str(e)[:80]}" for e in entries if not isinstance(e, dict)]
    out = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        s = _placed(e, index, sole)
        if s is None:
            problems.append(f"entry names no step: episode {e.get('episode', '(missing)')} #{e.get('step')}"
                            + (f" field {e.get('field')}" if e.get("field") else ""))
        elif e.get("field"):
            out.append({"episode": s.episode, "t": s.t, "field": str(e["field"]), "value": str(e["value"])})
    return out, problems


def field_order(annotations: list[dict], names: list[str]) -> list[str]:
    """The declared fields, then every other field the annotations name, in the order the episodes introduce it."""
    order, seen = list(names), set(names)
    for e in sorted(annotations, key=lambda e: (int(e["episode"]), int(e["t"]))):
        if e["field"] not in seen:
            order.append(e["field"])
            seen.add(e["field"])
    return order


def annotated_memory(steps: list[DecisionStep], annotations: list[dict], names: list[str]) -> tuple[dict, dict]:
    """Per step, the annotated memory carried into it (m~_{t-1}) and the one after it (m~_t).

    An annotation at a step is written from that step's segment, so it is read at the next step. The policy's own
    updates are merged with the same `policy_turn.carry`.
    """
    at = collections.defaultdict(list)
    for e in annotations:
        at[(int(e["episode"]), int(e["t"]))].append(e)
    order = field_order(annotations, names)
    by_ep = collections.defaultdict(list)
    for s in steps:
        by_ep[s.episode].append(s)
    carried, after = {}, {}
    for seq in by_ep.values():
        state: dict = {}
        for s in sorted(seq, key=lambda s: s.t):
            carried[s.key] = dict(state)
            state = policy_turn.carry(order, state, {e["field"]: e["value"] for e in at.get(s.key, [])})
            after[s.key] = dict(state)
    return carried, after


async def forward(cfg: LoopConfig, task: str, eps: list[int], annotator_and_policy_prompts: dict, client: LLMClient
                  ) -> tuple[dict, dict, list[str]]:
    """Step 2. One turn with the minibatch and phi, and no theta. Returns the annotated memory carried into and after
    each step (`annotated_memory`) and the problems of the answer.

    One turn for the minibatch and not one per episode, so phi names each fact the same way across the episodes.
    """
    parts, steps = evidence.episodes_turn(cfg, task, sorted(eps),
                                          opening=variables.annotator_prompt_block(annotator_and_policy_prompts),
                                          closing=blocks.OUTPUT_FORMAT)
    tag = ".".join(f"ep{e}" for e in sorted(eps))
    response = await ask(client, system_prompts(cfg.ablate).annotator, "\n".join(parts), f"{task}.annotate.{tag}")
    answer = parse_object(response.text, close=True)
    if answer is None:
        annotations, problems = [], [f"ep{e}: the annotator gave no answer" for e in eps]
    else:
        annotations, problems = read_annotations(answer, steps)
    carried, after = annotated_memory(steps, annotations, variables.field_names(annotator_and_policy_prompts))
    return carried, after, problems


def _failure_block(failed: list[dict], critiques: dict, number: dict, ablate: str | None) -> str:
    """One entry a step, with what the policy did there and what its critique says was missing. A text the critique
    said nothing about gets no line."""
    out = []
    for f in failed:
        key = (f["step"][0], f["step"][2])
        g = critiques.get(key) or {}
        body = []
        # An ablation withholds the half of the answer its principle grades.
        if ablate != "sufficiency":
            body.append(f"the subgoal a policy has written: {f.get('subgoal') or '(nothing)'}")
        if ablate != "predictability":
            body.append(f"the memory a policy has written:  {f.get('memory_written') or '(nothing)'}")
        said = [(label, " ".join(str(g.get(k, "")).split())) for label, k in REPORTED]
        body += [f"{label}: {text}" for label, text in said if text]
        if not any(text for _, text in said):
            body.append("it named nothing that was missing")
        out.append(f"ep{key[0]} #{number.get(key, '?')}\n" + "\n".join(f"  {b}" for b in body))
    return "\n\n".join(out)


def _backward_turn(annotator_and_policy_prompts: dict, failed: list[dict], critiques: dict, number: dict,
                   annotated: str, unused: dict[str, str], ablate: str | None) -> str:
    """The user turn of `backward`."""
    # phi first, then theta. That order is what BACKWARD_INPUT promises.
    body = [variables.annotator_prompt_block(annotator_and_policy_prompts),
            variables.policy_prompts_block(annotator_and_policy_prompts).rstrip("\n")]
    if annotated:
        body.append(annotated.rstrip("\n"))
    if failed:
        body.append(fill(blocks.BACKWARD_FAILURES,
                         failures=_failure_block(failed, critiques, number, ablate)).rstrip("\n"))
    if unused:
        body.append(fill(blocks.BACKWARD_UNUSED_FIELDS,
                         fields="\n".join(f"    {k}\n      {v}" for k, v in sorted(unused.items()))).rstrip("\n"))
    body.append(system_prompts(ablate).backward_output_format.rstrip("\n"))
    return "\n\n".join(body)


def _entries(answer: dict, key: str) -> list[dict]:
    """The answer's entries under `key` as {field, description}, empty ones left out."""
    out = []
    for item in answer.get(key) or []:
        if not isinstance(item, dict):
            continue
        need, name = " ".join(str(item.get("description", "")).split()), str(item.get("name", "")).strip()
        if need or name:
            out.append({"field": name, "description": need})
    return out


async def backward(cfg: LoopConfig, task: str, annotator_and_policy_prompts: dict, failed: list[dict], critiques: dict,
                   number: dict, client: LLMClient, annotated: str, unused: dict[str, str]) -> dict:
    """Step 6. The critiques of the failed steps and the unused fields merged into one proposal for phi and theta,
    the blog's d phi and d theta. One call an iteration, so two steps asking for the same change give one entry. A
    call that fails gives no proposal."""
    if not failed and not unused:
        return empty()
    user = _backward_turn(annotator_and_policy_prompts, failed, critiques, number, annotated, unused, cfg.ablate)
    try:
        response = await ask(client, system_prompts(cfg.ablate).annotator_backward, user, f"{task}.annotate.backward")
        answer = parse_object(response.text, close=True) or {}
    except Exception as exc:
        print(f"[annotator] {task}: backward failed ({exc})", flush=True)
        return empty()
    proposal = {SUBGOAL: " ".join(str(answer.get(SUBGOAL) or "").split()), MEMORY: _entries(answer, MEMORY),
                ANNOTATOR_PROMPT: _entries(answer, ANNOTATOR_PROMPT), "refused": prose(answer.get("refused"))}
    # What the critiques said about the policy prompt. A proposal that shares almost no words with it is noted.
    complaints = "\n".join(" ".join(str(g.get(k, "")).split())
                           for g in critiques.values() for k in PROMPT_KEYS if str(g.get(k, "")).strip())
    note = feedback.untraced(proposal, complaints)
    if note:
        print(f"[annotator] {task}: {note}", flush=True)
        proposal["untraced"] = note
    return proposal


def critique(proposal: dict) -> str:
    """The proposal as the optimizer reads it. The annotator prompt, then the policy prompt for memory, then the
    policy prompt for subgoal, then what was refused. Each field block names its half, since both carry the same
    field names."""
    out = [f"field {f['field']}, ANNOTATOR PROMPT\n  " + f["description"] for f in proposal.get(ANNOTATOR_PROMPT) or []]
    out += [f"field {f['field']}, POLICY PROMPT\n  " + f["description"] for f in proposal.get(MEMORY) or []]
    if (proposal.get(SUBGOAL) or "").strip():
        out.append("the policy prompt for subgoal\n  " + proposal[SUBGOAL])
    if (proposal.get("refused") or "").strip():
        out.append("refused\n  " + proposal["refused"])
    return "\n\n".join(out)
