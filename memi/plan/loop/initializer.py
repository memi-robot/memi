"""The initializer of blog Section 2.3. One call writes the initial phi and theta from a few training episodes."""
from __future__ import annotations

from ...llm import LLMClient
from ...llm.parse import parse_object
from ...text import ANNOTATOR_PROMPT, MEMORY, fill
from ..prompts import initializer as blocks
from ..prompts import system_prompts
from ..variables import FIELDS, SUBGOAL
from . import evidence
from .config import LoopConfig, ask


def from_answer(answer: dict | None) -> dict | None:
    """phi and theta from the initializer's answer, or None without one.

    The answer's two field lists, the annotator prompt and the policy prompt for memory, are joined on the field
    name. A name in only one list gets its one text on both sides and is reported under `name_mismatch`.
    """
    if not answer:
        return None

    def entries(items) -> dict[str, dict]:
        out = {}
        for f in items or []:
            if isinstance(f, dict) and str(f.get("name", "")).strip():
                out[str(f["name"]).strip()] = f
        return out

    by_policy_prompt, by_annotator_prompt = entries(answer.get(MEMORY)), entries(answer.get(ANNOTATOR_PROMPT))
    # The policy prompt's order first, since the memory line is written in it.
    names = list(by_policy_prompt) + [n for n in by_annotator_prompt if n not in by_policy_prompt]
    fields, missing = [], []
    for name in names:
        p_, r_ = by_policy_prompt.get(name, {}), by_annotator_prompt.get(name, {})
        if not p_ or not r_:
            missing.append(f"{name}: named only by the "
                           f"{'annotator prompt' if r_ else 'policy prompt'}")
        text = str(p_.get("description", "") or r_.get("description", ""))
        annot = str(r_.get("description", "") or text)
        fields.append({"name": name, "description": text, "annotation": annot})

    out = {k: v for k, v in answer.items()
           if k not in (FIELDS, MEMORY, ANNOTATOR_PROMPT, SUBGOAL) and not isinstance(v, dict)}
    out[FIELDS] = fields
    if missing:
        out["name_mismatch"] = missing
    out[SUBGOAL] = " ".join(str(answer.get(SUBGOAL) or "").split())
    return out


async def initialize(cfg: LoopConfig, task: str, eps: list[int], client: LLMClient) -> tuple[dict | None, dict]:
    """phi and theta, or None when the reply does not parse, and the record of the call. The record keeps the reply,
    because a reply that did not parse is the one most worth reading."""
    system = system_prompts().initializer
    closing = fill(blocks.OUTPUT_FORMAT, scope=blocks.BOTH_EPISODES if len(eps) > 1 else "")
    parts, steps = evidence.episodes_turn(cfg, task, eps, closing=closing)
    response = await ask(client, system, "\n".join(parts), f"{task}.initialize")
    meta = {"episodes": list(eps), "shown_steps": len(steps), "system_chars": len(system),
            "user_chars": sum(len(p) for p in parts), "model": client.model, "usage": dict(response.usage or {}),
            "reasoning": getattr(response, "reasoning", ""), "text": response.text}
    return from_answer(parse_object(response.text, close=True)), meta
