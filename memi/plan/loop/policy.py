"""The policy pi_theta of Eq. (1) at decision steps, steps 3 and 5 of the loop.

`forward` answers every step under teacher forcing, with the annotated memory carried in. `backward` puts a step it
got wrong to it again, with its answer beside the recorded one, and its critique says what was missing.
"""
from __future__ import annotations

import asyncio

from ...llm import CallFailed, LLMClient
from ...llm.parse import parse_object
from ...text import fill, memory_line, said
from .. import policy_turn
from ..prompts import POLICY_SYSTEM, system_prompts
from ..variables import field_names
from .config import POLICY_CALL_TIMEOUT, LoopConfig, ask
from .episodes import DecisionStep, key_of


def user_turn(step: DecisionStep, memory: str, has_demo: bool, policy_prompt: dict) -> str:
    """The policy's user turn at `step`, built from `policy_prompt` as the rollouts build it from an exported entry,
    so the loop scores exactly what the rollouts run."""
    return policy_turn.build_user(step.goal, policy_turn.entry(policy_prompt), step.menu, step.segment, memory,
                                  "first" if step.demo else "decide", saw_demo=has_demo)


async def _call(client: LLMClient, system: str, user: str, purpose: str, what: str) -> str:
    """One policy call. A call that fails raises CallFailed, which stops the run."""
    try:
        response = await ask(client, system, user, purpose, POLICY_CALL_TIMEOUT)
    except Exception as exc:
        raise CallFailed(f"{purpose}: {what} failed: {type(exc).__name__}: {exc}") from exc
    return response.text or ""


async def forward(steps: list[DecisionStep], carried: dict, client: LLMClient, policy_prompt: dict, has_demo: bool,
                  concurrency: int) -> list[dict]:
    """Step 3. The answer at every step under teacher forcing, in the order of `steps`, as {parsed JSON, raw,
    memory_in, used_fields}. `carried` is the annotated memory carried into each step. One fresh call per step."""
    entry = policy_turn.entry(policy_prompt)
    lines = {k: policy_turn.memory_text(entry, v) for k, v in carried.items()}
    gate = asyncio.Semaphore(concurrency)

    async def one(s: DecisionStep) -> dict:
        async with gate:
            text = await _call(client, POLICY_SYSTEM, user_turn(s, lines.get(s.key, ""), has_demo, policy_prompt),
                               f"{s.task}.policy.ep{s.episode}.t{s.t}", "policy call")
        parsed = parse_object(text) if text.strip() else None
        answer = dict(parsed) if isinstance(parsed, dict) else {}
        answer["raw"] = text
        answer["memory_in"] = lines.get(s.key, "")
        answer["used_fields"] = answer.get("used_fields") or []
        return answer

    return list(await asyncio.gather(*[one(s) for s in steps]))


async def _critique(cfg: LoopConfig, step: DecisionStep, row: dict, client: LLMClient, policy_prompt: dict,
                    has_demo: bool, expected: str) -> dict:
    """The critique of one failed step. `expected` is the annotated memory after it, as a line."""
    sub, mem = row.get("subgoal_wrong"), row.get("memory_wrong")
    # An ablation hides one half of the comparison, and the turn does not say that half disagreed.
    if cfg.ablate == "sufficiency":
        sub = False
    elif cfg.ablate == "predictability":
        mem = False
    which = ("Both your subgoal and your memory" if sub and mem else "Your subgoal" if sub
             else "Your memory" if mem else "Your answer")
    block = system_prompts(cfg.ablate).backward_comparison
    values = dict(
        what_disagreed=which,
        answered_subgoal=row.get("subgoal", "") or "(nothing)",
        answered_memory=policy_turn.memory_text(
            policy_turn.entry(policy_prompt), {k: str(v) for k, v in (row.get("memory_written") or {}).items()})
        or "(nothing)",
        recorded_subgoal=step.subgoal or "(none)",
        recorded_memory=expected or "(none)")
    # An ablated block has lost the lines of its half, and their placeholders with them.
    if cfg.ablate:
        values = {k: v for k, v in values.items() if "{" + k + "}" in block}
    user = user_turn(step, row.get("memory_in", ""), has_demo, policy_prompt) + "\n\n" + fill(block, **values)
    text = await _call(client, system_prompts(cfg.ablate).policy_backward, user,
                       f"{step.task}.backward.ep{step.episode}.t{step.t}", "policy backward call")
    answer = parse_object(text) or {}
    missing = said(answer, "missing_from_memory")
    return {"step": step.step, "memory_in": row.get("memory_in", ""), "answered": row.get("subgoal", ""),
            "recorded": step.subgoal, "missing_from_memory": missing,
            "missing_from_policy_prompt_for_memory": said(answer, "missing_from_policy_prompt_for_memory"),
            "missing_from_policy_prompt_for_subgoal": said(answer, "missing_from_policy_prompt_for_subgoal"),
            "missing": missing}


async def backward(cfg: LoopConfig, failed: list[dict], steps: list[DecisionStep], after: dict,
                   client: LLMClient, policy_prompt: dict, has_demo: bool) -> dict:
    """Step 5. {step key: critique} for every failed row. A critique says what the memory lacked (d m~) and what each
    policy prompt lacked (d theta)."""
    gate = asyncio.Semaphore(cfg.concurrency)
    by_key = {s.key: s for s in steps}
    names = field_names(policy_prompt)

    async def one(row: dict) -> dict:
        key = key_of(row["step"])
        async with gate:
            return await _critique(cfg, by_key[key], row, client, policy_prompt, has_demo,
                                   memory_line(names, after.get(key) or {}))

    got = await asyncio.gather(*[one(r) for r in failed]) if failed else []
    return {key_of(g["step"]): g for g in got}
