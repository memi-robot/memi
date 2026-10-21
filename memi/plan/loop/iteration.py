"""One iteration of the loop (blog Section 2.3), the seven steps of Figure 5 in order.

    1 sample episodes      a fresh minibatch B^(k) of training episodes
    2 annotator.forward    the annotated memory m~_t of every step, from phi
    3 policy.forward       the memory m_t, subgoal a^_t and used fields U_t of every step, from theta and m~_{t-1}
    4 loss                 the terms L_suf, L_pred and L_nec of sufficiency, predictability and necessity
    5 policy.backward      the critiques d m~ and d theta of each failed step
    6 annotator.backward   the critiques merged into d phi and d theta, one proposal for both prompts
    7 optimizer.step       phi and theta revised, with the optimization history in view

The policy's forward pass runs on the policy model. Every other role runs on the offline model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ...llm import LLMClient
from ...text import memory_line
from .. import variables
from . import annotator, episodes, evidence, loss, optimizer, policy
from .config import LoopConfig


@dataclass
class Iteration:
    """What one iteration saw and wrote."""
    eps: list[int]
    rows: list[dict]                  # one graded answer per decision step
    carried: dict                     # the annotated memory carried into each step
    scored: dict                      # the loss record
    critiques: dict                   # step key -> the policy's critique
    proposal: dict                    # annotator.backward's merged critiques
    critique: str                     # the proposal as the optimizer reads it
    revised: dict | None = None       # phi and theta after the optimizer, None in a quiet iteration
    meta: dict = field(default_factory=dict)                  # the optimizer's call record
    scored_for_optimizer: dict = field(default_factory=dict)  # the counts the optimizer's history keeps


def models(offline: LLMClient, policy_client: LLMClient) -> dict:
    """The model and effort each role ran at."""
    def of(b):
        return {"provider": b.provider, "model": b.model, "reasoning_effort": b.effort}
    return {"annotator": of(offline), "policy_forward": of(policy_client), "policy_backward": of(offline),
            "optimizer": of(offline)}


def _withheld(cfg: LoopConfig, rows: list[dict], names: list[str], number: dict) -> tuple[list[dict], dict]:
    """The failed steps and the unused fields sent backward. An ablation withholds its principle's signal.

    Without sufficiency a wrong subgoal alone is no failure. Without predictability a wrong memory alone is none, and
    an unused field does not say where it was the only wrong field. Without necessity no field is reported unused.
    """
    keep = {"sufficiency": "memory_wrong", "predictability": "subgoal_wrong"}.get(cfg.ablate or "")
    failed = [r for r in loss.failed(rows) if not keep or r.get(keep)]
    unused = ({} if cfg.ablate == "necessity"
              else loss.unused_fields(rows, names, number, sole=cfg.ablate != "predictability"))
    return failed, unused


def annotated_turn(cfg: LoopConfig, task: str, eps: list[int], carried: dict, names: list[str]) -> str:
    """The episodes with the annotated memory carried in, shown only at the steps where it changes."""
    last: dict[int, str] = {}
    changes: dict[tuple, str] = {}
    for key, values in sorted(carried.items()):
        line = memory_line(names, values)
        if line != last.get(key[0], ""):
            changes[key] = last[key[0]] = line
    return "\n".join(evidence.episodes_turn(cfg, task, sorted(eps), notes=changes,
                                            note_head="MEMORY CARRIED IN")[0])


async def iteration(cfg: LoopConfig, task: str, sample: Callable[[], list[int]], annotator_and_policy_prompts: dict,
                    history: list[dict], initial: dict, offline: LLMClient, policy_client: LLMClient,
                    has_demo: bool, record: Callable[[Iteration], None]) -> Iteration:
    """One iteration from phi and theta. `record` writes what the iteration saw before the optimizer is called. With
    an empty critique the optimizer is not called, and the iteration is quiet."""
    names = variables.field_names(annotator_and_policy_prompts)
    policy_prompt = variables.policy_prompt(annotator_and_policy_prompts)
    # Forward
    eps = sample()
    steps = [s for e in eps for s in episodes.steps(cfg, task, e)]
    number = {s.key: s.n for s in steps}
    carried, after, problems = await annotator.forward(cfg, task, eps, annotator_and_policy_prompts, offline)
    answers = await policy.forward(steps, carried, policy_client, policy_prompt, has_demo, cfg.concurrency)
    scored = loss.loss(answers, steps, carried, after, names, number, eps, problems)
    # Backward
    failed, unused = _withheld(cfg, answers, names, number)
    critiques = await policy.backward(cfg, failed, steps, after, offline, policy_prompt, has_demo)
    proposal = await annotator.backward(cfg, task, annotator_and_policy_prompts, failed, critiques, number,
                                        offline, annotated_turn(cfg, task, eps, carried, names), unused)
    it = Iteration(eps, answers, carried, scored, critiques, proposal, annotator.critique(proposal))
    record(it)
    # Revise
    if it.critique.strip():
        it.scored_for_optimizer = loss.for_optimizer(scored, answers, critiques,
                                                     tuple(k for _, k in annotator.REPORTED))
        it.revised, it.meta = await optimizer.step(cfg, task, eps, annotator_and_policy_prompts,
                                                   history + [{"scored": it.scored_for_optimizer}], initial,
                                                   it.critique, offline)
    return it
