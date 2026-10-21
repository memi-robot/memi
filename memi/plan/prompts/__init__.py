"""Every word the Plan roles and the deployed policy are told, one module per role, named as in memi/act/prompts.

`system_prompts(ablate)` joins the blocks into the offline roles' system prompts, with one principle removed when
`ablate` names it (`memi.plan.ablation`).
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

from ...text import assemble
from ..ablation import ABLATIONS, PRINCIPLES
from . import annotator, context, initializer, optimizer, policy

#: The policy is told the platform and its own role, and nothing else. No ablation touches its turns.
POLICY_SYSTEM = assemble(context.DEMONSTRATION, context.CONVENTIONS, policy.ROLE, policy.INPUT)
#: The template of the policy's user turn at one decision step (`policy_turn.build_user`).
POLICY_USER = assemble(policy.STEP, policy.OUTPUT_FORMAT)

#: The blocks an ablation rewrites, by the name the ablation table gives them.
_ABLATED = {"context.PRINCIPLES": context.PRINCIPLES, "context.LOOP_OVERVIEW": context.LOOP_OVERVIEW,
            "annotator.BACKWARD_INPUT": annotator.BACKWARD_INPUT,
            "annotator.BACKWARD_OUTPUT_FORMAT": annotator.BACKWARD_OUTPUT_FORMAT,
            "policy.BACKWARD_COMPARISON": policy.BACKWARD_COMPARISON}


@dataclass(frozen=True)
class Prompts:
    """The offline roles' system turns, and the two user-turn blocks an ablation rewrites."""
    initializer: str
    annotator: str
    annotator_backward: str
    policy_backward: str
    optimizer: str
    backward_output_format: str
    backward_comparison: str


@functools.cache
def system_prompts(ablate: str | None = None) -> Prompts:
    """The prompts with the principle `ablate` removed, or the full ones. A passage not found raises."""
    if ablate is not None and ablate not in PRINCIPLES:
        raise ValueError(f"unknown principle {ablate!r}")
    b = dict(_ABLATED)
    for name, old, new in ABLATIONS.get(ablate or "", []):
        if old not in b[name]:
            raise ValueError(f"ablation {ablate}: passage not found in {name}: {old[:60]!r}")
        b[name] = b[name].replace(old, new, 1)
    shared = (context.WHAT_THIS_IS_FOR, context.POLICY_CONTRACT, context.MEMORY_CONTRACT, b["context.PRINCIPLES"],
              b["context.LOOP_OVERVIEW"], context.DEMONSTRATION, context.CONVENTIONS)
    return Prompts(
        initializer=assemble(*shared, initializer.ROLE),
        annotator=assemble(*shared, annotator.ROLE, annotator.FITTING, annotator.INPUT),
        annotator_backward=assemble(*shared, annotator.BACKWARD_ROLE, b["annotator.BACKWARD_INPUT"]),
        policy_backward=assemble(*shared, policy.BACKWARD_ROLE, policy.BACKWARD_INPUT),
        optimizer=assemble(*shared, optimizer.ROLE, optimizer.INPUT),
        backward_output_format=b["annotator.BACKWARD_OUTPUT_FORMAT"],
        backward_comparison=b["policy.BACKWARD_COMPARISON"])


def _check() -> None:
    """Every passage of the ablation table is found, so the table cannot drift from the blocks."""
    for p in PRINCIPLES:
        system_prompts(p)


_check()
