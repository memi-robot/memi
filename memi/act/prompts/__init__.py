"""The system prompts of the Act roles and the deployed policy, one module per role, block for block as in
memi/plan/prompts. They differ from Plan's where the Act policy does, with waypoints, waits, frames, the robot state
and keyframes."""
from __future__ import annotations

from ...text import assemble
from . import annotator, context, initializer, optimizer, policy

#: What every offline role reads first.
SHARED = (context.WHAT_THIS_IS_FOR, context.CYCLES, context.POLICY_CONTRACT, context.MEMORY_CONTRACT,
          context.PRINCIPLES, context.LOOP_OVERVIEW, context.DEMONSTRATION, context.CONVENTIONS)


def initializer_system(task: str) -> str:
    return assemble(*SHARED, context.robot_notes(task), initializer.ROLE)


def annotator_system(task: str) -> str:
    return assemble(*SHARED, context.robot_notes(task), annotator.ROLE, annotator.FITTING, annotator.INPUT)


def annotator_backward_system(task: str) -> str:
    return assemble(*SHARED, context.robot_notes(task), annotator.BACKWARD_ROLE, annotator.BACKWARD_INPUT)


def optimizer_system() -> str:
    return assemble(*SHARED, optimizer.ROLE, optimizer.INPUT)


def policy_system(task: str) -> str:
    """The policy is told the cycles, the platform and its own role. The other shared blocks are about the policy
    prompts, and the policy is following them."""
    return assemble(context.CYCLES, context.DEMONSTRATION, context.CONVENTIONS, context.robot_notes(task),
                    policy.ROLE, policy.INPUT)


def policy_backward_system(task: str) -> str:
    return assemble(*SHARED, context.robot_notes(task), policy.BACKWARD_ROLE, policy.BACKWARD_INPUT)
