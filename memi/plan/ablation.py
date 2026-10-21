"""The principle ablations of blog Section 3.4, the passages `--ablate <principle>` rewrites in the offline prompts.

The loop also withholds the removed principle's feedback from the offline roles, wherever `loop/` reads `cfg.ablate`.
The policy's forward turn is unchanged and all three terms are still logged, so the conditions differ only in what
the loop learns from.
"""
from __future__ import annotations

PRINCIPLES = ("sufficiency", "predictability", "necessity")


_DEF_THREE = "Three conditions decide what a good annotator prompt and a good policy prompt are."
_DEF_TWO = "Two conditions decide what a good annotator prompt and a good policy prompt are."

_DEF_ATOM = "fails, so none of the three conditions can be checked on it."
_DEF_ATOM_TWO = "fails, so neither condition can be checked on it."

_DEF_SUF = """
  SUFFICIENCY    the memory and what the current step shows together predict
                 the next subgoal. That is the menu option, and the object its
                 point selects.
"""

_DEF_PRE = """
  PREDICTABILITY the memory and what the current step shows together predict what the
                 memory itself should hold at that step.
"""

_DEF_NEC = """
  NECESSITY      every field holds a fact that no later step recovers when the
                 information is needed.
"""

_OVERVIEW_CHECK = """\
      |                   answers that step's subgoal and that step's memory.
      |                   Where its answer differs from the recorded one, the
      |                   policy prompts did not settle that step."""


def _overview(half: str) -> str:
    return f"""\
      |                   answers that step's subgoal and that step's memory.
      |                   Where its {half} differs from the recorded one, the
      |                   policy prompts did not settle that step."""


_BACK_WHICH = "and which of the three conditions it breaks."
_BACK_WHICH_TWO = "and which of the two conditions it breaks."

_BACK_SUF = """
    SUFFICIENCY, when the current subgoal could not be chosen from the past
    memory and what the current step shows."""

_BACK_PRE = """ PREDICTABILITY, when the current
    memory could not be chosen from the past memory and what the current step
    shows. That covers the
    case where the value you wrote and the value the policy prompt asks the policy for
    are two different answers to one field at one step."""

_BACK_NEC = """ NECESSITY, when the
    step or the goal instruction already gave the information."""


#: Per principle, the passages it rewrites as (block, passage in the full prompt, what stands there instead).
#: `prompts.system_prompts` applies them and raises if a passage is not found.
ABLATIONS: dict[str, list[tuple[str, str, str]]] = {
    "sufficiency": [
        ("context.PRINCIPLES", _DEF_THREE, _DEF_TWO),
        ("context.PRINCIPLES", _DEF_SUF, ""),
        ("context.PRINCIPLES", _DEF_ATOM, _DEF_ATOM_TWO),
        ("context.LOOP_OVERVIEW", _OVERVIEW_CHECK, _overview("memory")),
        ("annotator.BACKWARD_OUTPUT_FORMAT", _BACK_WHICH, _BACK_WHICH_TWO),
        ("annotator.BACKWARD_OUTPUT_FORMAT", _BACK_SUF, ""),
        ("annotator.BACKWARD_OUTPUT_FORMAT", """how the observation changed, what subgoal the policy gave, and
    what the annotated memory said.""", """how the observation changed, and what the annotated
    memory said."""),
        ("policy.BACKWARD_COMPARISON", "  subgoal      {answered_subgoal}\n", ""),
        ("policy.BACKWARD_COMPARISON", "  subgoal      {recorded_subgoal}\n", ""),
    ],
    "predictability": [
        ("context.PRINCIPLES", _DEF_THREE, _DEF_TWO),
        ("context.PRINCIPLES", _DEF_PRE, ""),
        ("context.PRINCIPLES", _DEF_ATOM, _DEF_ATOM_TWO),
        ("context.LOOP_OVERVIEW", _OVERVIEW_CHECK, _overview("subgoal")),
        ("annotator.BACKWARD_OUTPUT_FORMAT", _BACK_WHICH, _BACK_WHICH_TWO),
        ("annotator.BACKWARD_OUTPUT_FORMAT", _BACK_PRE, ""),
        ("policy.BACKWARD_COMPARISON", "  memory       {answered_memory}\n", ""),
        ("policy.BACKWARD_COMPARISON", "  memory       {recorded_memory}\n", ""),
    ],
    "necessity": [
        ("context.PRINCIPLES", _DEF_THREE, _DEF_TWO),
        ("context.PRINCIPLES", _DEF_NEC, ""),
        ("context.PRINCIPLES", _DEF_ATOM, _DEF_ATOM_TWO),
        ("annotator.BACKWARD_OUTPUT_FORMAT", _BACK_WHICH, _BACK_WHICH_TWO),
        ("annotator.BACKWARD_OUTPUT_FORMAT", _BACK_NEC, ""),
        ("annotator.BACKWARD_INPUT", """ Last, the fields it carried and never
read.""", ""),
    ],
}
