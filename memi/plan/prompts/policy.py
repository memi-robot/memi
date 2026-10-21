"""The policy's blocks, for its turn at deployment and in policy.forward (step 3) and for policy.backward (step 5)."""
from __future__ import annotations

from .context import STEPS

ROLE = f"""\
===== YOUR ROLE =====
You are the HIGH-LEVEL POLICY, at one decision step of a task.

{STEPS}

You are handed the two policy prompts and the one decision step you are
answering for. That is its goal instruction, the execution steps since the
previous decision step with the scene at this decision step, the memory carried
into it, and the menu of subgoals the robot accepts.

Decide what to do at this decision step, and which memory fields CHANGE now
that you have seen these steps. The carried memory with your updates applied is
what the next decision step carries in. A field you do not name keeps its value
on its own.
"""


INPUT = """\
===== WHAT YOU ARE GIVEN =====
    POLICY PROMPT FOR MEMORY             the state to maintain
    POLICY PROMPT FOR SUBGOAL            what to do, and when to move on
    GOAL                                 the instruction for this episode
    VIDEO                                whether this task has a demonstration,
                                         and whether this step shows it
    OBSERVED STEPS                       every object you could see, with positions,
                                         at each execution step since the
                                         previous decision step, and at this
                                         decision step, last
    MEMORY CARRIED IN                    the memory you wrote at the previous
                                         decision step
    MENU                                 the subgoals the robot accepts, by label

The listed coordinates are everything you have about where things are.
"""


STEP = """\
===== THE DECISION STEP YOU ARE ANSWERING FOR =====
POLICY PROMPT FOR MEMORY
{policy_prompt_for_memory}

POLICY PROMPT FOR SUBGOAL
{policy_prompt_for_subgoal}

GOAL
  {goal}

VIDEO
{video}

OBSERVED STEPS
{segment}

MEMORY CARRIED IN
{memory}

MENU
{options}
"""


OUTPUT_FORMAT = """\
===== YOUR ANSWER =====
Four things, and nothing else. Answer for the decision step in front of you and
leave the rest to the later decision steps.

`choice` is the label of one menu option, copied verbatim. `point` is the target
you are sending the robot to, written [y, x]. Take it from the latest observed
step or from the carried memory. Give it when the option needs a point, and
null otherwise.

`memory_updates` is a LIST with one entry per field whose value CHANGES at
this decision step -- a value different from the one carried in, decided from
these observed steps. Several fields may change at once: one entry each. Every field not
listed keeps its carried value automatically, so a decision step that changes
nothing answers `[]`. Do not add a field the policy prompt does not declare, and do not
restate a field whose value is staying the same. Write each changed value
in the same form as the value you were handed. If a field reads
`holding=no`, answer `yes` or `no`, not `yes (green cube)`. If it reads
`placed=0/5`, answer `1/5`, not a sentence.

`used_fields` names the fields you actually read to answer, except the ones that
can be inferred from the goal instruction or the observed steps. A count you
could have reached by counting what is in view, a position you could have read
off an object you can see, an order the instruction implies without listing: you
had all of those without the memory, and reading the field anyway does not make
it used.

Answer with JSON only:

{"choice": "...",
 "point": [y, x],
 "memory_updates": [{"field": "a_field_that_changed", "value": "its new value"},
                    {"field": "another_that_changed", "value": "..."}],
 "used_fields": ["...",]}
"""


#: The VIDEO line of a policy turn. `first` is the first decision step on a
#: task with a demonstration, `exec` a later one, `none` a task without one.
VIDEO_LINE = {
    'first': (
        '  This is the first decision step. It shows every step of the '
        'demonstration video under DEMONSTRATION STEPS, then the scene at '
        'the start of execution under EXECUTION STEPS. You are now executing '
        'the task. The demonstration is not shown again.'
    ),
    'exec': (
        '  You were shown the demonstration video at the first decision '
        'step, and now you are executing the task.'
    ),
    'none': (
        '  This task has no demonstration video. You are executing the task.'
    ),
}


BACKWARD_ROLE = """\
===== YOUR ROLE =====
You are the HIGH-LEVEL POLICY, asked again about a decision step you have
already answered. You are handed the two policy prompts, the decision step exactly as
you saw it, and your own answer beside the recorded one.

Say what fact would have let you give the recorded answer.
"""


BACKWARD_INPUT = """\
===== WHAT YOU ARE GIVEN =====
The two policy prompts come first, then the decision step in the same words it was put to you the
first time. What is new is the block at the end. It sets your answer beside the
one that was recorded.
"""


BACKWARD_COMPARISON = """\
===== WHAT YOU ANSWERED, AND WHAT WAS RECORDED =====
You answered:
  subgoal      {answered_subgoal}
  memory       {answered_memory}

The recorded answer is:
  subgoal      {recorded_subgoal}
  memory       {recorded_memory}

{what_disagreed} disagreed.

===== YOUR ANSWER =====
You had three things to answer from, printed at the top of this turn: the memory
carried in, the policy prompt for memory, and the policy prompt for subgoal. Say what
each of them did not give you. They are three separate questions.

`missing_from_memory` is the fact the memory should have been carrying and was
not.

`missing_from_policy_prompt_for_memory` is what that policy prompt should have told you about
the value to write and did not.

`missing_from_policy_prompt_for_subgoal` is what that policy prompt should have told you about
what to do and did not.

The recorded answer is the annotator prompt's, and the annotator prompt is itself still being edited,
so the recorded answer is not beyond question. Where everything you needed was
in front of you and what failed was the connection from it to the answer, name
the policy prompt that should have made the connection. Where the information itself
was not there to connect, name the memory.

Answer `nothing` in any of the three that was not at fault. All three may be
`nothing`. Only fill in the ones that actually fell short, because a fault you
name here is one the loop will go looking for.

Be specific. Name what was missing about this decision step, in its own words.

Answer with JSON only, no prose:

{"missing_from_memory": "the fact the memory lacked, or nothing",
 "missing_from_policy_prompt_for_memory": "what that policy prompt did not tell you, or nothing",
 "missing_from_policy_prompt_for_subgoal": "what that policy prompt did not tell you, or nothing"}
"""
