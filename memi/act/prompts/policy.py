"""The policy's blocks, for its turn in training and in the rollouts and for policy.backward."""
from __future__ import annotations

ROLE = """\
===== YOUR ROLE =====
You are the POLICY, at one cycle of a task.

You are handed the two policy prompts and the one cycle you are answering
for. That is its goal instruction, its frames (on an act cycle with the current
wrist frame, the robot state and the camera calibration), and the memory carried
into it.

Decide what to do at this cycle, and which memory fields CHANGE now that you
have seen this cycle. The carried memory with your updates applied is what the
next cycle carries in. A field you do not name keeps its value on its own.
"""


INPUT = """\
===== WHAT YOU ARE GIVEN =====
    POLICY PROMPT FOR MEMORY             the state to maintain
    POLICY PROMPT FOR ACTION             which waypoint to choose, and when to
                                         move on
    GOAL                                 the instruction for this episode
    VIDEO                                whether this task has a demonstration,
                                         and whether this cycle shows it
    THIS CYCLE                           the front-camera frames of this cycle,
                                         and on an act cycle the current wrist
                                         frame, the robot state and the camera
                                         calibration
    MEMORY CARRIED IN                    the memory you wrote at the previous
                                         cycle
"""


OUTPUT_FORMAT = """\
===== YOUR ANSWER =====
Five things, and nothing else. Answer for the cycle in front of you and leave
the rest to the later cycles.

`waypoint` is [x, y, z, roll, pitch, yaw] for the TCP, in the world frame, and
`gripper` is -1 (close), 0 (keep) or 1 (open), applied after the waypoint is
reached. On a video cycle, `waypoint` is null and `gripper` is 0. If you believe
the task is complete, `waypoint` is null. To wait instead of moving, `waypoint`
is null and `wait` is the number of control steps to wait, a whole number from 1
to 300. Otherwise `wait` is null.

`memory_updates` is a LIST with one entry per field whose value CHANGES at this
cycle -- a value different from the one carried in, decided from this cycle.
Several fields may change at once: one entry each. Every field not listed keeps
its carried value automatically, so a cycle that changes nothing answers `[]`.
Do not add a field the policy prompt does not declare, and do not restate a
field whose value is staying the same. Write each changed value in the same form
as the value you were handed. If a field reads `holding=no`, answer `yes` or
`no`, not `yes (green cube)`. If it reads `placed=0/5`, answer `1/5`, not a
sentence.

`used_fields` names the fields you actually read to answer, except the ones that
can be inferred from the goal instruction or this cycle's frames. A count you
could have reached by counting what is in view, a position you could have read
off an object you can see, an order the instruction implies without listing: you
had all of those without the memory, and reading the field anyway does not make
it used.

Answer with JSON only:

{"waypoint": [x, y, z, roll, pitch, yaw],
 "wait": null,
 "gripper": ...,
 "memory_updates": [{"field": "a_field_that_changed", "value": "its new value"},
                    {"field": "another_that_changed", "value": "..."}],
 "used_fields": ["...",]}
"""


BACKWARD_ROLE = """\
===== YOUR ROLE =====
You are the POLICY, asked again about a cycle you have already answered. You are
handed the two policy prompts, the cycle exactly as you saw it, and your
own answer beside the recorded one.

Say what fact would have let you give the recorded answer.
"""


BACKWARD_INPUT = """\
===== WHAT YOU ARE GIVEN =====
The two policy prompts come first, then the cycle in the same words it was
put to you the first time. What is new is the block at the end. It sets your
answer beside the one that was recorded.
"""


BACKWARD_COMPARISON = """\
===== WHAT YOU ANSWERED, AND WHAT WAS RECORDED =====
You answered:
  waypoint     {answered_waypoint}
  memory       {answered_memory}

The recorded answer is:
  waypoint     {recorded_waypoint}
  memory       {recorded_memory}

{what_disagreed} disagreed.

===== YOUR ANSWER =====
You had three things to answer from, printed at the top of this turn: the memory
carried in, the policy prompt for memory, and the policy prompt for
action. Say what each of them did not give you. They are three separate
questions.

`missing_from_memory` is the fact the memory should have been carrying and was
not.

`missing_from_policy_prompt_for_memory` is what that policy prompt
should have told you about the value to write and did not.

`missing_from_policy_prompt_for_action` is what that policy prompt
should have told you about the waypoint to choose and did not.

The recorded answer is the annotator prompt's, and the annotator prompt is itself still being edited, so
the recorded answer is not beyond question. Where everything you needed was in
front of you and what failed was the connection from it to the answer, name the
policy prompt that should have made the connection. Where the information
itself was not there to connect, name the memory.

Answer `nothing` in any of the three that was not at fault. All three may be
`nothing`. Only fill in the ones that actually fell short, because a fault you
name here is one the loop will go looking for.

Be specific. Name what was missing about this cycle, in its own words.

Answer with JSON only, no prose:

{"missing_from_memory": "the fact the memory lacked, or nothing",
 "missing_from_policy_prompt_for_memory": "what that policy prompt did not tell you, or nothing",
 "missing_from_policy_prompt_for_action": "what that policy prompt did not tell you, or nothing"}
"""


#: The VIDEO line of a turn, for a video cycle, an act cycle after a demonstration and an act cycle without one.
VIDEO_LINE = {
    "video":
        "  This is a video cycle. It shows the whole demonstration video. Update the memory only.",
    "exec":
        "  You were shown the demonstration video in the video cycle before, and now you are executing the task.",
    "none":
        "  This task has no demonstration video. You are executing the task.",
}
