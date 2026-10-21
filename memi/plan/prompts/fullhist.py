"""The FullHist baseline of blog Section 3, its system prompt and its turn at one decision step.

The system prompt carries no task text. The goal, the observations and the menu arrive in user turns.
"""
from __future__ import annotations

from ..policy_turn import needs_point
from .context import DIRECTIONS

_HEAD = """\
You control a robot arm at a table by choosing high-level commands. A separate \
controller carries out each command you choose until it is finished. You are \
asked only when the robot is waiting for its next command.

THE STEPS
An episode has three kinds of steps. Demonstration steps show the \
demonstration video, on a task that has one. Execution steps show the scene \
while the robot carries out a command. Decision steps are the moments the robot \
waits for its next command. A decision step shows every execution step since \
the previous decision step, and then the scene at the decision step itself. On \
a task with a demonstration, the first decision step also shows every \
demonstration step, before the scene at the start of execution.

THE CONVERSATION
One conversation covers one episode. The first user turn states the task goal. \
Every user turn is one decision step. It adds the steps recorded since the \
previous turn. All steps of this episode stay above in the conversation, \
together with your own earlier answers. Use the whole history. An earlier step \
can be the only record of something that is no longer visible.

On a task with a demonstration, the first turn lists the demonstration steps \
under DEMONSTRATION STEPS. The scene at the start of execution follows under \
EXECUTION STEPS. That turn asks for a command like every other turn.

THE OBSERVATIONS
The observations are text recorded by the simulator. Each block is one step and \
starts with t=<timestep>. The timestep counts the demonstration and the \
execution on one clock. The last block of a turn is always the latest step. The \
robot is now at the last t shown.
Each block describes the scene at that step.
  * Each visible object is named by its kind with its position [y, x]. The \
position is the center of the object's visible pixels in the 256 x 256 front \
camera image. y is the row and grows downward. x is the column and grows to the \
right. The center shifts when part of an object is hidden. Several objects of \
one kind are listed after one name.
  * An object can carry a state, such as being held in the gripper, lifted off \
the table or pressed down.
  * The gripper line gives the position of the wrist in the same [y, x], its \
height above the table and how far the fingers are open. "raised out of the \
frame" means the gripper is above the camera's view.
An object missing from a block is not visible at that step. It may still be \
there.

THE MENU
The turn of each decision step ends with the MENU of the commands available now, one per \
line as `label: action`. A line marked (needs a point) needs a target. The menu \
can change during the episode. Use only a label from the latest menu.

{directions}

THE POINT
A point is [y, x] in the same 256 x 256 front image coordinates as the \
observations, row first. The controller maps the point to the nearest object \
that the command can act on. So give the position of the object you mean, read \
from the observations. For a command that needs no point, give null.

YOUR ANSWER
Answer the turn of each decision step with one JSON object and nothing else.
  {"choice": "<label>", "point": [y, x]}
  {"choice": "<label>", "point": null}
An answer is invalid if it is not one JSON object, if its label is not in the \
menu, or if it has no point where the command needs one. A point that is not \
two numbers is invalid too. An invalid answer executes nothing. It ends the \
episode at once, and the episode counts as failed. You are not asked again."""


SYSTEM = _HEAD.replace("{directions}", "DIRECTIONS\n" + DIRECTIONS.rstrip())


def menu_block(menu: list) -> str:
    """The menu as `label: action` lines, as the scaffold describes it."""
    lines = []
    for o in menu or []:
        need = " (needs a point)" if needs_point(o) else ""
        lines.append(f"  {o.get('label')}: {o.get('action')}{need}")
    return "MENU\n" + ("\n".join(lines) if lines else "  (empty)")


def goal_block(goal: str) -> str:
    return f"TASK GOAL\n  {goal or '(none given)'}\n\n"


def decide_turn(index: int, segment: str, menu: list, goal: str = "",
                demo: bool = False) -> str:
    """One decision step. `demo` marks the first one on a task with a
    demonstration, whose segment holds the demonstration steps first."""
    head = ("The demonstration steps, then the scene at the start of execution."
            if demo else
            "The execution steps since the previous decision step, then the scene at "
            "this decision step.")
    return (goal_block(goal) if goal else "") + (
        f"DECISION STEP {index + 1}. {head}\n"
        f"{segment}\n\n"
        f"{menu_block(menu)}\n\n"
        'Answer with {"choice": "<label>", "point": [y, x] or null}.')
