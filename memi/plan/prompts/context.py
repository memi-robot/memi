"""The blocks every offline role reads first, the platform blocks the policy reads too, and the shared headings."""
from __future__ import annotations

ANNOTATOR_PROMPT_HEAD = 'THE ANNOTATOR PROMPT'
POLICY_PROMPT_HEAD = 'THE POLICY PROMPT FOR MEMORY'

#: The three kinds of steps, stated once in each system prompt that needs them.
STEPS = """\
An episode has three kinds of steps. Demonstration steps show the
demonstration video, on a task that has one. Execution steps show the scene
while the robot carries out a command. Decision steps are the moments the robot waits for its next command. A decision step shows
every execution step since the previous decision step, and then the scene at
the decision step itself. On a task with a demonstration, the first decision
step also shows every demonstration step, before the scene at the start of
execution."""


WHAT_THIS_IS_FOR = f"""\
===== WHAT THIS IS FOR =====
A high-level policy drives a robot by choosing one subgoal at a time from a
menu, such as pick up the cube, press the button, or drop it onto the target. A
subgoal is one menu option, with a target point where the option needs one. An
oracle controller executes whichever subgoal it is given to its end. Only then
is the policy asked again. That moment is a DECISION STEP.

{STEPS}

At each decision step the policy has the goal instruction and what that
decision step shows. It has nothing else of the episode. It has no earlier
observation and no record of what it has already done.

Frequently that is not enough to choose. What settles the choice is often an
event that has already happened and is no longer visible. Which container came
down over which cube, how many of them are in the bin, which button has been
pressed. The policy therefore has to write itself a line as it goes and read that
line back at the next decision step. That line is the MEMORY, and it
is everything it carries.

"""


POLICY_CONTRACT = """\
===== WHAT THE POLICY HAS AT A DECISION STEP =====
At run time the high-level policy has five things and no others: the policy prompt for
memory, the policy prompt for subgoal, the goal instruction, the execution
steps since the previous decision step with the scene at this decision step,
and the memory it wrote at the previous decision step. It never sees the
annotator prompt. The goal instruction is given again at every decision step,
unchanged. The policy cannot look back at an observation from before the
previous decision step. It remembers nothing beyond that memory.
"""


MEMORY_CONTRACT = """\
===== WHEN A VALUE IS WRITTEN AND WHEN IT IS READ =====
A value recorded at a step describes what that step shows. It is read at the
step after it, never at the step it was written, and it is the whole of what
the policy carries forward.
"""


PRINCIPLES = """\
===== WHAT DEFINES A GOOD ANNOTATOR PROMPT AND A GOOD POLICY PROMPT =====
Three conditions decide what a good annotator prompt and a good policy prompt are.

  SUFFICIENCY    the memory and what the current step shows together predict
                 the next subgoal. That is the menu option, and the object its
                 point selects.

  PREDICTABILITY the memory and what the current step shows together predict what the
                 memory itself should hold at that step.

  NECESSITY      every field holds a fact that no later step recovers when the
                 information is needed.

And one rule of form: one field, one fact. A field that packs several
unrelated facts into one value hides which of them is read and which of them
fails, so none of the three conditions can be checked on it. The one caveat
is that this does not forbid a field from holding an ordered list.
"""


LOOP_OVERVIEW = """\
===== THE ROLES AND THE LOOP =====
We have high-level trajectories of the task performed correctly. Each trajectory
has one goal instruction and one menu. Its steps are its decision steps. On a
task with a demonstration, the first decision step also shows the
demonstration steps. Each decision step also carries the subgoal that was
correct there. That subgoal is a menu option, with the target point where the option
needs one. No step carries a memory.

Given the high-level trajectory, three documents are optimized.

  The ANNOTATOR PROMPT defines what value each field takes at each step of
  a recorded episode. It says which fields the memory has, what each one holds,
  and at which moment each value changes. Its reader reads the whole episode, so
  it can name a moment by what happens around it.

  The POLICY PROMPT FOR MEMORY defines how the policy keeps those same fields while
  it runs. It has the same fields under the same names, and its values are
  written the same way. Its reader has what one step shows and the memory it
  carried in. So every moment it names has to be one that can be told from what
  that step shows.

  The POLICY PROMPT FOR SUBGOAL says what to do next and when. For each subgoal it
  says when to move to which other subgoal, and which object to point at where
  the option needs a point. It states each condition in terms of what the step
  shows and what the memory now holds. It is a set of rules to
  apply in whatever situation the policy is standing in, rather than a sequence
  to walk through.

The INITIALIZER runs once, before the first iteration. It is shown one
minibatch of episodes and writes all three.

An iteration then runs five calls, in this order.

    ANNOTATOR.FORWARD()   One call. It reads the annotator prompt and the whole
      |                   minibatch, and fills the memory fields in at every
      |                   step of every episode. It sees each episode end to
      |                   end, which the policy never does. It is shown neither
      |                   policy prompt.
      v
    POLICY.FORWARD()      One call per step, each a fresh session with no
      |                   history. It has what the deployed policy has: the two
      |                   policy prompts, the goal instruction, the menu,
      |                   what the step shows, and the memory carried in. It
      |                   answers that step's subgoal and that step's memory.
      |                   Where its answer differs from the recorded one, the
      |                   policy prompts did not settle that step.
      v
    POLICY.BACKWARD()     One call for each step it got wrong. The step is put
      |                   to it again, this time with its own answer beside the
      |                   recorded one, and it says what information was
      |                   missing: an unclear memory transition condition, an
      |                   unclear subgoal transition condition, not enough kept
      |                   in the memory, and so on.
      v
    ANNOTATOR.BACKWARD()  One call, and the one that makes a correction out of
      |                   all that. The policy answered one step at a time and
      |                   saw only that step, so what arrives here is a separate
      |                   proposal per failed step, each made without knowing
      |                   what the others reported. This call reads them
      |                   together against the whole episode and turns them into
      |                   one correction on the THREE documents.
      v
    OPTIMIZER.STEP()      One call. It considers all past edits, and commits
                          the edit.
"""


DEMONSTRATION = """\
===== DEMONSTRATION AND EXECUTION =====
Where a task has a demonstration, the first decision step also shows every
demonstration step, before the scene at the start of execution. The policy is
shown them exactly as it is shown its own attempt. At that decision step it
has the goal instruction, the demonstration steps, the scene at the start of
execution and an empty memory, and nothing else. It answers its first subgoal
and the memory together.

It sees the demonstration once. Later decision steps do not show it again. So
everything the execution will need has to be written down at the first
decision step. Nothing can be recalled afterwards.
"""


#: The planner takes a point as [y, x], so every position is written [y, x].
COORDINATES = """\
===== COORDINATES AND HANDEDNESS =====
Positions are written [y, x]: y measured down from the top of the image, x across
from the left. Write them that way too, wherever they go: in a point, in a
memory field, and inside a list. [66, 105], never <66, 105> and never
{"y": 66, "x": 105}.
"""

#: The robot-frame directions. The FullHist prompt has them too.
DIRECTIONS = """\
Directions (left, right, forward, backward) are the robot's and not the camera's,
so the robot's right-hand side appears on the LEFT of the image.

Stated as arithmetic:

  the robot's RIGHT is the SMALLER x.     the robot's LEFT is the LARGER x.
  the robot's FORWARD is the LARGER y.    the robot's BACKWARD is the SMALLER y.

So forward moves DOWN the image, which is the opposite of the usual reading, and
a diagonal is the two combined: forward-left is larger y and larger x.

Of two candidates at [84, 72] and [77, 180], the right-side one is [84, 72]
because 72 is the smaller x, and the forward one is also [84, 72] because 84 is
the larger y.

Likewise, the NEAR end of something is the end close to the robot, not to the
camera, so it is the end with the SMALLER y. The FAR end has the LARGER y.
"""

CONVENTIONS = COORDINATES + "\n" + DIRECTIONS


#: How an observation is written. Part of EPISODES_INTRO.
LEGEND = """\
For each timestep, the observation is written from the recording. The gripper
shows where the hand is, meaning the wrist the fingers hang from and not the
fingertips, in the same [y, x], with its height and how far the fingers are
apart. The opening is worth reading to tell whether the robot closed on nothing
or grasped an object. The object shows its class and the coordinates, which is
the center of its visible pixels, so it moves when part of the object is hidden,
and it wobbles a little from step to step for no reason at all."""


#: What opens every episode turn.
EPISODES_INTRO = f"""\
Episodes of one task. Each opens with a banner naming it, the instruction it
was given and the menu of subgoals the robot accepts. Its steps are numbered
from #1 within that episode. Each step is a decision step. It shows every
execution step since the previous decision step, then the scene at the decision
step itself, then the subgoal chosen there. On a task with a demonstration, the
first decision step shows every demonstration step first, under
DEMONSTRATION STEPS, then the scene at the start of execution, under
EXECUTION STEPS.

{LEGEND}"""


#: The two policy prompts, in the order a step is answered in.
POLICY_PROMPTS = """\
{policy_prompt_for_memory}

===== THE POLICY PROMPT FOR SUBGOAL =====
{policy_prompt_for_subgoal}
"""
