"""The blocks every role shares, on what the memory is for, the cycles, the three principles of blog Section 2.2, the
loop and the robot."""
from __future__ import annotations

from .. import config

ANNOTATOR_PROMPT_HEAD = "THE ANNOTATOR PROMPT"
POLICY_PROMPT_HEAD = "THE POLICY PROMPT FOR MEMORY"

WHAT_THIS_IS_FOR = """\
===== WHAT THIS IS FOR =====
A policy drives a robot arm end to end. At every act cycle it chooses the next
end-effector waypoint (a position and an orientation) and a gripper command. A
motion planner only moves the hand to that waypoint. Only then is the policy
asked again.

At each cycle the policy has the goal instruction and what that cycle shows. It
has nothing else of the episode. It has no earlier frame and no record of what
it has already done.

Frequently that is not enough to choose. What settles the choice is often an
event that has already happened and is no longer visible. Which container came
down over which cube, how many of them are in the bin, which button has been
pressed. The policy therefore has to write itself a line as it goes and read
that line back at the next cycle. That line is the MEMORY, and it is everything
it carries.
"""


CYCLES = """\
===== THE CYCLES OF AN EPISODE =====
An episode has two kinds of cycles. The video cycle, on a task with a
demonstration, shows the whole demonstration video at once, its front-camera
frames one every 16 control steps plus the last, numbered by control step from
the start of the demonstration. Act cycles are the moments the robot waits for
its next waypoint, each time it has reached the previous one or finished a wait.
An act cycle shows the front-camera frames of the motion since the previous
cycle, one every 16 control steps plus the last, numbered by control steps since
the previous cycle, and then the current wrist-camera frame and the robot's own
state. The first act cycle, and one after a waypoint that moved nothing, shows
only the current frame, numbered 0. On a task with a demonstration, the video
cycle comes first, before the first act cycle.

Every frame shown carries a label. A frame of the demonstration is V<step>, its
control step from the start of the demonstration. A frame of the execution is
E<step>, control steps since the execution started, as the robot state counts
them: the motion frames of an act cycle and its current frame, whose wrist image
shares its label. A memory field can hold such a label, or an ordered list of
them, a keyframe. Every frame it names is then shown again with the memory
carried in, at every later cycle, as its front-camera and wrist-camera image. So
the memory can keep images of what was seen earlier, in the demonstration or in
the execution, instead of describing it.
"""


POLICY_CONTRACT = """\
===== WHAT THE POLICY HAS AT A CYCLE =====
At run time the policy has these things and no others: the policy prompt
for memory, the policy prompt for action, the goal instruction, the frames
of this cycle (on an act cycle with the current wrist frame, the robot state and
the camera calibration), and the memory it wrote at the previous cycle. It never
sees the annotator prompt. The goal instruction is given again at every cycle,
unchanged. The policy cannot look back at a frame from before this cycle. It
remembers nothing beyond that memory.
"""


MEMORY_CONTRACT = """\
===== WHEN A VALUE IS WRITTEN AND WHEN IT IS READ =====
A value recorded at a cycle describes what that cycle shows. It is read at the
cycle after it, never at the cycle it was written, and it is the whole of what
the policy carries forward.
"""


PRINCIPLES = """\
===== WHAT DEFINES A GOOD ANNOTATOR PROMPT AND A GOOD POLICY PROMPT =====
Three conditions decide what a good annotator prompt and a good policy prompt are.

  SUFFICIENCY    the memory and what the current cycle shows together predict
                 the next waypoint. That is its position within 1 cm of the
                 recorded one, and whether the gripper is open or closed after
                 it.

  PREDICTABILITY the memory and what the current cycle shows together predict
                 what the memory itself should hold at that cycle. Each field is
                 compared by the value type the annotator prompt gives it.

  NECESSITY      every field holds a fact that no later cycle recovers when the
                 information is needed.

And one rule of form: one field, one fact. A field that packs several unrelated
facts into one value hides which of them is read and which of them fails, so
none of the three conditions can be checked on it. The one caveat is that this
does not forbid a field from holding an ordered list.
"""


LOOP_OVERVIEW = """\
===== THE ROLES AND THE LOOP =====
We have trajectories of the task performed correctly. Each trajectory has one
goal instruction. Its cycles are its video cycle, on a task with a
demonstration, then its act cycles. Each act cycle also carries the waypoint
that was correct there and the gripper state after it. No cycle carries a
memory.

Given the trajectories, three texts are optimized.

  The ANNOTATOR PROMPT defines what value each field takes at each cycle
  of a recorded episode. It says which fields the memory has, what each one
  holds, and at which moment each value changes. Its reader reads the whole
  episode, so it can name a moment by what happens around it.

  The POLICY PROMPT FOR MEMORY defines how the policy keeps those same
  fields while it runs. It has the same fields under the same names, and its
  values are written the same way. Its reader has what one cycle shows and the
  memory it carried in. So every moment it names has to be one that can be told
  from what that cycle shows.

  The POLICY PROMPT FOR ACTION says which waypoint to choose next and when.
  For each part of the task it says how to place the hand, and when to move on
  to the next part. It states each condition in terms of what the cycle shows
  and what the memory now holds. It is a set of rules to apply in whatever
  situation the policy is standing in, rather than a sequence to walk through.

The INITIALIZER runs once, before the first iteration. It is shown one
minibatch of episodes and writes all three.

An iteration then runs five calls, in this order.

    ANNOTATOR.FORWARD()   One call. It reads the annotator prompt and the whole
      |                   minibatch, and fills the memory fields in at every
      |                   cycle of every episode. It sees each episode end to
      |                   end, which the policy never does. It is shown neither
      |                   policy prompt.
      v
    POLICY.FORWARD()      One call per cycle, each a fresh session with no
      |                   history. It has what the deployed policy has: the two
      |                   policy prompts, the goal instruction, what the
      |                   cycle shows, and the memory carried in. It answers
      |                   that cycle's waypoint and that cycle's memory.
      |                   Where its answer differs from the recorded one, the
      |                   policy prompts did not settle that cycle.
      v
    POLICY.BACKWARD()     One call for each cycle it got wrong. The cycle is put
      |                   to it again, this time with its own answer beside the
      |                   recorded one, and it says what information was
      |                   missing: an unclear memory transition condition, an
      |                   unclear condition for the next waypoint, not enough
      |                   kept in the memory, and so on.
      v
    ANNOTATOR.BACKWARD()  One call, and the one that makes a correction out of
      |                   all that. The policy answered one cycle at a time and
      |                   saw only that cycle, so what arrives here is a separate
      |                   proposal per failed cycle, each made without knowing
      |                   what the others reported. This call reads them
      |                   together against the whole episode and turns them into
      |                   one correction on the THREE texts.
      v
    OPTIMIZER.STEP()      One call. It considers all past edits, and commits
                          the edit.
"""


DEMONSTRATION = """\
===== DEMONSTRATION AND EXECUTION =====
Where a task has a demonstration, the episode opens with one video cycle that
shows the whole demonstration, a frame every 16 control steps plus the last. The
policy is shown these frames exactly as it is shown its own attempt. At a video
cycle it has the goal instruction, those frames of the demonstration and the
memory it carried in, and nothing else. It answers the memory only.

It sees the demonstration once, in this cycle. Later cycles do not show it
again. So everything the execution will need has to be written down as it is
watched. Nothing can be recalled afterwards.
"""


#: The robot and its frames, in the base harness's words (SYSTEM and LR_NOTE of
#: harness/astra_waypoint_eval_v3.py).
CONVENTIONS = """\
===== COORDINATES AND HANDEDNESS =====
World frame, metres: the robot's base is at (-0.615, 0, 0) and faces +x; +y is
the robot's left, +z is up. A waypoint is [x, y, z, roll, pitch, yaw]: the robot
moves its TCP (the point between the fingertips) to (x, y, z) with orientation
(roll, pitch, yaw) in radians, using the same convention as the TCP pose
reported to you, along a planned path. Pointing straight down is roll = -3.1416,
pitch = 0; yaw turns the hand about the vertical axis.

Left and right: in task instructions, left and right are from the robot's point
of view: the robot's left is +y and its right is -y. The front camera faces the
robot (the robot is at the top of the front image), so the robot's right appears
on the LEFT side of the front image.

Frames come from the fixed front camera (256 x 256) and from the wrist camera.
The camera calibration is pinhole intrinsics K and world-to-camera extrinsics
[R|t] (OpenCV convention: camera x right, y down, z forward;
pixel = K (R X_world + t), dividing by depth). The wrist camera moves with the
hand; where its extrinsics are given, they are in the robot state, for the
current wrist frame. You get no depth and no object positions.
"""


#: The base harness's sentences on the gripper and on unreachable poses and its MODEL_NOTES, with this repo's notes
#: on waiting and reaching added. STICK_NOTE takes its marked line on the stick tasks.
ROBOT_NOTES = """\
===== THE ROBOT AND THIS INTERFACE =====
After reaching the waypoint the robot applies the gripper command: -1 = close,
1 = open, 0 = keep. The finger positions are 0.04 m each when fully open and 0
when closed. If a pose cannot be reached the episode fails. A waypoint identical
to the current TCP pose also fails: to only open or close the gripper, move the
TCP by at least 2 mm (for example 5 mm up or down).
STICK_NOTE
Notes on the robot and this interface:
- Time: the simulation runs at 20 control steps per second, and time passes only
  while the robot executes a motion. A motion takes longer the farther it goes:
  about 0.15-0.3 s for a few millimetres, 0.4-1.1 s for 1-5 cm and 1-2.5 s for
  10-30 cm. Everything else in the scene keeps moving during a motion, so to act
  at a particular moment, start the motion that much earlier. A small motion (a
  few millimetres) is a short way to let a little time pass.
- Gripper: it opens or closes only when you command it (-1 or 1); with 0 it
  keeps its current state, also after touching, pushing or pressing something.
- Waiting: an answer may also be a wait of k control steps. The robot then keeps
  its pose and gripper while k steps pass, and the next cycle shows the frames
  of the wait. Use a wait only when there is something to wait for.
- Reaching: the TCP reaches the commanded position only approximately, typically
  within about 2 mm after a short motion and within about 1 cm after a long one,
  so after a motion the reported TCP pose is near the waypoint, not exactly at
  it.
- If an action executes no steps, the next cycle says so explicitly: then the
  robot did not move, no time passed and the scene is unchanged.
The episode ends automatically when the environment judges success or failure,
or after 1300 control steps.
"""


#: On the stick tasks the TCP is the end of the stick (panda_stick.urdf puts panda_hand_tcp there).
STICK_NOTE = """\
In this task the robot holds a stick instead of a gripper: the gripper command
is ignored. Here the TCP is the lower free end of the stick, not a point between
fingertips: a waypoint (x, y, z) puts the end of the stick there, and the
reported TCP pose is where the end of the stick is.
"""


#: Heads the recorded episodes an offline role is handed.
EPISODES_INTRO = """\
Episodes of one task. Each opens with a banner naming it and the instruction it
was given. Its cycles are numbered from #1 within that episode. A video cycle
shows its frames of the demonstration. An act cycle shows the front-camera
frames of the motion since the previous cycle, then the current wrist-camera
frame and the robot state, then the waypoint chosen there and the gripper state
after it."""


#: How a field's value is compared with the annotated one. Part of every answer format that declares fields.
VALUE_TYPES_BLOCK = """\
Each field of the annotator prompt also says how a policy's value is
compared with the annotated one, in `value_type`:
  world_coordinate  numbers in metres in the world frame. Each number is the
                    same within 1 cm. The words around them must be equal.
  count             a whole number. It must be equal.
  boolean           true or false. It must be equal.
  text              words. They must be equal, ignoring case.
  other             anything else. Give `tolerance` too: the largest difference
                    between two numbers of the value that still means the same
                    thing, in the value's own units. The words around them must
                    be equal.
  keyframe          the label of one frame that was shown, V<step> or E<step>,
                    or an ordered list of them, such as [V80, V160, E44]. Two
                    labels are the same frame when the letter is equal and the
                    steps are within 16 of each other; two lists must have the
                    same length and the same frame at every position. Every
                    later cycle is shown every frame it names.
`tolerance` is read only for `other`."""


#: The two policy prompts, in the order a cycle is answered in.
POLICY_PROMPTS = """\
{policy_prompt_for_memory}

===== THE POLICY PROMPT FOR ACTION =====
{policy_prompt_for_action}
"""


def robot_notes(task: str) -> str:
    return ROBOT_NOTES.replace("STICK_NOTE\n", STICK_NOTE if task in config.STICK_TASKS else "")
