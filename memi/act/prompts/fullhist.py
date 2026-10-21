"""The FullHist baseline's system prompt, SYSTEM of harness/astra_waypoint_eval_v3.py with its lr_note variant. This
repo adds the wait answer and its own notes on the robot (context.ROBOT_NOTES).
"""
from __future__ import annotations

from .. import config
from . import context

SYSTEM = """You control a simulated Franka Panda robot arm on a table, by choosing end-effector waypoints.
World frame, metres: the robot's base is at (-0.615, 0, 0) and faces +x; +y is the robot's left, +z is up.

Each turn, answer with ONE JSON object:
{"description": "what happened in the latest frames and what you will do next",
 "waypoint": [x, y, z, roll, pitch, yaw],
 "wait": null,
 "gripper": -1, 0 or 1}
The robot moves its TCP (the point between the fingertips) to (x, y, z) with orientation (roll, pitch, yaw) in
radians, using the same convention as the TCP pose reported to you, along a planned path; then it applies the
gripper command: -1 = close, 1 = open, 0 = keep. Pointing straight down is roll = -3.1416, pitch = 0; yaw turns the
hand about the vertical axis. If a pose cannot be reached the episode fails. A waypoint identical to the current
TCP pose also fails: to only open or close the gripper, move the TCP by at least 2 mm (for example 5 mm up or down).
To wait instead of moving, answer "waypoint": null and "wait": the number of control steps to wait, a whole number
from 1 to 300. Otherwise "wait" is null.
STICK_NOTE
Observations: frames from the fixed front camera (256 x 256) in time order, one every FRAME_STRIDE control steps
(FRAME_DT s) of the video so far and of each motion or wait, then the current front and wrist images; the TCP pose; the finger
positions (0.04 m each = fully open, 0 = closed); the elapsed control steps; and the camera calibration: pinhole
intrinsics K and world-to-camera extrinsics [R|t] (OpenCV convention: camera x right, y down, z forward;
pixel = K (R X_world + t), dividing by depth). You get no depth and no object positions. The task may require
remembering what happened earlier in the video.

MODEL_NOTES
The episode ends automatically when the environment judges success or failure, or after 1300 control steps.
If you believe the task is complete, answer with "waypoint": null and "wait": null."""

#: The harness's LR_NOTE, which its lr_note variant adds to the notes.
LR_NOTE = ("- Left and right: in task instructions, left and right are from the robot's point of view: the robot's left "
           "is +y and its right is -y. The front camera faces the robot (the robot is at the top of the front image), so "
           "the robot's right appears on the LEFT side of the front image.\n")

#: The harness's words after an action that executed no steps.
NOOP_TEXT = ("The last action executed no steps: the robot did not move and no time passed, so the scene is unchanged. "
             "Choose the next action.")


def _notes() -> str:
    """The bulleted notes of context.ROBOT_NOTES (time, gripper, waiting, reaching, no-step actions), then LR_NOTE."""
    body = context.ROBOT_NOTES.split("Notes on the robot and this interface:\n", 1)[1]
    bullets = body.split("The episode ends automatically", 1)[0]
    # The harness has no cycles. What follows an action arrives as the next message.
    return "Notes on the robot and this interface:\n" + bullets.replace("the next cycle", "the next message") + LR_NOTE


def system(task: str) -> str:
    stick = context.STICK_NOTE if task in config.STICK_TASKS else ""
    return (SYSTEM.replace("STICK_NOTE\n", stick).replace("MODEL_NOTES\n", _notes())
            .replace("FRAME_STRIDE", str(config.FRAME_STRIDE)).replace("FRAME_DT", f"{config.FRAME_STRIDE / 20:g}"))
