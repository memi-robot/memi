"""The turn of the Act policy pi_theta of Eq. (1) at one cycle, and how its answer is read. Training and the rollouts
share both."""
from __future__ import annotations

import json
import math

import numpy as np

from ..llm import responses as llm
from ..text import memory_line
from . import config, episodes, memory, prompts, variables

#: What a cycle's step numbers count, the same for every role and in the rollouts.
STEPS_OF = {"video": " of the demonstration", "act": " since the previous cycle (last = current)"}
CYCLE_LINE = {"video": "the video cycle, the whole demonstration. Front-camera frames",
              "act": "an act cycle. Front-camera frames of the motion"}
#: Said after a waypoint that executed no steps. Adapted from the harness's NOOP_TEXT. Training never has one.
NOOP_TEXT = ("The last waypoint executed no steps: the robot did not move and no time passed, so the scene is "
             "unchanged.")


def policy_prefix(annotator_and_policy_prompts: dict) -> list[dict]:
    """The two policy prompts (theta), which every cycle of a run shares. They are sent once as a cached prefix."""
    return [llm.text(f"===== THE CYCLE YOU ARE ANSWERING FOR =====\nPOLICY PROMPT FOR MEMORY\n"
                     f"{variables.memory_prompt(annotator_and_policy_prompts)}\n\nPOLICY PROMPT FOR ACTION\n"
                     f"  {variables.action_text(annotator_and_policy_prompts)}")]


def policy_turn(episode: dict, cycle: dict, carried: dict, names: list[str],
                keyframes: dict[str, tuple[bytes, bytes]], noop: bool = False) -> list[dict]:
    """GOAL, VIDEO, THIS CYCLE, MEMORY CARRIED IN with its keyframes, then the answer format, in the order of the
    Plan turn (memi/plan/policy_turn.py). `episode` has `goal`, `calibration` and `has_demo`. `cycle` has `kind`,
    `frames` (front PNGs), `steps`, `labels`, `wrist` (PNG) and `state`. `keyframes` maps a frame label to its front
    and wrist image."""
    kind, labels = cycle["kind"], cycle["labels"]
    video = prompts.policy.VIDEO_LINE["video" if kind == "video" else ("exec" if episode["has_demo"] else "none")]
    out = [llm.text(f"GOAL\n  {episode['goal']}\n\nVIDEO\n{video}")]
    if noop:
        out.append(llm.text(NOOP_TEXT))
    out.append(llm.text(f"THIS CYCLE\n  {CYCLE_LINE[kind]} at steps {cycle['steps']}{STEPS_OF[kind]}:"))
    for png, lab in zip(cycle["frames"], labels):
        out += [llm.text(f"  {lab}:"), llm.image(png)]
    if kind == "act":
        out += [llm.text(f"  current wrist camera ({labels[-1]}):"), llm.image(cycle["wrist"]),
                llm.text("  robot state: " + json.dumps(cycle["state"])),
                llm.text("  camera calibration: " + json.dumps(episode["calibration"]))]
    out.append(llm.text(f"MEMORY CARRIED IN\n  {memory_line(names, carried)}"))
    for n in names:
        labs = memory.keyframe_labels(carried.get(n, "")) or []
        for j, lab in enumerate(labs):
            at = f"{n}[{j}]" if len(labs) > 1 else n
            if lab in keyframes:
                out += [llm.text(f"  keyframe {at} = {lab}, front and wrist camera:"),
                        llm.image(keyframes[lab][0]), llm.image(keyframes[lab][1])]
            else:
                out.append(llm.text(f"  keyframe {at} = {lab}: no frame with this label was shown"))
    out.append(llm.text(prompts.policy.OUTPUT_FORMAT))
    return out


def training_turn(ep: dict, c: dict, carried: dict, names: list[str]) -> list[dict]:
    """A recorded cycle as the deployed policy would see it, with the keyframes its memory names."""
    keyframes = {}
    for v in carried.values():
        for lab in memory.keyframe_labels(v) or []:
            t = episodes.find_frame(ep, lab, upto=c["k"])
            if t is not None:
                keyframes[lab] = (episodes.front_png(ep, t), episodes.wrist_png(ep, t))
    cycle = {"kind": c["kind"], "frames": [episodes.front_png(ep, t) for t in c["frames"]], "steps": c["labels"],
             "labels": episodes.frame_labels(c),
             "wrist": episodes.wrist_png(ep, c["t"]) if c["kind"] == "act" else None, "state": c.get("state")}
    return policy_turn(ep, cycle, carried, names, keyframes)


def parse_action(answer: dict) -> tuple[str, object]:
    """("wait", k), ("move", 7-vector), ("complete", None) or ("invalid", None). A wait is rounded and capped at
    config.MAX_WAIT. Training grades with this rule too."""
    wp, g, k = answer.get("waypoint"), answer.get("gripper", 0), answer.get("wait")
    if wp is None and isinstance(k, (int, float)) and not isinstance(k, bool) and k >= 1 and math.isfinite(k):
        return "wait", int(min(round(k), config.MAX_WAIT))
    if "waypoint" in answer and wp is None:
        return "complete", None
    if isinstance(wp, list) and len(wp) == 6 and all(isinstance(v, (int, float)) for v in wp) and g in (-1, 0, 1):
        return "move", np.array([*wp, g], dtype=np.float64)
    return "invalid", None
