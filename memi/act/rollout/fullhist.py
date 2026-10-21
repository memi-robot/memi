"""One test episode of the FullHist baseline, the whole episode in one chained conversation as the base harness runs it.

It follows run_episode and observation of harness/astra_waypoint_eval_v3.py. The first message holds the goal and
every frame the reset returns, frames are labelled by their index, the elapsed control steps count the demonstration,
the front camera calibration comes once and the wrist camera's with every message, and the answer carries a free-text
description. Unlike the harness and like rollout/episode.py, it shows one frame every config.FRAME_STRIDE control
steps, runs at reasoning effort config.EFFORT, offers the wait and stops at config.MAX_ACT_CALLS answers.
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ...llm import responses as llm
from .. import config, episodes
from ..policy_turn import parse_action
from ..prompts import fullhist as P
from . import sim
from .episode import write_result


def _state(obs: dict, info: dict, first: bool) -> dict:
    """The harness's state_text. TCP pose, fingers, elapsed control steps with the demonstration, the wrist camera's
    calibration, and the front camera's on the first message."""
    s = {"tcp_pose_xyz_rpy": episodes.rounded(np.asarray(obs["eef_state_list"][-1]).reshape(-1)),
         "finger_positions_m": episodes.rounded(np.asarray(obs["gripper_state_list"][-1]).reshape(-1)),
         "elapsed_control_steps": sim.elapsed_steps(info),
         "wrist_camera_K": episodes.rounded(np.asarray(info["wrist_camera_intrinsic"]).reshape(3, 3), 2),
         "wrist_camera_extrinsic_world_to_camera":
             episodes.rounded(np.asarray(obs["wrist_camera_extrinsic_list"][-1]).reshape(3, 4))}
    if first:
        s["front_camera_K"] = episodes.rounded(np.asarray(info["front_camera_intrinsic"]).reshape(3, 3), 2)
        s["front_camera_extrinsic_world_to_camera"] = episodes.rounded(
            np.asarray(obs["front_camera_extrinsic_list"][-1]).reshape(3, 4))
    return s


def _observation(obs: dict, info: dict, label: str, first: bool = False) -> list[dict]:
    """The harness's observation. Frames one in every FRAME_STRIDE plus the last, labelled by index, then the wrist
    frame and the state."""
    fr, wr = obs["front_rgb_list"], obs["wrist_rgb_list"]
    idx = episodes.sample_steps(0, len(fr) - 1)
    out = [llm.text(f"{label}: front-camera frames {idx} of {len(fr)} (last = current):")]
    out += [llm.image(sim.png(fr[i])) for i in idx]
    out += [llm.text("Current wrist camera:"), llm.image(sim.png(wr[-1])),
            llm.text("Robot state and calibration: " + json.dumps(_state(obs, info, first)))]
    return out


def run_episode(task: str, episode: int, client: llm.Client, out: Path, meta: dict) -> dict:
    """Runs the episode and writes out/result.json with the fields of rollout/episode.py, out/episode.mp4 and the
    exchange log out/exchange.jsonl."""
    import imageio
    out.mkdir(parents=True, exist_ok=True)
    env = sim.make_env(task, episode)
    obs, info = env.reset()
    goal = info["task_goal"][0]
    system = P.system(task)
    video = []

    def record(o):
        for f, w in zip(o["front_rgb_list"], o["wrist_rgb_list"]):
            video.append(np.hstack([sim.uint8(f), sim.uint8(w)]))

    record(obs)
    calls, steps, cost, previous = [], [], 0.0, None
    log = open(out / "exchange.jsonl", "w")      # every request and answer, images as their sha256

    def ask(content):
        nonlocal cost, previous
        logged = [x if x["type"] != "input_image" else
                  {"type": "input_image", "sha256": hashlib.sha256(base64.b64decode(x["image_url"].split(",", 1)[1])).hexdigest()}
                  for x in content]
        log.write(json.dumps({"request": {"model": client.model, "effort": client.effort, "previous_response_id": previous,
                                          "instructions": system if previous is None else None, "input": logged}}) + "\n")
        r = client.ask_chained(system, content, purpose=f"{task}.rollout.fullhist", previous=previous)
        log.write(json.dumps({"response": {"id": r["id"], "text": r["text"], "usage": r["usage"]}}) + "\n")
        log.flush()
        previous = r["id"]
        cost += r["cost"]
        answer = r["json"] if isinstance(r["json"], dict) else None
        calls.append({"kind": "act", "answer": answer if answer is not None else r["text"][-400:],
                      "cost": round(r["cost"], 5), "s": r["s"]})
        return answer

    content = [llm.text(f"Task goal: {goal}")] + _observation(obs, info, "Video so far", first=True)
    status, end_reason, noops = "ongoing", None, 0
    t0 = time.time()
    while True:
        if len(calls) >= config.MAX_ACT_CALLS:
            status, end_reason = "model_call_limit", f"limit:{config.MAX_ACT_CALLS}_model_calls"
            break
        answer = ask(content)
        if answer is None:
            status, end_reason = "unparseable", "model_answer:unparseable"
            break
        kind, arg = parse_action(answer)
        if kind == "complete":
            status, end_reason = "declared_complete", "model_answer:declared_complete"
            break
        if kind == "invalid":
            status, end_reason = "invalid_action", "model_answer:invalid_action"
            break
        if kind == "wait":
            did, label = {"wait": arg}, "Frames recorded during that wait"
            new_obs, rew, term, trunc, new_info = sim.wait(env, arg)
        else:
            did, label = {"action": arg.tolist()}, "Frames recorded during that motion"
            new_obs, rew, term, trunc, new_info = env.step(arg)
        if new_obs is None or (new_info or {}).get("status") == "error":
            steps.append({**did, "error": (new_info or {}).get("error_message")})
            status, end_reason = "error", "env_error"
            break
        k = len(new_obs.get("front_rgb_list") or [])
        if k == 0:
            noops += 1
            steps.append({**did, "frames": 0, "noop": True})
            if noops >= config.MAX_CONSECUTIVE_NOOPS:
                status, end_reason = "no_progress", f"limit:{config.MAX_CONSECUTIVE_NOOPS}_consecutive_noops"
                break
            content = [llm.text(P.NOOP_TEXT), llm.text("Current front camera:"),
                       llm.image(sim.png(obs["front_rgb_list"][-1])), llm.text("Current wrist camera:"),
                       llm.image(sim.png(obs["wrist_rgb_list"][-1])),
                       llm.text("Robot state and calibration: " + json.dumps(_state(obs, info, False)))]
            continue
        noops = 0
        obs, info = new_obs, new_info
        record(obs)
        st = info.get("status", "unknown")
        steps.append({**did, "frames": k, "status": st, "elapsed_steps": sim.elapsed_steps(info),
                      "tcp_after": episodes.rounded(np.asarray(obs["eef_state_list"][-1]).reshape(-1))})
        if bool(term) or bool(trunc) or st in ("success", "fail", "timeout", "error"):
            status, end_reason = st, f"env:{st}"
            break
        content = _observation(obs, info, label)
    env.close()
    log.close()
    if video:
        imageio.mimsave(out / "episode.mp4", video, fps=30)
    row = {"task": task, "episode": episode, "dataset": config.DATASET, **meta, "goal": goal, "status": status,
           "success": status == "success", "end_reason": end_reason, "model_calls": len(calls),
           "act_calls": len(calls), "video_calls": 0,
           "waypoints_executed": sum(1 for s in steps if s.get("frames")), "steps": steps, "calls": calls,
           "cost_usd": round(cost, 4), "episode_s": round(time.time() - t0, 1)}
    write_result(out, row)
    return row
