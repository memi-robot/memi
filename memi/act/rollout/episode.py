"""One Init or MEMI test episode of the Act policy, deployed as in blog Section 2.1. Every call is a fresh session, and
the policy conditions on its own memory m_{t-1} from the empty m_0.

The protocol is the base harness's (harness/astra_waypoint_eval_v3.py). Success is what the environment reports, and
the episode also ends when the model declares completion, after config.MAX_CONSECUTIVE_NOOPS actions in a row that
execute no steps, at an unusable answer and at config.MAX_ACT_CALLS act calls.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from ...llm import responses as llm
from .. import config, episodes, memory, policy_turn, prompts, variables
from ..policy_turn import parse_action
from . import sim


def write_result(out: Path, row: dict) -> None:
    """out/result.json, written whole under another name and then renamed into place."""
    tmp = out / f".result.json.{os.getpid()}"
    tmp.write_text(json.dumps(row, indent=1, default=str))
    os.replace(tmp, out / "result.json")


def read_result(p: Path) -> dict | None:
    """A result, or None if there is none or it cannot be read."""
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def run_episode(task: str, episode: int, annotator_and_policy_prompts: dict, client: llm.Client, out: Path,
                meta: dict) -> dict:
    """Runs the episode and writes out/result.json and out/episode.mp4. `meta` is recorded with the result."""
    import imageio
    names = variables.field_names(annotator_and_policy_prompts)
    out.mkdir(parents=True, exist_ok=True)
    env = sim.make_env(task, episode)
    obs, info = env.reset()
    goal = info["task_goal"][0]
    calibration = sim.calibration(obs, info)
    video = []

    def record(o):
        for f, w in zip(o["front_rgb_list"], o["wrist_rgb_list"]):
            video.append(np.hstack([sim.uint8(f), sim.uint8(w)]))

    record(obs)
    start = sim.elapsed_steps(info)             # the demonstration's steps
    carried: dict = {}
    calls, steps, cost = [], [], 0.0
    shown: dict = {}                           # every labelled frame shown so far, as its raw front and wrist frame

    def ask(kind, frames, frame_steps, labels, wrist, state, noop=False):
        nonlocal carried, cost
        keyframes = {lab: (sim.png(shown[lab][0]), sim.png(shown[lab][1]))
                     for v in carried.values() for lab in memory.keyframe_labels(v) or [] if lab in shown}
        content = policy_turn.policy_turn(
            {"goal": goal, "calibration": calibration, "has_demo": has_demo},
            {"kind": kind, "frames": frames, "steps": frame_steps, "labels": labels, "wrist": wrist, "state": state},
            carried, names, keyframes, noop)
        r = client.ask(prompts.policy_system(task), content, purpose=f"{task}.rollout",
                       prefix=policy_turn.policy_prefix(annotator_and_policy_prompts))
        cost += r["cost"]
        answer = r["json"] if isinstance(r["json"], dict) else None
        before = dict(carried)
        if answer is not None:
            carried = memory.apply_updates(carried, answer.get("memory_updates"), names)
        calls.append({"kind": kind, "frames": frame_steps, "memory_in": before, "memory_out": dict(carried),
                      "answer": answer if answer is not None else r["text"][-400:],
                      "used": (answer or {}).get("used_fields"), "cost": round(r["cost"], 5), "s": r["s"]})
        return answer

    # The reset returns the demonstration's frames, then the scene. The demonstration is one video cycle.
    fr, wr = obs["front_rgb_list"], obs["wrist_rgb_list"]
    has_demo = len(fr) > 1
    if has_demo:
        idx = episodes.sample_steps(0, len(fr) - 2)
        labels = [f"V{i}" for i in idx]
        shown.update({lab: (fr[i], wr[i]) for lab, i in zip(labels, idx)})
        ask("video", [sim.png(fr[i]) for i in idx], idx, labels, None, None)
    cur_frames, cur_steps, cur_raw = [sim.png(fr[-1])], [0], [(fr[-1], wr[-1])]
    status, end_reason, noops, noop = "ongoing", None, 0, False
    t0 = time.time()
    while True:
        if sum(c["kind"] == "act" for c in calls) >= config.MAX_ACT_CALLS:
            status, end_reason = "model_call_limit", f"limit:{config.MAX_ACT_CALLS}_model_calls"
            break
        state = sim.robot_state(obs, info, start)
        now = state["elapsed_control_steps"]
        labels = episodes.exec_labels(now, cur_steps[-1], cur_steps)
        shown.update(zip(labels, cur_raw))
        answer = ask("act", cur_frames, cur_steps, labels, sim.png(obs["wrist_rgb_list"][-1]), state, noop)
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
            did = {"wait": arg}
            new_obs, rew, term, trunc, new_info = sim.wait(env, arg)
        else:
            did = {"action": arg.tolist()}
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
            cur_frames, cur_steps, noop = [sim.png(obs["front_rgb_list"][-1])], [0], True
            cur_raw = [(obs["front_rgb_list"][-1], obs["wrist_rgb_list"][-1])]
            continue
        noops, noop = 0, False
        obs, info = new_obs, new_info
        record(obs)
        st = info.get("status", "unknown")
        steps.append({**did, "frames": k, "status": st, "elapsed_steps": sim.elapsed_steps(info),
                      "tcp_after": episodes.rounded(np.asarray(obs["eef_state_list"][-1]).reshape(-1))})
        if bool(term) or bool(trunc) or st in ("success", "fail", "timeout", "error"):
            status, end_reason = st, f"env:{st}"
            break
        idx = episodes.sample_steps(0, k - 1)          # frame i is the scene after control step i + 1
        cur_frames, cur_steps = [sim.png(obs["front_rgb_list"][i]) for i in idx], [i + 1 for i in idx]
        cur_raw = [(obs["front_rgb_list"][i], obs["wrist_rgb_list"][i]) for i in idx]
    env.close()
    if video:
        imageio.mimsave(out / "episode.mp4", video, fps=30)
    row = {"task": task, "episode": episode, "dataset": config.DATASET, **meta, "goal": goal, "status": status,
           "success": status == "success", "end_reason": end_reason, "model_calls": len(calls),
           "act_calls": sum(c["kind"] == "act" for c in calls), "video_calls": sum(c["kind"] == "video" for c in calls),
           "waypoints_executed": sum(1 for s in steps if s.get("frames")), "steps": steps, "calls": calls,
           "cost_usd": round(cost, 4), "episode_s": round(time.time() - t0, 1)}
    write_result(out, row)
    return row
