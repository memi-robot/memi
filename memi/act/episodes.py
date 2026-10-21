"""RoboMME's recorded training episodes as the cycles the Act policy is called at, built once from the h5 files.

    ep = load("BinFill", 12)        # {"task", "episode", "goal", "has_demo", "calibration", "cycles": [...]}

The video cycle shows the whole demonstration, and the policy only updates its memory there. An act cycle is a moment
the robot waits for its next waypoint, and its target a_t is the expert's next waypoint or wait.
"""
from __future__ import annotations

import json
import random

import numpy as np

from ..paths import H5_DIR
from . import config

#: A waypoint change of less than this, with the same gripper state, is the expert reissuing the same pose, in metres.
MERGE_M = 0.002
#: The stick solver records a waypoint when the stick arrives there, not when the motion to it starts. RouteStick
#: records the destination at the start, as the gripper tasks do.
ARRIVAL_LABEL_TASKS = ("PatternLock",)
#: A still wait shorter than this many control steps is not one (a gripper closing or settling takes ~6-10).
WAIT_MIN = 32
#: Tasks whose last command pushes past where the part seats. The expert commands a stroke the contact stops short
#: of, and the task succeeds at the pose reached. That pose, not the unreached command, is the target.
SEATED_AT_SUCCESS_TASKS = ("InsertPeg",)
#: Fingers below FINGERS_OPEN_M are closed. Fingers that widen by more than OPENING_M in one step are the expert's
#: unlabelled opening. Both in metres.
FINGERS_OPEN_M, OPENING_M = 0.03, 0.003
#: The TCP has arrived within ARRIVED_M of a waypoint and has left it beyond LEFT_M, in metres.
ARRIVED_M, LEFT_M = 0.02, 0.035
#: A TCP step shorter than MOVING_M is no motion. A hold within STILL_M over three steps that stays within HELD_M is a
#: still wait, in metres.
MOVING_M, STILL_M, HELD_M = 0.001, 0.002, 0.005
#: The number of recorded episodes of each task.
RECORDED_EPISODES = 100


def _h5(task: str):
    import h5py
    return h5py.File(H5_DIR / f"record_dataset_{task}.h5", "r")


def sample_steps(start: int, end: int) -> list[int]:
    """Timesteps start..end shown to the policy, one every config.FRAME_STRIDE plus the last."""
    return sorted(set(range(start, end + 1, config.FRAME_STRIDE)) | {end})


def training_ids(task: str) -> list[int]:
    """Training episodes, without RoboMME's deliberate fail-and-recover demonstrations."""
    p = config.EPISODE_CACHE / task / "train_episodes.json"
    if p.exists():
        return json.loads(p.read_text())
    keep = []
    with _h5(task) as f:
        for e in range(RECORDED_EPISODES):
            s = f[f"episode_{e}/setup"]
            if "fail_recover_mode" not in s or str(s["fail_recover_mode"][()]) in ("None", "b'None'"):
                keep.append(e)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(keep))
    return keep


def minibatches(task: str) -> dict[str, list[int]]:
    """The minibatches B^(k), disjoint episodes for the initializer and each iteration, drawn once with a fixed
    seed."""
    eps = training_ids(task)
    rng = random.Random(f"{config.SEED}:{task}")
    rng.shuffle(eps)
    n0, n = config.INIT_EPISODES, config.EPISODES_PER_ITERATION
    out = {"init": eps[:n0]}
    for r in range(config.ITERATIONS):
        out[f"iteration{r + 1}"] = eps[n0 + r * n: n0 + (r + 1) * n]
    return out


def rounded(a, nd: int = 4) -> list:
    """A number or an array as the policy reads it, here and in the rollouts. The h5 holds float32, which would print
    with long tails."""
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def png(rgb) -> bytes:
    """An RGB frame as the PNG the policy is shown, here and in the rollouts."""
    import cv2
    return cv2.imencode(".png", cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))[1].tobytes()


def robot_state(eef, fingers, elapsed: int, wrist_extrinsic) -> dict:
    """The robot state an act cycle shows. TCP pose, fingers, control steps of the execution and wrist camera pose."""
    return {"tcp_pose_xyz_rpy": rounded(eef), "finger_positions_m": rounded(fingers),
            "elapsed_control_steps": elapsed,
            "wrist_camera_extrinsic_world_to_camera": rounded(np.asarray(wrist_extrinsic).reshape(3, 4))}


def calibration(front_k, front_extrinsic, wrist_k) -> dict:
    return {"front_camera_K": rounded(np.asarray(front_k).reshape(3, 3), 2),
            "front_camera_extrinsic_world_to_camera": rounded(np.asarray(front_extrinsic).reshape(3, 4)),
            "wrist_camera_K": rounded(np.asarray(wrist_k).reshape(3, 3), 2)}


def exec_labels(now: int, current: int, steps: list[int]) -> list[str]:
    """The labels of execution frames, E<control steps since the execution started>. `now` is that count at the
    current frame, whose own step is `current`."""
    return [f"E{now - (current - s)}" for s in steps]


def _gripper_state(v: float) -> int:
    return 1 if v > 0 else -1


def act_cycles(task: str, wp: list, fingers: list, tcp: list, exec_start: int, n: int, success: int | None
               ) -> list[tuple]:
    """(observed timestep, target timestep, waypoint or ("wait", k), gripper after) per act cycle.

    `wp` is the expert's waypoint at every timestep (`action/waypoint_action`), x, y, z, roll, pitch, yaw and the
    gripper state, 1 open and -1 closed. A cycle is observed at the last timestep before the expert's next command
    starts to act, as the policy is at test, since the recording stores each observation after its first control step.
    """
    if task in ARRIVAL_LABEL_TASKS:
        cycles = _arrival_cycles(wp, exec_start, n)
    else:
        cycles = _with_waits(_with_retreats(_departure_cycles(wp, fingers, exec_start, n), tcp, n), tcp, n)
    if task in SEATED_AT_SUCCESS_TASKS:
        cycles = _seated_at_success(cycles, tcp, success)
    return cycles


def _arrival_cycles(wp: list, exec_start: int, n: int) -> list[tuple]:
    """The motion that starts at one waypoint change ends at the waypoint recorded at the next change."""
    out, changes, last = [], [], None
    for i in range(exec_start, n):
        w = wp[i]
        if np.all(np.isfinite(w)) and (last is None or np.linalg.norm(w[:3] - last[:3]) >= MERGE_M):
            changes.append(i)
            last = w
    starts = [exec_start] + changes[:-1]
    for s, e in zip(starts, changes):
        out.append((max(s - 1, exec_start), e, wp[e][:6], _gripper_state(wp[e][6])))
    return out


def _departure_cycles(wp: list, fingers: list, exec_start: int, n: int) -> list[tuple]:
    """A cycle at each waypoint change. The expert opens the gripper in place before it records the next waypoint,
    under the old closed label, so that opening is its own cycle. It has the same waypoint, with the gripper open
    after."""
    out, last, opening = [], None, False
    for i in range(exec_start, n):
        w = wp[i]
        if not np.all(np.isfinite(w)):
            continue
        g = _gripper_state(w[6])
        if last is not None and np.linalg.norm(w[:3] - last[0][:3]) < MERGE_M and g == last[1] == -1 \
                and i > exec_start and fingers[i - 1] < FINGERS_OPEN_M and fingers[i] - fingers[i - 1] > OPENING_M:
            opening = True
        if opening:
            g = 1
        if last is None or np.linalg.norm(w[:3] - last[0][:3]) >= MERGE_M or g != last[1]:
            if last is not None and np.linalg.norm(w[:3] - last[0][:3]) >= MERGE_M:
                opening = False
                g = _gripper_state(w[6])
            out.append((max(i - 1, exec_start), i, w[:6], g))
            last = (w, g)
    return out


def _with_retreats(cycles: list, tcp: list, n: int) -> list:
    """Each cycle followed by the unrecorded move away from its waypoint (a lift or a shift after a release), if the
    expert made one before the next waypoint was recorded. The move starts where the TCP, having reached the
    waypoint, starts to leave, and its target is the pose it ends at."""
    out = []
    for j, (o, e, w, g) in enumerate(cycles):
        out.append((o, e, w, g))
        end = cycles[j + 1][1] if j + 1 < len(cycles) else n
        dist = [float(np.linalg.norm(np.asarray(tcp[i][:3]) - np.asarray(w[:3]))) for i in range(e, end)]
        arrived = next((k for k, d in enumerate(dist) if d < ARRIVED_M), None)
        left = None if arrived is None else next((k for k in range(arrived, len(dist)) if dist[k] > LEFT_M), None)
        if left is None:
            continue
        k = left
        while (k - 1 > arrived
               and np.linalg.norm(np.asarray(tcp[e + k][:3]) - np.asarray(tcp[e + k - 1][:3])) > MOVING_M):
            k -= 1
        out.append((e + k - 1, end - 1, list(np.asarray(tcp[end - 1][:6], dtype=float)), g))
    return out


def _seated_at_success(cycles: list, tcp: list, success: int | None) -> list:
    """The command active at the success timestep gets the TCP pose recorded there as its target."""
    if success is None:
        return cycles
    k = max((j for j, c in enumerate(cycles) if not isinstance(c[2], tuple) and c[1] <= success), default=None)
    if k is None:
        return cycles
    o, e, w, g = cycles[k]
    return cycles[:k] + [(o, e, list(np.asarray(tcp[success][:6], dtype=float)), g)] + cycles[k + 1:]


def _with_waits(cycles: list, tcp: list, n: int) -> list:
    """Each cycle followed by wait cycles where the expert, having arrived, holds still for at least WAIT_MIN steps
    (StopCube waits for the cube). One comes at the start of the wait and one every config.WAIT_EVERY steps after,
    each a wait until the next, so the policy sees the scene in between."""
    out = []
    for j, (o, e, w, g) in enumerate(cycles):
        out.append((o, e, w, g))
        nxt = cycles[j + 1][0] if j + 1 < len(cycles) else n - 1
        pos = [np.asarray(tcp[i][:3]) for i in range(e, nxt + 1)]
        still = next((k for k in range(len(pos) - 3)
                      if np.linalg.norm(pos[k] - np.asarray(w[:3])) < ARRIVED_M
                      and max(np.linalg.norm(pos[k + d] - pos[k]) for d in (1, 2, 3)) < STILL_M), None)
        if still is None:
            continue
        a = e + still
        if nxt - a < WAIT_MIN or any(np.linalg.norm(pos[i - e] - pos[still]) > HELD_M for i in range(a, nxt + 1)):
            continue
        for h in range(a, nxt - 2, config.WAIT_EVERY):
            k = min(config.WAIT_EVERY, nxt - h)
            out.append((h, h + k, ("wait", k), g))
    return out


def load(task: str, ep: int) -> dict:
    """The episode as cycles, its frames cached as PNG files. Built once, then read from the cache.

    Each cycle keeps `frames`, the recorded timesteps it shows, and `labels`, the step numbers the policy is told,
    which match what the test environment gives. The video cycle counts steps from the start of the demonstration.
    An act cycle counts control steps since the previous cycle, and the first shows only the current frame (0).
    """
    d = config.EPISODE_CACHE / task / f"ep{ep}"
    meta = d / "episode.json"
    if meta.exists():
        return json.loads(meta.read_text())
    d.mkdir(parents=True, exist_ok=True)
    with _h5(task) as f:
        g = f[f"episode_{ep}"]
        goal = g["setup/task_goal"][()]
        goal = (goal[0] if np.ndim(goal) else goal)
        goal = goal.decode() if isinstance(goal, bytes) else str(goal)
        ts = sorted((k for k in g if k.startswith("timestep_")), key=lambda k: int(k.split("_")[1]))
        demo = [i for i, k in enumerate(ts) if bool(g[k]["info/is_video_demo"][()])]
        exec_start = (demo[-1] + 1) if demo else 0
        wp = [np.asarray(g[k]["action/waypoint_action"][()], dtype=float) for k in ts]
        fingers = [float(np.mean(g[k]["obs/gripper_state"][()])) for k in ts]
        tcp = [np.asarray(g[k]["obs/eef_state"][()], dtype=float) for k in ts]
        need: set[int] = set()
        cycles: list[dict] = []
        if demo:
            shown = sample_steps(0, len(demo) - 1)
            frames = [demo[p] for p in shown]
            cycles.append({"kind": "video", "t": frames[-1], "frames": frames, "labels": shown})
            need.update(frames)
        prev = None
        success = (next((i for i, k in enumerate(ts) if bool(np.asarray(g[k]["info/is_completed"][()]).reshape(-1)[0])),
                        None) if task in SEATED_AT_SUCCESS_TASKS else None)
        acts = act_cycles(task, wp, fingers, tcp, exec_start, len(ts), success)
        first = acts[0][0] if acts else exec_start
        for o, e, w, grip in acts:
            if prev is None or o <= prev:
                frames, labels = [o], [0]
            else:
                frames = sample_steps(prev + 1, o)
                labels = [t - prev for t in frames]
            ob = g[ts[o]]["obs"]
            before = cycles[-1]["target"]["gripper_after"] if cycles and cycles[-1]["kind"] == "act" \
                else (1 if fingers[o] > FINGERS_OPEN_M else -1)
            state = robot_state(ob["eef_state"][()], ob["gripper_state"][()], o - first,
                                ob["wrist_camera_extrinsic"][()])
            target = ({"waypoint": None, "wait": w[1], "gripper_after": grip} if isinstance(w, tuple)
                      else {"waypoint": rounded(w), "gripper_after": grip})
            cycles.append({"kind": "act", "t": o, "frames": frames, "labels": labels, "target_t": e, "state": state,
                           "gripper_before": before, "target": target})
            need.update(frames)
            need.add(o)
            prev = o
        for i in sorted(need):                 # a keyframe is shown as its front and its wrist frame
            ob = g[ts[i]]["obs"]
            (d / f"front_{i}.png").write_bytes(png(ob["front_rgb"][()]))
            (d / f"wrist_{i}.png").write_bytes(png(ob["wrist_rgb"][()]))
        cal = calibration(g["setup/front_camera_intrinsic"][()], g[ts[exec_start]]["obs/front_camera_extrinsic"][()],
                          g["setup/wrist_camera_intrinsic"][()])
    for i, c in enumerate(cycles, 1):
        c["k"] = i
    out = {"task": task, "episode": ep, "goal": goal, "has_demo": bool(demo), "calibration": cal,
           "cycles": cycles}
    meta.write_text(json.dumps(out))
    return out


def frame_labels(cycle: dict) -> list[str]:
    """The label of each frame a cycle shows. V<step> counts steps from the start of the demonstration, and E<step>
    counts control steps since the execution started, as the robot state does. The rollouts label their frames the
    same way."""
    if cycle["kind"] == "video":
        return [f"V{s}" for s in cycle["labels"]]
    return exec_labels(cycle["state"]["elapsed_control_steps"], cycle["t"], cycle["frames"])


def find_frame(ep: dict, label: str, upto: int) -> int | None:
    """The recorded timestep of a frame label, if a cycle up to cycle `upto` showed it. The rollouts likewise know
    only the frames shown so far."""
    for c in ep["cycles"]:
        if c["k"] > upto:
            break
        for t, lab in zip(c["frames"], frame_labels(c)):
            if lab == label:
                return t
    return None


def front_png(ep: dict, t: int) -> bytes:
    return (config.EPISODE_CACHE / ep["task"] / f"ep{ep['episode']}" / f"front_{t}.png").read_bytes()


def wrist_png(ep: dict, t: int) -> bytes:
    return (config.EPISODE_CACHE / ep["task"] / f"ep{ep['episode']}" / f"wrist_{t}.png").read_bytes()
