"""Build the Plan training data, data/decisions/<Task>.jsonl, from the RoboMME recordings in ROBOMME_H5_DIR and the
frame records.

    python -m memi.plan.data.build_decisions --tasks PickXtimes --frames-dir <frame records> --out data/decisions

The frame records (`<Task>_ep<N>.jsonl`, the visible objects at every simulator step) are not part of the release, so
this module documents how data/decisions was made. Each episode becomes one decision step per subgoal of its execution,
with the segment since the previous step, the recorded command and the objects its point can select.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

from ...paths import H5_DIR, resolve
from ..rollout import segments
from ..rollout.scene import SHOW_PX, arm_state, describe
from . import candidates

DEV = 10            # development episodes per task, out of 100


def split(task: str, episodes: list[int]) -> dict[int, str]:
    """A fixed 90/10 split, decided by a hash of the task and episode only."""
    order = sorted(episodes, key=lambda e: hashlib.sha256(f"{task}:{e}".encode()).hexdigest())
    dev = set(order[:DEV])
    return {e: ("dev" if e in dev else "train") for e in episodes}


def _text(x) -> str:
    """A string field of the recording as str."""
    x = x[()]
    if isinstance(x, bytes):
        return x.decode()
    if hasattr(x, "tolist"):
        x = x.tolist()
        if isinstance(x, list) and x:
            x = x[0]
        if isinstance(x, bytes):
            return x.decode()
    return str(x)


def arm_states(g) -> dict[int, dict]:
    """timestep -> the gripper's state, from one episode group of the recording."""
    import numpy as np
    K = np.array(g["setup"]["front_camera_intrinsic"]).tolist()
    out = {}
    for key in g:
        if not key.startswith("timestep_"):
            continue
        obs = g[key]["obs"]
        eef = np.array(obs["eef_state"]).tolist()
        E = np.array(obs["front_camera_extrinsic"]).tolist()
        fingers = np.array(obs["gripper_state"]).tolist()
        out[int(key.split("_")[1])] = arm_state(
            K, E, eef[:3], float(sum(fingers)), bool(np.array(obs["is_gripper_close"])))
    return out


def _nearest(records: list[dict], point) -> dict | None:
    """The visible object whose centroid is nearest `point`, kept for reference."""
    if not point:
        return None
    best = None
    for r in records:
        c = r.get("visible_centroid_yx")
        if not c:
            continue
        d = (c[0] - point[0]) ** 2 + (c[1] - point[1]) ** 2
        if best is None or d < best[0]:
            best = (d, r)
    if best is None:
        return None
    r = best[1]
    return {"actor": r.get("actor_name"), "type": r.get("object_type"),
            "at": r.get("visible_centroid_yx"), "distance": round(best[0] ** 0.5, 1)}


def _objects(records: list[dict]) -> list[dict]:
    """The visible objects at a step, kept for reference."""
    return [{"actor": r.get("actor_name"), "type": r.get("object_type"),
             "at": r.get("visible_centroid_yx")}
            for r in records if r.get("visible_centroid_yx")]


def _menu(task: str, menu: list) -> list:
    """The menu with `need_parameter` as the wrapper requires it, from `candidates.TARGET_PATTERNS`."""
    out = []
    for o in menu:
        takes = candidates.target_pattern(task, o.get("label"), o.get("action")) is not None
        out.append({**o, "need_parameter": takes})
    return out


def episode_steps(task: str, ep: int, h5, frames_dir: Path, report: dict | None = None) -> dict:
    """One episode's record of data/decisions, without its split. `report` sums the candidate and label counts."""
    group = h5[f"episode_{ep}"]
    goal = json.loads(_text(group["setup/task_goal"])) if _text(group["setup/task_goal"]).startswith("[") \
        else _text(group["setup/task_goal"])
    goal = goal[0] if isinstance(goal, list) else goal
    menu = _menu(task, json.loads(_text(group["setup/available_multi_choices"])))
    intrinsic = group["setup/front_camera_intrinsic"][()].tolist()
    keys = sorted((k for k in group if k.startswith("timestep_")), key=lambda k: int(k.split("_")[1]))
    video = [int(k.split("_")[1]) for k in keys if bool(group[k]["info/is_video_demo"][()])]
    bounds = []
    for k in keys:
        s = group[k]
        if bool(s["info/is_video_demo"][()]) or not bool(s["info/is_subgoal_boundary"][()]):
            continue
        cmd = json.loads(_text(s["action/choice_action"]))
        if cmd.get("choice"):
            bounds.append((int(k.split("_")[1]), cmd))
    # The last boundary is recorded after the task has succeeded, so it is not a decision.
    bounds = bounds[:-1]

    rows = [json.loads(l) for l in open(frames_dir / f"{task}_ep{ep}.jsonl")]
    rows.sort(key=lambda r: int(r["timestep"]))
    assert len(rows) == len(keys), f"{task} ep{ep}: {len(rows)} frame rows, {len(keys)} h5 steps"
    robot = arm_states(group)

    # The rollout's rendering, from the part of the episode that has happened, so an offline step reads like a
    # rollout call.
    def render(start: int, end: int) -> tuple[str, list[int]]:
        feats = describe(segments.as_seen(rows[:end], task), task, robot, move_px=SHOW_PX)
        picked = segments.picked(start, end)
        steps = [(t, feats.get((bool(rows[t].get("is_video_demo")), t), "")) for t in picked]
        return segments.render(steps), picked

    out, prev = [], 0
    demo_seg, demo_picked = "", []
    if video:
        demo_seg, demo_picked = render(0, len(video))
        prev = len(video)
    report = report if report is not None else {}
    for i, (t, cmd) in enumerate(bounds):
        segment, picked = render(prev, t + 1)
        shown_demo = demo_picked if i == 0 else []
        if shown_demo:
            segment = segments.with_demo(demo_seg, segment)
        point = cmd.get("point") or []
        records = rows[t].get("visible_object_records") or []
        obs = group[f"timestep_{t}/obs"]
        camera = {"intrinsic": intrinsic,
                  "extrinsic": obs["front_camera_extrinsic"][()].tolist(),
                  "image_shape": list(obs["front_rgb"].shape[:2])}
        # Each option that takes a point keeps its candidates, so the loop selects the object of a predicted point
        # as the wrapper would. A point is graded by that object, not by its pixels.
        by_label = {}
        for o in menu:
            pattern = candidates.target_pattern(task, o["label"])
            if pattern is None:
                continue
            got, counts = candidates.at_step(rows, t, pattern)
            for c in got:
                c["pixel_xy"] = candidates.project(c["xyz"], camera)
            by_label[o["label"]] = got
            for k, v in counts.items():
                report[k] = report.get(k, 0) + v
        choice = str(cmd["choice"]).strip().lower()
        matched = None
        if choice in by_label:
            matched = candidates.select(by_label[choice], point, camera)
            near = _nearest(records, point)
            report["points"] = report.get("points", 0) + 1
            report["no_match"] = report.get("no_match", 0) + (matched is None)
            report["same_as_nearest_visible"] = report.get("same_as_nearest_visible", 0) + bool(
                matched and near and near["actor"] == matched["name"])
            if matched:
                report.setdefault("match_distance", []).append(matched["distance"])
                report.setdefault("matched", {}).setdefault(choice, {})
                kind = re.sub(r"\d+", "#", matched["name"])
                report["matched"][choice][kind] = report["matched"][choice].get(kind, 0) + 1
            else:
                report.setdefault("no_match_at", []).append([ep, t])
        elif point:
            report["point_without_rule"] = report.get("point_without_rule", 0) + 1
        out.append({"kind": "decide", "k": i, "t": t, "segment": segment, "shown": picked,
                    "demo_shown": shown_demo,
                    "label": {"choice": cmd["choice"], "point": point, "object": matched,
                              "nearest_visible": _nearest(records, point)},
                    "candidates": by_label, "camera": camera,
                    "objects": _objects(records)})
        prev = t + 1
    return {"task": task, "episode": ep, "goal": goal, "menu": menu,
            "steps_total": len(keys), "video_steps": len(video), "steps": out}


def main(argv=None) -> int:
    import h5py
    ap = argparse.ArgumentParser(prog="python -m memi.plan.data.build_decisions", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--episodes", nargs="*", type=int, default=None)
    ap.add_argument("--out", default="data/decisions")
    ap.add_argument("--frames-dir", default="data/robomme_frame_records")
    a = ap.parse_args(argv)
    out, frames = resolve(a.out), resolve(a.frames_dir)
    out.mkdir(parents=True, exist_ok=True)
    for task in a.tasks:
        with h5py.File(H5_DIR / f"record_dataset_{task}.h5", "r") as f:
            all_eps = sorted(int(k.split("_")[1]) for k in f if k.startswith("episode_"))
            eps = a.episodes if a.episodes is not None else all_eps
            sp = split(task, all_eps)
            rep: dict = {}
            with open(out / f"{task}.jsonl", "w") as fh:
                for e in eps:
                    rec = episode_steps(task, e, f, frames, rep)
                    rec["split"] = sp[e]
                    fh.write(json.dumps(rec) + "\n")
        dist = rep.get("match_distance", [])
        print(f"[decisions] {task}: candidates now {rep.get('now', 0)}, from earlier frames "
              f"{rep.get('earlier', 0)}, never seen yet {rep.get('never', 0)}. Label points "
              f"{rep.get('points', 0)}, no match {rep.get('no_match', 0)}, same object as the "
              f"nearest visible {rep.get('same_as_nearest_visible', 0)}, match distance median "
              f"{statistics.median(dist) if dist else None} and max {max(dist) if dist else None}. "
              f"Points on options that take none {rep.get('point_without_rule', 0)}. Matched "
              f"{rep.get('matched', {})}", flush=True)
        if rep.get("no_match_at"):
            print(f"[decisions] {task}: no candidate matched the label point at [episode, t] "
                  f"{rep['no_match_at']}", flush=True)
        train = sum(sp[e] == "train" for e in eps)
        print(f"[decisions] {task}: {len(eps)} episodes, {train} train and {len(eps) - train} dev -> "
              f"{out / (task + '.jsonl')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
