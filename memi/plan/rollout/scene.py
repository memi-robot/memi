"""A step's text observation (blog Section 3.1), every visible object with its type and its position in the front
camera image. It states only what a person looking at the frame could state, so what the frame hides is left to memory.
"""
from __future__ import annotations

import math
import re
from typing import Any

SAME_PLACE_PX = 10.0  # below this two coordinates are the same place
# A centroid that moves less than this keeps its position. It is small because the policy's point is executed, and a
# position held longer would send the robot where the object was a moment ago.
SHOW_PX = 2.0
MOVE_M = 0.02      # below this a change of pose is jitter, not motion
HOLD_M = 0.06      # an object pose this close to the gripper's is in the hand
HOLD_PX = 26       # the same test in pixels, when poses are not recorded
GRIP_MIN = 5       # percent open. Below this the fingers are shut on nothing
GRIP_MAX = 90      # percent open. Above this they are open
LIFT_M = 0.05      # metres above an object's own resting height
PRESS_M = 0.002    # a cap that never travels this far was never pressed

# Drawn overlays, not things. A highlight is painted at the position it marks,
# so it inherits the height of whatever it is drawn on. Nothing here can be
# lifted, held or pressed.
MARKERS = {"target", "waypoint", "highlighted area", "highlighted target"}
HIGHLIGHTS = {"highlighted area", "highlighted target"}
#: Labels whose markers collapse to one, the one laid down last.
REDUCE_TO_FRONT = {"waypoint"}
# Tasks where position_highlight markers trace the path to follow, so they
# become a waypoint instead of being dropped as a static grid.
WAYPOINT_TASKS = {"routestick", "patternlock"}

# The gripper reads 0.04 per finger fully open on every task.
FINGER_OPEN = 0.04
IMAGE = 256        # the frame is 256 x 256. Outside it there is nothing to name.


def classify_object_type(actor_name: Any, task: str | None) -> str | None:
    """Map a raw simulator actor name to the object type the text uses, or None
    for an actor that is not named at all (the button base among them).

    peg_head and peg_tail are shared by InsertPeg (the peg) and MoveCube (the
    hooking stick), so the task name disambiguates them.
    """
    if not isinstance(actor_name, str):
        return None
    name = actor_name.lower()
    task_name = (task or "").lower()
    if not name:
        return None
    # Buttons. Only the pressable cap counts, not the base.
    if "button_cap" in name:
        return "button"
    if "button_base" in name:
        return None
    if "peg_head" in name or "peg_tail" in name:
        part = "head" if "head" in name else "tail"
        if task_name == "movecube":
            return f"stick for hooking {part}"
        return f"peg {part}"
    # Cubes, colour aware, as cube_*, target_cube_* and fixed_cube.
    if "cube" in name:
        for color in ("red", "blue", "green"):
            if color in name:
                return f"{color} cube"
        return "cube"
    # MoveCube's goal marker is a target.
    if "goal_site" in name:
        return "target"
    if "target" in name:
        return "target"
    # Receptacles. Order matters, since board_with_hole also contains "hole".
    if "board_with_hole" in name:
        return "opaque bin"
    if "box_with_hole" in name or "aperture" in name or "hole" in name:
        return "aperture box"
    if "bin" in name or "container" in name or "cover" in name:
        return "container"
    if "position_highlight" in name:
        return "waypoint" if task_name in WAYPOINT_TASKS else None
    if "highlight_disk" in name:
        return "highlighted target" if task_name in WAYPOINT_TASKS else "highlighted area"
    return None


def _actor_index(actor_name: Any) -> int:
    match = re.search(r"(\d+)\s*$", str(actor_name))
    return int(match.group(1)) if match else -1


def _front_point(items: list[tuple[Any, list[int]]]) -> list[int]:
    """The marker with the highest index, which is the one laid down last."""
    return max(items, key=lambda item: _actor_index(item[0]))[1]


def _records(row: dict) -> list[dict]:
    v = row.get("visible_object_records")
    return v if isinstance(v, list) else []


def _covers(disk: dict, cur: dict, task: str, targets: set = frozenset()) -> str:
    """What to call a highlight, named after the thing under it.

    A disk drawn over a target is a highlighted target. One drawn beside a cube
    is a highlighted area. Targets do not move, so a position that was ever a
    target still is one, even while the disk covers it.
    """
    p = tuple(disk.get("visible_centroid_yx", [0, 0]))
    if any(math.dist(p, q) <= SAME_PLACE_PX for q in targets):
        return "highlighted target"
    near = [(math.dist(p, tuple(o["visible_centroid_yx"])), lab)
            for o in cur.values()
            for lab in [classify_object_type(o.get("actor_name"), task)]
            if lab and lab not in HIGHLIGHTS and lab not in REDUCE_TO_FRONT
            and math.dist(p, tuple(o["visible_centroid_yx"])) <= SAME_PLACE_PX]
    if near and min(near)[1] == "target":
        return "highlighted target"
    return "highlighted area"


def _pts(pts: list[str]) -> str:
    """Coordinates in one fixed order."""
    return ", ".join(sorted(pts, key=lambda p: tuple(
        int(v) for v in p.strip("<>").split(", "))))


def _merge_ends(plain: dict[str, list[str]]) -> dict[str, list[str]]:
    """MoveCube's hooking stick is one object recorded as two ends.

    A peg's ends are not merged. InsertPeg asks the robot to grasp the same end
    it grasped before, so which is which is the task.
    """
    heads = [k for k in plain if k.startswith("stick for hooking")]
    if len(heads) < 2:
        return plain
    pts = [tuple(int(v) for v in p.strip("<>").split(", "))
           for k in heads for p in plain[k]]
    out = {k: v for k, v in plain.items() if k not in heads}
    out["stick for hooking"] = [f"<{sum(p[0] for p in pts) // len(pts)}, "
                                f"{sum(p[1] for p in pts) // len(pts)}>"]
    return out


def project(K, E, xyz) -> tuple[int, int]:
    """World metres -> image <y, x>, with the camera's intrinsic K and extrinsic E."""
    cam = [sum(E[r][c] * (list(xyz) + [1.0])[c] for c in range(4)) for r in range(3)]
    pix = [sum(K[r][c] * cam[c] for c in range(3)) for r in range(3)]
    if abs(pix[2]) < 1e-9:
        return (0, 0)
    return (int(round(pix[1] / pix[2])), int(round(pix[0] / pix[2])))


def arm_state(K, E, tcp_xyz, opening: float, closed: bool) -> dict:
    """The gripper at one step. Where the hand is in the image and in metres, and
    how far the fingers are apart (the sum of the two finger joints)."""
    return {"at": project(K, E, tcp_xyz),
            "xyz": tuple(float(v) for v in tcp_xyz[:3]),
            "height": float(tcp_xyz[2]),
            "opening": float(opening),
            "closed": bool(closed)}


def percent_open(state: dict) -> int:
    """Fingers as a percentage of fully open."""
    return round(100 * state["opening"] / (2 * FINGER_OPEN))


def gripper_line(state: dict | None) -> str:
    """One line for the gripper.

    A gripper raised clear of the table projects outside the image, and is
    said in words. The finger opening tells a grasp that took hold (the
    fingers stop at the object's width) from one closed on nothing (0%).
    """
    if not state:
        return ""
    y, x = state["at"]
    inside = 0 <= y < IMAGE and 0 <= x < IMAGE
    if not inside:
        return f"    gripper raised out of the frame  fingers {percent_open(state)}% open"
    return (f"    gripper at <{y}, {x}>  height {state['height']:.3f}  "
            f"fingers {percent_open(state)}% open")


def describe(rows: list[dict], task: str, robot: dict,
             move_px: float = SHOW_PX) -> dict[tuple[bool, int], str]:
    """(is_demo, timestep) -> the text observation of that step.

    A row is one simulator step with `timestep`, `is_video_demo`, `visible_object_records` (per object segmentation,
    pixel centroid and pose) and `gripper` ("open" or "closed"). `robot` maps a timestep to `arm_state`, from the
    simulator in a rollout and from the recording in the dataset builder. Every statistic is computed over the rows
    given. Callers pass the part of the episode that has happened, so no step is described with a later frame. `rows`
    is sorted in place by timestep.

    Each object is named by type and place, never by simulator identity. Its position is held until it moves more than
    `move_px`, which keeps jitter out. A state clause says it is held (the fingers are shut on it, the hand is beside
    it and it is not where it started), lifted (well above its resting height) or pressed (a button cap pushed down
    into its base).
    """
    # By timestep alone. The demonstration occupies the low timesteps and
    # execution continues from where it stops.
    rows.sort(key=lambda r: int(r["timestep"]))

    # Resting height per object, the lowest pose it is ever seen at.
    rest: dict[str, float] = {}
    for r in rows:
        for o in _records(r):
            z = o.get("actor_pose", {}).get("xyz", [0, 0, 0])[2]
            name = o["actor_name"]
            rest[name] = min(rest.get(name, z), z)

    # Every position a target ever holds. At the moment a disk lights a target
    # the target is under it and out of the current list.
    target_spots = {tuple(o.get("visible_centroid_yx", [0, 0]))
                    for r in rows for o in _records(r)
                    if classify_object_type(o.get("actor_name"), task) == "target"}

    # Where each object was first seen and how far it ever rose or moved.
    start_px: dict[str, tuple[int, int]] = {}
    start_pose: dict[str, tuple] = {}
    travel: dict[str, float] = {}
    moved_pose: dict[str, float] = {}
    moved_px: dict[str, float] = {}
    for r in rows:
        for o in _records(r):
            n = o["actor_name"]
            c = tuple(o.get("visible_centroid_yx", [0, 0]))
            pose = tuple(o.get("actor_pose", {}).get("xyz", [0, 0, 0]))
            start_px.setdefault(n, c)
            start_pose.setdefault(n, pose)
            travel[n] = max(travel.get(n, 0.0), pose[2] - rest.get(n, pose[2]))
            moved_pose[n] = max(moved_pose.get(n, 0.0), math.dist(pose, start_pose[n]))
            moved_px[n] = max(moved_px.get(n, 0.0), math.dist(c, start_px[n]))
    # A pose that never updates while the object visibly moves is not recorded,
    # and its centroid has to answer instead.
    unreliable = {n for n in start_pose
                  if moved_pose.get(n, 0.0) <= 1e-9 and moved_px.get(n, 0.0) > SAME_PLACE_PX}

    stable: dict[str, tuple[int, int]] = {}
    out: dict[tuple[bool, int], str] = {}
    for r in rows:
        by_label: dict[str, list[tuple[str, tuple[int, int], str]]] = {}
        held_at: dict[str, float] = {}
        for o in _records(r):
            name = o["actor_name"]
            label = classify_object_type(name, task)
            if label is None:
                continue
            if label in HIGHLIGHTS:
                label = _covers(o, {x["actor_name"]: x for x in _records(r)},
                                task, target_spots)
            cy, cx = o.get("visible_centroid_yx", [0, 0])
            prev = stable.get(name)
            if prev is None or math.dist(prev, (cy, cx)) > move_px:
                stable[name] = (cy, cx)
            y, x = stable[name]
            z = o.get("actor_pose", {}).get("xyz", [0, 0, 0])[2]
            # Held, then lifted, never about a marker.
            note = ""
            arm_now = robot.get(int(r["timestep"]))
            if arm_now is not None and label not in MARKERS:
                pct = percent_open(arm_now)
                # Pose when it is recorded, pixels when it is not. An arm
                # passing in front of a thing is close in pixels and far in
                # metres.
                pose = tuple(o.get("actor_pose", {}).get("xyz", [0, 0, 0]))
                near = (math.dist((cy, cx), tuple(arm_now["at"])) < HOLD_PX
                        if name in unreliable
                        else math.dist(pose, tuple(arm_now["xyz"])) < HOLD_M)
                # Not where it started, which separates the thing in the hand
                # from a thing the hand is carrying another past.
                disturbed = (math.dist((cy, cx), start_px.get(name, (cy, cx))) > SAME_PLACE_PX
                             if name in unreliable
                             else math.dist(pose, start_pose.get(name, pose)) > MOVE_M)
                if GRIP_MIN <= pct <= GRIP_MAX and near and disturbed:
                    note = " is held in the gripper"
                    held_at[name] = math.dist((cy, cx), tuple(arm_now["at"]))
            if not note and z - rest.get(name, z) > LIFT_M and label not in MARKERS:
                note = " is lifted off the table"
            if (name.endswith("_cap") and travel.get(name, 0.0) > PRESS_M
                    and z - rest.get(name, z) < 1e-6):
                note = " is pressed down"
            by_label.setdefault(label, []).append((name, (y, x), note))
        # One thing is in the hand. Both ends of a peg can fall inside the
        # radius, so the nearest to the hand keeps the clause.
        if len(held_at) > 1:
            keep = min(held_at, key=held_at.get)
            by_label = {lab: [(n, p, "" if n in held_at and n != keep else note)
                              for n, p, note in items]
                        for lab, items in by_label.items()}
        parts, by_kind = [], {}
        for label, items in by_label.items():
            if label in REDUCE_TO_FRONT:
                pt = _front_point([(n, list(p)) for n, p, _ in items])
                parts.append(f"{label} at <{pt[0]}, {pt[1]}>")
                continue
            # One line per kind, coordinates after it.
            plain = [f"<{y}, {x}>" for _, (y, x), note in items if not note]
            if plain:
                by_kind[label] = plain
            # Anything with a state clause keeps its own entry, so the clause
            # says which coordinate it is about.
            for _, (y, x), note in items:
                if note:
                    parts.append(f"{label} at <{y}, {x}>{note}")
        for label, pts in _merge_ends(by_kind).items():
            parts.append(f"{label} at " + _pts(pts))
        # The gripper first. Without an arm state the row's open/closed word is used.
        arm = robot.get(int(r["timestep"]))
        gripper = (gripper_line(arm) if arm else
                   "    gripper is " + (r.get("gripper", "") or "in an unknown state"))
        key = "\n".join([gripper] + ["    " + p for p in sorted(parts)])
        out[(bool(r.get("is_video_demo")), int(r["timestep"]))] = key
    return out
