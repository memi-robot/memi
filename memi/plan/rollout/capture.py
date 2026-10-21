"""The object records of every simulator step of a Plan episode, the demonstration's and the execution's, taken while
the step happens. `scene.describe` turns them into text observations.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .scene import arm_state

#: The subgoal of the demonstration steps the wrapper drops before it hands the demonstration over. Every step is
#: recorded and then filtered the same way, so the rows stay aligned with the demonstration.
NO_RECORD = "NO RECORD"

#: Actors that are never objects in the scene.
EXCLUDED_NAME_PARTS = ("panda", "robot", "finger", "gripper", "camera")
BACKGROUND_ACTOR_NAMES = {"ground", "table-workspace"}


def _to_list(x) -> list:
    """A tensor, array or sequence as a python list, with leading dimensions of size one removed."""
    if x is None:
        return []
    for attr in ("detach", "cpu"):
        if hasattr(x, attr):
            x = getattr(x, attr)()
    if hasattr(x, "tolist"):
        x = x.tolist()
    while isinstance(x, list) and len(x) == 1 and isinstance(x[0], list):
        x = x[0]
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _first(x):
    """The scalar behind a batched field. ManiSkill hands back shape-(1, ...)
    tensors even for a single env, and a subgoal string arrives as a one-item
    list."""
    v = x
    if hasattr(v, "tolist"):
        v = v.tolist()
    while isinstance(v, (list, tuple)) and v:
        v = v[0]
    return v


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _plain(value: Any) -> Any:
    """`value` with numpy scalars and arrays turned into python ones."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _actor_name(actor: Any) -> str:
    name = getattr(actor, "name", None)
    return type(actor).__name__ if name is None else str(name)


def _classify_actor(name: str, task: str | None) -> str | None:
    """The coarse type a detection is sorted by. The scene text does not read it."""
    normalized = name.lower()
    task_name = (task or "").lower()
    if "button" in normalized or "cap" in normalized:
        return "button"
    if "board_with_hole" in normalized:
        return "opaque_bin"
    if "box_with_hole" in normalized or "aperture" in normalized or "hole" in normalized:
        return "aperture_box"
    if "bin" in normalized:
        return "opaque_bin"
    if "container" in normalized or "cover" in normalized:
        return "container"
    if "highlight" in normalized or "area" in normalized or "disk" in normalized:
        return "highlighted_area"
    if "target" in normalized and "cube" not in normalized:
        return "target"
    if "goal_site" in normalized or "goalsite" in normalized or "goal" in normalized:
        return "target"
    if "peg" in normalized:
        if task_name == "movecube":
            return "stick_for_hooking"
        return "peg"
    if "stick" in normalized:
        if task_name == "routestick":
            return "stick_for_routing"
        return "stick_for_hooking"
    if "cube" in normalized:
        for color in ("red", "blue", "green"):
            if color in normalized:
                return f"{color} cube"
        return "cube"
    return None


def _pose(actor: Any) -> dict[str, list[float]] | None:
    pose = getattr(actor, "pose", None)
    if pose is None and hasattr(actor, "get_pose"):
        try:
            pose = actor.get_pose()
        except Exception:
            return None
    if pose is None:
        return None
    result: dict[str, list[float]] = {}
    for key in ("p", "q"):
        raw = getattr(pose, key, None)
        if raw is None:
            continue
        arr = _to_numpy(raw).reshape(-1)
        result["xyz" if key == "p" else "quat"] = [float(x) for x in arr if np.isfinite(x)]
    return result or None


def _chain(env: Any):
    """The env, every env it wraps, and the unwrapped env, each once."""
    seen: set[int] = set()
    current = env
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "env", None)
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and id(unwrapped) not in seen:
        yield unwrapped


def _segmentation_ids(env: Any) -> dict[int, Any] | None:
    """A segmentation id -> actor map, built from the live scene.

    ManiSkill caches `segmentation_id_map` after reconfigure. RoboMME creates
    highlight_disk actors during `step`, so the cached map can miss visible ids.
    """
    def actor_id(actor):
        try:
            return int(actor._objs[0].per_scene_id)
        except Exception:
            return None

    def link_id(link):
        try:
            return int(link._objs[0].entity.per_scene_id)
        except Exception:
            return None

    for layer in _chain(env):
        scene = getattr(layer, "scene", None)
        if scene is None:
            continue
        mapping: dict[int, Any] = {}
        actors = getattr(scene, "actors", None)
        for actor in (actors.values() if isinstance(actors, dict) else actors or []):
            seg_id = actor_id(actor)
            if seg_id is not None:
                mapping[seg_id] = actor
        arts = getattr(scene, "articulations", None)
        for art in (arts.values() if isinstance(arts, dict) else arts or []):
            for link in getattr(art, "links", []) or []:
                seg_id = link_id(link)
                if seg_id is not None:
                    mapping[seg_id] = link
        if mapping:
            return mapping
    for layer in _chain(env):
        if hasattr(layer, "segmentation_id_map"):
            cached = getattr(layer, "segmentation_id_map")
            return cached if isinstance(cached, dict) else None
    return None


def visible_objects(obs: dict, env: Any, task: str) -> list[dict[str, Any]]:
    """Every object with visible pixels in the front camera. Each has its name,
    segmentation ids, pixel box and centroid [y, x], pixel count and pose."""
    try:
        segmentation = obs["sensor_data"]["base_camera"]["segmentation"]
    except Exception:
        return []
    arr = _to_numpy(segmentation)
    if arr.ndim > 2:
        arr = arr[0]
    segmentation = np.asarray(arr).squeeze()
    id_map = _segmentation_ids(env)
    if not isinstance(id_map, dict):
        return []
    actor_to_ids: dict[int, tuple[Any, list[int]]] = {}
    for raw_id, actor in sorted(id_map.items()):
        try:
            seg_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        name = _actor_name(actor).lower()
        if name in BACKGROUND_ACTOR_NAMES or any(p in name for p in EXCLUDED_NAME_PARTS):
            continue
        actor_to_ids.setdefault(id(actor), (actor, []))[1].append(seg_id)

    detections = []
    for actor, ids in actor_to_ids.values():
        mask = np.isin(segmentation, ids)
        if not np.any(mask):
            continue
        ys, xs = np.nonzero(mask)
        name = _actor_name(actor)
        detections.append({
            "actor_name": name,
            "segmentation_ids": ids,
            "bbox_pixels_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "visible_centroid_yx": [int(ys.mean()), int(xs.mean())],
            "visible_pixel_count": int(ys.size),
            "actor_pose": _pose(actor),
            "object_type": _classify_actor(name, task) or "unknown_object",
        })
    detections.sort(key=lambda item: (str(item.get("object_type", "")),
                                      str(item["actor_name"]),
                                      tuple(item["segmentation_ids"])))
    return detections


class Capture:
    """One episode's scene records, taken as the simulator produces them.

    Poses are read from the live env, so they have to be taken at the step itself. A segmentation image replayed later
    would pair the right pixels with the poses of a later moment.
    """

    def __init__(self, task: str, env: Any):
        self.task = task
        self.env = env
        self.rows: list[dict] = []        # every sim step of the demonstration
        self.exec_rows: list[dict] = []   # every sim step of the execution
        self.phase = "watching"
        self._original_step = None
        self._wrapper = None
        self.errors: list[str] = []

    def _note(self, msg: str) -> None:
        if msg not in self.errors:
            self.errors.append(msg)

    def _arm(self, obs: dict) -> dict | None:
        """The gripper. The tool centre is projected with the frame's camera
        matrices, and the finger opening is the sum of the two finger joints."""
        try:
            agent = self.env.unwrapped.agent
            tcp = _to_list(agent.tcp.pose.p)[:3]
            qpos = _to_list(agent.robot.get_qpos())
            opening = float(qpos[-1] + qpos[-2])
            cam = obs["sensor_param"]["base_camera"]
            K = _to_list(cam["intrinsic_cv"])
            E = _to_list(cam["extrinsic_cv"])
            # `_to_list` keeps the rows of a batched matrix. A matrix that arrives flat is cut into rows.
            if K and not isinstance(K[0], list):
                K = [K[i * 3:(i + 1) * 3] for i in range(3)]
            if E and not isinstance(E[0], list):
                E = [E[i * 4:(i + 1) * 4] for i in range(3)]
            return arm_state(K, E, tcp, opening, opening < 0.02)
        except Exception as exc:
            self._note(f"arm: {type(exc).__name__}: {exc}")
            return None

    def record(self, obs: Any, info: Any, timestep: int, demo: bool) -> dict:
        """One row in the shape `scene.describe` reads."""
        row: dict = {"timestep": int(timestep), "is_video_demo": bool(demo)}
        try:
            row["visible_object_records"] = _plain(
                visible_objects(obs if isinstance(obs, dict) else {}, self.env, self.task))
        except Exception as exc:
            self._note(f"objects: {type(exc).__name__}: {exc}")
            row["capture_error"] = f"{type(exc).__name__}: {exc}"
            row["visible_object_records"] = []
        if isinstance(obs, dict):
            row["arm"] = self._arm(obs)
        # Read by the NO_RECORD filter.
        sub = _first(info.get("simple_subgoal_online")) if isinstance(info, dict) else None
        row["subgoal_online"] = sub if isinstance(sub, str) else None
        g = row.get("arm")
        row["gripper"] = "open" if g is None or g["opening"] > 0.02 else "closed"
        return row

    def _demo_wrapper(self):
        """The DemonstrationWrapper in the chain, found by the methods that identify it.

        The builder inserts other wrappers depending on the action space, and
        always a FailAwareWrapper on top, so its depth is not fixed.
        """
        env, seen = self.env, 0
        while env is not None and seen < 12:
            if hasattr(env, "_step_batch") and hasattr(env, "_augment_obs_and_info"):
                return env
            env = getattr(env, "env", None)
            seen += 1
        return None

    def install(self) -> None:
        """Shadow `_step_batch` on the demonstration wrapper for this episode.

        The demonstration is generated inside `reset()`, so it can only be recorded from inside the wrapper's step.
        `_step_batch` returns the row after `_augment_obs_and_info` has added `simple_subgoal_online`, the field the
        demonstration is filtered on. The innermost env's `step` lacks it. The instance attribute wins over the class
        method, so every path lands here, and `remove` puts the object back as it was. Patching the class would
        outlive the episode.
        """
        if self._original_step is not None:
            return
        w = self._demo_wrapper()
        if w is None:
            self._note("no DemonstrationWrapper in the chain")
            return
        self._wrapper, self._original_step = w, w._step_batch
        original = self._original_step

        def stepped(action, *a, **kw):
            # At termination `_step_batch` calls itself once more for a last frame that the caller drops. The flag
            # is set for that whole inner call, so it is read before the call.
            extra = bool(getattr(w, "_doing_extra_step", False))
            out = original(action, *a, **kw)
            if not extra:
                try:
                    # The batch's obs carries no segmentation, so the scene is
                    # rendered again here.
                    raw = self.env.unwrapped.get_obs()
                    demo = self.phase == "watching"
                    rows = self.rows if demo else self.exec_rows
                    rows.append(self.record(raw, out[-1], len(rows), demo))
                except Exception as exc:      # capture must never kill a rollout
                    self._note(f"hook: {type(exc).__name__}: {exc}")
            return out

        w._step_batch = stepped

    def remove(self) -> None:
        if self._original_step is not None and self._wrapper is not None:
            try:
                del self._wrapper._step_batch
            except AttributeError:
                self._wrapper._step_batch = self._original_step
        self._original_step = self._wrapper = None

    def start_execution(self) -> None:
        """From here on, recorded steps are execution steps. The hook stays on,
        because the planner's commands step the env inside the wrapper."""
        self.phase = "execution"
