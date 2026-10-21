"""The objects a menu option's point can select, and the one the planner's wrapper selects for a point. The loop
grades a predicted point by that object.
"""
from __future__ import annotations

import functools
import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ...paths import ROOT

IMAGE_SHAPE = (256, 256)

CUBE = r"cube_(?:red|green|blue)_\d+"
BIN = r"bin_\d+"
TARGET = r"target_\d+"
PEG = r"peg_(?:head|tail)"

#: task -> label -> (action, actor name pattern), for every option with an `available` list. RoboMME sets `available` in
#: robomme_env/utils/vqa_options.py from fixed env attributes, and the pattern matches those actors' names in the frame
#: records. An option not listed has no `available` and takes no point. A task missing here is an error. Each comment
#: gives the vqa_options.py line, then where the env names the actors.
TARGET_PATTERNS: dict[str, dict[str, tuple[str, str]]] = {
    # vqa_options.py:83 env.all_cubes. BinFill.py:328 cube_<colour>_<idx>.
    "BinFill": {"a": ("pick up the cube", CUBE)},
    # vqa_options.py:131 env.spawned_bins. ButtonUnmask.py:200 bin_<i>.
    "ButtonUnmask": {"b": ("pick up the container", BIN)},
    # vqa_options.py:174 env.spawned_bins. ButtonUnmaskSwap.py:282 bin_<i>.
    "ButtonUnmaskSwap": {"c": ("pick up the container", BIN)},
    # vqa_options.py:197 env.peg_heads + env.peg_tails. InsertPeg.py:185 builds
    # 3 pegs. object_generation.py:75 and :90 name every link peg_head / peg_tail.
    "InsertPeg": {"a": ("pick up the peg by grasping one end", PEG)},
    # vqa_options.py:233-300, no option has `available`.
    "MoveCube": {},
    # vqa_options.py:303-457, no option has `available`.
    "PatternLock": {},
    # vqa_options.py:614 env.all_cubes. PickHighlight.py:210 cube_<colour>_<idx>.
    "PickHighlight": {"b": ("pick up the highlighted cube", CUBE)},
    # vqa_options.py:638 env.all_cubes. PickXtimes.py:231 cube_<colour>_<idx>.
    "PickXtimes": {"a": ("pick up the cube", CUBE)},
    # vqa_options.py:459-584, no option has `available`.
    "RouteStick": {},
    # vqa_options.py:790-881, no option has `available`.
    "StopCube": {},
    # vqa_options.py:669 env.all_cubes. SwingXtimes.py:228 cube_<colour>_<idx>.
    # vqa_options.py:685 [env.target_right, env.target_left]. SwingXtimes.py:259
    # and :280 temp_target_0 / temp_target_1.
    "SwingXtimes": {"a": ("pick up the cube", CUBE),
                    "b": ("move to the top of the target", r"temp_target_[01]")},
    # vqa_options.py:758 env.all_cubes. VideoPlaceButton.py:252 cube_<colour>_<idx>.
    # vqa_options.py:771 env.targets. VideoPlaceButton.py:287 target_<i>.
    "VideoPlaceButton": {"a": ("pick up the cube", CUBE), "b": ("drop onto", TARGET)},
    # vqa_options.py:720 env.all_cubes. VideoPlaceOrder.py:253 cube_<colour>_<idx>.
    # vqa_options.py:733 env.targets. VideoPlaceOrder.py:286 target_<i>.
    "VideoPlaceOrder": {"a": ("pick up the cube", CUBE), "b": ("drop onto", TARGET)},
    # vqa_options.py:51 env.spawned_cubes. VideoRepick.py:221 cube_<colour>_<idx>
    # on hard, VideoRepick.py:271 bin_<i> otherwise (cubes named bin_<i>).
    "VideoRepick": {"a": ("pick up the cube", rf"{CUBE}|{BIN}")},
    # vqa_options.py:891 env.spawned_bins. VideoUnmask.py:182 bin_<i>.
    "VideoUnmask": {"a": ("pick up the container", BIN)},
    # vqa_options.py:910 env.spawned_bins. VideoUnmaskSwap.py:212 bin_<i>.
    "VideoUnmaskSwap": {"a": ("pick up the container", BIN)},
}


def target_pattern(task: str, label: str, action: str | None = None) -> str | None:
    """The actor name pattern of `label`'s `available`, or None when the option
    has none. Raises for a task without patterns or an action that does not match."""
    if task not in TARGET_PATTERNS:
        raise KeyError(f"no target patterns for task {task!r}")
    got = TARGET_PATTERNS[task].get(str(label).strip().lower())
    if got is None:
        return None
    if action is not None and action != got[0]:
        raise ValueError(f"{task} option {label!r} is {action!r} here and "
                         f"{got[0]!r} in TARGET_PATTERNS. The table is out of date.")
    return got[1]


MAPPING = Path("robomme_env") / "utils" / "choice_action_mapping.py"


def _benchmark() -> Path:
    """The `robomme` package directory, found without importing it.

    The installed package if there is one, else the submodule. Raises when
    neither holds the mapping file.
    """
    spec = importlib.util.find_spec("robomme")
    places = []
    if spec is not None and spec.submodule_search_locations:
        places.append(Path(list(spec.submodule_search_locations)[0]))
    places.append(ROOT / "third_party" / "robomme_benchmark" / "src" / "robomme")
    for p in places:
        if (p / MAPPING).exists():
            return p
    raise FileNotFoundError(
        f"the RoboMME benchmark is missing. The loop grades points with its "
        f"{MAPPING.as_posix()}, and neither an installed robomme package nor "
        f"third_party/robomme_benchmark/src has it. Run `git submodule update --init` "
        f"in {ROOT}.")


@functools.lru_cache(maxsize=1)
def _mapping():
    """The benchmark's choice_action_mapping, loaded by path. It needs numpy only.

    Importing it through the package would import the simulator too.
    """
    path = _benchmark() / MAPPING
    spec = importlib.util.spec_from_file_location("_robomme_choice_action_mapping", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wrapper_pixel(point) -> list[int] | None:
    """[y, x] as the wrapper's `_resolve_command` hands it on, [x, y] rounded."""
    try:
        y, x = float(point[0]), float(point[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return [int(np.rint(x)), int(np.rint(y))]


def project(xyz, camera: dict) -> list[int] | None:
    """The benchmark's projection, [x, y], or None out of the image."""
    return _mapping().project_world_to_pixel(
        xyz, intrinsic_cv=camera["intrinsic"], extrinsic_cv=camera["extrinsic"],
        image_shape=camera.get("image_shape") or IMAGE_SHAPE)


class _Actor:
    """A recorded candidate in the shape the benchmark reads, with `.name` and
    `.pose.p`. Hashed by identity, as the benchmark de-duplicates actors."""

    def __init__(self, cand: dict):
        self.name, self.pose, self.cand = cand["name"], SimpleNamespace(p=cand["xyz"]), cand


def select(cands: list[dict], point, camera: dict) -> dict | None:
    """The candidate the wrapper selects for `point` ([y, x]), or None.

    `cands` are {"name", "seg", "xyz"}. They go to the benchmark's own `select_target_with_pixel` as actors with that
    name and pose, in list order, so no math is reimplemented. It projects each candidate into the front camera, skips
    those outside the image and takes the one nearest the point.
    """
    pixel = wrapper_pixel(point)
    if pixel is None or not cands:
        return None
    actors = [_Actor(c) for c in cands]
    got = _mapping().select_target_with_pixel(
        available=actors, pixel_like=pixel, intrinsic_cv=camera["intrinsic"],
        extrinsic_cv=camera["extrinsic"], image_shape=camera.get("image_shape") or IMAGE_SHAPE)
    if got is None:
        return None
    c = got["obj"].cand
    return {"name": c["name"], "seg": c.get("seg"), "pixel_xy": got["projected_pixel"],
            "distance": round(float(got["match_distance"]), 2)}


def key(c: dict | None):
    """A candidate's identity, its name and segmentation id."""
    return None if c is None else (c.get("name"), c.get("seg"))


def _seg(rec: dict):
    ids = rec.get("segmentation_ids") or []
    return int(ids[0]) if ids else None


def at_step(rows: list[dict], t: int, pattern: str) -> tuple[list[dict], dict]:
    """The candidates matching `pattern` at row `t`, and where their poses came from.

    A candidate's pose is its record at `t` when it is visible there, else its
    most recent earlier record. One never seen up to `t` (but seen later in the
    episode) is skipped. Returns (candidates, {"now", "earlier", "never"}).
    """
    rx = re.compile(rf"^(?:{pattern})$")
    last: dict = {}
    universe: set = set()
    for i, r in enumerate(rows):
        for rec in r.get("visible_object_records") or []:
            name = rec.get("actor_name") or ""
            xyz = (rec.get("actor_pose") or {}).get("xyz")
            if not rx.match(name) or xyz is None:
                continue
            # InsertPeg names the links of all three pegs peg_head and peg_tail. The segmentation id tells them apart.
            k = (name, _seg(rec))
            universe.add(k)
            if i <= t:
                last[k] = (i, xyz)
    out, stats = [], {"now": 0, "earlier": 0, "never": 0}
    for k in sorted(universe, key=lambda k: (k[0], -1 if k[1] is None else k[1])):
        if k not in last:
            stats["never"] += 1
            continue
        i, xyz = last[k]
        stats["now" if i == t else "earlier"] += 1
        out.append({"name": k[0], "seg": k[1], "xyz": [float(v) for v in xyz],
                    **({} if i == t else {"from_t": i})})
    return out, stats
