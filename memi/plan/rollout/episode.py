"""One Plan test episode. The policy is called at each decision point with the segment since the previous one, and the
low-level planner provided with RoboMME executes the subgoal it returns.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from ...llm import CallFailed
from . import segments
from .capture import NO_RECORD, Capture
from .scene import SHOW_PX, describe
from .video import VideoWriter

#: Policy calls per episode, after which it ends as `decision_budget`.
MAX_DECISIONS = 50


@dataclass
class Decision:
    """One policy call and what the env did with its command, as the episode log keeps it."""
    index: int
    t_from: int
    t_to: int
    steps: int
    segment: str
    demo_steps: int = 0             # demonstration steps in the segment, at the first decision only
    command: dict | None = None
    result: str = ""
    error: str = ""
    batch: int = -1                 # frames the env step returned
    captured: int = -1              # execution rows the capture recorded during the command
    terminated: bool = False
    status: str = ""
    seconds: float = 0.0


@dataclass
class EpisodeLog:
    """One episode's row of episodes.jsonl."""
    task: str
    episode: int
    split: str
    goal: str = ""
    menu: list = field(default_factory=list)
    demo: dict = field(default_factory=dict)   # the demonstration steps the first decision shows
    decisions: list = field(default_factory=list)
    outcome: str = "unknown"
    error: str = ""
    capture_errors: list = field(default_factory=list)
    demo_rows: int = -1
    reset_batch: int = -1
    policy: dict = field(default_factory=dict)
    seconds: float = 0.0
    video: dict = field(default_factory=dict)


class EpisodeTimeline:
    """The episode's rows and their rendering, on one clock.

    Demonstration rows come first, filtered and renumbered exactly as the
    wrapper hands the video over. Execution rows continue the same clock.
    """

    def __init__(self, task: str):
        self.task = task
        self.rows: list[dict] = []

    def extend(self, rows: list[dict], demo: bool) -> tuple[int, int]:
        """Append `rows` on the episode clock. Returns their range [start, end)."""
        start = len(self.rows)
        for r in rows:
            self.rows.append(dict(r, is_video_demo=demo, timestep=len(self.rows)))
        return start, len(self.rows)

    def text(self, start: int, end: int) -> tuple[str, int]:
        """The segment of the steps in [start, end), and how many steps it shows."""
        robot = {int(r["timestep"]): r["arm"] for r in self.rows[:end] if r.get("arm")}
        feats = describe(segments.as_seen(self.rows[:end], self.task), self.task,
                         robot, move_px=SHOW_PX)
        steps = [(t, feats.get((self.rows[t]["is_video_demo"], t), ""))
                 for t in segments.picked(start, end)]
        return segments.render(steps), len(steps)


def _last(x):
    """A terminal flag as a python value. Batched flags arrive as tensors."""
    if hasattr(x, "reshape") and hasattr(x, "numel"):
        return x.reshape(-1)[-1].item() if x.numel() else False
    return x


# Messages the planner's wrapper raises before it executes anything. They end the episode as `api_error`.
REJECTED = ("requires a target pixel", "could not match any available candidate")
# Motion planner failures, in RoboMME's words. The wrapper raises the first. The planner's own errors reach
# FailAwareWrapper unwrapped as the other two.
PLANNER = ("Oracle solve failed", "planning exhausted", "Fail to parameterize path")


def _menu(info) -> list:
    return list((info or {}).get("available_multi_choices") or [])


def _batch(obs) -> int:
    """How many frames a step returned."""
    return len(next((v for v in obs.values() if isinstance(v, list)), [])) \
        if isinstance(obs, dict) else 0


def on_menu(cmd, menu: list) -> bool:
    """True when the wrapper's `_resolve_command` resolves `cmd`'s choice to a menu option, by the rule of
    `policy_turn._label_ok`."""
    choice = cmd.get("choice") if isinstance(cmd, dict) else None
    if not isinstance(choice, str) or not choice.strip():
        return False
    return any(o.get("label") == choice.strip().lower() for o in menu)


def follow_wrapper_rule(env) -> bool:
    """Make every menu's `need_parameter` say what the wrapper's `step` requires. False without a planner wrapper.

    The planner's wrapper lists `need_parameter = bool(option["available"])`, but it asks for a point whenever the
    option has an `available` key (`requires_target = "available" in option`), even an empty one. Its
    `_build_step_options` is shadowed on this env to apply that rule.
    """
    w, seen = env, 0
    while w is not None and seen < 12:
        if hasattr(w, "_build_step_options") and hasattr(w, "_resolve_command"):
            break
        w, seen = getattr(w, "env", None), seen + 1
    else:
        return False
    if w is None:
        return False
    orig = w._build_step_options

    def build():
        selected, options = orig()
        for shown, raw in zip(w.available_options, options):
            shown["need_parameter"] = "available" in raw
        return selected, options

    w._build_step_options = build
    return True


def run_episode(task: str, episode: int, split: str, policy,
                video_path: Path | None = None) -> EpisodeLog:
    """One episode and its log. With `video_path`, every camera frame also goes to that mp4.

    A decision's result is `executed`, `no_op`, `api_error` (an answer the wrapper cannot execute), `planner_error` (a
    planning failure of the low-level planner) or `harness_error`. An error ends the episode, and the policy is not
    asked again.
    """
    import robomme.robomme_env  # noqa: F401  (registers the tasks)
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    log = EpisodeLog(task=task, episode=episode, split=split)
    t0 = time.time()
    env = BenchmarkEnvBuilder(env_id=task, dataset=split, action_space="multi_choice",
                              gui_render=False).make_env_for_episode(episode)
    cap = Capture(task, env)
    timeline = EpisodeTimeline(task)
    film = VideoWriter(video_path)
    try:
        if not follow_wrapper_rule(env):
            cap._note("no planner wrapper found; need_parameter is the wrapper's bool(available)")
        cap.install()
        obs, info = env.reset()
        film.add(obs)
        goal = getattr(env.unwrapped, "language_goal", "") or info.get("task_goal", "") or ""
        # The env keeps a list of paraphrases. The first one is the instruction.
        log.goal = str(goal[0] if isinstance(goal, (list, tuple)) and goal else goal)
        log.menu = _menu(info)
        # The reset batch is the video followed by the frame execution starts
        # from. Frame i < n - 1 is the video.
        demo = [r for r in cap.rows if r.get("subgoal_online") != NO_RECORD] or list(cap.rows)
        n_reset = _batch(obs)
        video, first = (demo[:-1], demo[-1:]) if n_reset > 1 and demo else ([], demo[-1:])
        cap.start_execution()
        policy.start(task=task, goal=log.goal, menu=log.menu, has_demo=bool(video))
        demo_seg, demo_n = "", 0
        if video:
            a, b = timeline.extend(video, demo=True)
            demo_seg, demo_n = timeline.text(a, b)
            log.demo = {"t_from": a, "t_to": b - 1, "steps": demo_n, "chars": len(demo_seg)}
        if not first:
            # No video and no reset frame captured. Start from the live scene.
            first = [cap.record(env.unwrapped.get_obs(), info, 0, False)]
        start, end = timeline.extend(first, demo=False)
        log.demo_rows, log.reset_batch = len(demo), n_reset

        for i in range(MAX_DECISIONS):
            # An invalid answer ends the episode, so every call after the first
            # follows an executed command or a no_op. A command that captured
            # no rows shows the decision step again.
            seg, n = timeline.text(start, end) if end > start else timeline.text(end - 1, end)
            shown_demo = demo_n if i == 0 else 0
            if shown_demo:
                seg = segments.with_demo(demo_seg, seg)
            d = Decision(index=i, t_from=start, t_to=end - 1, steps=n, segment=seg,
                         demo_steps=shown_demo)
            ts = time.time()
            cmd = policy.decide(seg, log.menu)
            d.command = cmd
            before = len(cap.exec_rows)
            obs, _, term, trunc, info = env.step(cmd)
            film.add(obs)
            info = info or {}
            # The builder always wraps the env in FailAwareWrapper, which turns every exception of `step` into
            # info["status"] == "error". Errors are therefore classified from `info`.
            if info.get("status") == "error":
                msg = str(info.get("error_message", ""))
                d.error = f"{info.get('exception_type', '')}: {msg}"[:400]
                if any(m in msg for m in REJECTED):
                    d.result = "api_error"
                else:
                    d.result = "planner_error" if any(m in msg for m in PLANNER) else "harness_error"
            else:
                batch = _batch(obs)
                d.batch = batch
                # The wrapper puts the menu it resolved this choice against in `info`.
                if batch == 0 and not on_menu(cmd, _menu(info) or log.menu):
                    # A choice outside the menu or no readable choice. The wrapper
                    # executes nothing and returns an empty batch.
                    d.result = "api_error"
                elif batch == 0:
                    # A menu choice whose solve ran no sim step, such as StopCube's "remain static" after its last
                    # checkpoint. The scene did not change, so the next call shows the same step again.
                    d.result = "no_op"
                    log.menu = _menu(info) or log.menu
                else:
                    d.result = "executed"
                    d.terminated = bool(_last(term)) or bool(_last(trunc))
                    d.status = str(info.get("status", ""))
                    log.menu = _menu(info) or log.menu
            new = cap.exec_rows[before:]
            d.captured = len(new)
            start, end = timeline.extend(new, demo=False)
            d.seconds = round(time.time() - ts, 2)
            log.decisions.append(d.__dict__)
            if d.result in ("api_error", "planner_error", "harness_error"):
                log.outcome = d.result
                break
            if d.terminated:
                log.outcome = d.status or "terminated"
                break
        else:
            log.outcome = "decision_budget"
    except CallFailed:
        raise
    except Exception:
        log.error = traceback.format_exc()[-2000:]
        log.outcome = "harness_error"
    finally:
        log.capture_errors = list(cap.errors)
        log.policy = policy.summary()
        log.seconds = round(time.time() - t0, 1)
        film.close()
        if video_path:
            log.video = {"file": Path(video_path).name, "frames": film.frames, "error": film.error}
        try:
            cap.remove()
            env.close()
        except Exception:
            pass
    return log


def dumps(log: EpisodeLog) -> str:
    return json.dumps(log.__dict__, default=str)
