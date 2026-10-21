"""Run Plan test episodes and append one row per episode to <out>/episodes.jsonl.

    python -m memi.plan.rollout --policy memory --model gpt-6-luna \
        --policy-prompt artifacts/gpt/policy_prompts/optimized_luna.json --split test \
        --tasks PickXtimes --episodes 0 1 2 --out outputs/results/luna/memi --workers 4

Each episode runs in its own process, since SAPIEN cannot build a second environment in one process. A rerun skips the
finished episodes and refuses an --out that run_settings.json says was written with other settings.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from ...llm import CallFailed
from ...paths import ROOT, load_env, resolve
from .policies import POLICIES, PolicySettings, make

#: Exit code of an episode process that stopped on CallFailed.
EXIT_CALL_FAILED = 3


def settings_record(split: str, policy: str, s: PolicySettings) -> dict:
    """What decides an episode's outcome besides (task, episode)."""
    policy_prompt = s.policy_prompt if policy == "memory" else ""
    path = resolve(policy_prompt) if policy_prompt else None
    return {"split": split, "policy": policy, "model": s.model, "effort": s.effort, "policy_prompt": policy_prompt,
            "policy_prompt_sha": (hashlib.sha256(path.read_bytes()).hexdigest()[:16]
                                  if path and path.exists() else "")}


def check_settings(out: Path, want: dict) -> str:
    """'' when `out` may take episodes run with `want`, else why not. A new `out` stores `want` in run_settings.json."""
    path = out / "run_settings.json"
    if path.exists():
        have = json.loads(path.read_text())
        if have != want:
            diff = {k: (have.get(k), want.get(k)) for k in sorted(set(have) | set(want))
                    if have.get(k) != want.get(k)}
            return (f"{out} was written with other settings {diff} (was, now). "
                    "Use another --out.")
        return ""
    path.write_text(json.dumps(want, indent=1))
    return ""


def rows(out: Path) -> dict[tuple[str, int], dict]:
    """The last row per (task, episode). A truncated line is skipped."""
    p = out / "episodes.jsonl"
    last: dict[tuple[str, int], dict] = {}
    if p.exists():
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            last[(r["task"], r["episode"])] = r
    return last


def done(out: Path) -> set[tuple[str, int]]:
    """Episodes with a final outcome. Harness errors and crashes are rerun."""
    return {k for k, r in rows(out).items()
            if r.get("outcome") not in ("harness_error", "process_crash")}


def _part(out: Path, task: str, episode: int) -> Path:
    return out / "parts" / f"{task}_ep{episode}.json"


def one(task: str, episode: int, split: str, policy: str, s: PolicySettings,
        out: Path, video: bool = True) -> None:
    """Run one episode in this process and write its row to its part file.
    With `video`, the camera frames go to `<out>/videos/<Task>_ep<n>.mp4`."""
    from .episode import dumps, run_episode
    film = out / "videos" / f"{task}_ep{episode}.mp4" if video else None
    log = run_episode(task, episode, split, make(policy, task, episode, s), video_path=film)
    part = _part(out, task, episode)
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_text(dumps(log))
    n = len(log.decisions)
    bad = sum(d["result"] not in ("executed", "no_op") for d in log.decisions)
    u = (log.policy or {}).get("usage") or {}
    tok = (f"  tokens in {u.get('input_tokens', 0)} (cached {u.get('cached_tokens', 0)}) "
           f"out {u.get('output_tokens', 0)}") if u else ""
    print(f"[rollout] {task} ep{episode} -> {log.outcome}  decisions {n} (invalid {bad})  "
          f"{log.seconds}s{tok}{'  ERR ' + log.error.splitlines()[-1] if log.error else ''}",
          flush=True)


def _child_args(a: argparse.Namespace, task: str, episode: int) -> list[str]:
    argv = [sys.executable, "-m", "memi.plan.rollout", "--one", task, str(episode),
            "--split", a.split, "--policy", a.policy, "--out", str(a.out),
            "--model", a.model, "--effort", a.effort, "--max-tokens", str(a.max_tokens),
            "--timeout", str(a.timeout)]
    if a.policy_prompt:
        argv += ["--policy-prompt", a.policy_prompt]
    if a.no_video:
        argv.append("--no-video")
    return argv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m memi.plan.rollout", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+")
    ap.add_argument("--episodes", nargs="+", type=int)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--policy", required=True, choices=list(POLICIES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1,
                    help="episode processes run at once")
    ap.add_argument("--model", required=True, help="the policy model, e.g. gpt-6-luna")
    ap.add_argument("--effort", default="low", help="the policy's reasoning effort")
    ap.add_argument("--policy-prompt", default="",
                    help="the exported policy prompts, for the memory policy")
    ap.add_argument("--max-tokens", type=int, default=PolicySettings.max_tokens)
    ap.add_argument("--timeout", type=float, default=PolicySettings.timeout)
    ap.add_argument("--no-video", action="store_true",
                    help="do not save the episode videos under <out>/videos/")
    ap.add_argument("--one", nargs=2, metavar=("TASK", "EPISODE"), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    load_env()
    s = PolicySettings(model=a.model, effort=a.effort, policy_prompt=a.policy_prompt,
                       max_tokens=a.max_tokens, timeout=a.timeout)
    a.out = resolve(a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.one:
        try:
            one(a.one[0], int(a.one[1]), a.split, a.policy, s, a.out, video=not a.no_video)
        except CallFailed as exc:
            print(f"[rollout] {a.one[0]} ep{a.one[1]} stopped: LLM call failed ({exc})",
                  flush=True)
            return EXIT_CALL_FAILED
        return 0
    if not a.tasks or a.episodes is None:
        ap.error("--tasks and --episodes are required")
    why = check_settings(a.out, settings_record(a.split, a.policy, s))
    if why:
        print(f"[rollout] REFUSED: {why}", flush=True)
        return 2
    finished = done(a.out)
    todo = []
    # dict.fromkeys keeps the order and runs a repeated (task, episode) once.
    for task, e in dict.fromkeys((t, e) for t in a.tasks for e in a.episodes):
        if (task, e) in finished:
            print(f"[rollout] {task} ep{e} already done", flush=True)
        else:
            todo.append((task, e))

    lock = threading.Lock()
    stop = threading.Event()

    def run(task: str, e: int) -> int:
        if stop.is_set():
            return 0
        part = _part(a.out, task, e)
        part.unlink(missing_ok=True)
        rc = subprocess.run(_child_args(a, task, e), cwd=ROOT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode
        if rc == EXIT_CALL_FAILED:
            stop.set()
            return rc
        with lock:
            with open(a.out / "episodes.jsonl", "a") as fh:
                if part.exists():
                    fh.write(part.read_text().rstrip("\n") + "\n")
                    part.unlink()
                else:
                    # The process died before writing its row. Recorded, so the
                    # episode stays in the denominator and is visible.
                    fh.write(json.dumps({"task": task, "episode": e, "split": a.split,
                                         "outcome": "process_crash", "returncode": rc}) + "\n")
                    print(f"[rollout] {task} ep{e} -> process_crash (rc {rc})", flush=True)
        return rc

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        list(pool.map(lambda te: run(*te), todo))
    if stop.is_set():
        print("[rollout] STOPPED: an LLM call failed after its retries. Fix the "
              "settings and rerun. Finished episodes are kept.", flush=True)
        return EXIT_CALL_FAILED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
