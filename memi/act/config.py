"""Paths and settings of the Act action type (blog Section 3.2)."""
from __future__ import annotations

from ..paths import ROOT

#: The released optimization runs, one directory per task.
RELEASED = ROOT / "artifacts" / "act" / "runs" / "opt_astra_act"
#: Where the optimization writes by default, relative to the repository root.
RUNS_DIR = "outputs/artifacts/act/runs/opt_astra_act"
CACHE = ROOT / ".cache" / "e2e"              # the cost ledger and stored prompt prefixes
EPISODE_CACHE = ROOT / "data" / "e2e" / "episodes"

STICK_TASKS = ("PatternLock", "RouteStick")

# GPT-6 Astra is the policy and runs every offline role. It is the only model of the Act action type.
ASTRA = "gpt-6-astra"
EFFORT = "low"

# What a cycle shows and how an answer is checked. The prompts state FRAME_STRIDE, POS_TOL_M, MAX_WAIT and
# MAX_STEPS in words (prompts/context.py, prompts/policy.py, prompts/fullhist.py), so a change goes there too.
FRAME_STRIDE = 16              # one front frame every this many control steps, plus the last
POS_TOL_M = 0.01               # a waypoint this close to the recorded one meets sufficiency
WAIT_EVERY = 30                # an expert's still wait is split into waits of at most this many steps
WAIT_TOL = 4                   # a wait is right within this many steps of the recorded one
MAX_WAIT = 300                 # the longest wait an answer may ask for

# The initializer and each of the ITERATIONS iterations get their own minibatch of episodes.
INIT_EPISODES = 5
ITERATIONS = 5
EPISODES_PER_ITERATION = 5
SEED = 7
WORKERS = 8
IMAGE_CAP = 450                # images per request (the API takes at most 500)
ANNOTATOR_ATTEMPTS = 3         # an unparsed annotation would mark every memory write wrong

# The rollouts, as in the base harness except the call cap (150 there).
DATASET = "test"
MAX_STEPS = 1300
MAX_CONSECUTIVE_NOOPS = 3
MAX_ACT_CALLS = 50


def settings() -> dict:
    """What a run records and a resume must match."""
    return {"policy_model": ASTRA, "offline_model": ASTRA, "effort": EFFORT,
            "frame_stride": FRAME_STRIDE, "pos_tol_m": POS_TOL_M, "wait_every": WAIT_EVERY, "wait_tol": WAIT_TOL}
