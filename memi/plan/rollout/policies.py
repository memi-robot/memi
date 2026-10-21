"""The two Plan test policies of blog Section 3.1, FullHist and the memory policy that Init and MEMI run."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass

from ...llm.parse import parse_object
from ...llm.rollout_client import TOKEN_FIELDS, RolloutClient
from ...paths import resolve
from .. import policy_turn
from ..policy_turn import parse_answer
from ..prompts import POLICY_SYSTEM
from ..prompts.fullhist import SYSTEM as FULL_HISTORY_SYSTEM
from ..prompts.fullhist import decide_turn
from ..variables import FIELDS, field_names


def sha(text: str) -> str:
    """A prompt's hash, as the rollout logs record it."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


@dataclass
class PolicySettings:
    """How an LLM policy calls its model."""
    model: str
    effort: str = "low"
    policy_prompt: str = ""   # the exported policy prompts, for the memory policy
    max_tokens: int = 4096
    timeout: float = 180.0


class _LLMPolicy:
    """The calls every policy has. `start(task, goal, menu, has_demo)` once per episode, `decide(segment, menu)` at each
    decision point, which returns the command {"choice", "point"}, and `summary()` for the episode log.

    Every call is logged with its raw answer, its `parse_answer` status and its usage. A call that fails after its
    retries raises CallFailed, which stops the run.
    """

    name = "llm"
    cache_name = "llm"    # the policy's name in the prompt cache key
    system = ""

    def __init__(self, task: str, episode: int, settings: PolicySettings):
        self.task, self.episode = task, episode
        self.chat = RolloutClient(settings.model, settings.effort,
                                  max_tokens=settings.max_tokens, timeout=settings.timeout)
        self.goal = ""
        self.calls: list[dict] = []
        self.n_decide = 0

    def _purpose(self) -> str:
        # The prompt cache key, as the released rollouts sent it. The client drops segments like ep3 from it, so
        # "episode3" keeps one key per episode conversation.
        return f"oracle.{self.cache_name}.{self.task}.episode{self.episode}"

    def _log(self, text: str, rec: dict, status: str, **extra) -> None:
        self.calls.append({"kind": "decide", "index": len(self.calls), "status": status,
                           "raw": text, **rec, **extra})

    def summary(self) -> dict:
        """The policy's part of the episode log, its calls and their summed usage."""
        usage = {k: sum(int(c.get(k) or 0) for c in self.calls) for k in TOKEN_FIELDS}
        usage["latency"] = round(sum(float(c.get("latency") or 0) for c in self.calls), 3)
        usage["retries"] = sum(int(c.get("retries") or 0) for c in self.calls)
        return {
            "name": self.name,
            "model": self.chat.model,
            "effort": self.chat.effort,
            "system_sha": sha(self.system),
            "calls": dict(Counter(c["kind"] for c in self.calls)),
            "models_returned": sorted({c.get("model_returned", "") for c in self.calls} - {""}),
            "statuses": dict(Counter(c["status"] for c in self.calls)),
            "usage": usage,
            "log": self.calls,
        }


class FullObservationHistoryPolicy(_LLMPolicy):
    """FullHist, the full observation history in one conversation per episode. Each decision appends its segment as a
    user turn, so the model's own earlier answers stay in context."""

    name = "fullhist"
    cache_name = "full_observation_history"
    system = FULL_HISTORY_SYSTEM

    def start(self, task, goal, menu, has_demo=False):
        self.task, self.goal = task, goal
        self.has_demo = bool(has_demo)
        self.messages: list[dict] = []
        self.n_decide = 0

    def _goal(self) -> str:
        """The goal goes into the first user turn only."""
        return "" if self.messages else self.goal

    def _send(self, user: str) -> tuple[str, dict]:
        self.messages.append({"role": "user", "content": user})
        # An answer that comes back is kept, even an invalid one, which ends the episode.
        text, rec = self.chat.send(self.system, self.messages, self._purpose())
        if text:
            self.messages.append({"role": "assistant", "content": text})
        return text, rec

    def decide(self, segment, menu):
        menu = list(menu or [])
        demo = self.has_demo and self.n_decide == 0
        text, rec = self._send(decide_turn(self.n_decide, segment, menu, self._goal(), demo=demo))
        self.n_decide += 1
        cmd, status = parse_answer(text, menu)
        self._log(text, rec, status, command=cmd, turns=len(self.messages))
        return cmd

    def summary(self):
        return dict(super().summary(), turns=len(getattr(self, "messages", [])))


class MEMIPolicy(_LLMPolicy):
    """The memory policy of Init and MEMI, Eq. (1) at deployment, (m_t, a-hat_t) = pi_theta(q, o_t, m_{t-1}).

    theta is the task's entry in an exported policy prompt file. The memory starts empty (m_0) and is the policy's own.
    Each call is a fresh turn that `policy_turn` builds and merges, the turn the loop's policy.forward sends for the
    same policy prompt, step and memory.
    """

    name = "memory"
    cache_name = "schema_memory"
    system = POLICY_SYSTEM

    def __init__(self, task: str, episode: int, settings: PolicySettings):
        super().__init__(task, episode, settings)
        self.policy_prompt_file = settings.policy_prompt
        self.policy_prompt: dict = {}
        self.memory: dict = {}

    def _policy_prompt(self, task: str) -> dict:
        if not self.policy_prompt_file:
            raise RuntimeError("the memory policy needs a policy prompt file. Pass --policy-prompt.")
        raw = json.loads(resolve(self.policy_prompt_file).read_text())
        if task not in raw:
            raise KeyError(f"no policy prompt for {task!r} in {self.policy_prompt_file}. "
                           f"Present: {sorted(raw)}")
        if FIELDS not in raw[task]:
            raise KeyError(f"{self.policy_prompt_file}: the entry for {task!r} has no `{FIELDS}`. "
                           "Export it with python -m memi.plan.export.")
        return raw[task]

    def start(self, task, goal, menu, has_demo=False):
        self.task, self.goal = task, goal
        self.policy_prompt = self._policy_prompt(task)
        self.memory = {}
        self.n_decide = 0
        # Whether this episode has a demonstration. Its steps come in the first
        # decision step, and the prompt says so on every turn.
        self.saw_demo = bool(has_demo)

    def decide(self, segment, menu):
        menu = list(menu or [])
        phase = "first" if self.saw_demo and self.n_decide == 0 else "decide"
        before = policy_turn.memory_text(self.policy_prompt, self.memory)
        user = policy_turn.build_user(self.goal, self.policy_prompt, menu, segment, before,
                                      phase, saw_demo=self.saw_demo)
        text, rec = self.chat.send(self.system, [{"role": "user", "content": user}],
                                   self._purpose())
        updates = policy_turn.updates(parse_object(text) if text else None)
        # The answer names only the fields that change. The merge keeps the
        # declared fields only.
        self.memory = policy_turn.carry(field_names(self.policy_prompt), self.memory, updates)
        rec = dict(rec, memory_before=before,
                   memory_after=policy_turn.memory_text(self.policy_prompt, self.memory),
                   memory_updates=updates)
        self.n_decide += 1
        cmd, status = parse_answer(text, menu)
        self._log(text, rec, status, command=cmd)
        return cmd

    def summary(self):
        return dict(super().summary(), policy_prompt=self.policy_prompt_file,
                    memory=dict(self.memory),
                    memory_line=policy_turn.memory_text(self.policy_prompt, self.memory))


POLICIES = {"fullhist": FullObservationHistoryPolicy, "memory": MEMIPolicy}


def make(name: str, task: str, episode: int, settings: PolicySettings):
    return POLICIES[name](task, episode, settings)
