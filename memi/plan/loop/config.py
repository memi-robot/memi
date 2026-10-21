"""The settings of one Plan optimization run, the flags `init` and `optimize` share, and the model calls.

Relative paths are read against the repository root.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path

from ...llm import LLMClient, LLMResponse, budget
from ...llm.client import EFFORTS
from ...paths import resolve

#: The model of the offline roles (initializer, annotator, optimizer, and the policy's backward pass) in the
#: released runs.
OFFLINE_MODEL = "gpt-6-astra"
#: How long one offline call may take before it is treated as hung, in seconds. Every client's request timeout too.
OFFLINE_TIMEOUT = 1800.0
#: How long one call of the policy may take, in seconds.
POLICY_CALL_TIMEOUT = 300.0


@dataclass
class LoopConfig:
    """Where one run reads and writes, and how."""
    #: Policy prompts, annotator prompts, step records and traces go under `<runs_dir>/<Task>/`.
    runs_dir: str = "outputs/artifacts/gpt/runs"
    #: The decision dataset, `<decisions_dir>/<Task>.jsonl`.
    decisions_dir: str = "data/decisions"
    #: Policy calls in flight at once, forward and backward.
    concurrency: int = 4
    #: The principle this run removes (`memi.plan.ablation`), or None for the full loop.
    ablate: str | None = None

    def task_dir(self, task: str) -> Path:
        return resolve(self.runs_dir) / task

    def decisions(self, task: str) -> Path:
        return resolve(self.decisions_dir) / f"{task}.jsonl"


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """The flags `init` and `optimize` share."""
    ap.add_argument("--task", required=True)
    ap.add_argument("--runs-dir", default=LoopConfig.runs_dir,
                    help="policy prompts, annotator prompts and traces go under <runs-dir>/<Task>/")
    ap.add_argument("--decisions-dir", default=LoopConfig.decisions_dir,
                    help="the decision dataset, <decisions-dir>/<Task>.jsonl")
    ap.add_argument("--model", default=OFFLINE_MODEL, help="the offline roles' model")
    ap.add_argument("--effort", choices=EFFORTS, default="low",
                    help="the offline roles' reasoning effort")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="the answer budget. By default it follows the effort")


def client(model: str, effort: str, max_tokens: int | None, trace_path: Path) -> LLMClient:
    """A client that records every call in `trace_path`. The budget follows `effort` unless `max_tokens` is given."""
    return LLMClient(model, effort=effort, max_tokens=max_tokens or budget(effort), timeout=OFFLINE_TIMEOUT,
                     trace_path=trace_path)


async def ask(client_: LLMClient, system: str, user: str, purpose: str,
              timeout: float = OFFLINE_TIMEOUT) -> LLMResponse:
    """One model call, cut at `timeout` seconds."""
    return await asyncio.wait_for(client_.complete(system=system, user=user, purpose=purpose), timeout)
