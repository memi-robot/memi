"""The initializer, initialize(E) of Algorithm 1. One call writes phi and theta from one minibatch of recorded
episodes."""
from __future__ import annotations

from ...llm import responses as llm
from .. import episodes, evidence, prompts, variables


def initialize(task: str, episode_ids: list[int], client: llm.Client) -> tuple[dict, dict]:
    """phi and theta, and the record of the call. Raises CallFailed when the answer is not JSON."""
    content = [llm.text(prompts.context.EPISODES_INTRO)]
    for e in episode_ids:
        content += evidence.recorded_episode(episodes.load(task, e))
    content.append(llm.text(prompts.initializer.OUTPUT_FORMAT))
    r = client.ask(prompts.initializer_system(task), content, purpose=f"{task}.initialize")
    if not isinstance(r["json"], dict):
        raise llm.CallFailed(f"initializer answer is not JSON: {r['text'][:300]}")
    return variables.from_answer(r["json"]), {"text": r["text"], "usage": r["usage"], "cost": r["cost"]}
