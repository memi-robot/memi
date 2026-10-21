"""The policy pi_theta of Eq. (1) under teacher forcing, steps 3 and 5 of Figure 5.

`forward` answers every cycle with the annotated memory carried in. `backward` puts each failed cycle to the policy
again with its answer beside the recorded one, and its critique says what the memory and each policy prompt lacked.
"""
from __future__ import annotations

import concurrent.futures as cf

from ...llm import responses as llm
from ...text import fill, memory_line
from .. import config, memory, policy_turn, prompts, variables
from ..evidence import gripper_word


def _forward(task: str, ep: dict, c: dict, carried: dict, annotated_after: dict, annotator_and_policy_prompts: dict,
             client: llm.Client) -> dict:
    names = variables.field_names(annotator_and_policy_prompts)
    r = client.ask(prompts.policy_system(task), policy_turn.training_turn(ep, c, carried, names),
                   purpose=f"{task}.policy.forward", prefix=policy_turn.policy_prefix(annotator_and_policy_prompts))
    answer = r["json"] if isinstance(r["json"], dict) else {}
    after = memory.apply_updates(carried, answer.get("memory_updates"), names)
    return {"episode": ep["episode"], "k": c["k"], "kind": c["kind"], "carried": carried, "answer": answer,
            "mem_after": after, "ann_after": annotated_after, "grade": None, "cost": r["cost"],
            "used": [u for u in answer.get("used_fields") or [] if u in names]}


def forward(task: str, eps: list[dict], annotated: dict, annotator_and_policy_prompts: dict, client: llm.Client
            ) -> list[dict]:
    """Step 3. Every cycle of every episode, answered with the annotated memory carried in. The loss fills each
    `grade`."""
    jobs = []
    for ep in eps:
        after, carried = annotated[ep["episode"]], {}
        for c in ep["cycles"]:
            jobs.append((ep, c, dict(carried), after.get(c["k"], {})))
            carried = after.get(c["k"], carried)
    with cf.ThreadPoolExecutor(config.WORKERS) as pool:
        return list(pool.map(lambda j: _forward(task, *j, annotator_and_policy_prompts, client), jobs))


def recorded_action(c: dict) -> str:
    tg = c.get("target")
    if not tg:
        return "null (a video cycle)"
    if tg.get("wait"):
        return f"a wait of {tg['wait']} control steps in place (waypoint null, wait {tg['wait']})"
    return f"{tg['waypoint']}, gripper {gripper_word(tg['gripper_after'])} after it"


def _critique(task: str, ep: dict, c: dict, annotator_and_policy_prompts: dict, step: dict, client: llm.Client) -> dict:
    """The cycle put again exactly as it was, with the answer beside the recorded one."""
    names = variables.field_names(annotator_and_policy_prompts)
    answer, g = step["answer"], step["grade"]
    wp_bad, mem_bad = g["waypoint_wrong"], bool(g["memory_wrong"])
    what = ("Both your waypoint and your memory" if wp_bad and mem_bad else "Your waypoint" if wp_bad
            else "Your memory" if mem_bad else "Your answer")
    answered = (f"null, wait {answer['wait']}" if "wait" in answer and answer.get("waypoint") is None
                else f"{answer.get('waypoint')}, gripper command {answer.get('gripper')}")
    content = policy_turn.training_turn(ep, c, step["carried"], names)
    content.append(llm.text(fill(
        prompts.policy.BACKWARD_COMPARISON, answered_waypoint=answered,
        answered_memory=memory_line(names, step["mem_after"]), recorded_waypoint=recorded_action(c),
        recorded_memory=memory_line(names, step["ann_after"]), what_disagreed=what)))
    r = client.ask(prompts.policy_backward_system(task), content, purpose=f"{task}.policy.backward",
                   prefix=policy_turn.policy_prefix(annotator_and_policy_prompts))
    return {"critique": r["json"] if isinstance(r["json"], dict) else {}, "cost": r["cost"]}


def backward(task: str, eps: list[dict], failed: list[dict], annotator_and_policy_prompts: dict, client: llm.Client
             ) -> list[dict]:
    """Step 5. The critique of every failed cycle (d m~ and d theta), in the order of `failed`."""
    by_ep = {ep["episode"]: ep for ep in eps}
    with cf.ThreadPoolExecutor(config.WORKERS) as pool:
        return list(pool.map(lambda s: _critique(task, by_ep[s["episode"]], by_ep[s["episode"]]["cycles"][s["k"] - 1],
                                                 annotator_and_policy_prompts, s, client), failed))
