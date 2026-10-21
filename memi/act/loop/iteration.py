"""One iteration of the Act loop (blog Section 2.3), the seven steps of Figure 5 in order.

    1 sample episodes      the minibatch B^(k) of training episodes (`episodes.minibatches`)
    2 annotator.forward    the annotated memory m~_t of every cycle, from phi
    3 policy.forward       the memory m_t, action a^_t and used fields U_t of every cycle, from theta and m~_{t-1}
    4 loss                 L_suf, L_pred and L_nec, sufficiency, predictability and necessity
    5 policy.backward      the critiques d m~ and d theta of each failed cycle
    6 annotator.backward   the critiques merged into one proposal for phi and theta, d phi and d theta
    7 optimizer.step       phi and theta revised, with the optimization history in view

The policy's forward pass runs on its own client. Every other role runs on the offline client.
"""
from __future__ import annotations

from ...llm import responses as llm
from .. import episodes
from . import annotator, loss, optimizer, policy


def iteration(task: str, k: int, episode_ids: list[int], annotator_and_policy_prompts: dict, history: list[dict],
              initial: dict, offline: llm.Client, policy_client: llm.Client) -> dict:
    """One iteration on its minibatch, as the record iteration{k}.json holds it."""
    # Forward
    eps = [episodes.load(task, e) for e in episode_ids]
    annotated = annotator.forward(task, eps, annotator_and_policy_prompts, offline)
    steps = policy.forward(task, eps, annotated["after"], annotator_and_policy_prompts, policy_client)
    score, failed, unused = loss.loss(task, eps, steps, annotator_and_policy_prompts)
    # Backward
    critiques = policy.backward(task, eps, failed, annotator_and_policy_prompts, offline)
    failures = [{**s, "critique": r["critique"]} for s, r in zip(failed, critiques)]
    prop = (annotator.backward(task, annotator_and_policy_prompts, eps, annotated["after"], failures, unused, offline)
            if failures or unused else {"proposal": {}, "text": "", "cost": 0.0})
    critique = annotator.critique(prop["proposal"])
    # Revise
    if critique.strip():
        opt = optimizer.step(task, annotator_and_policy_prompts, history + [{"score": score}], initial, critique,
                             offline)
    else:                                  # a quiet iteration commits nothing and calls no optimizer
        opt = {"texts": annotator_and_policy_prompts, "revision": {}, "applied": None, "not_applied": [], "text": "",
               "cost": 0.0}
    cost = {"annotator.forward": annotated["cost"], "policy.forward": sum(s["cost"] for s in steps),
            "policy.backward": sum(r["cost"] for r in critiques), "annotator.backward": prop["cost"],
            "optimizer.step": opt["cost"]}
    return {"iteration": k, "episodes": episode_ids, "score": score, "annotations": annotated, "steps": steps,
            "failures": failures, "n_failed": len(failed), "unused_fields": unused, "proposal": prop,
            "critique": critique, "optimizer": {key: v for key, v in opt.items() if key != "texts"},
            "texts": opt["texts"], "applied": opt["applied"], "not_applied": opt["not_applied"],
            "cost": cost, "cost_total": round(sum(cost.values()), 4)}
