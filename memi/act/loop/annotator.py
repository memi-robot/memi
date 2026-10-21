"""The annotator alpha_phi of Eq. (2), steps 2 and 6 of Figure 5.

`forward` annotates the memory of a minibatch from the annotator prompt (phi), and `backward` merges the policy's
critiques of the failed cycles into one proposal for phi and theta.
"""
from __future__ import annotations

import collections

from ...llm import responses as llm
from ...text import ANNOTATOR_PROMPT, MEMORY, fill, memory_line, prose, said
from .. import config, evidence, memory, prompts, variables


def forward(task: str, eps: list[dict], annotator_and_policy_prompts: dict, client: llm.Client) -> dict:
    """Step 2. The annotated memory after each cycle of each episode (`after`), with the replies and their cost. A
    field keeps its value until an entry changes it. One call reads phi and the whole minibatch, or one per group of
    consecutive episodes when its images pass config.IMAGE_CAP."""
    groups, n = [[]], 0
    for ep in eps:
        items = evidence.recorded_episode(ep)
        k = llm.n_images(items)
        if groups[-1] and n + k > config.IMAGE_CAP:
            groups.append([])
            n = 0
        groups[-1].append(items)
        n += k
    cost, entries, replies = 0.0, [], []
    for group in groups:
        content = [llm.text(variables.annotator_prompt_block(annotator_and_policy_prompts)),
                   llm.text(prompts.context.EPISODES_INTRO)]
        content += [x for items in group for x in items]
        content.append(llm.text(prompts.annotator.OUTPUT_FORMAT))
        for _ in range(config.ANNOTATOR_ATTEMPTS):
            r = client.ask(prompts.annotator_system(task), content, purpose=f"{task}.annotator.forward")
            cost += r["cost"]
            if isinstance(r["json"], dict) and isinstance(r["json"].get("memory"), list):
                break
        got = r["json"] if isinstance(r["json"], dict) else {}
        entries += got.get("memory") or []
        replies.append(r["text"])
    names = variables.field_names(annotator_and_policy_prompts)
    at = collections.defaultdict(lambda: collections.defaultdict(dict))
    for e in entries:
        if not (isinstance(e, dict) and e.get("field") in names):
            continue
        try:
            at[int(e["episode"])][int(e["cycle"])][e["field"]] = memory.value_text(e.get("value", ""))
        except (KeyError, TypeError, ValueError):
            continue
    after = {}
    for ep in eps:
        state, per = {}, {}
        for c in ep["cycles"]:
            state = {**state, **at[ep["episode"]].get(c["k"], {})}
            per[c["k"]] = {n: state[n] for n in names if n in state}
        after[ep["episode"]] = per
    return {"after": after, "cost": cost, "text": "\n\n".join(replies), "calls": len(groups)}


#: What the policy's critique names, under the label the annotator reads it by.
REPORTED = (("the memory lacked", "missing_from_memory"),
            ("the policy prompt for memory lacked", "missing_from_policy_prompt_for_memory"),
            ("the policy prompt for action lacked", "missing_from_policy_prompt_for_action"))


def answered_action(answer: dict) -> str:
    if "wait" in answer and answer.get("waypoint") is None:
        return f"null, wait {answer['wait']}"
    return f"{answer.get('waypoint')}, gripper {answer.get('gripper')}" if "waypoint" in answer else "(nothing)"


def failure_block(failures: list[dict], names: list[str]) -> str:
    """One entry per failed cycle, with what the policy did there and what its critique says was missing. What it
    carried and what was correct there are in the annotated episodes above, at this cycle number."""
    out = []
    for f in failures:
        rep = f["critique"]
        body = [f"the waypoint a policy has written: {answered_action(f['answer'])}",
                f"the memory a policy has written:  {memory_line(names, f['mem_after'])}"]
        reported = [(label, said(rep, k)) for label, k in REPORTED]
        reported = [(label, s) for label, s in reported if s]
        body += [f"{label}: {s}" for label, s in reported]
        if not reported:
            body.append("it named nothing that was missing")
        out.append(f"ep{f['episode']} #{f['k']}\n" + "\n".join(f"  {b}" for b in body))
    return "\n\n".join(out)


def _shown_cycles(eps: list[dict], failures: list[dict]) -> dict[int, set]:
    """The cycles whose images are sent when the episodes pass the image cap. These are the failed cycles and the one
    before each, in order, while they fit."""
    cycles = {(ep["episode"], c["k"]): c for ep in eps for c in ep["cycles"]}
    show, n = set(), 0
    for f in failures:
        for key in ((f["episode"], f["k"] - 1), (f["episode"], f["k"])):
            c = cycles.get(key)
            if c is None or key in show:
                continue
            k = len(c["frames"]) + (c["kind"] == "act")
            if n + k <= config.IMAGE_CAP:
                show.add(key)
                n += k
    return {ep["episode"]: {k for e, k in show if e == ep["episode"]} for ep in eps}


def backward(task: str, annotator_and_policy_prompts: dict, eps: list[dict], annotated: dict, failures: list[dict],
             unused: dict, client: llm.Client) -> dict:
    """Step 6. The critiques of the failed cycles and the unused fields merged into one proposal, the critiques d phi
    and d theta, with the answer text."""
    names = variables.field_names(annotator_and_policy_prompts)
    content = [llm.text(variables.annotator_prompt_block(annotator_and_policy_prompts)),
               llm.text(variables.policy_prompts_block(annotator_and_policy_prompts)),
               llm.text(prompts.context.EPISODES_INTRO)]
    body = [evidence.recorded_episode(ep, annotated[ep["episode"]], names) for ep in eps]
    if sum(llm.n_images(b) for b in body) > config.IMAGE_CAP:
        show = _shown_cycles(eps, failures)
        body = [evidence.recorded_episode(ep, annotated[ep["episode"]], names, show=show[ep["episode"]]) for ep in eps]
    content += [x for b in body for x in b]
    if failures:
        content.append(llm.text(fill(prompts.annotator.BACKWARD_FAILURES, failures=failure_block(failures, names))))
    if unused:
        content.append(llm.text(fill(prompts.annotator.BACKWARD_UNUSED_FIELDS, fields="\n".join(
            f"    {k}\n      {v}" for k, v in sorted(unused.items())))))
    content.append(llm.text(prompts.annotator.BACKWARD_OUTPUT_FORMAT))
    r = client.ask(prompts.annotator_backward_system(task), content, purpose=f"{task}.annotator.backward")
    return {"proposal": r["json"] if isinstance(r["json"], dict) else {}, "text": r["text"], "cost": r["cost"]}


def critique(p: dict) -> str:
    """The proposal as the optimizer reads it, in the order and headings of memi/plan/loop/annotator.critique."""
    out = []
    for s in p.get(ANNOTATOR_PROMPT) or []:
        if isinstance(s, dict):
            how = f" (value_type: {variables.compared_as(s)})" if s.get("value_type") else ""
            if prose(s.get("description")) or how:
                out.append(f"field {s.get('name')}, ANNOTATOR PROMPT{how}\n  {prose(s.get('description'))}")
    for s in p.get(MEMORY) or []:
        if isinstance(s, dict) and prose(s.get("description")):
            out.append(f"field {s.get('name')}, POLICY PROMPT\n  {prose(s.get('description'))}")
    if prose(p.get(variables.ACTION)):
        out.append(f"the policy prompt for action\n  {prose(p.get(variables.ACTION))}")
    if prose(p.get("refused")):
        out.append(f"refused\n  {prose(p.get('refused'))}")
    return "\n\n".join(out)
