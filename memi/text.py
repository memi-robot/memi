"""Text helpers both action types share. Prompt placeholders and blocks, prose answers, the memory line and the quoted
passage edits of optimizer.step."""
from __future__ import annotations

import difflib

#: The keys under which the roles answer the annotator prompt and the policy prompt for memory.
ANNOTATOR_PROMPT = "annotator_prompt"
MEMORY = "policy_prompt_for_memory"


def fill(text: str, **values) -> str:
    """Put values into a prompt's `{name}` placeholders.

    `replace`, never `format`, because these prompts hold literal JSON braces.
    A value without its placeholder raises, since a renamed placeholder would
    otherwise reach the model unfilled.
    """
    for name, value in values.items():
        token = "{" + name + "}"
        if token not in text:
            raise KeyError(f"no {token} in this prompt")
        text = text.replace(token, str(value))
    return text


def assemble(*blocks: str) -> str:
    """One blank line between blocks, and nothing else joining them."""
    return "\n".join(b.strip("\n") + "\n" for b in blocks if b.strip())


def heading(name: str) -> str:
    """The marker between one part of a turn and the next."""
    return f"===== {name} ====="


#: Answers that decline, which the prompt offers per field. said() turns them into an empty string.
DECLINED = ("", "nothing", "none", "n/a", "na", "-")


def said(answer: dict, key: str) -> str:
    """One field of a backward answer, whitespace folded, empty if declined."""
    text = " ".join(str(answer.get(key, "")).split())
    return "" if text.lower().strip(" .") in DECLINED else text


def prose(value) -> str:
    """One account, however the answer shaped it. A list is joined."""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(x) for x in value if str(x).strip())
    return " ".join(str(value or "").split())


def memory_line(names: list[str], values: dict) -> str:
    """`a=1; b=[80, 126]`, every field in order, an empty one as `name=`.

    A field the policy invented does not enter the line, and one it dropped
    shows up as empty.
    """
    return "; ".join(f"{n}={values.get(n, '')}" for n in names)


#: How close a quote must be to a passage before it is treated as that passage. High, because the repair rewrites
#: the span it matches, so a quote that is only broadly similar must miss and be reported.
NEAR = 0.93


def _near_span(text: str, want: str):
    """(lo, hi, ratio) of the span of `text` that `want` most nearly matches, or None below NEAR.

    The ratio is taken over the span the matching blocks cover and not over the
    whole text, so a short quote cannot match a long text.
    """
    if not want or not text:
        return None
    blocks = [b for b in difflib.SequenceMatcher(None, text, want, autojunk=False).get_matching_blocks() if b.size]
    if not blocks:
        return None
    lo, hi = blocks[0].a, blocks[-1].a + blocks[-1].size
    ratio = difflib.SequenceMatcher(None, text[lo:hi], want, autojunk=False).ratio()
    return (lo, hi, ratio) if ratio >= NEAR else None


def splice(text: str, edits: list[dict]) -> tuple[str, list[str], list[dict]]:
    """`text` with each quoted passage `modify_from` replaced by `modify_to`, the quotes that missed, and the edits
    that applied.

    An edit with no quote is appended. A missing quote is not applied, since
    appending it would add a second copy of a sentence that says something
    slightly different. A near quote is applied over the span it nearly
    matches, and its applied edit keeps the passage as it really read.
    """
    problems, applied = [], []
    for e in edits or []:
        want, to = str(e.get("modify_from", "")).strip(), str(e.get("modify_to", "")).strip()
        if not want:
            if to:
                text = (text + "\n\n" + to).strip()
                applied.append(dict(e))
            continue
        if want in text:
            text = text.replace(want, to, 1)
            applied.append(dict(e))
            continue
        near = _near_span(text, want)
        if near:
            lo, hi, ratio = near
            applied.append({**e, "modify_from": text[lo:hi], "repaired_from": want, "similarity": round(ratio, 4)})
            text = text[:lo] + to + text[hi:]
            continue
        problems.append(" ".join(want.split())[:120])
    return text, problems, applied
