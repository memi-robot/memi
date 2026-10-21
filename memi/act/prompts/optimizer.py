"""The optimizer's blocks for optimizer.step, its role, input, history, report and answer format."""
from __future__ import annotations

from ...text import fill
from .context import VALUE_TYPES_BLOCK

ROLE = """\
===== YOUR ROLE =====
You are the OPTIMIZER. You are handed the annotator prompt and the two policy prompts
as they were first written. Then every report since, with the edit that was
committed under it. Then the three as they stand now, and the report on them.
You are not shown the episodes.

Decide the edits that settle the reported failures. You answer the edits
themselves, not the full texts again.

Each report was written by the annotator, which filled the memory in over that
iteration's episodes and then read back the cycles the policy could not answer.
Yours is the commit. You are the only role that sees every earlier edit beside
this one, so an edit that settles this iteration by undoing an earlier one is
yours to catch.

The report is a proposal, not an instruction. It was written from one
iteration's episodes, so it often describes that case rather than the rule
behind it. Read it against the earlier reports before committing it. Where
several reports are the same failure in different words, write the one rule that
covers all of them instead of a clause for each. A policy prompt that gains an
exception every iteration grows without becoming any more general, and the next
episode will be one more exception.

The text you write is used on episodes you are not shown. Write it from what the
recorded episodes have in common, rather than memorizing values that hold in
only one of them.
"""


INPUT = """\
===== WHAT YOU ARE GIVEN =====
The annotator prompt and the two policy prompts as they were initialized. Then every
iteration since, with what it scored, the report it produced and the edit that
was committed under it. The last of those is this iteration, and it carries only
what it scored, because its report is printed in full below and its edit is what
you are being asked for. Then the three as they stand now, in full. Last comes
this iteration's report.
"""


HISTORY = """\
===== THE ANNOTATOR PROMPT, THE POLICY PROMPTS, AND HOW THEY GOT HERE =====
{history}
"""


REPORT = """\
===== THE REPORT OF THIS ITERATION =====
The annotator prompt, policy prompt for memory, and the policy
prompt for action were checked cycle by cycle by a policy that had only
what the deployed one will have. Below is one block per thing the annotator prompt and
the policy prompts have to change.

{critique}

Edit the three so those are settled and the earlier reports stay settled. An
edit that answers this one by undoing an earlier one has settled nothing, so
check each against what was committed before it. An earlier edit can itself be
wrong, though. If the scores got worse after it was committed, or this
iteration's report shows with evidence that it causes these failures, revert or
replace it, and say which edit you undo and why. Do not undo an earlier edit only
to settle one episode of this iteration. Adding a field adds it to both halves:
the annotator has to fill it and the policy has to keep it.
"""


OUTPUT_FORMAT = fill("""\
===== YOUR ANSWER =====
The edits to apply, and nothing else. Do not write the policy prompts out
again.

What you commit is what the policy will follow, not an account of what went
wrong. The report argues for its edit by describing the failure; the policy
prompts say what to do. A field whose description begins "Failures occurred
when..." tells the policy nothing it can act on.

An edit to something the policy prompts already say is a passage of it
quoted as it reads now, and what should stand there instead.

  modify_from   the passage as it reads now, quoted word for word, so that it
                can be found and replaced. It has to appear above exactly as you
                write it here.
  modify_to     what should stand there instead. Leave it empty to delete the
                passage. To add something that is not said yet, quote the
                passage the new words should follow and repeat it here with the
                new words after it.

`annotator_prompt` holds one entry per field that changes on the annotator
prompt side. `policy_prompt_for_memory` holds one entry per field that changes on
the policy prompt side. A field that changes on both sides gets an entry in
each list. A field whose filling is right, and whose description to the policy
is not, gets an entry only under `policy_prompt_for_memory`.

`kind` says which change the entry is. What else the entry carries follows from
that.

  modify   `edits`, one passage edit per part of that field's text that changes.
           Quote from that field, on that side, and not from the other. The
           `e.g.` value closing it is part of that text: an edit that changes
           what the value looks like changes that clause in the same edit.
  add      `description` in full, there being nothing yet to quote from, closing
           with its `e.g.` value. Add the field on both sides.
  delete   nothing but the field's name. Delete the field on both sides. A field
           the annotator does not fill is one the policy cannot be checked on.

An add or a modify under `annotator_prompt` may also set `value_type` and
`tolerance`, to change how that field is compared. An add there gives them.

{value_types}

`policy_prompt_for_action` holds one passage edit per part of it that
changes, and a text may change in several places at once: give one entry per
part that changes rather than folding several changes into one quote that spans
the text between them. The same goes for a field's `edits`.

`refused` is one account, in prose, of what you are not committing. Name the
changes the report asked for and say why they stay out. The report comes from
one iteration and you have read all the past iterations. Committing nothing is
an answer.

Answer with JSON only, no prose:

{"annotator_prompt": [{"kind": "modify", "field": "...",
      "edits": [{"modify_from": "...", "modify_to": "..."}, ...]},
     {"kind": "add", "field": "...", "description": "...", "value_type": "...", "tolerance": 0},
     {"kind": "delete", "field": "..."}, ...],
 "policy_prompt_for_memory": [{"kind": "modify", "field": "...",
      "edits": [{"modify_from": "...", "modify_to": "..."}, ...]},
     {"kind": "add", "field": "...", "description": "..."},
     {"kind": "delete", "field": "..."}, ...],
 "policy_prompt_for_action": [{"modify_from": "...", "modify_to": "..."}, ...],
 "refused": "..."}
""", value_types=VALUE_TYPES_BLOCK)
