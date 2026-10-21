"""The annotator's blocks, for annotator.forward (step 2 of the loop) and annotator.backward (step 6)."""
from __future__ import annotations

ROLE = """\
===== YOUR ROLE =====
You are the ANNOTATOR. You are handed the annotator prompt and a minibatch
of episodes of the same task.

Read the annotator prompt and fill the memory fields in. Every field it names, at every step
of every episode, takes the value the annotator prompt gives it there.
"""


FITTING = """\
===== FILLING THE FIELDS AS THE ANNOTATOR PROMPT WRITES THEM =====
Fill them as the annotator prompt stands. Do not adapt it to the episode in any way: not the
names, not the totals, not which fields exist. A field named red_quota is about
red here too, and one described as out of the five required is out of five here
too.

If a field names something this episode does not have, leave that field unset.
If this episode needs a fact and no field holds it, leave the fact out. The annotator
prompt has to fix both, and you are asked afterwards what it should say instead.

Do not rename a field, do not add one, do not leave out one you can fill, and do
not pack a fact into another field's value instead.
"""


INPUT = """\
===== WHAT YOU ARE GIVEN =====
The annotator prompt comes first, one heading per field, then the episodes
to fill in. You are not shown either policy prompt.
"""


OUTPUT_FORMAT = """\
===== YOUR ANSWER =====
The steps of each episode are numbered #1, #2, and so on, within that episode,
and an entry names the episode and that number.

Finish with JSON only, no prose around it.

{"memory": [{"episode": 63, "step": 1,  "field": "...", "value": "..."},
            {"episode": 63, "step": 1,  "field": "...", "value": "..."},
            {"episode": 63, "step": 6,  "field": "...", "value": "..."},
            {"episode": 88, "step": 1,  "field": "...", "value": "..."}]}

`field` is always the annotator prompt's own name.

The policy carries every field at every step, so the answer must set at step #1
of every episode every field that has a value there, with whatever the policy
can honestly say, which may be `unknown` or a count of zero. After that give an
entry only where a value changes; a field keeps its value between entries.
"""


BACKWARD_ROLE = """\
===== YOUR ROLE =====
You are the ANNOTATOR, being asked about the memory you already filled in. You
are handed the annotator prompt you filled it from, the two policy prompts the policy was given, the
episodes as you annotated them, and the steps where a fresh policy could not
answer. Each of those steps carries the fact it says its memory lacked, and what
the policy prompt for subgoal or policy prompt for memory lacked.

Take everything it reported and say what the three documents have to become: the
annotator prompt, the policy prompt for memory, and the policy prompt for
subgoal. The policy answered one step at a time and saw only that step, so it wrote
one step's trouble in one step's words. You have the whole minibatch, so write
what holds across these episodes rather than one clause for each step that
reported something.

The policy reads what one step shows and the memory it carried in, while you
read the whole episode. So the annotator prompt can describe the
annotation by comparing what two steps show. The policy prompt for memory and the policy prompt for subgoal
cannot.
"""


BACKWARD_INPUT = """\
===== WHAT YOU ARE GIVEN =====
The annotator prompt comes first, then the two policy prompts the policy
was reading, then the episodes as you annotated them. Then the failed steps, each
with what the policy says was missing from the memory, from the policy prompt for
memory and from the policy prompt for subgoal. Last, the fields it carried and never
read.
"""


BACKWARD_FAILURES = """\
===== THE STEPS THAT DID NOT SETTLE =====
Each step a fresh policy could not answer. What it did there, and what it says
was missing from each of the three documents, in its own words. A document it
said nothing about is one it found sufficient at that step.

{failures}
"""


BACKWARD_UNUSED_FIELDS = """\
===== THE MEMORY THAT WAS CARRIED AND NOT READ =====
At every step the policy names the fields it read to answer. These were read at
few steps or none, so the policy carried them and decided nothing with them.
Each is a field, how often it was read, and at which steps:

{fields}

Each one is a field to drop, a field to fold into another, or a field to keep.
Look up the steps named above in the episodes and decide from them. Drop it when
nothing needs what it holds. Fold it when another field could hold the same
fact, and say so as a delete on this one and a modify on that one. Keep it when
those steps are the ones it exists for, and say why.

This block is the one thing you may act on that the policy did not report. A
field carried and never read is a fault no single step can see.
"""


BACKWARD_OUTPUT_FORMAT = """\
===== YOUR ANSWER =====
Three things: what the annotator prompt has to change, what the policy prompt for
memory has to change, and what the policy prompt for subgoal has to change. Leave out
any of them that does not have to change.

You are describing the changes. You are not making them. The role after you
holds every earlier edit beside this one and commits them all together.

`annotator_prompt` holds one entry per field whose filling has to change.
`policy_prompt_for_memory` holds one entry per field whose description to the policy
has to change. Write an entry in each list when both halves of a field have to change.

`policy_prompt_for_subgoal` is one account, in prose.

`refused` is one account, in prose, of what you read and are not passing on. Say
what was reported and why it stays out. Gather the steps that reported the same
thing and answer them together. The policy stood at one step, so it can report a
fault that contradicts the other steps in the episodes of the minibatch, or one
the policy prompts already cover. Write it so that a reader who never sees those steps
knows what was dropped. Leave it out when you are passing everything
on.

Write every account in the same order, the field entries and the policy prompt for
subgoal alike. The next role has nothing else to go on. It never sees an
observation, an episode or a step, and you have read the episodes end to end.

  - in what situation the failure occurred. Do not list every value at the step.
    Describe the part that mattered, in sentences. What the goal instruction
    asked for, how the observation changed, what subgoal the policy gave, and
    what the annotated memory said.
  - how the failure occurred, and which of the three conditions it breaks.
    SUFFICIENCY, when the current subgoal could not be chosen from the past
    memory and what the current step shows. PREDICTABILITY, when the current
    memory could not be chosen from the past memory and what the current step
    shows. That covers the
    case where the value you wrote and the value the policy prompt asks the policy for
    are two different answers to one field at one step. NECESSITY, when the
    step or the goal instruction already gave the information.
  - what has to change so the failure stops happening. For a field, say what it
    holds, the event at which it changes its value, and one example value. Under
    `policy_prompt_for_memory` the policy has to be able to recognise that event in
    what the step in front of it shows. Under `annotator_prompt` the event can be
    anything the episode shows.
    For the policy prompt for subgoal, say which of its rules sent the policy wrong
    and what should stand in its place, which rule it does not have and what that
    rule should say, and which rule it should no longer have at all.

Do not name a step or an episode. Whoever reads this cannot look them up. "The
gripper was over the bin with a red cube just released" carries. "At ep63 #45"
does not.

Answer with JSON only, no prose:

{"annotator_prompt": [{"name": "...", "description": "..."}],
 "policy_prompt_for_memory": [{"name": "...", "description": "..."}],
 "policy_prompt_for_subgoal": "...",
 "refused": "..."}
"""
