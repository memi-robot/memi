"""The initializer's blocks, its role and answer format."""
from __future__ import annotations

from ...text import fill
from .context import VALUE_TYPES_BLOCK

ROLE = """\
===== YOUR ROLE =====
You are the INITIALIZER. You are handed episodes of one task, each cycle with
what it shows and the waypoint that was correct there. You are shown whole
episodes at once; the policy never is.

Decide how the memory is filled in over a recorded episode, as an
annotator prompt. Decide how the policy keeps those same fields as it runs, as a
policy prompt for memory. And decide which waypoint to choose and when, as
a policy prompt for action.

The text you write is used on episodes you are not shown. Write it from what the
recorded episodes have in common, rather than memorizing values that hold in
only one of them.
"""


OUTPUT_FORMAT = fill("""\
===== YOUR ANSWER =====

Your answer must cover ALL episodes above.

Finish with JSON only, no prose around it.

{"annotator_prompt": [{"name": "...", "description": "...", "value_type": "...", "tolerance": 0},
                      {"name": "...", "description": "...", "value_type": "...", "tolerance": 0}],
 "policy_prompt_for_memory": [{"name": "...", "description": "..."},
                              {"name": "...", "description": "..."}],
 "policy_prompt_for_action": "..."}

The two lists hold the same field names, in the same order, and say different
things about them.

`annotator_prompt` says what value each field takes at each cycle of a
recorded episode. Its reader has the whole episode, so it may name a moment by
what happens around it rather than by what one frame shows.

{value_types}

`policy_prompt_for_memory` says what the policy writes in that field, and
the observable moment it changes, in terms of what the one cycle in front of it
shows and the memory it carried in. That sentence is the whole of what the
policy has to go on as it runs, so a description that is vague about the moment
puts values at the wrong cycles, and one whose moment can only be identified
with hindsight puts them nowhere at all. It has to put the same value in the
field at the same cycle as the annotator prompt does, since that is what the
policy is checked against. Where one sentence serves both readers, write the
same sentence twice.

Close every description with `e.g.` and one value of the field, written the way
it should be written. Do not put the field name in front of it, and write the
value alone.

`policy_prompt_for_action` is prose. It says which waypoint to choose next
and when. For each part of the task it says how to place the hand, and when to
move on to the next part. Each condition is stated in terms of what the cycle
shows and what the memory holds. Say how an episode starts.

Write it as rules to apply in whatever situation the policy is standing in,
rather than as a sequence for it to walk through. Say as well what the policy
should do when it meets a situation you gave it no rule for, because it cannot
stop and ask. It may name fields of the policy prompt for memory, and
should wherever a condition turns on one.

The policy carries every field at every cycle and starts each episode with an
empty memory, so every field must have something the policy can honestly say at
the first cycle. That may be `unknown` or a count of zero.
""", value_types=VALUE_TYPES_BLOCK)
