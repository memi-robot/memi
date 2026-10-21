"""The conditions of the project page, as configs/conditions.json defines them, named `<policy>/<method>`.

    python -m memi.conditions                       every condition, in the order of the page
    python -m memi.conditions luna                  the conditions of one policy
    python -m memi.conditions --tasks luna/memi_iteration1
                                                    the tasks a condition is released for, none when it is all 16
    python -m memi.conditions --model luna          the model of a policy

A condition gives the policy's model and reasoning effort, the prompts it follows (none for FullHist), the number of
test episodes per task and, when it is released for some tasks only, those `tasks`. A policy named `<name>_act` is of
the Act action type.
"""
from __future__ import annotations

import argparse
import json

from .paths import ROOT

PATH = ROOT / "configs" / "conditions.json"


def load() -> dict[str, dict]:
    """Every condition by name, in the order of the project page."""
    return json.loads(PATH.read_text())


def policies() -> list[str]:
    """Every policy, in the order of the project page."""
    return list(dict.fromkeys(name.split("/")[0] for name in load()))


def methods() -> list[str]:
    """Every method, in the order of the project page."""
    return list(dict.fromkeys(name.split("/")[1] for name in load()))


def is_act(name: str) -> bool:
    """Whether a condition or a policy is of the Act action type."""
    return name.split("/")[0].endswith("_act")


def model(policy: str) -> str:
    """The model of a policy, the same in each of its conditions."""
    return next(c["model"] for name, c in load().items() if name.split("/")[0] == policy)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m memi.conditions", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy", nargs="?", help="only the conditions of this policy")
    ap.add_argument("--tasks", metavar="CONDITION", help="print the tasks this condition is released for")
    ap.add_argument("--model", metavar="POLICY", help="print the model of this policy")
    a = ap.parse_args(argv)
    conditions = load()
    if a.model:
        if a.model not in policies():
            return 1
        print(model(a.model))
        return 0
    if a.tasks:
        if a.tasks not in conditions:
            return 1
        print(" ".join(conditions[a.tasks].get("tasks") or []))
        return 0
    for name in conditions:
        if a.policy in (None, name.split("/")[0]):
            print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
