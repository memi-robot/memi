"""Every number of the project page's figures, recomputed from the released rollout logs.

    python -m memi.report                      the released results, checked against the page
    python -m memi.report --root outputs       your own rollouts, in the same layout
    python -m memi.report --figure 5           one figure
    python -m memi.report --figure cost        what our runs cost, optimization and rollouts

It reads `<root>/results/<policy>/<method>/` of every condition and needs no API key and no simulator. With the
released results every figure ends with `matches the project page`, and the run ends with `ALL MATCH`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import conditions
from .llm.prices import PRICES, dollars
from .llm.rollout_client import usage_record
from .paths import ROOT
from .results import first_finished, half_up, records
from .tasks import SUITES

#: The tasks in the order of the project page.
PAGE_TASKS = [t for _, ts in SUITES for t in ts]
#: The policies of the figures, by their name on the page.
POLICIES = {"astra_plan": "GPT-6 Astra-Plan", "luna": "GPT-6 Luna", "fable": "Claude Fable 5.1",
            "haiku": "Claude Haiku 5.5", "astra_act": "GPT-6 Astra-Act"}
METHODS = {"fullhist": "FullHist", "init": "Init", "memi": "MEMI"}
#: The conditions of Figures 1 and 5, one per point.
FIGURE1_METHODS = ["init", "memi_iteration1", "memi_iteration2", "memi_iteration3", "memi_iteration4", "memi"]

#: The numbers the project page shows.
PAGE = {
    # Figure 1 shows Astra-Plan and Astra-Act. Luna's iterations, on StopCube only, are Figure 5.
    "fig1": {"astra_plan": [92.4, 98.3, 99.3, 99.0, 99.3, 99.4], "astra_act": [88.3, 94.3, 93.5, 95.5, 95.5, 97.0]},
    # Figure 5 follows the StopCube run of MEMI-Luna-Plan iteration by iteration.
    "fig5": [34, 12, 84, 84, 92, 94],
    "fig6": {
        "luna": {"fullhist": [64, 90, 36, 48, 98, 34, 98, 44, 96, 38, 60, 88, 88, 64, 54, 6],
                 "init": [62, 100, 94, 34, 100, 40, 90, 38, 90, 38, 28, 90, 98, 66, 74, 24],
                 "memi": [88, 98, 92, 94, 100, 82, 96, 48, 96, 76, 84, 90, 100, 52, 86, 84]},
        "astra_plan": {"fullhist": [100, 100, 30, 78, 100, 100, 100, 86, 100, 94, 86, 94, 96, 96, 100, 100],
                  "init": [100, 100, 100, 76, 100, 96, 98, 98, 100, 100, 24, 90, 100, 96, 100, 100],
                  "memi": [98, 100, 100, 100, 100, 98, 98, 100, 100, 100, 100, 100, 100, 96, 100, 100]},
        "haiku": {"fullhist": [58, 98, 12, 2, 86, 22, 92, 44, 100, 26, 56, 86, 80, 86, 52, 28],
                  "init": [58, 96, 70, 72, 100, 68, 90, 34, 68, 46, 28, 88, 98, 94, 94, 78],
                  "memi": [96, 100, 100, 86, 100, 96, 92, 82, 96, 48, 86, 86, 100, 92, 94, 98]},
        "fable": {"fullhist": [96, 100, 30, 68, 100, 96, 100, 62, 100, 72, 84, 90, 90, 96, 96, 94],
                  "init": [100, 98, 98, 82, 100, 98, 100, 96, 100, 46, 24, 90, 100, 94, 100, 92],
                  "memi": [88, 100, 96, 92, 100, 98, 98, 100, 100, 96, 94, 100, 100, 96, 96, 100]},
        "astra_act": {"fullhist": [98, 100, 100, 70, 90, 90, 92, 84, 100, 92, 88, 96, 50, 64, 78, 72],
                      "init": [98, 96, 96, 92, 100, 100, 74, 52, 100, 82, 66, 90, 98, 74, 100, 94],
                      "memi": [98, 100, 100, 96, 100, 96, 100, 92, 100, 92, 100, 100, 96, 84, 100, 98]}},
    "fig6_mean": {"luna": [62.9, 66.6, 85.4], "astra_plan": [91.3, 92.4, 99.4], "haiku": [58.0, 73.9, 90.8],
                  "fable": [85.9, 88.6, 97.1], "astra_act": [85.3, 88.3, 97.0]},
    "fig7": {"StopCube": [34, 94, 36], "RouteStick": [24, 84, 18], "VideoPlaceOrder": [1.79, 1.76, 2.19]},
}
#: Figure 7. The policy, the condition without one principle, and the panel's measure. Each condition is released
#: for one task.
ABLATIONS = [("luna", "memi_no_sufficiency", "success"),
             ("luna", "memi_no_predictability", "success"),
             ("haiku", "memi_no_necessity", "cost")]


def call_cost(c: dict, write_share: float = 0.0) -> float:
    """$ of one policy call. With `write_share`, that share of a GPT-6 call's uncached input counts as cache writes
    when the call records none."""
    inp, rd, wr = (int(c.get(k) or 0) for k in ("input_tokens", "cached_tokens", "cache_write_tokens"))
    if write_share and "cache_write_tokens" not in c and c["model"].startswith("gpt-"):
        wr = round(write_share * (inp - rd))
    return dollars(c["model"], inp, rd, wr, int(c.get("output_tokens") or 0))


def episodes(root: Path, policy: str, method: str, write_share: float = 0.0) -> dict[tuple[str, int], tuple[bool, float]]:
    """(task, episode) -> (success, API cost of the policy calls), from one condition's logs. An Act result records
    its cost, a Plan row records each policy call."""
    out = {}
    for k, r in first_finished(records(root / "results" / policy / method)).items():
        if "cost_usd" in r:
            cost = float(r["cost_usd"] or 0)
        else:
            log = (r.get("policy") or {}).get("log") or []
            cost = sum(call_cost(c, write_share) for c in log if isinstance(c, dict) and c.get("model") in PRICES)
        out[k] = (r.get("outcome") == "success", cost)
    return out


def rate(eps: dict, tasks: list[str] | None = None) -> float | None:
    """Success (%), unrounded."""
    vals = [s for (t, _), (s, _) in eps.items() if tasks is None or t in tasks]
    return 100 * sum(vals) / len(vals) if vals else None


def per_task(eps: dict) -> list[float | None]:
    return [rate(eps, [t]) for t in PAGE_TASKS]


def cost(eps: dict, tasks: list[str] | None = None) -> float:
    """$, unrounded."""
    return sum(c for (t, _), (_, c) in eps.items() if tasks is None or t in tasks)


def fmt(v, digits: int = 1) -> str:
    return "-" if v is None else str(half_up(v, digits))


def leaves(x):
    if isinstance(x, dict):
        for k in x:
            yield from leaves(x[k])
    elif isinstance(x, list):
        for v in x:
            yield from leaves(v)
    else:
        yield x


class Check:
    """Compares the numbers, rounded as the page rounds them, with the page's."""

    def __init__(self, enabled: bool):
        self.enabled, self.bad = enabled, 0

    def __call__(self, figure: str, got, want, digits: int) -> None:
        if not self.enabled:
            return
        g = list(leaves({k: got[k] for k in want} if isinstance(want, dict) else got))
        w = list(leaves(want))
        off = [(None if a is None else float(half_up(a, digits)), b) for a, b in zip(g, w)
               if a is None or half_up(a, digits) != half_up(b, digits)]
        if len(g) == len(w) and not off:
            print(f"{figure} matches the project page\n")
        else:
            self.bad += 1
            print(f"{figure} DIFFERS from the project page (got, page): {off[:5]}\n")


def figure1(root: Path, check: Check) -> None:
    print("Figure 1. Mean success (%) after each optimization iteration. Init and MEMI on 50 test")
    print("episodes per task, iterations 1 to 4 on 25.")
    heads = ["Init", "Iter 1", "Iter 2", "Iter 3", "Iter 4", "MEMI"]
    print(f"{'policy':26s}" + "".join(f"{x:>8s}" for x in heads) + f"{'FullHist':>10s}")
    got = {}
    for p in PAGE["fig1"]:
        name = POLICIES[p]
        got[p] = [rate(episodes(root, p, m)) for m in FIGURE1_METHODS]
        full = rate(episodes(root, p, "fullhist"))
        print(f"{name:26s}" + "".join(f"{fmt(v):>8s}" for v in got[p]) + f"{fmt(full):>10s}")
    print()
    check("Figure 1", got, PAGE["fig1"], 1)


def figure5(root: Path, check: Check) -> None:
    print("Figure 5. StopCube success (%) of MEMI-Luna-Plan after each iteration. Init and MEMI on 50 test")
    print("episodes, iterations 1 to 4 on 25.")
    got = [rate({k: v for k, v in episodes(root, "luna", m).items() if k[0] == "StopCube"}) for m in FIGURE1_METHODS]
    print(f"{'GPT-6 Luna, StopCube':26s}" + "".join(f"{fmt(v, 0):>8s}" for v in got))
    print()
    check("Figure 5", got, PAGE["fig5"], 0)


def figure6(root: Path, check: Check) -> None:
    print("Figure 6. Success (%) per task, 50 test episodes each, and the mean over 800 episodes.")
    got, means = {}, {}
    for p, name in POLICIES.items():
        print(f"\n{name}")
        print(f"{'task':18s}" + "".join(f"{m:>10s}" for m in METHODS.values()))
        eps = {m: episodes(root, p, m) for m in METHODS}
        got[p] = {m: per_task(e) for m, e in eps.items()}
        for i, t in enumerate(PAGE_TASKS):
            print(f"{t:18s}" + "".join(f"{fmt(got[p][m][i], 0):>10s}" for m in METHODS))
        means[p] = [rate(eps[m]) for m in METHODS]
        print(f"{'mean':18s}" + "".join(f"{fmt(v):>10s}" for v in means[p]))
    print()
    check("Figure 6 (per task)", got, PAGE["fig6"], 0)
    check("Figure 6 (means)", means, PAGE["fig6_mean"], 1)


def figure7(root: Path, check: Check) -> None:
    print("Figure 7. The optimization rerun with the feedback of one principle withheld, on the")
    print("task where it matters. Success (%), or the API cost per 1,000 rollouts in $ for necessity.")
    print(f"{'task':18s}{'policy':18s}{'Init':>8s}{'MEMI':>8s}{'without':>9s}  principle withheld")
    got = {}
    for p, abl, measure in ABLATIONS:
        task = conditions.load()[f"{p}/{abl}"]["tasks"][0]
        row = []
        for m in ("init", "memi", abl):
            eps = {k: v for k, v in episodes(root, p, m).items() if k[0] == task}
            if measure == "success":
                row.append(rate(eps))
            else:
                row.append(1000 * cost(eps) / len(eps) if eps else None)
        got[task] = row
        d = 0 if measure == "success" else 2
        print(f"{task:18s}{POLICIES[p]:18s}" + "".join(f"{fmt(v, d):>8s}" for v in row[:2])
              + f"{fmt(row[2], d):>9s}  {abl.removeprefix('memi_no_')}")
    print()
    for task, want in PAGE["fig7"].items():
        check(f"Figure 7 ({task})", got[task], want, 0 if task != "VideoPlaceOrder" else 2)


def trace_cost(model: str, usage: dict) -> float:
    """$ of one recorded optimization call, from its chat-completions usage."""
    return call_cost({"model": model, **usage_record(usage)})


def release_cost(root: Path, check: Check) -> None:
    # The GPT-6 rollout logs record no cache writes. The optimization traces of the same models do, so the share
    # of uncached input written to the cache there stands in for the rollouts.
    uncached = written = 0
    for p in (root / "artifacts" / "gpt" / "runs").glob("*/*/provider_trace.jsonl"):
        for line in open(p):
            u = json.loads(line).get("usage") or {}
            d = u.get("prompt_tokens_details") or {}
            uncached += (u.get("prompt_tokens") or 0) - (d.get("cached_tokens") or 0)
            written += d.get("cache_write_tokens") or 0
    share = written / uncached if uncached else 0.0
    print("API cost of the released runs, in $ at the list prices in memi/llm/prices.py. A rerun")
    print("from the released policy prompts pays only the rollouts. The GPT-6 rollouts count")
    print(f"{100 * share:.1f}% of their uncached input as cache writes, the share in the optimization traces.")
    print(f"{'setting':32s}{'optimization':>14s}{'rollouts':>10s}{'total':>10s}")
    families = {"GPT-6 Astra-Plan and Luna": ("gpt", ["astra_plan", "luna"]),
                "Claude Fable 5.1 and Haiku 5.5": ("claude", ["fable", "haiku"])}
    for name, (fam, policies) in families.items():
        opt = 0.0
        for p in (root / "artifacts" / fam / "runs").glob("*/*/provider_trace.jsonl"):
            for line in open(p):
                r = json.loads(line)
                if r.get("model") in PRICES and r.get("usage"):
                    opt += trace_cost(r["model"], r["usage"])
        roll = sum(cost(episodes(root, *name.split("/"), share)) for name in conditions.load()
                   if name.split("/")[0] in policies)
        print(f"{name:32s}{fmt(opt, 2):>14s}{fmt(roll, 2):>10s}{fmt(opt + roll, 2):>10s}")
    runs = root / "artifacts" / "act" / "runs" / "opt_astra_act"
    opt = sum(json.loads(p.read_text()).get("cost_total") or 0 for p in runs.glob("*/iteration*.json"))
    opt += sum(json.loads(p.read_text()).get("cost") or 0 for p in runs.glob("*/initializer.json"))
    roll = sum(cost(episodes(root, *name.split("/"))) for name in conditions.load()
               if name.startswith("astra_act/"))
    print(f"{'GPT-6 Astra-Act':32s}{fmt(opt, 2):>14s}{fmt(roll, 2):>10s}{fmt(opt + roll, 2):>10s}")
    print()


FIGURES = {"1": figure1, "5": figure5, "6": figure6, "7": figure7, "cost": release_cost}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m memi.report", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT),
                    help="the directory holding results/, the repository by default")
    ap.add_argument("--figure", choices=list(FIGURES), action="append",
                    help="only these figures. All by default")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    # Your own rollouts are not checked against the page, since the models sample their answers.
    check = Check(root == ROOT)
    for f in a.figure or list(FIGURES):
        FIGURES[f](root, check)
    if check.enabled:
        print("ALL MATCH" if not check.bad else f"{check.bad} FIGURES DIFFER")
    return 1 if check.bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
