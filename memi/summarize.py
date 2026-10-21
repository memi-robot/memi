"""The task x condition success tables of rollouts, one per policy, and their token and latency summary.

    python -m memi.summarize                         the released rollouts
    python -m memi.summarize outputs/results         new rollouts
    python -m memi.summarize results/luna results/haiku

A path is a condition directory `<policy>/<method>` that a rollout wrote, or a directory above such directories.
`python -m memi.report` checks the figures of the project page instead.
"""
from __future__ import annotations

import argparse
import collections
import statistics
from pathlib import Path

from . import conditions
from .paths import ROOT
from .results import first_finished, half_up, records


def order(condition: str) -> tuple:
    """Policies, then methods, in the order of the project page."""
    policies, methods = conditions.policies(), conditions.methods()
    policy, _, method = condition.rpartition("/")
    return (policies.index(policy) if policy in policies else len(policies), policy,
            methods.index(method) if method in methods else len(methods), method)


def compact(r: dict) -> dict:
    """What the tables count of one episode record."""
    usage = {k: v for k, v in ((r.get("policy") or {}).get("usage") or {}).items() if k != "retries"}
    return {"outcome": r.get("outcome"), "decisions": len(r.get("decisions") or []), "usage": usage}


def read(path: Path) -> dict[str, list[dict]]:
    """condition -> its records, from a condition directory and every directory below it."""
    rows = records(path)
    if rows:
        return {f"{path.parent.name}/{path.name}": rows}
    out: dict[str, list[dict]] = {}
    for p in sorted(path.iterdir()):
        if p.is_dir():
            for c, rs in read(p).items():
                out.setdefault(c, []).extend(rs)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m memi.summarize", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=[str(ROOT / "results")])
    a = ap.parse_args(argv)
    found: dict[str, list[dict]] = collections.defaultdict(list)
    for p in a.paths:
        for c, rs in read(Path(p)).items():
            found[c] += rs
    cells: dict[tuple[str, str], dict] = collections.defaultdict(dict)
    for c, rs in found.items():
        for (t, e), r in first_finished(rs).items():
            cells[(t, c)][e] = compact(r)
    if not cells:
        print("no rollouts found")
        return 1
    tasks = sorted({t for t, _ in cells})
    conds = sorted({c for _, c in cells}, key=order)
    groups: dict[str, list[str]] = {}
    for c in conds:
        groups.setdefault(c.rpartition("/")[0], []).append(c)
    total, other = collections.Counter(), collections.Counter()
    for policy, group in groups.items():
        width = max(8, *(len(c.rpartition("/")[2]) + 2 for c in group))
        print(f"{policy or 'condition':18s}" + "".join(f"{c.rpartition('/')[2]:>{width}s}" for c in group))
        for t in tasks:
            row = []
            for c in group:
                rs = list(cells[(t, c)].values())
                o = collections.Counter(r["outcome"] for r in rs)
                total[c] += o["success"]
                total[c + "/n"] += len(rs)
                for k, v in o.items():
                    if k not in ("success", "fail"):
                        other[(c, k)] += v
                row.append(f"{o['success']}/{len(rs)}" if rs else "-")
            print(f"{t:18s}" + "".join(f"{x:>{width}s}" for x in row))
        print(f"{'TOTAL':18s}" + "".join(f"{total[c]}/{total[c + '/n']}".rjust(width) for c in group))
        print(f"{'success rate':18s}" + "".join(
            (f"{half_up(100 * total[c] / total[c + '/n'], 1)}%" if total[c + "/n"] else "-").rjust(width)
            for c in group))
        print()
    if other:
        print("other endings: " + ", ".join(f"{c} {k} {v}" for (c, k), v in sorted(other.items())))

    print("\nper episode mean tokens: input (cached) output, reasoning | "
          "mean latency per decision call")
    # The Act logs carry no token counts.
    counted = [c for c in conds if any(r.get("usage") for t in tasks for r in cells[(t, c)].values())]
    for c in counted:
        rs = [r for t in tasks for r in cells[(t, c)].values()]
        u = [r.get("usage") or {} for r in rs]
        calls = sum(r["decisions"] for r in rs)
        lat = sum(float(x.get("latency") or 0) for x in u)

        def mean(k):
            return statistics.mean(float(x.get(k) or 0) for x in u) if u else 0.0
        print(f"{c:28s} in {mean('input_tokens') / 1000:6.1f}k ({mean('cached_tokens') / 1000:5.1f}k) "
              f"out {mean('output_tokens'):6.0f}, reasoning {mean('reasoning_tokens'):6.0f} | "
              f"{(lat / calls if calls else 0):5.2f}s over {calls} calls")

    print("\ntotal tokens in millions: input, cached input (share), output, reasoning")
    for c in counted:
        u = [r.get("usage") or {} for t in tasks for r in cells[(t, c)].values()]

        def tot(k):
            return sum(float(x.get(k) or 0) for x in u)
        share = 100 * tot("cached_tokens") / tot("input_tokens") if tot("input_tokens") else 0
        print(f"{c:28s} in {tot('input_tokens') / 1e6:7.2f}  cached {tot('cached_tokens') / 1e6:6.2f} "
              f"({share:3.0f}%)  out {tot('output_tokens') / 1e6:6.3f}  "
              f"reasoning {tot('reasoning_tokens') / 1e6:6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
