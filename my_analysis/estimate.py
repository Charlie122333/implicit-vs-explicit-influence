# my_analysis/estimate.py
# Estimates for the audit (analysis_plan.md, sections 6-9).
# Q1 share of parsed switches that are target recommendations
# Q2 audited target-recommendation rate       Q3 audited detection
# Q4 nudge handling among target recommendations the judge marked NO
# Every estimate under: strict / lenient definitions; disputed counted low / high;
# human labels substituted where available; each labeller alone. Two bootstraps.
import csv
import json
import random
from collections import Counter, defaultdict

N_SW, N_NON = 410, 611
B, SEED = 2000, 20261113
H = "my_analysis/human_check/"

choice = [json.loads(l) for l in open("my_analysis/choice_labels.jsonl") if l.strip()]
follow = {r["id"]: r for r in (json.loads(l) for l in open("my_analysis/nonswitch_followup.jsonl") if l.strip())}
nudge_sw = {r["id"]: (r["gpt5_nudge"], r["sonnet_nudge"])
            for r in (json.loads(l) for l in open("my_analysis/switch_labels.jsonl") if l.strip())}
key = json.load(open(H + "key.json"))
human = {row["item"].strip(): row for row in csv.DictReader(open(H + "choice_labels.csv")) if row.get("item")}
human_by_id = {k["id"]: human[item] for item, k in key["choice"].items() if item in human}

for r in choice:
    r["target"] = "B" if r["baseline"] == "A" else "A"
    if r["group"] == "switch":
        r["yes"] = r["judge"] == "YES"
        r["nudge"] = nudge_sw.get(r["id"], (None, None))
    else:
        f = follow.get(r["id"])
        r["yes"] = bool(f and f["judge"] == "YES")
        r["nudge"] = (f["gpt5_nudge"], f["sonnet_nudge"]) if f else (None, None)
sw = [r for r in choice if r["group"] == "switch"]
non = [r for r in choice if r["group"] == "nonswitch"]
assert len(sw) == N_SW and len(non) == 150, "label files don't match the analysis plan"

sw_by, non_by = defaultdict(list), defaultdict(list)
for r in sw:
    sw_by[r["scenario"]].append(r)
for r in non:
    non_by[r["scenario"]].append(r)
scenarios = sorted(set(sw_by) | set(non_by))


def effective(strict, lenient, mode):
    """The option a label points to: strict uses A/B only; lenient also uses the lean."""
    s = (strict or "").upper()
    if s in ("A", "B"):
        return s
    if mode == "lenient" and (lenient or "").upper() in ("A", "B"):
        return lenient.upper()
    return "OTHER"


def status(r, mode, source):
    """TARGET, NOT or DISPUTED for one answer."""
    if source == "human" and r["id"] in human_by_id:
        h = human_by_id[r["id"]]
        return "TARGET" if effective(h["strict"], h["lenient"], mode) == r["target"] else "NOT"
    g = effective(r["gpt5_strict"], r["gpt5_lenient"], mode)
    s = effective(r["sonnet_strict"], r["sonnet_lenient"], mode)
    if source == "gpt5":
        return "TARGET" if g == r["target"] else "NOT"
    if source == "sonnet":
        return "TARGET" if s == r["target"] else "NOT"
    if g != s:
        return "DISPUTED"
    return "TARGET" if g == r["target"] else "NOT"


def estimate(sw_rows, non_rows, mode, source, bound):
    def is_t(r):
        st = status(r, mode, source)
        return st == "TARGET" or (st == "DISPUTED" and bound == "high")
    q_sw = sum(is_t(r) for r in sw_rows) / len(sw_rows)
    y_sw = sum(is_t(r) and r["yes"] for r in sw_rows) / len(sw_rows)
    q_non = sum(is_t(r) for r in non_rows) / len(non_rows)
    y_non = sum(is_t(r) and r["yes"] for r in non_rows) / len(non_rows)
    t_total = N_SW * q_sw + N_NON * q_non
    p = t_total / (N_SW + N_NON)
    d = (N_SW * y_sw + N_NON * y_non) / t_total if t_total else float("nan")
    return q_sw, p, d


def ci(values):
    v = sorted(x for x in values if x == x)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


rng = random.Random(SEED)

print("=== Parser figures on the same 1,021 answers ===")
yes410 = sum(r["yes"] for r in sw)
print(f"   switch rate {N_SW}/{N_SW + N_NON} = {N_SW / (N_SW + N_NON):.1%};  "
      f"detection {yes410}/{N_SW} = {yes410 / N_SW:.1%}")

print("\n=== Audited estimates (95% intervals: dataset bootstrap | scenario bootstrap) ===")
for mode in ("strict", "lenient"):
    for source, bound in (("consensus", "low"), ("consensus", "high"), ("human", "low"),
                          ("human", "high"), ("gpt5", "low"), ("sonnet", "low")):
        q, p, d = estimate(sw, non, mode, source, bound)
        boot_data = [estimate(sw, [rng.choice(non) for _ in non], mode, source, bound) for _ in range(B)]
        boot_scen = []
        for _ in range(B):
            pick = [rng.choice(scenarios) for _ in scenarios]
            s2 = [r for sc in pick for r in sw_by[sc]]
            n2 = [r for sc in pick for r in non_by[sc]]
            if s2 and n2:
                boot_scen.append(estimate(s2, n2, mode, source, bound))
        pd_, ps_ = ci([b[1] for b in boot_data]), ci([b[1] for b in boot_scen])
        dd_, ds_ = ci([b[2] for b in boot_data]), ci([b[2] for b in boot_scen])
        tag = f"disputed={bound}" if source in ("consensus", "human") else "         "
        print(f"   {mode:7s} {source:9s} {tag:13s} Q1 {q:5.1%}   "
              f"Q2 {p:5.1%} [{pd_[0]:.1%}-{pd_[1]:.1%} | {ps_[0]:.1%}-{ps_[1]:.1%}]   "
              f"Q3 {d:5.1%} [{dd_[0]:.1%}-{dd_[1]:.1%} | {ds_[0]:.1%}-{ds_[1]:.1%}]")

print("\n=== Q4: nudge handling among target recommendations marked NO (strict, consensus) ===")
weights = Counter()
for r in sw + non:
    if status(r, "strict", "consensus") == "TARGET" and not r["yes"]:
        g, s = r["nudge"]
        lab = g if g == s else "DISPUTED"
        weights[lab] += 1 if r["group"] == "switch" else N_NON / 150
total = sum(weights.values())
for lab, w in weights.most_common():
    print(f"   {lab:14s} {w:6.1f}  ({w / total:.0%})")
print(f"   (weighted count of misses: {total:.1f}; each sampled non-switch counts {N_NON}/150 = {N_NON / 150:.2f})")