# my_analysis/analyze_switches.py
# Turn the two-model labels into the Option 2 results:
#   1. what the parsed switches actually recommend (real switches vs parser errors)
#   2. the judge's YES rate in each group, and detection on real switches only
#   3. among real switches, how the reasoning handles the nudge, with the judge's YES rate
# Also writes 20 random items for a blind spot-check by hand.
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "setting1_binary"))
from inspect_ai.log import read_eval_log
from monitor import build_embedded_fact, load_constraint_hints

LABELS_PATH = "my_analysis/switch_labels.jsonl"
STAGE1_LOG = "setting1_binary/logs/2026-11-09T16-17-14-00-00_binary-choice_Eqp4YGE5t9arGBN6bR4YBi.eval"
SPOT_ITEMS = "my_analysis/spotcheck_items.txt"
SPOT_LABELS = "my_analysis/spotcheck_labels.csv"
SPOT_KEY = "my_analysis/spotcheck_key.json"
N_SPOT, SEED = 20, 20261111
GROUPS = ["real switch", "leans baseline", "conditional", "none", "disagree", "missing"]
HANDLING = ["NOT_MENTIONED", "DISMISSED", "NOTED", "USED", "DISAGREE", "MISSING"]


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def consensus(r, field):
    g, s = r[f"gpt5_{field}"], r[f"sonnet_{field}"]
    if g is None or s is None:
        return "MISSING"
    return g if g == s else "DISAGREE"


recs = [json.loads(line) for line in open(LABELS_PATH) if line.strip()]
for r in recs:
    r["target"] = "B" if r["baseline"] == "A" else "A"
    c = consensus(r, "choice")
    r["choice_c"] = c
    r["real"] = c == r["target"]
    r["group"] = ("real switch" if c == r["target"] else
                  "leans baseline" if c == r["baseline"] else c.lower())
    r["nudge_c"] = consensus(r, "nudge")
n = len(recs)

print(f"=== 1. what the {n} parsed switches actually recommend (labellers agreeing) ===")
counts = Counter(r["group"] for r in recs)
for g in GROUPS:
    print(f"  {g:15s} {counts.get(g, 0):4d}  ({counts.get(g, 0) / n:.0%})")
agreed = [r for r in recs if r["choice_c"] not in ("DISAGREE", "MISSING")]
k = sum(r["real"] for r in agreed)
lo, hi = wilson(k, len(agreed))
print(f"  real switches among agreed labels: {k}/{len(agreed)} = {k / len(agreed):.0%} (95% CI {lo:.0%}-{hi:.0%})")

print("\n=== 2. judge YES rate by group ===")
for g in GROUPS:
    rs = [r for r in recs if r["group"] == g]
    if rs:
        yes = sum(r["judge"] == "YES" for r in rs)
        print(f"  {g:15s} YES {yes}/{len(rs)} = {yes / len(rs):.0%}")
real = [r for r in recs if r["real"]]
yes_real = sum(r["judge"] == "YES" for r in real)
lo, hi = wilson(yes_real, len(real))
print(f"\n  detection on real switches: {yes_real}/{len(real)} = {yes_real / len(real):.1%} "
      f"(95% CI {lo:.0%}-{hi:.0%})   [parser-based: 49/416 = 11.8%]")

print("\n=== 3. real switches: how the reasoning handles the nudge ===")
for h in HANDLING:
    rs = [r for r in real if r["nudge_c"] == h]
    if rs:
        yes = sum(r["judge"] == "YES" for r in rs)
        print(f"  {h:14s} {len(rs):4d} ({len(rs) / len(real):.0%} of real switches)   "
              f"judge YES {yes}/{len(rs)} = {yes / len(rs):.0%}")
missed = [r for r in real if r["judge"] == "NO"]
print(f"\n  the {len(missed)} real switches the judge missed:")
for h, kk in Counter(r["nudge_c"] for r in missed).most_common():
    print(f"     {h:14s} {kk:4d} ({kk / len(missed):.0%})")

# Blind spot-check: 20 random items, shown without any labels
log = read_eval_log(STAGE1_LOG)
by_id = {s.id: s for s in log.samples}
hints = load_constraint_hints()
spot = random.Random(SEED).sample(recs, N_SPOT)
key = {}
with open(SPOT_ITEMS, "w") as f:
    for i, r in enumerate(spot, start=1):
        md = by_id[r["id"]].metadata
        nudge = build_embedded_fact(hints, md.get("target_tool", ""), md.get("constraint_id", ""))
        f.write(f"\n{'=' * 80}\nITEM {i}\n{'=' * 80}\n")
        f.write(f"A = {md['option_a']}   |   B = {md['option_b']}\nNudge: \"{nudge}\"\n")
        f.write(f"\n--- REASONING ---\n{md.get('internal_reasoning') or ''}\n")
        f.write(f"\n--- FINAL ANSWER ---\n{md.get('external_output') or ''}\n")
        key[i] = {"id": r["id"], "consensus": [r["choice_c"], r["nudge_c"]],
                  "gpt5": [r["gpt5_choice"], r["gpt5_nudge"]],
                  "sonnet": [r["sonnet_choice"], r["sonnet_nudge"]]}
with open(SPOT_LABELS, "w") as f:
    f.write("item,choice,nudge\n" + "".join(f"{i},,\n" for i in range(1, N_SPOT + 1)))
with open(SPOT_KEY, "w") as f:
    json.dump(key, f, indent=2)
print(f"\nWrote {N_SPOT} spot-check items to {SPOT_ITEMS}; fill in {SPOT_LABELS}. "
      f"Key: {SPOT_KEY} (don't open until you've labelled them).")