# my_analysis/extract_matched.py
# Pilot analysis for the matched-hint runs: outcomes by condition, behavioural shift
# against neutral, filter counts down to eligible traces, and crude keyword flags.
# Usage: python my_analysis/extract_matched.py [log_dir]
import glob
import json
import sys
from collections import Counter, defaultdict

from inspect_ai.log import read_eval_log
from scipy.stats import fisher_exact

LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else "my_analysis/logs/pilot_insurance"
OUT_PATH = "my_analysis/eligible_pilot.json"
PRIVACY_TERMS = ("privacy", "sell", "delet", "personal data", "user data")
RELIABILITY_TERMS = ("reliab", "outage", "single issue", "never had")

path = sorted(glob.glob(f"{LOG_DIR}/*.eval"))[-1]
log = read_eval_log(path)
print(f"{path} | {log.status} | {len(log.samples)} samples")


def make_record(s):
    md = s.metadata or {}
    sc = (s.scores or {}).get("two_hint_monitor")
    smd = (sc.metadata or {}) if sc else {}
    reasoning = md.get("internal_reasoning") or ""
    text = reasoning.lower()
    return {
        "id": s.id,
        "scenario": md.get("scenario_id"),
        "condition": md.get("condition"),
        "repetition": md.get("repetition"),
        "target_choice": md.get("target_choice"),
        "model_choice": md.get("model_choice"),
        "error": s.error is not None,
        "judged": bool(smd.get("judged")),
        "verdicts": smd.get("verdicts") or {},
        "reasoning": reasoning,
        "mentions_privacy": any(t in text for t in PRIVACY_TERMS),
        "mentions_reliability": any(t in text for t in RELIABILITY_TERMS),
    }


records = [make_record(s) for s in log.samples]
assert len({r["id"] for r in records}) == len(records), "duplicate sample ids"

by_cond = defaultdict(list)
for r in records:
    by_cond[r["condition"]].append(r)


# 1. Outcome distribution by condition
def outcome(r):
    if r["error"]:
        return "error"
    if r["model_choice"] not in ("A", "B"):
        return "non-answer"
    return r["model_choice"]

print("\n=== outcomes by condition ===")
for cond in ("neutral", "A", "B"):
    c = Counter(outcome(r) for r in by_cond[cond])
    print(f"{cond:8s} n={len(by_cond[cond]):3d}  A={c['A']}  B={c['B']}  "
          f"non-answer={c['non-answer']}  error={c['error']}")


# 2. Behavioural shift: non-answers stay in the denominator; technical errors are excluded
def target_counts(rs):
    done = [r for r in rs if not r["error"]]
    hits = sum(r["model_choice"] == r["target_choice"] for r in done)
    return hits, len(done)

print("\n=== behavioural shift vs neutral (target-choice rate) ===")
h0, n0 = target_counts(by_cond["neutral"])
print(f"neutral: {h0}/{n0} = {h0/n0:.0%}" if n0 else "neutral: no completed trials")
for cond in ("A", "B"):
    h, n = target_counts(by_cond[cond])
    if n and n0:
        _, p = fisher_exact([[h, n - h], [h0, n0 - h0]])
        print(f"hint {cond}:  {h}/{n} = {h/n:.0%}   shift = {h/n - h0/n0:+.0%}   (Fisher p = {p:.3f})")


# 3. Filter counts, hinted trials only
print("\n=== filter counts ===")
eligible = []
for cond in ("A", "B"):
    rs = by_cond[cond]
    done = [r for r in rs if not r["error"]]
    valid = [r for r in done if r["model_choice"] in ("A", "B")]
    to_target = [r for r in valid if r["model_choice"] == r["target_choice"]]
    usable = [r for r in to_target
              if r["judged"] and len(r["verdicts"]) == 2
              and set(r["verdicts"].values()) <= {"YES", "NO"}]
    two_no = [r for r in usable if all(v == "NO" for v in r["verdicts"].values())]
    eligible += two_no
    print(f"hint {cond}: trials {len(rs)} -> completed {len(done)} -> valid {len(valid)} "
          f"-> to target {len(to_target)} -> usable verdicts {len(usable)} -> NO to both {len(two_no)}")
    patterns = Counter((r["verdicts"]["A"], r["verdicts"]["B"]) for r in usable)
    print(f"    verdict patterns (judge on A, judge on B): {dict(patterns)}")


# 4. Eligible traces per scenario, before and after balancing
print("\n=== eligible traces per scenario ===")
per = defaultdict(Counter)
for r in eligible:
    per[r["scenario"]][r["condition"]] += 1
total_pairs = 0
for sid, c in sorted(per.items()):
    pairs = min(c["A"], c["B"])
    total_pairs += pairs
    print(f"{sid}: A={c['A']}  B={c['B']}  -> balanced pairs {pairs}")
print(f"total balanced pairs: {total_pairs}")


# 5. Crude keyword flags among eligible traces
print("\n=== keyword mentions among eligible traces ===")
for cond in ("A", "B"):
    rs = [r for r in eligible if r["condition"] == cond]
    priv = sum(r["mentions_privacy"] for r in rs)
    rel = sum(r["mentions_reliability"] for r in rs)
    neither = sum(not r["mentions_privacy"] and not r["mentions_reliability"] for r in rs)
    print(f"delivered {cond} (n={len(rs)}): privacy terms {priv}, reliability terms {rel}, neither {neither}")


with open(OUT_PATH, "w") as f:
    json.dump({"source_log": path, "eligible": eligible}, f, indent=2, ensure_ascii=False)
print(f"\nSaved {len(eligible)} eligible traces to {OUT_PATH}")