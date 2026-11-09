import json
from collections import Counter
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import NOANSWER


LOG_PATH = "setting1_binary/logs/2026-11-09T16-17-14-00-00_binary-choice_Eqp4YGE5t9arGBN6bR4YBi.eval"
EXPECTED_N = 1042
SHOW_INVALID = True     # print the invalid-choice samples for inspection


log = read_eval_log(LOG_PATH)
print(f"status: {log.status}, samples: {len(log.samples)}")

ids = [s.id for s in log.samples]
assert len(ids) == EXPECTED_N, f"expected {EXPECTED_N} samples, got {len(ids)}"

dupes = [i for i, count in Counter(ids).items() if count > 1]
assert not dupes, f"duplicate ids: {dupes}"

n_errors = sum(1 for s in log.samples if s.error is not None)
print(f"samples with errors: {n_errors}")


def make_record(s):
    md = s.metadata or {}
    scores = s.scores or {}
    score = scores.get("nudge_monitor")
    return {
        "id": s.id,
        "scenario": md.get("scenario_id"),
        "theme": md.get("constraint_theme"),
        "direction": md.get("direction"),
        "baseline_choice": md.get("baseline_choice"),
        "model_choice": md.get("model_choice"),
        "switched": md.get("switched"),
        "switched_recomputed": md.get("model_choice") in ("A", "B") and md.get("model_choice") != md.get("baseline_choice"),
        "has_reasoning": bool((md.get("internal_reasoning") or "").strip()),
        "verdict": score.value if score is not None else None,
        "error": s.error is not None,
    }

records = [make_record(s) for s in log.samples]
print(records[0])

def classify(r):
    if r["error"]:
        return "1_errored"
    if r["model_choice"] not in ("A", "B"):
        return "2_invalid_choice"
    if not r["switched"]:
        return "3_no_switch"
    if r["verdict"] == NOANSWER or not r["has_reasoning"]:
        return "4_switched_no_trace"
    if r["verdict"] == "YES":
        return "5_yes"
    if r["verdict"] == "NO":
        return "6_no"
    if r["verdict"] == "UNKNOWN":
        return "7_unknown"
    return "other"


cats = Counter(classify(r) for r in records)
print("\ncategory counts:")
for name in sorted(cats):
    print(f"  {name}: {cats[name]}")

assert sum(cats.values()) == EXPECTED_N, "categories don't add up to the sample count"
assert cats["other"] == 0, "some records didn't fit any category"

mismatch = sum(1 for r in records if bool(r["switched"]) != r["switched_recomputed"])
print(f"stored vs recomputed switch disagreements: {mismatch}")


def show(label, num, den):
    if den == 0:
        print(f"{label}: undefined (denominator is 0)")
    else:
        print(f"{label}: {num}/{den} = {num/den:.1%}")


n_switched = cats["4_switched_no_trace"] + cats["5_yes"] + cats["6_no"] + cats["7_unknown"]
n_valid = cats["3_no_switch"] + n_switched
n_scorable = cats["5_yes"] + cats["6_no"]

print()
show("switch rate", n_switched, n_valid)
show("detection among switches", cats["5_yes"], n_scorable)


# Section 8: inspect the invalid-choice samples
if SHOW_INVALID:
    by_id = {s.id: s for s in log.samples}
    invalid = [r for r in records if classify(r) == "2_invalid_choice"]
    print(f"\n=== {len(invalid)} invalid-choice samples ===")
    for r in invalid:
        answer = by_id[r["id"]].metadata.get("external_output") or ""
        print(f"\n{r['id']} | model_choice: {r['model_choice']!r}")
        print("  ...", answer[-400:])