# my_analysis/check_smoke.py
# Print the key fields for each sample in the newest smoke-test log,
# then show the full reasoning and both judge explanations for one sample.
import glob
from inspect_ai.log import read_eval_log

INSPECT_ID = "insurance_plan__B__00"    # the sample to look at in detail

path = sorted(glob.glob("my_analysis/logs/smoke/*.eval"))[-1]   # newest log in the folder
log = read_eval_log(path)
print(path, "|", log.status, "|", len(log.samples), "samples\n")

for s in log.samples:
    sc = (s.scores or {}).get("two_hint_monitor")
    smd = (sc.metadata or {}) if sc else {}
    print(f"{s.id} | choice: {s.metadata.get('model_choice')} | switched: {s.metadata.get('switched')} "
          f"| judged: {smd.get('judged')} | verdicts: {smd.get('verdicts')}")
    for cand, desc in (smd.get("descriptions_shown") or {}).items():
        print(f"    judge shown for {cand}: {desc}")

# After the loop (no indentation): look at one sample in detail
target = next(x for x in log.samples if x.id == INSPECT_ID)
sc = target.scores["two_hint_monitor"]
smd = sc.metadata or {}

print(f"\n=== sample: {target.id} ===")
print("score metadata keys:", sorted(smd.keys()))
print("\n=== reasoning ===\n", target.metadata["internal_reasoning"])
for cand in ("A", "B"):
    print(f"\n=== judge on {cand} ===\n", (smd.get("explanations") or {}).get(cand))