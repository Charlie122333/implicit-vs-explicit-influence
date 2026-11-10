# my_analysis/compare_human.py
# Compare human labels with the models' labels (analysis_plan.md, section 5).
# Reports random and targeted samples separately, the decision that matters most
# (is it a target recommendation?), the baseline-favouring check, and every disagreement.
import csv
import json

KEY = "my_analysis/human_check/key.json"
CHOICE_CSV = "my_analysis/human_check/choice_labels.csv"
NUDGE_CSV = "my_analysis/human_check/nudge_labels.csv"
STRICT = {"A", "B", "CONDITIONAL", "NONE"}
LENIENT = {"A", "B", "UNRESOLVED"}
NUDGE = {"NOT_MENTIONED", "DISMISSED", "NOTED", "USED"}

key = json.load(open(KEY))


def read(path, cols):
    rows = {}
    for row in csv.DictReader(open(path)):
        if not row.get("item"):
            continue
        rows[row["item"].strip()] = {c: (row.get(c) or "").strip().upper() for c in cols}
    return rows


# ---------------- choice task ----------------
human = read(CHOICE_CSV, ["strict", "lenient"])
bad = [i for i, r in human.items() if r["strict"] not in STRICT
       or (r["strict"] in ("CONDITIONAL", "NONE") and r["lenient"] not in LENIENT)]
missing = [i for i in key["choice"] if i not in human]
if bad or missing:
    print(f"Fix {CHOICE_CSV} first. Invalid rows: {bad}  Missing items: {missing}")
    raise SystemExit(1)

rows = []
for item, k in key["choice"].items():
    h = human[item]
    target = "B" if k["baseline"] == "A" else "A"
    rows.append({"item": item, "sample": k["sample"], "baseline": k["baseline"], "target": target,
                 "human": h["strict"], "human_len": h["lenient"], "consensus": k["consensus"],
                 "gpt5": k["gpt5"][0], "sonnet": k["sonnet"][0]})

print("=== CHOICE: agreement with the models, by sample ===")
for sample in ["random_switch", "random_nonswitch", "targeted_baseline", "targeted_disputed",
               "targeted_conditional"]:
    rs = [r for r in rows if r["sample"] == sample]
    agreed = [r for r in rs if r["consensus"] != "DISPUTED"]
    print(f"\n{sample} (n={len(rs)})")
    print(f"   human = model consensus:   {sum(r['human'] == r['consensus'] for r in agreed)}/{len(agreed)}"
          f"  (items where the models agreed)")
    print(f"   human = GPT-5:             {sum(r['human'] == r['gpt5'] for r in rs)}/{len(rs)}")
    print(f"   human = Sonnet:            {sum(r['human'] == r['sonnet'] for r in rs)}/{len(rs)}")
    same = sum((r["human"] == r["target"]) == (r["consensus"] == r["target"]) for r in agreed)
    print(f"   same answer to 'is it a target recommendation?': {same}/{len(agreed)}")

random_rows = [r for r in rows if r["sample"].startswith("random") and r["consensus"] != "DISPUTED"]
print("\n=== CHOICE: random sample, human (rows) vs model consensus (columns) ===")
cats = ["A", "B", "CONDITIONAL", "NONE"]
print("            " + "".join(f"{c:>13}" for c in cats))
for hc in cats:
    print(f"{hc:12s}" + "".join(f"{sum(r['human'] == hc and r['consensus'] == mc for r in random_rows):>13}"
                               for mc in cats))

tb = [r for r in rows if r["sample"] == "targeted_baseline"]
print(f"\n=== The striking claim: of {len(tb)} parsed switches the models say favour the BASELINE ===")
print(f"   human also says baseline:    {sum(r['human'] == r['baseline'] for r in tb)}")
print(f"   human says target:           {sum(r['human'] == r['target'] for r in tb)}")
print(f"   human says CONDITIONAL/NONE: {sum(r['human'] in ('CONDITIONAL', 'NONE') for r in tb)}")

print("\n=== CHOICE: every disagreement with the model consensus ===")
for r in rows:
    if r["consensus"] != "DISPUTED" and r["human"] != r["consensus"]:
        print(f"   {r['item']:4s} {r['sample']:20s} human={r['human']:12s} models={r['consensus']:12s}"
              f" baseline={r['baseline']} target={r['target']}")

disp = [r for r in rows if r["consensus"] == "DISPUTED"]
print(f"\n=== Disputed items (models disagreed): how the human decided ({len(disp)}) ===")
for r in disp:
    side = "gpt5" if r["human"] == r["gpt5"] else "sonnet" if r["human"] == r["sonnet"] else "neither"
    print(f"   {r['item']:4s} human={r['human']:12s} gpt5={r['gpt5']!s:12s} sonnet={r['sonnet']!s:12s} -> sides with {side}")

# ---------------- nudge task ----------------
nudge_h = read(NUDGE_CSV, ["nudge"])
filled = {i: r["nudge"] for i, r in nudge_h.items() if r["nudge"]}
if not filled:
    print("\n(nudge labels not filled in yet: skipping the nudge comparison)")
else:
    invalid = [i for i, v in filled.items() if v not in NUDGE]
    if invalid:
        print(f"\nFix these rows in {NUDGE_CSV}: {invalid}")
        raise SystemExit(1)
    print(f"\n=== NUDGE: {len(filled)} of {len(key['nudge'])} labelled ===")
    agree_cons, agree_any, n_cons = 0, 0, 0
    for item, k in key["nudge"].items():
        if item not in filled:
            continue
        h, g, s = filled[item], k["gpt5"], k["sonnet"]
        if g == s:
            n_cons += 1
            agree_cons += h == g
        agree_any += h in (g, s)
        if not (h == g == s):
            print(f"   {item:4s} human={h:14s} gpt5={g!s:14s} sonnet={s!s:14s}")
    print(f"   human = model consensus: {agree_cons}/{n_cons} (items where the models agreed)")
    print(f"   human matches at least one model: {agree_any}/{len(filled)}")