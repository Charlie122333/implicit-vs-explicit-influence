# my_analysis/parser_checks.py
# Follow-up checks on the authors' choice parser:
#   A. show the text around each keyword match for the flagged pilot answers
#      (are "ppo"/"hmo" genuine, or inside other words?)
#   B. Stage 1, pooled: how often a tail keyword counts only because it sits inside another word
#   C. print the tails that name only option B, for manual classification
#      (a lean toward B, or discussion of the hint?)
#   D. keyboard_purchase: compare the stored answer with the model's stored output
import glob
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "setting1_binary"))
from inspect_ai.log import read_eval_log
from choice_extraction import CHOICE_KEYWORDS, extract_choice_natural

STAGE1_LOG = "setting1_binary/logs/2026-11-09T16-17-14-00-00_binary-choice_Eqp4YGE5t9arGBN6bR4YBi.eval"
PILOT_DIR = "my_analysis/logs/pilot_insurance"
FLAGGED_IDS = ["insurance_plan__neutral__05", "insurance_plan__neutral__09",
               "insurance_plan__A__16", "insurance_plan__B__11", "insurance_plan__neutral__03"]


def tail_of(text):
    return text[-500:] if len(text) > 500 else text


def matches(text, kw, width=40):
    """Every occurrence of kw in text: (is_whole_word, surrounding text)."""
    out = []
    for m in re.finditer(re.escape(kw), text):
        s, e = m.start(), m.end()
        whole = not (s > 0 and text[s - 1].isalpha()) and not (e < len(text) and text[e].isalpha())
        out.append((whole, text[max(0, s - width):e + width].replace("\n", " ")))
    return out


stage1 = read_eval_log(STAGE1_LOG)
pilot = read_eval_log(sorted(glob.glob(f"{PILOT_DIR}/*.eval"))[-1])
pilot_by_id = {s.id: s for s in pilot.samples}

print("=== A. keyword matches in the tail, flagged pilot answers ===")
for sid in FLAGGED_IDS:
    md = pilot_by_id[sid].metadata
    tail = tail_of((md.get("external_output") or "").lower())
    kws = CHOICE_KEYWORDS[md["scenario_id"]]
    print(f"\n{sid}")
    for side in ("a", "b"):
        for kw in kws[side]:
            for whole, ctx in matches(tail, kw):
                tag = "whole word" if whole else "INSIDE ANOTHER WORD"
                print(f"   [{side.upper()}] '{kw}' ({tag}): ...{ctx}...")

print("\n=== B. Stage 1 (pooled): tail keywords present ONLY inside other words ===")
only_embedded = Counter()
answers_affected = 0
for s in stage1.samples:
    md = s.metadata or {}
    kws = CHOICE_KEYWORDS.get(md.get("scenario_id", ""))
    if not kws:
        continue
    tail = tail_of((md.get("external_output") or "").lower())
    affected = False
    for side in ("a", "b"):
        for kw in kws[side]:
            found = matches(tail, kw)
            if found and not any(whole for whole, _ in found):
                only_embedded[kw] += 1
                affected = True
    answers_affected += affected
print(f"answers where a tail keyword counts only through another word: {answers_affected} of {len(stage1.samples)}")
for kw, n in only_embedded.most_common():
    print(f"   '{kw}': {n}")

print("\n=== C. pilot tails that name only option B (classify by hand) ===")
for s in pilot.samples:
    md = s.metadata
    kws = CHOICE_KEYWORDS[md["scenario_id"]]
    text = md.get("external_output") or ""
    tail = tail_of(text.lower())
    if any(k in tail for k in kws["b"]) and not any(k in tail for k in kws["a"]):
        print(f"\n--- {s.id} (condition {md['condition']}, parsed {md['model_choice']}) ---")
        print(text[-500:])

print("\n=== D. keyboard_purchase: stored answer vs model output ===")
for s in stage1.samples:
    md = s.metadata or {}
    if md.get("scenario_id") != "keyboard_purchase":
        continue
    stored_text = md.get("external_output") or ""
    now = extract_choice_natural(stored_text, "keyboard_purchase")
    if now == md.get("model_choice"):
        continue
    completion = (s.output.completion if s.output else "") or ""
    print(f"\n{s.id}")
    print(f"   stored choice {md.get('model_choice')} | baseline {md.get('baseline_choice')} | switched {md.get('switched')}")
    print(f"   parser on stored answer ({len(stored_text)} chars): {now}")
    print(f"   parser on output.completion ({len(completion)} chars): {extract_choice_natural(completion, 'keyboard_purchase')}")
    print(f"   identical texts: {stored_text == completion}")
    print(f"   end of stored answer: ...{stored_text[-300:]}")
