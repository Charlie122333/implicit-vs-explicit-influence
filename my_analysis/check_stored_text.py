# my_analysis/check_stored_text.py
# Rebuild each answer from the log's stored model message, exactly as the authors'
# extractor does, and compare it with the stored external_output. Reports how many
# stored answers differ, whether the parser's label changes, and the block structure
# of the keyboard_purchase answers.
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "setting1_binary"))
from inspect_ai.log import read_eval_log
from choice_extraction import extract_choice_natural

STAGE1_LOG = "setting1_binary/logs/2026-11-09T16-17-14-00-00_binary-choice_Eqp4YGE5t9arGBN6bR4YBi.eval"
RETRY_START = "2026-11-09T16:17"   # the retry log was created at 16:17:14


def rebuild_external(s):
    """Join the text blocks of the final model message, as extract_reasoning_from_state does."""
    msg = s.output.message if s.output else None
    if msg is None:
        return None, []
    content = msg.content
    if isinstance(content, str):
        return content, [("str", len(content))]
    parts, blocks = [], []
    for b in content:
        kind = type(b).__name__
        if kind == "ContentReasoning":
            blocks.append((kind, len(b.summary or b.reasoning or "")))
        elif hasattr(b, "text"):
            parts.append(b.text)
            blocks.append((kind, len(b.text)))
        else:
            blocks.append((kind, 0))
    return "\n".join(parts), blocks


def first_event_time(s):
    try:
        return str(s.events[0].timestamp)
    except Exception:
        return "unknown"


log = read_eval_log(STAGE1_LOG)
differ, label_changes = [], []
for s in log.samples:
    md = s.metadata or {}
    stored = md.get("external_output") or ""
    rebuilt, blocks = rebuild_external(s)
    if rebuilt is None:
        continue
    if rebuilt != stored:
        sid = md.get("scenario_id", "")
        on_stored = extract_choice_natural(stored, sid)
        on_rebuilt = extract_choice_natural(rebuilt, sid)
        row = (s.id, len(stored), len(rebuilt), md.get("model_choice"), on_stored, on_rebuilt,
               blocks, first_event_time(s))
        differ.append(row)
        if on_stored != on_rebuilt:
            label_changes.append(row)

print(f"Stage 1: {len(log.samples)} samples; stored answer differs from rebuilt answer in {len(differ)}")
print(f"  of those, the parser's label differs between the two texts in {len(label_changes)}")
if differ:
    print(f"  stored lengths of the differing answers: min {min(r[1] for r in differ)}, "
          f"max {max(r[1] for r in differ)}")
    print(f"  rebuilt lengths of the differing answers: min {min(r[2] for r in differ)}, "
          f"max {max(r[2] for r in differ)}")
    ran = Counter("first run" if r[7] < RETRY_START else "retry" for r in differ)
    print(f"  when the differing samples ran (by first event time): {dict(ran)}")
    by_scenario = Counter(r[0].split("__")[0] for r in differ)
    print(f"  scenarios involved (pooled count): {len(by_scenario)}")

stored_choice_matches_rebuilt = sum(1 for r in differ if r[3] == r[5])
print(f"  stored choice equals the parser on the REBUILT text in {stored_choice_matches_rebuilt} of {len(differ)}")

print("\n=== keyboard_purchase samples with a label change ===")
for r in label_changes:
    if r[0].startswith("keyboard_purchase"):
        print(f"\n{r[0]}  (first event {r[7]})")
        print(f"   stored {r[1]} chars -> parser {r[4]} | rebuilt {r[2]} chars -> parser {r[5]} | stored choice {r[3]}")
        print(f"   message blocks: {r[6]}")