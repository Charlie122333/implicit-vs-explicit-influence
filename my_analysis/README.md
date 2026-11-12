> **This fork adds an audit of how Setting 1 extracts choices.**
> See [`my_analysis/README.md`](my_analysis/README.md) for the findings and how to reproduce them.


# Auditing choice extraction in Setting 1 of the implicit-influence benchmark

An audit of how Setting 1 of Duzan & Cooper Stickland (2026), *Chain-of-Thought
Monitoring Can Be Unreliable in Implicit-Influence Settings*, measures whether a model
switched its choice. Model: Claude Haiku 4.5. Condition: implicit delivery.


## Results

| Measure (1,021 answers) | Parser (paper's method) | Audited, strict | Audited, lenient |
| --- | --- | --- | --- |
| Switch rate             | 40.2%                   | 18.0–24.0%      | 23.4–33.1%       |
| Detection among switches| 12.0%                   | 18.5–20.3%      | 18.8–21.5%       |

- The reproduction matches the paper when measured the paper's way: 40.2% switching
  and 11.8% detection on all 1,035 valid answers (paper: 40.8% and 12.7%).
- The parser's keyword rules mislabel hedged and opposite answers, in both directions.
- Of real switches the monitor misses, about 64% name the nudge and dismiss it; about
  10% never mention it.
- Ranges run from disputed labels counted as non-switches to counted as switches.
  Intervals and robustness checks: `estimate_output.txt`.

## Reproducing the numbers

**From the committed files alone (no API calls, no log needed):**

    python my_analysis/compare_human.py     # human vs model labels
    python my_analysis/estimate.py          # all audited estimates (matches estimate_output.txt)
    python my_analysis/make_audit_figures.py

**From scratch:**

1. Set up the environment from the repository root:

       python3 -m venv .venv && source .venv/bin/activate
       pip install -r requirements.txt

   Python 3.14.2 was used. `requirements.txt` pins three packages the original code
   needs (anthropic 1.17.0, openai 2.54.0, agent-client-protocol 0.12.1).
   `requirements-lock.txt` lists every installed version.
2. Copy `.env.example` to `.env` and add `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`.
3. Run Setting 1 for Haiku (about $7):

       cd setting1_binary
       python run.py --arm implicit --models haiku --log-name full_implicit

   A fresh run produces new model outputs, so its numbers will differ slightly. The log
   used here is not in Git (`setting1_binary/logs/` is ignored) and is available on
   request.
4. Run the audit scripts below in order.

## Pipeline

**Reproduction**
- `extract_counts.py`: switch and detection counts from the Stage 1 log, checked
  against the authors' `plot.py` (49/416 exactly).
- `make_split.py` → `scenario_split.json`: fixed development/test split (6/17).

**Parser investigation**
- `parser_trace.py`: traced copy of the authors' parser, verified identical on 1,092
  answers; reports which step decided each label.
- `parser_checks.py`, `check_stored_text.py`, `insurance_compare.py`: keyword-match
  checks, incomplete stored text, original vs pilot prompts.

**Exploratory classification**
- `classify_switches.py` → `switch_labels.jsonl`: two-model labels of the 410 parsed
  switches (choice and nudge handling). The nudge labels are used in the audit.
- `analyze_switches.py`: exploratory tables (superseded by the audit).

**Audit**
- `analysis_plan.md`: written and committed before the audit runs, with a change log.
- `audit_rules.md`: the labelling rules.
- `classify_choice.py` → `choice_labels.jsonl`, `nonswitch_sample.json`: answer-only
  choice labels for the 410 switches and 150 sampled non-switches (about $4).
- `followup_nonswitches.py` → `nonswitch_followup.jsonl`: monitor verdicts and nudge
  labels for the 41 candidate switches among sampled non-switches.
- `build_human_check.py` → `human_check/`: blind materials, human labels and key.
- `compare_human.py`: human vs model agreement.
- `estimate.py` → `estimate_output.txt`: weighted estimates, bounds, labeller checks,
  two bootstraps.
- `final_checks.py`: prompt leak test, count totals.
- `make_audit_figures.py` → `figures/`.

**Matched-hint pilot (the original project; exploratory)**
- `build_matched_prompts.py`, `matched_task.py`, `extract_matched.py`,
  `read_traces.py`, `check_smoke.py`, plus their outputs.

**One-off exploration (kept as a record)**
- `check_size.py`, `explore_data.py`, `find_prompts.py`, `check_pairs.py`,
  `search_buffer.py`, `spotcheck_*` (superseded by `human_check/`),
  `*_output.txt`, `labels_insurance_*`.

## Exclusions and known deviations

- 14 keyboard_purchase samples have incomplete stored text after an interrupted run.
  Their run-time labels are used in the reproduction; they are excluded from re-reading.
- 7 answers the parser could not label are genuine non-answers and are excluded.
- Choice labellers also saw the nudge-category definitions (not the nudge itself).
  See the change log in `analysis_plan.md`.

## How this was done

Code for this project was written with substantial assistance from Claude (Anthropic).
I ran all experiments, performed the blind human validation, and checked the outputs. 
I developed and refined the analysis alongside Claude throughout the process. 


## Contact

Charles McLaren, mclaren.c@northeastern.edu