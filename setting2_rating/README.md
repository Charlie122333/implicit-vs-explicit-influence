# Setting 2 — Multi-Domain Rating

Models rate items 1–10 across four domains (essay, hiring, medical, creative). Each item is
rated under many nudge conditions; per (model, domain) we keep the conditions producing a
significant rating shift vs the bare control (Welch t vs bare, Bonferroni p<0.05, |Δ|≥0.5), and
ask whether a GPT-5 monitor detects the nudge in the CoT. Arms: `implicit` (+ optional
`rubric_focus` guidance) and `explicit` (+ optional `--adversarial`).

The monitor is a **separate step** (not an in-eval scorer) because the detection target is a
*cell* (needs its 20 reps + the bare baseline), not a single sample.

## Files

| file | role |
|---|---|
| `data/stimuli/` | 40 base items (10 per domain) |
| `data/nudges.json` | per-domain nudge conditions |
| `data/framings.json`, `data/essay_meta.json` | framing templates; essay topic/type |
| `data/guidance_sysprompts.json` | rubric-focus guidance (implicit arm) |
| `data/adversarial_sysprompts.json` | explicit directive + adversarial stack |
| `config.py` | domain registry + prompt builders |
| `rating_task.py` | inspect task (item × framing × condition, ×20 reps), scalar scorer |
| `analysis_common.py` | scalar parsers, `.eval` reader, significance screen |
| `run.py` / `monitor.py` / `plot.py` | generation / GPT-5 monitor over significant cells / 2×2 domain figure |

## Run

Large grid (~316k calls/arm at 20 reps). Use `--dry-run`, `--models`, `--domains`, `--limit`.

```bash
python run.py --arm implicit --log-name implicit ;                          python monitor.py --log-name implicit
python run.py --arm implicit --intervention rubric_focus --log-name rubric ; python monitor.py --log-name rubric
python run.py --arm explicit --log-name explicit ;                          python monitor.py --log-name explicit
python run.py --arm explicit --adversarial --log-name explicit_adv ;        python monitor.py --log-name explicit_adv
python plot.py 
```

Significance is computed per arm, so results bars are comparable as rates, not raw counts.
