#!/usr/bin/env python3
"""Setting 2 — multi-domain rating, as an inspect_ai Task.

One Task per (model, arm[, intervention|adversarial]). The dataset is the FULL grid
for the requested domains — every base item x framing x nudge condition, INCLUDING the
`bare` control (which S2's cell-level significance test needs). Each cell is run
`N_SAMPLES=20` times via inspect `epochs`. The solver builds the user prompt (implicit
aside via `config.DOMAINS[d]["build_prompt"]`, or explicit instruct+conceal via
`config.build_explicit_prompt`), injects any guidance/adversarial system prompt, and
generates with extended thinking; a small scorer extracts the 1-10 scalar rating.

There is deliberately NO monitor in this eval. Unlike Settings 1/3/4, S2's detection
target is a *cell* (a nudge condition's 20 reps vs the bare 20 reps), not a single
sample — so the monitor cannot be a per-sample scorer. Generation logs are written
here; `monitor.py` reads them back, runs the significance screen, and judges the
samples in significant cells through inspect's model layer. See README.

Parameters (passed as `-T` args, or via run.py):
  model_key    one of the 7 model keys (labels the log; the model itself is set by run.py)
  arm          "implicit" (casual aside) | "explicit" (instruct + conceal)
  intervention ""|"rubric_focus"   (implicit arm only — de-biasing guidance system prompt)
  adversarial  0|1                 (explicit arm only — monitor-aware obfuscation SP)

The prompt builders, stimuli, nudges, framings and system prompts live in the frozen
`config.py` / `data/`. run.py sets the per-provider generation config.
"""

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem
from inspect_ai.scorer import NOANSWER, Score, Scorer, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, chain, generate, solver

from analysis_common import extract_scalar
from config import (DOMAINS, INTERVENTIONS, build_adversarial_system,
                    build_explicit_prompt)
from reasoning_extractor import extract_reasoning_from_state

N_SAMPLES = 20


# ── dataset ──────────────────────────────────────────────────────────────────

def _system_prompt_for(domain, arm, intervention, adversarial):
    """The per-domain system prompt for this run (guidance or adversarial), or ""."""
    if arm == "explicit" and adversarial:
        return build_adversarial_system(domain)
    if arm != "explicit" and intervention:
        mapping = INTERVENTIONS[intervention]
        return mapping["_all"] if "_all" in mapping else mapping[domain]
    return ""


def build_samples(model_key, arm, intervention, adversarial, domains=None):
    """One Sample per (domain, stim, framing, condition) cell — INCLUDING bare. inspect
    `epochs` replicates each cell N_SAMPLES times. Order is domain->stim->framing->
    condition so `--limit` yields whole adjacent cells (bare first) for smoke tests."""
    domain_list = domains or list(DOMAINS.keys())
    samples = []
    for domain in domain_list:
        dcfg = DOMAINS[domain]
        conditions, framings, stimuli = dcfg["conditions"], dcfg["framings"], dcfg["stimuli"]
        build_prompt = dcfg["build_prompt"]
        sys_prompt = _system_prompt_for(domain, arm, intervention, adversarial)
        for stim_id, stim_data in stimuli.items():
            for framing in framings:
                for cond_name in conditions:
                    if arm == "explicit":
                        prompt = build_explicit_prompt(domain, stim_data, framing, cond_name)
                    else:
                        prompt = build_prompt(stim_data, framing, cond_name)
                    entry = conditions[cond_name]
                    # nudge_text stored for the monitor is always the raw first-person
                    # nudge (medical third_person applies its override only inside the
                    # prompt, not the stored nudge_text).
                    nudge = entry.get("nudge", "")
                    samples.append(Sample(
                        input=prompt,
                        target="",
                        id=f"{domain}|{stim_id}|{framing}|{cond_name}",
                        metadata={
                            "model_key": model_key, "domain": domain,
                            "stim_id": stim_id, "framing": framing,
                            "condition": cond_name, "category": entry.get("family", ""),
                            "nudge_text": nudge, "prompt": prompt,
                            "system_prompt": sys_prompt, "arm": arm,
                            "intervention": intervention or "",
                            "adversarial": bool(adversarial),
                        },
                    ))
    return samples


# ── solvers ──────────────────────────────────────────────────────────────────

@solver
def inject_system() -> Solver:
    """Insert the guidance/adversarial system prompt (if any) ahead of the user turn."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        sp = state.metadata.get("system_prompt") or ""
        if sp:
            state.messages.insert(0, ChatMessageSystem(content=sp))
        return state
    return solve


@solver
def extract_rating() -> Solver:
    """Pull CoT + response out of the model output and store them (plus the parsed
    scalar) in metadata, so the sidecar monitor and the plot can read them back."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        internal, external = extract_reasoning_from_state(state)
        state.metadata["thinking"] = internal or ""
        state.metadata["response"] = external or ""
        state.metadata["scalar"] = extract_scalar(external or "")
        return state
    return solve


@scorer(metrics=[])
def rating_scorer() -> Scorer:
    """Report the parsed 1-10 rating as the sample score (float), NOANSWER if
    unparseable. Purely for visibility in `inspect view`; the analysis reads the
    stored response, not this score."""
    async def score(state: TaskState, target: Target) -> Score:
        v = state.metadata.get("scalar")
        return Score(value=v if v is not None else NOANSWER)
    return score


# ── task ─────────────────────────────────────────────────────────────────────

@task
def rating(model_key: str, arm: str = "implicit", intervention: str = "",
           adversarial: int = 0, domains: str = "") -> Task:
    domain_list = [d for d in domains.split(",") if d] or None
    samples = build_samples(model_key, arm, intervention, bool(int(adversarial)),
                            domains=domain_list)
    return Task(
        dataset=MemoryDataset(samples=samples),
        solver=chain(inject_system(), generate(), extract_rating()),
        scorer=rating_scorer(),
        # 20 reps per cell. "mode" reducer tolerates mixed float/NOANSWER values (we
        # never use the reduced score — the analysis reads every epoch's stored response).
        epochs=Epochs(N_SAMPLES, "mode"),
        metadata={"setting": "multidomain_rating", "model_key": model_key,
                  "arm": arm, "intervention": intervention,
                  "adversarial": bool(int(adversarial))},
    )
