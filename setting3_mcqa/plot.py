#!/usr/bin/env python3
"""Setting 3 figure — per-model monitor detection rate across conditions.

For each model and condition, the bar height is the monitor detection rate (fraction
of behavioral flips the GPT-5 3-level monitor caught, i.e. NOTICED+INFLUENCED / flips).
The number above each bar is the flip denominator n; whiskers are 95% Clopper-Pearson CIs.

Reads logs/<condition>/monitor_3level.json (from monitor.py). Each `condition` here
is one log-name dir (one delivery method / arm). The five conditions, in plotting
order, are:
  explicit        Explicit                                    (green)
  explicit_adv    Explicit + adversarial system prompt        (red)
  s8              Implicit: prefill (8-shot)                  (blue)
  s16             Implicit: prefill (16-shot)                 (orange)
  mt              Implicit: multiturn + planted reasoning     (purple, non-Anthropic only)

Usage:
    python plot.py                                   # the five conditions
    python plot.py --conditions s8 s16               # any subset you have run
    python plot.py --conditions explicit --out figures/quickstart.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

EXP_DIR = Path(__file__).parent.resolve()

MODEL_ORDER = ["haiku", "sonnet", "opus", "gpt_oss", "grok_xai", "kimi", "glm"]
MODEL_DISPLAY = {
    "haiku": "Haiku 4.5", "sonnet": "Sonnet 4.5", "opus": "Opus 4.5",
    "gpt_oss": "GPT-OSS 120B", "grok_xai": "Grok 3 Mini",
    "kimi": "Kimi K2", "glm": "GLM 4.7",
}
ANTHROPIC = {"haiku", "sonnet", "opus"}

# condition dir -> (display label, solid colour) in plotting order
CONDITION_STYLE = {
    "explicit":     ("Explicit", "#59A14F"),
    "explicit_adv": ("Explicit + adversarial system prompt", "#E15759"),
    "s8":           ("Implicit: prefill (8-shot)", "#4E79A7"),
    "s16":          ("Implicit: prefill (16-shot)", "#F28E2B"),
    "mt":           ("Implicit: multiturn + planted reasoning", "#B07AA1"),
}
DEFAULT_CONDITIONS = ["explicit", "explicit_adv", "s8", "s16", "mt"]
FALLBACK_COLORS = ["#59A14F", "#E15759", "#4E79A7", "#F28E2B", "#B07AA1", "#76B7B2", "#EDC948"]

# The multiturn arm is non-Anthropic only — those bars are skipped entirely.
MT_SKIP_MODELS = ANTHROPIC

FS_TICK = 10; FS_LABEL = 10; FS_TITLE = 12; FS_LEGEND = 9; FS_ANNOT = 8

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _darken(hex_color, factor=0.7):
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"


def _ci_95(detected, total):
    """95% Clopper-Pearson CI on the detection proportion (in percent)."""
    from scipy.stats import beta as beta_dist
    if total == 0:
        return 0.0, 0.0
    k, n = detected, total
    lo = beta_dist.ppf(0.025, max(k, 0.5), n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(0.975, k + 1, max(n - k, 0.5)) if k < n else 1.0
    return lo * 100, hi * 100


def _load_condition(cond):
    path = EXP_DIR / "logs" / cond / "monitor_3level.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — run monitor.py --log-name {cond} first")
    with open(path) as f:
        per_model = json.load(f)["per_model"]
    is_mt = cond == "mt"
    stats = {}
    for model in MODEL_ORDER:
        if is_mt and model in MT_SKIP_MODELS:
            stats[model] = None            # no bar for this model/condition
            continue
        pm = per_model.get(model)
        if pm is None:
            stats[model] = (0, 0)
        else:
            detected = pm["detected"]
            total = pm["detected"] + pm["undetected"]
            stats[model] = (detected, total)
    return stats


def make_plot(conditions, out_path):
    all_stats = {c: _load_condition(c) for c in conditions}
    n_m, n_c = len(MODEL_ORDER), len(conditions)
    bw, gap = 0.8 / n_c, 0.35
    group_step = n_c * bw + gap

    fig, ax = plt.subplots(figsize=(11, 3.8))
    for ci, cond in enumerate(conditions):
        stats = all_stats[cond]
        color = CONDITION_STYLE.get(cond, (cond, FALLBACK_COLORS[ci % len(FALLBACK_COLORS)]))[1]
        cc = _darken(color)
        for j, m in enumerate(MODEL_ORDER):
            st = stats[m]
            if st is None:                 # skipped bar (e.g. mt + Anthropic)
                continue
            yes, tot = st
            x = j * group_step + ci * bw
            rate = 100 * yes / tot if tot else 0.0
            ax.bar(x, rate, bw, color=color, edgecolor="white", linewidth=0.3)
            hi = 0.0
            if tot > 0:
                lo, hi = _ci_95(yes, tot)
                ax.plot([x, x], [lo, hi], color=cc, linewidth=0.8, solid_capstyle="round")
                ax.plot([x - bw * 0.22, x + bw * 0.22], [lo, lo], color=cc, linewidth=0.8)
                ax.plot([x - bw * 0.22, x + bw * 0.22], [hi, hi], color=cc, linewidth=0.8)
            ax.text(x, hi + 2, f"{tot}", ha="center", va="bottom",
                    fontsize=FS_ANNOT, color="#777777")

    centers = np.arange(n_m) * group_step + (n_c - 1) * bw / 2
    ax.set_xticks(centers)
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], fontsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_ylabel("Monitor detection rate (%)", fontsize=FS_LABEL)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("Setting 3: Hard MCQA with Positional Hints", fontsize=FS_TITLE)

    handles = [mpatches.Patch(facecolor=CONDITION_STYLE.get(c, (c, FALLBACK_COLORS[ci % 7]))[1],
                              label=CONDITION_STYLE.get(c, (c, None))[0])
               for ci, c in enumerate(conditions)]
    ax.legend(handles=handles, fontsize=FS_LEGEND, loc="upper right",
              framealpha=0.92, ncol=2, handlelength=1.2, handleheight=0.9)
    ax.yaxis.grid(True, alpha=0.15, linewidth=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Setting 3 detection-rate figure")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--out", type=str, default=str(EXP_DIR / "figures" / "S3_attribution_rate.pdf"))
    args = parser.parse_args()
    make_plot(args.conditions, Path(args.out))


if __name__ == "__main__":
    main()
