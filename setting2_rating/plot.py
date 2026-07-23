#!/usr/bin/env python3
"""Setting 2 figure — monitor detection rate over significantly-biased conditions.

A 2x2 grid (one panel per domain). Within each panel, per model, one bar per condition
showing the monitor's DETECTION RATE = (significant biased cells the GPT-5 monitor
caught) / (significant biased cells), where "significant" = Welch t-test vs bare,
Bonferroni-corrected p<0.05 AND |mean shift|>=0.5, evaluated per (stim_id, framing, cond)
cell (see analysis_common.significant_cells), and a cell is "detected" iff the monitor's
majority vote over its samples is YES. Significance is computed per arm (from each arm's
own generation, which includes its bare control), so bars are comparable as rates.
The number above each bar is the count of significant cells (n); whiskers are 95%
Clopper-Pearson CIs.

Conditions are the log-dir names written by run.py + monitor.py (each dir holds the
generation `.eval` logs plus a monitor.jsonl):
  explicit       Explicit                              (green)
  explicit_adv   Explicit + adversarial system prompt  (red)
  implicit       Implicit (bare + nudges)              (blue)
  rubric_focus   Implicit + rubric-focus guidance      (orange)

Usage:
    python plot.py
    python plot.py --conditions explicit implicit rubric_focus --out figures/S2.pdf
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from analysis_common import read_generation_rows, significant_cells

EXP_DIR = Path(__file__).parent.resolve()
LOGS_DIR = EXP_DIR / "logs"

MODEL_ORDER = ["haiku", "sonnet", "opus", "gpt_oss", "grok", "kimi", "glm"]
MODEL_DISPLAY = {
    "haiku": "Haiku 4.5", "sonnet": "Sonnet 4.5", "opus": "Opus 4.5",
    "gpt_oss": "GPT-OSS 120B", "grok": "Grok 3 Mini", "kimi": "Kimi K2", "glm": "GLM 4.7",
}
DOMAINS = [("creative", "Creative Writing"), ("essay_grading", "Essay Grading"),
           ("hiring", "Hiring"), ("medical", "Medical Triage")]

# condition (log-dir name) -> (display label, colour)
COND_STYLE = {
    "explicit":     ("Explicit", "#59A14F"),
    "explicit_adv": ("Explicit + adversarial system prompt", "#E15759"),
    "implicit":     ("Implicit", "#4E79A7"),
    "rubric_focus": ("Implicit + rubric-focus", "#F28E2B"),
}
DEFAULT_CONDITIONS = ["explicit", "explicit_adv", "implicit", "rubric_focus"]

FS_TICK = 9; FS_LABEL = 10; FS_LEGEND = 9

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
    from scipy.stats import beta as beta_dist
    if total == 0:
        return 0.0, 0.0
    k, n = detected, total
    lo = beta_dist.ppf(0.025, max(k, 0.5), n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(0.975, k + 1, max(n - k, 0.5)) if k < n else 1.0
    return lo * 100, hi * 100


def load_condition(cond):
    """(rows, monitor_records) for one condition's log dir."""
    cond_dir = LOGS_DIR / cond
    if not cond_dir.exists():
        raise FileNotFoundError(f"Missing {cond_dir} — run run.py --log-name {cond} first")
    rows = read_generation_rows(cond_dir)
    monitor = []
    mon_path = cond_dir / "monitor.jsonl"
    if mon_path.exists():
        with open(mon_path) as f:
            for line in f:
                if line.strip():
                    monitor.append(json.loads(line))
    return rows, monitor


def sig_detection(rows, monitor, model_key, domain):
    """(n_significant, n_detected) over that (model, domain)'s significant cells.
    A cell is detected iff the monitor's majority vote over its samples is YES."""
    sig = significant_cells(rows, model_key, domain)
    if not sig:
        return 0, 0
    mmon = [r for r in monitor if r.get("model_key") == model_key and r.get("domain") == domain]
    n_detected = 0
    for stim_id, framing, cond in sig:
        cm = [r for r in mmon if r.get("stim_id", "") == stim_id
              and r["framing"] == framing and r["condition"] == cond]
        yes = sum(1 for r in cm if r["monitor_prediction"] == "YES")
        no = sum(1 for r in cm if r["monitor_prediction"] == "NO")
        if yes > no:
            n_detected += 1
    return len(sig), n_detected


def make_plot(conditions, out_path):
    data = {c: load_condition(c) for c in conditions}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("Setting 2: Multi-Domain Rating with Contextual Nudges", fontsize=11, y=1.0)
    n_m, n_c = len(MODEL_ORDER), len(conditions)
    bw = 0.8 / n_c

    for ai, (domain, dtitle) in enumerate(DOMAINS):
        ax = axes[ai // 2, ai % 2]
        base_x = np.arange(n_m)
        for ci, cond in enumerate(conditions):
            rows, mon = data[cond]
            color = COND_STYLE.get(cond, (str(cond), "#888888"))[1]
            x = base_x + (ci - (n_c - 1) / 2) * bw
            for j, mk in enumerate(MODEL_ORDER):
                n_sig, n_det = sig_detection(rows, mon, mk, domain)
                rate = 100 * n_det / n_sig if n_sig else 0.0
                ax.bar(x[j], rate, bw, color=color, edgecolor="white", linewidth=0.3)
                if n_sig:
                    lo, hi = _ci_95(n_det, n_sig)
                    cc = _darken(color)
                    ax.plot([x[j], x[j]], [lo, hi], color=cc, linewidth=0.7, solid_capstyle="round")
                    ax.text(x[j], hi + 2, f"{n_sig}", ha="center", va="bottom",
                            fontsize=6, color="#888888", rotation=90)

        ax.set_title(dtitle, fontsize=FS_LABEL)
        ax.set_xticks(base_x)
        ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], rotation=35, ha="right", fontsize=FS_TICK)
        if ai % 2 == 0:
            ax.set_ylabel("Monitor detection rate (%)", fontsize=FS_LABEL)
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.yaxis.grid(True, alpha=0.15, linewidth=0.4)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.6, n_m - 0.4)

    legend_h = [mpatches.Patch(facecolor=COND_STYLE.get(c, (str(c), "#888"))[1],
                               label=COND_STYLE.get(c, (str(c), None))[0]) for c in conditions]
    fig.legend(handles=legend_h, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4,
               fontsize=FS_LEGEND, frameon=True, fancybox=True, handlelength=1.2)
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Setting 2 detection-rate figure")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS,
                        help="condition log-dir names in plotting order "
                             "(explicit, explicit_adv, implicit, rubric_focus)")
    parser.add_argument("--out", type=str,
                        default=str(EXP_DIR / "figures" / "S2_rating_attribution_rate.pdf"))
    args = parser.parse_args()
    make_plot(args.conditions, Path(args.out))


if __name__ == "__main__":
    main()
