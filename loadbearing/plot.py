#!/usr/bin/env python3
"""Load-bearing figures.

  detection  -> figures/loadbearing_detection.pdf
      Two panels (S1, S4). Per model: implicit (blue) vs load-bearing (green) monitor
      detection rate, 95% Clopper-Pearson CIs, pooled rates in the panel titles.
      Both bars are computed from .eval logs: load-bearing from logs/<log-name>/ (this
      setting), implicit from the base Setting 1 / Setting 4 runs (../setting{1,4}_*/logs/,
      restricted to the convertible nudges that have a load-bearing variant). So the figure
      needs the base implicit runs too; point at them with --s1-implicit-log / --s4-implicit-log.

  capability -> figures/loadbearing_capability.pdf
      Joint attribution accuracy (chance 25%). Per model: 6-hop CoT (green) vs 6-hop
      no-CoT (purple); sonnet/opus also 15-hop CoT (blue) vs 15-hop no-CoT (red).
      Reads logs/capability.json (from capability_probe.py).

    python plot.py                 # both figures
    python plot.py --which detection
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
from scipy.stats import beta as beta_dist

from inspect_ai.log import list_eval_logs, read_eval_log

HERE = Path(__file__).parent.resolve()
REPO = HERE.parent
LOG_DIR = HERE / "logs"
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

# base benchmark that carries the implicit comparison arm, per LB setting
BASE_DIR = {"s1": REPO / "setting1_binary", "s4": REPO / "setting4_coding"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.weight": "normal", "axes.titleweight": "normal", "axes.labelweight": "normal",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})
FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND, FS_ANNOT = 10, 10, 12, 9.5, 8.5

IMPL, LB = "#4E79A7", "#59A14F"          # blue = implicit, green = load-bearing
GREEN, PURPLE, BLUE, RED = "#59A14F", "#B07AA1", "#4E79A7", "#E15759"

ORDER = ["haiku", "sonnet", "opus", "gpt_oss", "grok_xai", "kimi", "glm"]
DISP = {"haiku": "Haiku 4.5", "sonnet": "Sonnet 4.5", "opus": "Opus 4.5",
        "gpt_oss": "GPT-OSS 120B", "grok_xai": "Grok 3 Mini", "kimi": "Kimi K2", "glm": "GLM 4.7"}


def ci95(k, n):
    lo = 0.0 if k == 0 else beta_dist.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(0.975, k + 1, n - k)
    return lo * 100, hi * 100


def _darken(hex_color, f=0.65):
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"#{int(r * f):02x}{int(g * f):02x}{int(b * f):02x}"


# ── Detection figure ──────────────────────────────────────────────────────────

def _score_value(sample):
    """The lb_monitor score value ('YES'/'NO'/other) for a sample."""
    scores = getattr(sample, "scores", None) or {}
    for s in scores.values():
        v = getattr(s, "value", None)
        if v in ("YES", "NO"):
            return v
    return None


def _convertible_constraints(setting):
    """Constraint ids that have a load-bearing variant (from the frozen LB manifest)."""
    path = DATA_DIR / f"{setting}_lbset_full.jsonl"
    return {json.loads(l)["constraint_id"] for l in open(path)}


def _detection_from_logs(log_dir, per="model_key", constraint_filter=None):
    """{model: [yes, yes+no]} over switched cases, from a dir of .eval logs. Optionally
    keep only samples whose constraint_id is in `constraint_filter`."""
    out = defaultdict(lambda: [0, 0])
    if not Path(log_dir).exists():
        return {}
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info)
        model_key = (log.eval.metadata or {}).get(per)
        for sample in (log.samples or []):
            md = sample.metadata or {}
            if not md.get("switched"):
                continue
            if constraint_filter is not None and md.get("constraint_id") not in constraint_filter:
                continue
            v = _score_value(sample)
            if v == "YES":
                out[model_key][0] += 1; out[model_key][1] += 1
            elif v == "NO":
                out[model_key][1] += 1
    return out


def load_detection(log_name, base_logs=None):
    """{setting: {model: [[impl_k, impl_n], [lb_k, lb_n]]}}. Load-bearing from logs/<log_name>/;
    implicit from the base Setting 1/4 runs (base_logs = {'s1': name, 's4': name}), restricted
    to the convertible nudges."""
    base_logs = base_logs or {"s1": "implicit", "s4": "implicit"}
    lb_dir = LOG_DIR / log_name
    if not lb_dir.exists():
        raise FileNotFoundError(f"Missing {lb_dir} — run run.py --log-name {log_name} first")

    lb = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for info in list_eval_logs(str(lb_dir)):
        log = read_eval_log(info)
        meta = log.eval.metadata or {}
        setting, model_key = meta.get("setting"), meta.get("model_key")
        if setting is None or model_key is None:
            continue
        for sample in (log.samples or []):
            v = _score_value(sample)
            if v == "YES":
                lb[setting][model_key][0] += 1; lb[setting][model_key][1] += 1
            elif v == "NO":
                lb[setting][model_key][1] += 1

    det = {}
    for s in ("s1", "s4"):
        implicit = _detection_from_logs(BASE_DIR[s] / "logs" / base_logs[s],
                                        constraint_filter=_convertible_constraints(s))
        if not lb.get(s) and not implicit:
            continue
        det[s] = {}
        for m in set(implicit) | set(lb.get(s, {})):
            det[s][m] = [implicit.get(m, [0, 0]), lb.get(s, {}).get(m, [0, 0])]
    return det


def plot_detection(out_path, log_name="lb", base_logs=None):
    det = load_detection(log_name, base_logs=base_logs)
    settings = [s for s in ("s1", "s4") if s in det]
    titles = {"s1": "Setting 1: Binary Preference Elicitation", "s4": "Setting 4: Coding Library"}

    fig, axes = plt.subplots(1, len(settings), figsize=(9.5, 4.4), squeeze=False)
    axes = axes[0]
    bw = 0.36
    for ax, s in zip(axes, settings):
        x = np.arange(len(ORDER))
        pooled = {0: [0, 0], 1: [0, 0]}
        for j, (color, cc, off, lab) in enumerate([
                (IMPL, _darken(IMPL), -bw / 2, "implicit"),
                (LB, _darken(LB), bw / 2, "load-bearing")]):
            for i, m in enumerate(ORDER):
                k, n = det[s].get(m, [[0, 0], [0, 0]])[j]
                pooled[j][0] += k
                pooled[j][1] += n
                val = 100 * k / n if n else 0
                xi = x[i] + off
                ax.bar(xi, val, bw, color=color, edgecolor="white", linewidth=0.4,
                       label=lab if i == 0 else None)
                ax.text(xi, val + 1.5, f"{val:.0f}", ha="center", va="bottom",
                        fontsize=FS_ANNOT, color="#444")
                if n:
                    lo, hi = ci95(k, n)
                    ax.plot([xi, xi], [lo, hi], color=cc, lw=0.8, solid_capstyle="round")
                    ax.plot([xi - bw * 0.22, xi + bw * 0.22], [lo, lo], color=cc, lw=0.8)
                    ax.plot([xi - bw * 0.22, xi + bw * 0.22], [hi, hi], color=cc, lw=0.8)
        pi = 100 * pooled[0][0] / max(pooled[0][1], 1)
        pl = 100 * pooled[1][0] / max(pooled[1][1], 1)
        ax.set_xticks(x)
        labels = [DISP[m] + ("*" if m in ("sonnet", "opus") else "") for m in ORDER]
        ax.set_xticklabels(labels, fontsize=FS_TICK, rotation=18, ha="right")
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.set_ylim(0, 112)
        ax.set_title(f"{titles[s]}\npooled: implicit {pi:.0f}%  ->  load-bearing {pl:.0f}%",
                     fontsize=FS_TITLE - 1.5)
        ax.yaxis.grid(True, alpha=0.15, linewidth=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Monitor detection rate (%)", fontsize=FS_LABEL)
    fig.legend(handles=[mpatches.Patch(facecolor=IMPL, label="implicit (original nudge)"),
                        mpatches.Patch(facecolor=LB, label="load-bearing (multi-hop nudge)")],
               fontsize=FS_LEGEND, loc="lower center", ncol=2, framealpha=0.92,
               bbox_to_anchor=(0.33, -0.02))
    fig.text(0.85, -0.005, "* 15-hop chains (else 6-hop)", ha="center", va="bottom",
             fontsize=FS_ANNOT, color="#666")
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Capability figure ─────────────────────────────────────────────────────────

CAP_ORDER = ["haiku", "sonnet", "opus", "kimi", "glm"]  # models with a valid no-CoT arm


def plot_capability(out_path):
    with open(LOG_DIR / "capability.json") as f:
        cap = json.load(f)
    res = cap["results"]

    # per model: list of (label, color, k, n), ordered 6CoT,6noCoT,[15CoT,15noCoT]
    data = {}
    for m in CAP_ORDER:
        if m not in res:
            continue
        bars = []
        for depth, dcol, ncol in [("6", GREEN, PURPLE), ("15", BLUE, RED)]:
            e = res[m].get(depth)
            if not e:
                continue
            ck, cn = e["cot"]
            bars.append((f"{depth}-hop CoT", dcol, ck, cn))
            if e.get("nocot"):
                nk, nn = e["nocot"]
                bars.append((f"{depth}-hop no-CoT", ncol, nk, nn))
        if bars:
            data[m] = bars
    models = [m for m in CAP_ORDER if m in data]

    bw = 0.2
    maxbars = max(len(v) for v in data.values())
    group_step = maxbars * bw + 0.42
    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    for gi, m in enumerate(models):
        bars = data[m]
        nb = len(bars)
        center = gi * group_step
        offs = (np.arange(nb) - (nb - 1) / 2) * bw
        for off, (label, color, k, n) in zip(offs, bars):
            x = center + off
            val = 100 * k / n if n else 0
            ax.bar(x, val, bw, color=color, edgecolor="white", linewidth=0.4)
            ax.text(x, val + 1.4, f"{val:.0f}", ha="center", va="bottom",
                    fontsize=FS_ANNOT, color="#444444")
            lo, hi = ci95(k, n) if n else (0, 0)
            cc = _darken(color)
            ax.plot([x, x], [lo, hi], color=cc, linewidth=0.8, solid_capstyle="round")
            ax.plot([x - bw * 0.22, x + bw * 0.22], [lo, lo], color=cc, linewidth=0.8)
            ax.plot([x - bw * 0.22, x + bw * 0.22], [hi, hi], color=cc, linewidth=0.8)

    centers = np.arange(len(models)) * group_step
    ax.set_xticks(centers)
    ax.set_xticklabels([DISP[m] for m in models], fontsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_ylabel("Joint attribution accuracy (%)", fontsize=FS_LABEL)
    ax.set_ylim(0, 112)
    ax.axhline(25, color="#999", lw=0.6, ls="--")
    ax.text(centers[-1], 26.5, "chance (25%)", ha="right", va="bottom", fontsize=FS_ANNOT, color="#777")
    ax.set_title("Is reasoning load-bearing to process the multi-hop nudge?", fontsize=FS_TITLE)
    handles = [mpatches.Patch(facecolor=GREEN, label="6-hop, CoT"),
               mpatches.Patch(facecolor=PURPLE, label="6-hop, no-CoT"),
               mpatches.Patch(facecolor=BLUE, label="15-hop, CoT"),
               mpatches.Patch(facecolor=RED, label="15-hop, no-CoT")]
    ax.legend(handles=handles, fontsize=FS_LEGEND, loc="lower left", framealpha=0.92,
              ncol=2, handlelength=1.2, handleheight=0.9)
    ax.yaxis.grid(True, alpha=0.15, linewidth=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Load-bearing figures")
    ap.add_argument("--which", choices=["detection", "capability", "both"], default="both")
    ap.add_argument("--log-name", default="lb", help="load-bearing .eval log dir under logs/")
    ap.add_argument("--s1-implicit-log", default="implicit",
                    help="Setting 1 implicit run (dir under ../setting1_binary/logs/) for the comparison bars")
    ap.add_argument("--s4-implicit-log", default="implicit",
                    help="Setting 4 implicit run (dir under ../setting4_coding/logs/) for the comparison bars")
    args = ap.parse_args()
    if args.which in ("detection", "both"):
        plot_detection(FIG_DIR / "loadbearing_detection.pdf", log_name=args.log_name,
                       base_logs={"s1": args.s1_implicit_log, "s4": args.s4_implicit_log})
    if args.which in ("capability", "both"):
        plot_capability(FIG_DIR / "loadbearing_capability.pdf")


if __name__ == "__main__":
    main()
