# my_analysis/make_audit_figures.py
# Two figures for the write-up:
#   1. What the 410 parsed switches actually recommend (strict, answer-only, both models agreeing)
#   2. Parser vs audited switch rate and detection (strict, consensus, disputed counted low)
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("my_analysis/figures")
OUT.mkdir(exist_ok=True)

# ---------------- Figure 1: what parsed switches actually are ----------------
recs = [json.loads(l) for l in open("my_analysis/choice_labels.jsonl") if l.strip()]
switches = [r for r in recs if r["group"] == "switch"]


def category(r):
    target = "B" if r["baseline"] == "A" else "A"
    g, s = r["gpt5_strict"], r["sonnet_strict"]
    if g != s:
        return "Disputed"
    if g == target:
        return "Real switch"
    if g == r["baseline"]:
        return "Favours baseline"
    if g == "CONDITIONAL":
        return "Conditional"
    return "No recommendation"


order = ["Real switch", "Favours baseline", "Conditional", "No recommendation", "Disputed"]
colours = ["#2b6cb0", "#c3c8cf", "#a4abb5", "#868e9a", "#e1e4e8"]
counts = Counter(category(r) for r in switches)
assert sum(counts.values()) == len(switches) == 410, "counts don't add up to 410"
print("Figure 1 counts:", {k: counts[k] for k in order})

fig, ax = plt.subplots(figsize=(9, 1.9))
start = 0
for name, colour in zip(order, colours):
    width = counts[name]
    ax.barh(0, width, left=start, color=colour, label=name, edgecolor="white")
    if width >= 25:   # label only segments wide enough to hold text
        ax.text(start + width / 2, 0, f"{width}\n({width / 410:.0%})", ha="center", va="center",
                fontsize=9, color="white" if name == "Real switch" else "black")
    start += width
ax.set_xlim(0, 410)
ax.set_yticks([])
ax.set_xlabel("Parsed switches (n = 410), by what the final answer actually recommends")
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.6), frameon=False, fontsize=9)
fig.savefig(OUT / "parsed_switches.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 2: parser vs audited ----------------
# From estimate_output.txt: strict labels, both models agreeing, disputed counted low;
# 95% intervals from the bootstrap over the 150 sampled non-switches.
labels = ["Switch rate", "Detection among switches"]
parser = [40.2, 12.0]
audited = [18.0, 20.3]
lower = [14.8, 13.2]
upper = [21.6, 28.6]
yerr = [[a - lo for a, lo in zip(audited, lower)],
        [hi - a for a, hi in zip(audited, upper)]]

x = np.arange(len(labels))
w = 0.35
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar(x - w / 2, parser, w, label="Parser (as in the paper)", color="#a4abb5")
ax.bar(x + w / 2, audited, w, yerr=yerr, capsize=4, label="Audited (strict)", color="#2b6cb0")
for xi, p in zip(x, parser):
    ax.text(xi - w / 2, p + 1, f"{p:.1f}%", ha="center", fontsize=9)
for xi, a, hi in zip(x, audited, upper):
    ax.text(xi + w / 2, hi + 1, f"{a:.1f}%", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Percent")
ax.set_ylim(0, 50)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=9)
ax.set_title("Haiku 4.5, Setting 1, implicit delivery (1,021 answers)", fontsize=10)
fig.savefig(OUT / "parser_vs_audited.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved {OUT / 'parsed_switches.png'} and {OUT / 'parser_vs_audited.png'}")