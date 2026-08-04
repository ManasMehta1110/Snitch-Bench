"""Build the headline evidence-grounding comparison figure: frontier baseline
(aggregate, ~0%) vs. trained overseer (multi-seed mean +/- std), per class,
for both trace distributions. Uses the numbers already reported in
paper/results_baselines_draft.md / README.md -- kept as a script rather than
a notebook cell so the figure is regeneratable if numbers change.
"""
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ["REWARD_HACKER", "LAZY", "DECEIVER"]

# Aggregate frontier-model baseline evidence-bonus hit rate per class
# (from the 10 model x dataset configurations, paper/results_baselines_draft.md).
BASELINE = {"REWARD_HACKER": 0.0, "LAZY": 0.0, "DECEIVER": 0.0006}

# Trained overseer, LR=2e-5, 3-seed mean +/- std, per trace distribution.
TRAINED = {
    "original": {
        "REWARD_HACKER": (0.128, 0.137),
        "LAZY": (0.283, 0.148),
        "DECEIVER": (0.741, 0.151),
    },
    "realistic": {
        "REWARD_HACKER": (0.0, 0.0),  # undefined for 2/3 seeds; shown as 0 with a note
        "LAZY": (0.345, 0.522),
        "DECEIVER": (0.520, 0.145),
    },
}

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

x = np.arange(len(CLASSES))
width = 0.32  # < 0.35 so paired bars have a visible gap rather than touching edge-to-edge
bar_gap = 0.02

for ax, (dist_name, dist_label) in zip(
    axes, [("original", "Original (rule-based) traces"), ("realistic", "Realistic (LLM-generated) traces")]
):
    baseline_vals = [BASELINE[c] * 100 for c in CLASSES]
    trained_means = [TRAINED[dist_name][c][0] * 100 for c in CLASSES]
    trained_stds = [TRAINED[dist_name][c][1] * 100 for c in CLASSES]

    ax.bar(x - width / 2 - bar_gap / 2, baseline_vals, width, label="Frontier baseline (aggregate)",
           color="#b0b0b0", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2 + bar_gap / 2, trained_means, width, yerr=trained_stds, capsize=4,
           label="Trained overseer (3-seed mean $\\pm$ std)",
           color="#2b6cb0", edgecolor="black", linewidth=0.5, error_kw=dict(elinewidth=1.1))

    # The baseline bars are genuinely ~0%, which without a label is visually
    # indistinguishable from missing data -- label each one explicitly, offset
    # above the x-axis line so it never sits on top of the tick labels below it.
    for i, v in enumerate(baseline_vals):
        ax.annotate(f"{v:.2g}%" if v > 0 else "0%", xy=(i - width / 2 - bar_gap / 2, 3),
                    ha="center", va="bottom", fontsize=10, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels(["REWARD\nHACKER", "LAZY", "DECEIVER"], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title(dist_label, fontsize=15)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.6, len(CLASSES) - 1 + 0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Evidence-bonus hit rate (%)\n(among correctly-classified verdicts)", fontsize=13)

# Short, purely-vertical annotation anchored directly above the empty
# REWARD_HACKER column in the realistic panel -- avoids the earlier diagonal
# arrow that cut across the LAZY bar and the "0%" label.
axes[1].annotate(
    "metric undefined here:\nmodel rarely classifies this\nclass correctly on realistic traces",
    xy=(0, 3), xytext=(0, 30),
    ha="center", va="bottom", fontsize=10.5, color="#555555",
    arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8,
                     connectionstyle="arc3,rad=0"),
)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=False, fontsize=13)

fig.tight_layout(rect=[0, 0, 1, 0.93])

out_path = "figures/evidence_grounding_comparison.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved {out_path}")
