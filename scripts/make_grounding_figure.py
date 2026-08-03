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

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

x = np.arange(len(CLASSES))
width = 0.35

for ax, (dist_name, dist_label) in zip(
    axes, [("original", "Original (rule-based) traces"), ("realistic", "Realistic (LLM-generated) traces")]
):
    baseline_vals = [BASELINE[c] * 100 for c in CLASSES]
    trained_means = [TRAINED[dist_name][c][0] * 100 for c in CLASSES]
    trained_stds = [TRAINED[dist_name][c][1] * 100 for c in CLASSES]

    ax.bar(x - width / 2, baseline_vals, width, label="Frontier baseline (aggregate)",
           color="#b0b0b0", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, trained_means, width, yerr=trained_stds, capsize=4,
           label="Trained overseer (3-seed mean $\\pm$ std)",
           color="#2b6cb0", edgecolor="black", linewidth=0.5)

    # The baseline bars are genuinely ~0%, which without a label is visually
    # indistinguishable from missing data -- label each one explicitly.
    for i, v in enumerate(baseline_vals):
        ax.annotate(f"{v:.2g}%" if v > 0 else "0%", xy=(i - width / 2, 1.5),
                    ha="center", fontsize=7.5, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels(["REWARD\nHACKER", "LAZY", "DECEIVER"])
    ax.set_title(dist_label, fontsize=11)
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Evidence-bonus hit rate (%)\n(among correctly-classified verdicts)")
axes[1].annotate(
    "metric undefined here:\nmodel rarely classifies\nthis class correctly\non realistic traces",
    xy=(0 + width / 2, 1), xytext=(0.55, 45),
    fontsize=7.5, ha="left", color="#555555",
    arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8),
)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False, fontsize=9)

fig.suptitle("")
fig.tight_layout(rect=[0, 0, 1, 0.94])

out_path = "figures/evidence_grounding_comparison.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved {out_path}")
