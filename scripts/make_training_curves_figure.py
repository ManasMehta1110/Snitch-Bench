"""Build a training-curves figure: held-out eval reward vs. training step,
for both learning rates (2e-5 vs 5e-6), all 3 seeds each, both trace
distributions. Pulls real per-step log_history from each checkpoint's
trainer_state.json (already downloaded locally under checkpoints/).
"""
import json
import glob
import matplotlib.pyplot as plt

SEEDS = [1, 2, 3]
LRS = [("lr2e5", "LR=2e-5", "#2b6cb0"), ("lr5e6", "LR=5e-6", "#c05621")]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

for ax, (dist_name, dist_label) in zip(
    axes, [("original", "Original (rule-based) traces"), ("realistic", "Realistic (LLM-generated) traces")]
):
    for lr_key, lr_label, color in LRS:
        for i, seed in enumerate(SEEDS):
            ckpt_key = f"{dist_name}_seed{seed}_{lr_key}"
            state_path = f"checkpoints/{ckpt_key}/trainer_state.json"
            try:
                state = json.load(open(state_path))
            except FileNotFoundError:
                continue
            log_history = state["log_history"]
            points = sorted(
                [(e["step"], e["eval_reward"]) for e in log_history if "eval_reward" in e]
            )
            if not points:
                continue
            steps, rewards = zip(*points)
            ax.plot(steps, rewards, color=color, linewidth=1.4, alpha=0.85,
                     label=lr_label if i == 0 else None)

    ax.set_title(dist_label, fontsize=14)
    ax.set_xlabel("Training step", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Held-out eval reward\n(mean over held-out v3 traces)", fontsize=12)
axes[0].legend(loc="upper left", frameon=False, fontsize=11)

fig.tight_layout()
out_path = "figures/training_curves_multiseed.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved {out_path}")
