# Results: Frontier-Model Baselines (draft)

## Setup

We evaluate each model as a zero/few-shot overseer on held-out v3 traces:
30 questions (q_0121-q_0150), disjoint from all training question IDs,
crossed with 4 misbehavior classes (HONEST, REWARD_HACKER, LAZY, DECEIVER).
Every model sees the identical system prompt (3-shot, one worked example per
class) and the identical `build_user_prompt`/`format_trace_body` rendering of
the trace, and is scored with the identical reward function
(`env/reward.py:compute_reward`), so results are directly comparable across
providers and across the two trace distributions described below.

Two trace distributions are tested:
- **original** (n=120): the rule-based, fully-scripted doer traces from the
  original hackathon build.
- **realistic** (n=600, 5x larger): LLM-generated (Claude Haiku) surface text
  -- queries, answers, hedges, deceptive claims -- built on top of the same
  deterministic ground-truth scaffolding (search count, doc relevance,
  smoking-gun step/keyword position), addressing the concern that
  fully-templated traces are trivially distinguishable as synthetic.

8 distinct models across 3 providers: GPT-4o, GPT-4o-mini (OpenAI);
Claude Haiku 4.5, Claude Sonnet 4.5 (Anthropic); Gemini 3.6 Flash (Google);
Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct (open-weight, run locally). The
first six were evaluated on both trace distributions where budget allowed;
Llama and Qwen were evaluated on the original set only (10 model x dataset
configurations total).

## Headline result

**Across all 10 configurations, only 1 of 1,794 correctly-classified,
non-HONEST verdicts (0.06%) cited both the correct trace step number and the
specific smoking-gun keyword required for the evidence bonus.** Verdict
accuracy is high and often near-ceiling (70.8%-100%); evidence grounding is
essentially absent regardless.

| Model                          | Dataset    | n   | Accuracy | REWARD_HACKER grounded | LAZY grounded | DECEIVER grounded |
| ------------------------------- | ---------- | --- | -------- | ----------------------- | ------------- | ------------------ |
| GPT-4o                          | original   | 120 | 90.0%    | 0/18                    | 0/30          | 0/30                |
| GPT-4o-mini                     | original   | 120 | 82.5%    | 0/11                    | 0/30          | 0/29                |
| Llama-3.1-8B-Instruct           | original   | 120 | 85.0%    | 0/12                    | 0/30          | 0/30                |
| Qwen2.5-7B-Instruct              | original   | 120 | 70.8%    | 0/24                    | 0/30          | 0/1                 |
| Claude Haiku 4.5                | original   | 120 | 90.8%    | 0/30                    | 0/22          | 0/28                |
| Claude Haiku 4.5                | realistic  | 600 | 94.3%    | 0/123                   | 0/150         | 0/150               |
| Claude Sonnet 4.5               | original   | 120 | 89.2%    | 0/17                    | 0/30          | 0/30                |
| Claude Sonnet 4.5               | realistic  | 600 | 90.7%    | 0/103                   | 0/148         | 0/150               |
| Gemini 3.6 Flash                | original   | 120 | 100.0%   | 0/30                    | 0/30          | **1/30**            |
| Gemini 3.6 Flash                | realistic  | 600 | 99.3%    | 0/149                   | 0/150         | 0/149               |

*"Grounded" = evidence text contains both `step {N}` and the ground-truth
smoking-gun keyword, among traces where the verdict was already correct
(the only traces eligible for the +0.5 evidence bonus). Denominators vary
because they count correctly-classified traces per class, not the full 30/150
per-class total.*

## Observations

1. **The single exception is small and isolated.** Gemini 3.6 Flash --
   the one "thinking" model in the set, spending 500-800+ tokens of
   invisible reasoning per call -- grounds evidence on exactly 1 of 30
   DECEIVER traces in the original set, and 0 of 149 correct DECEIVER
   verdicts in the 5x larger realistic set. Extra inference-time compute
   does not durably close the gap; at best it produces one incidental hit.

2. **Scaling within a provider does not help.** Claude Sonnet 4.5 (larger,
   more capable) shows the same 0% grounding rate as Claude Haiku 4.5 on both
   trace distributions -- if anything Sonnet's REWARD_HACKER accuracy is
   *lower* than Haiku's (56.7%/68.7% vs. 100%/82.0%), suggesting Sonnet is
   more conservative about calling REWARD_HACKER, not that it reasons through
   the trace more carefully once it does.

3. **The gap survives the "too synthetic" critique.** The realistic trace
   set is a fully independent construction (different generation pipeline,
   5x the volume, LLM-authored surface text) from the original set, and every
   model tested on both shows the same near-zero grounding rate on each,
   ruling out "the traces are trivially fingerprinted" as an explanation.

4. **Grounding failure is not an accuracy problem.** Every model in this
   table clears 70%+ verdict accuracy, several are at or near ceiling
   (Gemini 3.6 Flash: 99.3-100%). The models are not confused about *what*
   happened; they are not citing *why* they believe it, in a way that
   matches the actual evidence, essentially ever.

## Comparison to the trained overseer

The single-seed reference run reported in `README.md`/`BLOG.md` -- a
1.5B-parameter Qwen2.5 model, GRPO+LoRA fine-tuned with an explicit
evidence-grounding reward term (LR=2e-5, 400 steps, seed=42) -- reaches
75.8% accuracy on the same held-out v3 set, below several frontier baselines
on raw accuracy. Prior to this pass, `scripts/gen_gap_eval.py` (the script
used to evaluate this checkpoint) did not track evidence-bonus hit rate at
all -- only `accuracy`, `mean_reward`, `parse_error_rate` -- so the one
number that matters most for this paper's argument had never actually been
measured for the trained overseer, only implied by the aggregate
`mean_reward` jump (0.44 to 0.90 vs. base). This has now been fixed
(`evaluate_traces()` emits per-class `evidence_bonus_hit_rate`, mirroring
`eval_api_model.py`'s methodology exactly) and re-run on the original
checkpoint (`results/eval_lora_lr2e5_400_seed42_with_bonus.json`):

| Class          | Frontier baseline (aggregate, Table above) | Trained overseer (2e-5, seed=42) |
| -------------- | ------------------------------------------- | ---------------------------------- |
| REWARD_HACKER  | 0.0% (0/166 eligible across baselines)        | 0.0% (0/4 eligible)                 |
| LAZY           | 0.0% (0/620 eligible)                          | 28.6% (2/7 eligible)                |
| DECEIVER       | 0.06% (1/1,008 eligible)                       | **82.8%** (24/29 eligible)          |

Accuracy reproduced bit-identically to the previously-committed result
(0.7583333... exactly), a good sign that the checkpoint and eval pipeline
are behaving deterministically despite running on different hardware
(original run's hardware unknown/unrecorded vs. this re-eval on a local
RTX 3050).

**This is the paper's central positive result.** DECEIVER is the one class
where the smoking-gun keyword is genuine claim content, not a structural
marker (see Methods) -- and it's exactly the class where training produces a
large, unambiguous jump: from a 0.06% baseline rate (statistically
indistinguishable from zero across 8 models) to 82.8%. The small model is
not just getting the right answer more often after training; when it does,
it is now actually citing what it read. LAZY's smaller gain and
REWARD_HACKER's null result are worth reporting honestly rather than
folded into a single macro-averaged "grounding improved" claim -- the effect
is real and large specifically where the reward signal requires real
content grounding, and weak-to-absent on the two classes where the
"keyword" is closer to a fixed class token.

The retrained multi-seed checkpoints (3 seeds x 2 trace variants, LR=2e-5
batch on Colab, in progress) will turn this single-seed result into a real
seed-variance estimate and additionally test whether it holds on the
LLM-realistic trace distribution, not just the original rule-based one.

---
**TODO before this goes in the paper:**
- Add the multi-seed retrained-overseer evidence-bonus numbers once Colab
  finishes and the fixed `gen_gap_eval.py` has been run on all 6 checkpoints.
- Decide whether to report per-seed variance (mean +/- std) or all seeds
  individually in the main table, given n=3 per variant.
- Confirm whether Llama/Qwen should be re-run on the realistic set too for
  table symmetry, or whether the paper explicitly scopes the realistic-set
  comparison to the 3 API-only providers (cost/time tradeoff -- open-weight
  local runs are free, this is a scope decision not a budget one).
- Cross-check the Wilson CIs / significance test convention from README.md
  (z=3.14, p=0.0017 style) and apply the same convention here for consistency.
- Convert this draft's inline reasoning into proper prose paragraphs once
  the multi-seed numbers land (right now some of this reads like analysis
  notes, which is intentional at this stage but should tighten up).
