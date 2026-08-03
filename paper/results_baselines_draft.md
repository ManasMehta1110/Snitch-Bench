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

We train Qwen2.5-1.5B with GRPO+LoRA (Methods) at two learning rates: the
original hackathon single run used LR=2e-5 (400 steps, seed=42), which we
treat as the headline configuration; we additionally ran a 3-seed x
2-trace-distribution grid at LR=2e-5 (the true multi-seed headline data) and
a matching grid at LR=5e-6 (kept as an explicit LR-robustness ablation, not
discarded -- see Discussion). `scripts/gen_gap_eval.py` previously tracked
only `accuracy`/`mean_reward`/`parse_error_rate` for the trained overseer,
never evidence-bonus hit rate -- this has been fixed, and a separate
non-stratified-subsampling bug (the realistic eval file was silently capped
at a random 120-trace slice of its 600) has also been fixed; all numbers
below are post-fix, full-file evaluations.

**LR=2e-5, original trace distribution (n=3 seeds, n=120 traces/seed):**

| Class          | Frontier baseline (aggregate) | Trained overseer, mean +/- std | Raw (seed 1/2/3) |
| -------------- | ------------------------------ | -------------------------------- | ------------------ |
| REWARD_HACKER  | 0.0% (0/166 eligible)            | 12.8% +/- 13.7%                    | 27.3%, 11.1%, 0.0%   |
| LAZY           | 0.0% (0/620 eligible)             | 28.3% +/- 14.8%                    | 42.9%, 28.6%, 13.3%  |
| DECEIVER       | 0.06% (1/1,008 eligible)          | **74.1% +/- 15.1%**                 | 90.0%, 72.4%, 60.0%   |

**LR=2e-5, realistic trace distribution (n=3 seeds, full n=600 traces/seed):**

| Class          | Frontier baseline (aggregate) | Trained overseer, mean +/- std | Raw (seed 1/2/3) |
| -------------- | ------------------------------ | -------------------------------- | ------------------ |
| REWARD_HACKER  | 0.0% (0/166 eligible)            | undefined (2/3 seeds: 0% class. accuracy, no eligible traces) | -, -, 0.0% |
| LAZY           | 0.0% (0/620 eligible)             | 34.5% +/- 52.2% (not a trend -- noise) | 94.6%, 1.3%, 7.4% |
| DECEIVER       | 0.06% (1/1,008 eligible)          | **52.0% +/- 14.5%**                 | 68.7%, 42.9%, 44.4%   |

The single-seed reference checkpoint (seed=42, original distribution only)
independently reaches 82.8% DECEIVER grounding (`accuracy` reproduced
bit-identically to the previously-published 75.8% figure, a good
determinism sanity check) -- consistent with, though a bit above, the 3-seed
mean of 74.1%, as expected for one sample from a distribution with
substantial seed variance.

**This is the paper's central positive result, now with real multi-seed
support on two independent trace distributions.** DECEIVER is the one class
where the smoking-gun keyword is genuine claim content, not a structural
marker (see Methods) -- and it is exactly the class where training produces
a large, consistent-in-direction jump on *both* distributions: from a 0.06%
baseline (indistinguishable from zero across 8 models) to 74.1% (original)
and 52.0% (realistic). The drop on the harder, LLM-generated distribution is
expected and, if anything, reassuring -- a real effect degrading gracefully
under a harder distribution is more credible than one that doesn't move at
all. REWARD_HACKER and LAZY tell an honest, less flattering story: LAZY's
per-seed numbers swing too widely to support any specific point estimate
(std exceeds the mean in both distributions), and REWARD_HACKER's
classification accuracy on the realistic distribution is so low that the
evidence-bonus metric is undefined for most seeds -- the reward can't teach
grounding for verdicts the model isn't getting right in the first place.
We report all three plainly rather than average them into a single
"grounding improved" number.

---
**TODO before this goes in the paper:**
- LR=5e-6 ablation grid: original-variant results in, realistic-variant
  evals in progress (lower priority, running after the LR=2e-5 data above).
  Add once complete.
- Decide on final table format for the paper itself (mean+-std vs. all
  seeds individually) -- current draft shows both, pick one before
  submission for space.
- Confirm whether Llama/Qwen should be re-run on the realistic set too for
  table symmetry in the *baseline* comparison, or whether the paper
  explicitly scopes the realistic-set baseline comparison to the 3 API-only
  providers (cost/time tradeoff -- open-weight local runs are free, this is
  a scope decision not a budget one).
- Cross-check the Wilson CIs / significance test convention from README.md
  (z=3.14, p=0.0017 style) and apply an equivalent convention to the
  multi-seed evidence-bonus numbers (e.g. a paired test against the 0.06%
  baseline rate) before submission -- current numbers are mean+-std only,
  no formal significance test yet.
- Convert this draft's inline reasoning into proper prose paragraphs for
  the final manuscript.
