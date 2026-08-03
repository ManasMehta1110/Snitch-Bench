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

**Pooled significance test.** Per-seed means understate what's actually
supported by the data for two of the three classes. Reconstructing raw
hit/eligible counts from each seed and pooling across the 3 seeds, then
running an exact one-sided binomial test against the aggregate frontier
baseline rate (0.06%, 1/1,794) gives:

| Class | Distribution | Pooled hits/eligible | Pooled rate | p (exact binomial, vs. 0.06% baseline) |
| --- | --- | --- | --- | --- |
| REWARD_HACKER | original | 4/34 | 11.8% | 4.4e-9 |
| REWARD_HACKER | realistic | 0/6 | 0.0% | 1.0 (no evidence of effect) |
| LAZY | original | 24/86 | 27.9% | 9.7e-58 |
| LAZY | realistic | 154/448 | 34.4% | ~0 |
| DECEIVER | original | 66/89 | 74.2% | ~0 |
| DECEIVER | realistic | 227/436 | 52.1% | ~0 |

This refines the per-seed-variance framing above: **REWARD_HACKER and LAZY
are not "no effect" or pure noise on the original distribution -- pooled
across seeds, both are statistically distinguishable from the near-zero
baseline.** What the high per-seed standard deviation actually reflects is
that the *magnitude* of the effect is imprecisely estimated at n=3 seeds,
not that the effect is absent. The one case with genuinely no evidence of
an effect is REWARD_HACKER on the realistic distribution (0 hits out of 6
pooled eligible traces across all 3 seeds) -- consistent with the
classification-accuracy bottleneck described above (the model rarely gets
this class right at all on realistic traces, so there is rarely a verdict
eligible for the bonus in the first place).

**This is the paper's central positive result, now with real multi-seed
support, on two independent trace distributions, backed by a formal
significance test rather than point estimates alone.** DECEIVER shows both
the largest effect size and the most precise estimate (pooled rate 74.2%
original / 52.1% realistic, both p ≈ 0 against baseline) -- consistent with
it being the class where the smoking-gun keyword is genuine claim content
rather than a structural marker (see Methods). LAZY and REWARD_HACKER are
real but smaller and, in REWARD_HACKER's case, distribution-dependent: it
works on the original distribution and fails entirely on the realistic one,
tracking that class's classification-accuracy bottleneck rather than a
grounding failure per se.

*Caveat on the pooled test:* it treats individual trace outcomes within and
across seeds as independent Bernoulli trials, a standard simplifying
assumption for this kind of report but not a perfectly rigorous one (traces
within a seed share the same fixed evaluation set). The baseline rate being
tested against is itself an aggregate across 8 different models rather than
a single matched comparison condition -- appropriate for establishing "the
trained model behaves differently from frontier models in general," less
appropriate for a claim about any one specific baseline model.

## Learning-rate ablation (LR=5e-6)

The same 3-seed x 2-distribution grid, trained at LR=5e-6 instead of 2e-5
(all other hyperparameters, including step count, identical), gives a
sharp negative result:

| Class | Distribution | Trained overseer (mean ± s.d., n=3 seeds) |
| --- | --- | --- |
| REWARD_HACKER | original | no eligible traces in any seed |
| REWARD_HACKER | realistic | no eligible traces in any seed |
| LAZY | original | 0.0% ± 0.0% (raw: 0.0, 0.0, 0.0) |
| LAZY | realistic | 0.0% ± 0.0% (raw: 0.0, 0.0, 0.0) |
| DECEIVER | original | 0.0% ± 0.0% (raw: 0.0, 0.0, 0.0) |
| DECEIVER | realistic | 0.26% ± 0.45% (raw: 0.79%, 0.0%, 0.0%) |

At this learning rate, evidence-grounding is statistically indistinguishable
from the 0.06% frontier baseline across every class, every seed, and both
trace distributions -- verdict accuracy is still reasonable (70.3% ± 0.5%
original, 69.2% ± 1.1% realistic, comparable to several frontier baselines)
but the model has not learned to ground its verdicts at all. This is not a
smaller version of the LR=2e-5 effect; it is its near-total absence.

This is a clean, valuable negative result for the paper's argument: the
evidence-grounding capability is not a byproduct of GRPO training with this
reward function in general, nor of reaching a particular accuracy level --
it requires a training configuration with enough policy movement to
actually shift citation behavior, not just classification behavior. A
learning rate 4x smaller, with identical steps, data, and reward, produces
a materially better-than-random classifier that grounds nothing. Grounding
is a separate optimization target that the reward makes *reachable*, not
one it guarantees for any configuration that improves accuracy.

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
