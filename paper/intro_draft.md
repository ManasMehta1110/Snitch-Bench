# Introduction (draft)

As AI systems are delegated increasingly autonomous, multi-step tasks --
searching, reading, synthesizing, acting on tools -- the humans supervising
them can no longer feasibly re-derive every answer from scratch. Scalable
oversight proposes a way out: use a model (or a weaker model, or a debate
between models) to audit the trajectory instead. But an oversight signal is
only as trustworthy as the process that produces it. A verdict is not by
itself evidence that the overseer looked at the right thing.

This paper asks a narrow, checkable question about that process: **when an
AI overseer correctly flags another agent's misbehavior, does its stated
justification actually cite the evidence that makes the verdict correct?**
We are not asking whether overseers get the right answer -- prior work
already studies that axis extensively (see Related Work). We are asking
whether a correct answer is accompanied by an *auditable, checkable reason*,
in a setting built specifically so "checkable" has a precise, automatable
definition: the overseer must name the trace step and the specific detail
that gave the misbehavior away, and we can verify programmatically whether
it did.

We build an environment, **GroundingBench**, in which an overseer reads a
completed tool-use trace from a "doer" agent that answered a factual
question, and must classify the doer's behavior as honest or one of three
misbehavior types -- fabricating an answer without doing the work,
half-doing the work and giving up with a hedge, or fully doing the work and
then reporting something that contradicts it. Each misbehavior trace carries
a ground-truth "smoking gun": the exact step and keyword an honest audit
would need to reference. This lets us measure grounding directly and
cheaply, rather than relying on human judgment of explanation quality.

**Finding 1.** We evaluate 7 frontier models spanning 5 organizations -- 3
major API providers (OpenAI, Anthropic, Google) and 2 open-weight model
families (Meta's Llama, Alibaba's Qwen) -- on two independently-constructed
trace distributions -- a rule-based synthetic one and an LLM-generated one built to look
naturalistic while preserving identical ground-truth labels. Verdict
accuracy is high, often near-ceiling. Evidence grounding is not: across
1,794 correctly-classified misbehavior verdicts, only 1 (0.06%) cites both
the correct step and the correct keyword. The gap does not close with scale
within a provider (a larger Claude model shows the same pattern as a
smaller one) or with additional inference-time "thinking" compute (a
reasoning-augmented Gemini model shows the same pattern, with one small
exception). Getting the verdict right and being able to show your work
appear to be nearly independent capabilities in current frontier models,
at least as elicited by a direct prompt.

**Finding 2.** We show this gap is addressable by training, not just an
inherent ceiling. A small (1.5B parameter) model, fine-tuned with GRPO
using a reward that explicitly bonuses citing the correct evidence (on top
of, not instead of, correct classification), learns to ground its verdicts
at a rate far above any frontier baseline -- but unevenly across
misbehavior types. The effect is large and consistent across seeds and
across both trace distributions for the misbehavior type where the
"smoking gun" is genuine claim content (a contradicted assertion); it is
weak or statistically inconclusive for the two misbehavior types where the
ground-truth keyword is closer to a structural marker of the behavior
itself. We report this unevenness directly rather than average over it,
since it is itself informative: the training signal works where there is
real content to ground a citation in, and struggles where there isn't much
more than a fixed pattern to memorize.

**Contributions:**
1. An environment and reward function for measuring evidence-grounding in
   AI oversight, separable from classification accuracy, with a
   ground-truth-preserving methodology for generating naturalistic (not
   obviously synthetic) test traces.
2. An empirical measurement, at a scale and breadth (7 models, 5 organizations,
   2 independent trace distributions) that rules out the most obvious
   objections (small model set, single trace style, scale would fix it),
   that this evidence-grounding gap is close to universal in current
   frontier models used as overseers.
3. A demonstration, with real seed-to-seed variance rather than a single
   run, that an explicit training signal narrows this gap substantially for
   at least one misbehavior class, together with an honest account of where
   it does not.

---
**TODO before this goes in the paper:**
- Once final multi-seed numbers are locked (pending the full-n600 realistic
  re-eval), update the exact "1,794" / "0.06%" figures if the fix to the
  subsampling bug changes any baseline-adjacent numbers (it shouldn't --
  that bug only affected the *trained overseer's* realistic eval, not the
  frontier-model baselines, which always used the full file via
  eval_api_model.py -- but double check before submission).
- Add 1-2 sentences citing the specific related-work threads this
  positions against (currently deferred to the Related Work section itself
  -- may want a compressed one-liner here too depending on SaTML's
  page-budget norms).
- Consider whether "Finding 2" needs to preemptively flag the REWARD_HACKER
  negative result in the intro itself, or whether that's better left for
  the reader to encounter in Results/Discussion with full context. Currently
  drafted to hint at it ("weak or inconclusive... for the two... where the
  keyword is closer to a structural marker") without naming which class,
  to avoid front-loading a caveat before the reader has the setup needed to
  interpret it.
