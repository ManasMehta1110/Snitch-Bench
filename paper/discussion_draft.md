# Discussion (draft)

## Why does training help DECEIVER so much more than the other two classes?

The most direct reading of our results is that the evidence-grounding reward
term does not uniformly teach "cite your evidence" as a general skill -- it
teaches something closer to "cite the specific thing that makes this
verdict correct," and that only works to the extent there is a specific,
content-bearing thing to cite. For DECEIVER, the ground-truth keyword is
drawn from the doer's actual submitted claim: grounding this verdict
requires the overseer to have actually processed what the claim said and
connected it to a contradiction in the trace. For REWARD_HACKER and LAZY,
the ground-truth keyword is closer to a fixed structural marker (the
literal tool-call name, or a hedge-phrase vocabulary) that is present by
construction on every trace of that class regardless of question content --
there is comparatively little "there" there for a citation to be grounded
in, beyond recognizing the pattern at all.

If this reading is right, it has a somewhat uncomfortable implication for
scalable oversight more broadly: a reward that bonuses evidence citation
will preferentially teach genuine grounding exactly where genuine grounding
is easiest to fake as well as easiest to earn honestly, and will do least
good exactly where a structural, class-indicative marker would let a model
learn to "cite" without doing the harder work of connecting evidence to
claim. We do not have a clean way to distinguish "the model is now really
reading the claim" from "the model learned to always name the doer's first
non-stopword token when predicting DECEIVER" from the evidence-bonus metric
alone -- this is a real limitation of the metric, not just of the model,
and is worth flagging directly (see Limitations).

## What this does and doesn't say about weak-to-strong oversight

Our claims are about a specific, narrow, checkable form of grounding --
citing a step number and a keyword -- not about explanation quality,
faithfulness, or interpretability in general. A verdict can be
"grounded" by our metric while still being a fairly shallow pattern match,
and a verdict could plausibly reflect genuine understanding while still
missing the specific keyword our string-match checks for (we mitigate this
somewhat by requiring both a step reference and the keyword, and by manual
spot-checks during development, but it remains an approximation). What we
can say with more confidence is the negative result: current frontier
models, prompted directly, essentially never produce output that clears
even this checkable bar, and that this holds up across a wide model/provider
spread and two independently-built trace distributions. Whatever the true
rate of "real" grounding is, it's not being surfaced by a direct prompt in
current frontier models.

## Limitations

- **Single base model.** All training results are on a single 1.5B
  Qwen2.5 checkpoint. Whether the same reward signal produces the same
  differential effect (strong for content-groundable classes, weak for
  structural ones) on a larger or differently-pretrained base model is
  untested.
- **Three seeds.** Enough to see that DECEIVER's effect is real and that
  REWARD_HACKER's and LAZY's are not reliably distinguishable from noise at
  this sample size -- not enough for tight confidence intervals. The
  per-class eligible-trace counts (traces that are both correctly classified
  and thus eligible for the evidence bonus) are small for REWARD_HACKER and
  LAZY specifically (single digits to low double digits per seed), which is
  a sample-size problem independent of the number of seeds.
- **Doers are still deterministic policies**, even in the "realistic"
  variant -- an LLM authors the surface text, but which documents get read,
  how many tool calls occur, and whether the claim is correct are all fixed
  in advance by scripted logic. This is a controlled lab setting, not an
  adversarial one; a doer that adapted its behavior to what it expects an
  overseer to check for is a meaningfully harder and currently untested
  setting.
- **The evidence-bonus metric's own blind spot**, discussed above: it can't
  distinguish genuine claim-comprehension from having learned to name a
  salient token, especially since DECEIVER's keyword-extraction heuristic
  (first content word of the claim) is itself a fairly simple, learnable
  pattern.

## Future work

- Test whether the differential training effect (strong on content-bearing
  keywords, weak on structural ones) replicates on a larger base model, or
  whether scale narrows the gap between misbehavior classes even where
  prompting alone does not.
- A harder grounding check for DECEIVER that requires citing more than one
  content word -- e.g. a full contradicted phrase, or a citation checked by
  a secondary verifier model rather than substring match -- to reduce the
  chance the model is learning a shallow proxy for grounding rather than
  the real thing.
- Adaptive/adversarial doers, so the misbehavior itself is not drawn from a
  fixed script -- a harder and more realistic test of whether grounding
  transfers or was specific to this trace distribution's structure.
- Extend the same measurement (verdict-correct-but-evidence-absent) to
  other oversight-relevant domains -- code review, browsing-agent audits --
  where the analogous "smoking gun" would need a different, domain-specific
  definition.

---
**TODO before this goes in the paper:**
- This section leans on the "structural vs. content-bearing keyword"
  hypothesis fairly heavily as the explanation for the DECEIVER/REWARD_HACKER
  split -- worth considering whether a small additional analysis (e.g.
  checking whether the trained model's DECEIVER citations vary across
  traces vs. converge to a few repeated tokens) could turn this from a
  hypothesis into something closer to a tested claim before submission.
- Confirm final per-class eligible-trace counts once all evals (including
  the full-n600 realistic re-run) are locked, and cite the actual smallest-n
  case explicitly here rather than "single digits to low double digits."
