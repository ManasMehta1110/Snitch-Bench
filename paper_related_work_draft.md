# Related Work (draft)

Our work sits at the intersection of three lines of research: scalable oversight,
chain-of-thought faithfulness, and process-level verification. We position our
contribution against each below.

## Scalable oversight and weak-to-strong generalization

The scalable oversight literature asks whether a weaker supervisor (human or
model) can reliably evaluate a stronger model's outputs as capabilities scale
past the point of easy human verification. Burns et al. formalize this as
*weak-to-strong generalization*: fine-tuning a strong model on labels from a
weaker one, finding the strong model can partially generalize beyond the weak
supervisor's own competence. Bowman et al. study the closely related question
of whether weak LLM judges can reliably evaluate stronger LLM debaters,
motivating debate and consultancy protocols as scalable oversight mechanisms.
Most recent survey work (e.g. Towards Scalable Automated Alignment of LLMs)
treats the oversight *verdict* -- correct or incorrect -- as the object of
study.

Our work complicates this picture along a different axis: we hold the
overseer's classification accuracy roughly fixed (all evaluated frontier
models achieve 82-100% verdict accuracy across our misbehavior taxonomy) and
ask a separate question -- when the overseer is *correct*, does its accompanying
justification actually cite the evidence that makes the verdict correct?
Across every frontier model and both our synthetic and LLM-generated trace
sets, we find the answer is essentially never: across 10 model x dataset
baseline configurations (8 distinct models across 3 providers -- OpenAI,
Anthropic, Google -- evaluated on both a rule-based synthetic trace
distribution and an independently LLM-generated realistic one), only 1 of
1,794 correctly-classified misbehavior verdicts (0.06%) grounds its verdict
in both the correct step number and the specific evidence keyword. This
suggests weak-to-strong
oversight protocols that treat verdict accuracy as a proxy for trustworthy
supervision may be measuring the wrong thing -- a correct verdict is not
evidence that the supervision process itself is auditable or trustworthy.

## Chain-of-thought faithfulness

A separate line of work studies whether a model's *own* stated reasoning
matches the computation that actually produced its answer. Turpin et al. show
CoT explanations can be systematically unfaithful under biasing prompts;
Lanham et al. (Anthropic) measure faithfulness via interventions on the CoT
itself (truncation, paraphrasing, injected mistakes); more recent work extends
this to naturalistic (non-adversarially-biased) settings, finding unfaithfulness
is not confined to constructed bias conditions.

Our setting differs in an important way: CoT faithfulness studies the
reasoner auditing *itself* -- does the stated chain of thought reflect the
model's own internal decision process. We study a *third-party* overseer
auditing *someone else's* completed trajectory after the fact. The overseer
in our environment is not being asked to introspect on its own reasoning; it
is being asked to point to specific, checkable evidence in an external
artifact (a tool-use trace) that a human or automated grader could verify
independently. That the same qualitative failure -- confident, correct-sounding
output decoupled from the evidence that would justify it -- appears in this
distinct setting suggests the failure mode may be broader than either
literature currently frames it.

## Process reward models and step-level verification

Process reward models (PRMs) provide step-level supervision signals over a
generation, typically to guide or re-rank reasoning in math and code tasks.
Lightman et al. (PRM800K) show step-level human labels improve best-of-k
selection over outcome-only supervision; Math-Shepherd automates label
generation via Monte Carlo rollouts to remove the human-annotation
bottleneck; more recent verbalized PRMs (e.g. ThinkPRM) have the verifier
produce an explicit natural-language justification chain per step, closer in
spirit to what our overseer is asked to do.

PRMs are trained and evaluated on the *reasoning trace that produces an
answer* (typically math or code), where step correctness is often
automatically checkable (does the intermediate expression follow validly).
Our overseer instead audits a *completed agentic trajectory* -- a sequence of
tool calls and their results -- for one of four qualitatively different
misbehavior patterns (fabrication, incomplete-effort hedging, and
evidence-contradicting deception), where the "correct" step to cite is a
specific tool-use event rather than an arithmetic or logical operation. The
verbalized-justification design of models like ThinkPRM is close in spirit to
our evidence-citation requirement, but to our knowledge this citation-grounding
failure has not been measured directly in the agentic-oversight setting we
study.

## Positioning

Our contribution is not a new oversight protocol, faithfulness metric, or
process reward model. It is an empirical measurement, replicated across three
frontier model families (GPT, Claude, Gemini; 8 distinct models/versions) and
two independently-constructed trace distributions (rule-based synthetic and
LLM-generated realistic), that a specific and checkable form of oversight
failure -- correct verdicts without correct evidence grounding -- is close to
universal (0.06% hit rate, 1/1,794 eligible verdicts) in current frontier
models acting as overseers, and that this gap is *not* closed by scaling from
a smaller to a larger model within the same provider (Claude Haiku vs. Sonnet
show the same pattern) nor by using a "thinking" model with an order of
magnitude more inference-time compute (Gemini 3.6 Flash). We further show a
small (1.5B parameter) model, trained via GRPO with an explicit
evidence-grounding reward term, can be taught to close much of this gap --
suggesting the failure is one of training incentive rather than raw
capability.

---
**TODO before this goes in the paper:**
- Convert citations to proper `\cite{}` + BibTeX entries (currently author/year
  inline, need to pull exact venue/year/arXiv ID for each and build references.bib)
- Confirm the exact n counts once the full baseline table (all 6+ models x
  both trace sets) is finalized
- Add the multi-seed retraining result once Colab finishes, to firm up the
  "training incentive not raw capability" claim with real variance, not a
  single run
