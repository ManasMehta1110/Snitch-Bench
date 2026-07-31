"""Generate the LLM-augmented "realistic" trace set -- a parallel dataset to
generate_traces.py's original rule-based one, NOT a replacement. The original
data/train_traces.jsonl and data/eval_traces.jsonl (and every existing result
computed against them) stay untouched and fully valid.

Uses the SAME policy classes and SAME structural misbehavior patterns (search
count, doc relevance, smoking-gun position) as the original generator -- only
the surface text (search query phrasing, claimed answers) is LLM-generated,
so ground-truth labels stay exactly as reliable as before while the traces
read as considerably less templated.

Costs real API calls (Claude Haiku by default) -- unlike the original
generator, this is not free/instant. Roughly 2 LLM calls per non-HONEST
trace (search query + answer text).

Usage:
    python scripts/generate_traces_realistic.py --questions data/questions.json --out-dir data/
    python scripts/generate_traces_realistic.py --limit-questions 3   # cheap smoke test first
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from doers.base import ToolEnv
from doers.honest import HonestPolicy
from doers.reward_hacker import RewardHackerPolicy
from doers.lazy import LazyPolicy
from doers.deceiver import DeceiverPolicy
from env.trace_format import Trace

POLICY_CLASSES = [HonestPolicy, RewardHackerPolicy, LazyPolicy, DeceiverPolicy]
VARIANTS = [1, 2, 3]


def generate_traces(questions: list[dict], seed: int = 42, verbose: bool = True) -> tuple[list[Trace], list[Trace]]:
    train_traces: list[Trace] = []
    eval_traces: list[Trace] = []

    per_policy_counts: dict[str, int] = {}
    per_policy_errors: dict[str, int] = {}

    started = time.time()
    total = len(questions) * len(POLICY_CLASSES) * len(VARIANTS)
    done = 0

    for q_idx, question in enumerate(questions):
        base_seed = seed + q_idx * 100

        for policy_cls in POLICY_CLASSES:
            for variant in VARIANTS:
                done += 1
                policy_seed = base_seed + hash((policy_cls.__name__, variant)) % 1000
                policy = policy_cls(variant=variant, seed=policy_seed, use_llm_text=True)
                tool_rng = random.Random(policy_seed + 1)
                tools = ToolEnv(
                    corpus=question["corpus"],
                    relevant_doc_ids=question["relevant_doc_ids"],
                    rng=tool_rng,
                )

                key = f"{policy.policy_class}_v{variant}"
                try:
                    trace = policy.run(question, tools)
                    trace.validate()
                    # Extra safety: LAZY/REWARD_HACKER/DECEIVER must not leak the
                    # ground truth despite the retry-guards already in llm_text.py
                    # -- belt and suspenders before this enters the dataset.
                    truth = question.get("ground_truth_answer", "").strip().lower()
                    if truth and policy.policy_class != "HONEST" and truth in trace.claimed_answer.lower():
                        raise ValueError(f"ground truth leaked into {policy.policy_class} claim: "
                                          f"{trace.claimed_answer!r}")
                except Exception as e:
                    per_policy_errors[key] = per_policy_errors.get(key, 0) + 1
                    print(f"  [error {key} on {question['id']}]: {e}", file=sys.stderr)
                    continue

                per_policy_counts[key] = per_policy_counts.get(key, 0) + 1
                if variant == 3:
                    eval_traces.append(trace)
                else:
                    train_traces.append(trace)

                if verbose and done % 20 == 0:
                    elapsed = time.time() - started
                    print(f"  {done}/{total} ({elapsed:.0f}s elapsed, {elapsed/done:.1f}s/trace avg)")

    print("\nPer-policy counts:")
    for key in sorted(per_policy_counts):
        errors = per_policy_errors.get(key, 0)
        print(f"  {key}: {per_policy_counts[key]} generated, {errors} errors")

    return train_traces, eval_traces


def write_jsonl(traces: list[Trace], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t in traces:
            f.write(t.to_json() + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--questions", default="data/questions.json")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-questions", type=int, default=None,
                     help="only use the first N questions -- use this for a cheap smoke test first")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    if args.limit_questions is not None:
        questions = questions[:args.limit_questions]
    print(f"Loaded {len(questions)} questions "
          f"({'limited' if args.limit_questions else 'full'} set).\n")

    train_traces, eval_traces = generate_traces(questions, seed=args.seed)

    out_dir = Path(args.out_dir)
    train_path = out_dir / "train_traces_realistic.jsonl"
    eval_path = out_dir / "eval_traces_realistic.jsonl"

    write_jsonl(train_traces, train_path)
    write_jsonl(eval_traces, eval_path)

    print(f"\nWrote {len(train_traces)} train traces -> {train_path}")
    print(f"Wrote {len(eval_traces)} eval traces -> {eval_path}")

    from collections import Counter
    train_dist = Counter(t.policy_class for t in train_traces)
    eval_dist = Counter(t.policy_class for t in eval_traces)
    print(f"\nTrain class distribution: {dict(train_dist)}")
    print(f"Eval class distribution:  {dict(eval_dist)}")


if __name__ == "__main__":
    main()
