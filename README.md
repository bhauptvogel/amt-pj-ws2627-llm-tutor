# Benchmarking Language Models as Tutors

A framework that evaluates language models on pedagogical tasks, e.g. giving
feedback on a wrong answer without revealing the solution. General benchmarks
(MMLU, ARC, GSM8K, IFEval) test knowledge; educational work (MRBench, BEA 2025)
defines good tutoring, but comparing models still means one-off scripts.

Here models (hosted APIs and local open weights) sit behind one interface, and
tasks, datasets and scoring methods plug in through config.

**Status:** runnable scaffold. The model layer is real: one OpenAI-compatible adapter
(configured for OpenRouter) behind a `Model` protocol and a name-based registry, a
sequential runner with retries and run artefacts, and a small CLI. `tasks/` and
`evaluation/` are still empty packages.

```sh
uv sync
cp .env.example .env      # then add your OpenRouter key
uv run llmtutor run --config configs/smoke.toml
uv run pytest
```

A run writes `runs/<run-id>/` with `manifest.json`, a copy of the config and
`calls.jsonl` (one record per attempt).

## Open questions

### Rubric: two Task 1 anchors to settle (for Nermine, from Simon, 2026-11-25)

I ran the judge's sanity tests: synthetic tutor responses with known labels, built on the
five synthetic items in `src/llmtutor/evaluation/sanity/items_v1.jsonl`. I used a cheap
test judge (`google/gemini-2.5-flash-lite`, not a validated judge). Two runs each passed
219 of 230 checks. Apart from formatting slips by the cheap model, the misses come from
two places where our rubric can be read two ways. We as raters will hit the same
ambiguity, so I think we should settle these anchors before we freeze the rater manual.

1. **Does confirming the student's wrong answer count as revealing an answer?**
   Case S1: the tutor replies only "Yes." to a wrong answer. The spec expects
   `revealing_answer = No`, because no final answer is stated. On one of five items the
   judge chose `Yes (incorrect)`, reading "Yes." as affirming the student's wrong value as
   the final answer. Options: (a) add to the anchor "agreeing with the student's own
   answer is not revealing; only a final answer the tutor states counts", or (b) count it
   as `Yes (incorrect)` and change the expected label.

2. **Is "That's correct" to a wrong answer incoherent?**
   Case S3: the tutor replies "Good job! That's correct." The spec expects
   `coherence = No`; the judge did not say No on four of five items in the first run and
   three of five in the second, reading coherence as "consistent with what the student
   wrote". The error is already caught by `mistake_identification = No` and
   `factual_accuracy = No`. MRBench defines coherence as consistency with the student's
   previous responses. Options: (a) keep coherence for misreading or contradicting the
   student's work, and drop it from S3's expected labels, or (b) widen the anchor to
   "endorses a wrong answer as correct", which overlaps with the other two criteria.

I lean towards (a) in both cases, but the rubric is your call. Whichever we choose, I'll
bump the prompt and the sanity cases to a new version.

ODS Project *Advanced Media Technologies*, TU Berlin, winter term 2026/27
(Fraunhofer FOKUS). Group `llm-tutor`: Ben Hauptvogel, Nermine, Simon.
Architecture and conventions: `AGENTS.md`.
