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

ODS Project *Advanced Media Technologies*, TU Berlin, winter term 2026/27
(Fraunhofer FOKUS). Group `llm-tutor`: Ben Hauptvogel, Nermine, Simon.
Architecture and conventions: `AGENTS.md`.
