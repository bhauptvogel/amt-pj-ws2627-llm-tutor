# Benchmarking Language Models as Tutors

A framework that evaluates language models on pedagogical tasks, e.g. giving
feedback on a wrong answer without revealing the solution. General benchmarks
(MMLU, ARC, GSM8K, IFEval) test knowledge; educational work (MRBench, BEA 2025)
defines good tutoring, but comparing models still means one-off scripts.

Here models (hosted APIs and local open weights) sit behind one interface, and
tasks, datasets and scoring methods plug in through config.

**Status:** initialization. Conventions and tooling are set up, no framework code
yet.

```sh
uv sync
uv run pytest
```

ODS Project *Advanced Media Technologies*, TU Berlin, winter term 2026/27
(Fraunhofer FOKUS). Group `llm-tutor`: Ben Hauptvogel, Nermine, Simon.
Architecture and conventions: `AGENTS.md`.
