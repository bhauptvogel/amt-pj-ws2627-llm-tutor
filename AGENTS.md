# AGENTS.md

Modular benchmark for how well language models **tutor**: not whether they know
the answer, but whether they guide a student to it without giving it away. Models
(hosted APIs, local open weights) sit behind one interface; tasks, datasets and
scorers are plugins picked in config.

AMT PJ, TU Berlin, WiSe 2026/27 · supervisors Ghanime, Do, An (Fraunhofer FOKUS) ·
Option A, a single pedagogical benchmark. Group `llm-tutor`: Ben, Nermine, Simon.

| Layer | Code | Owner |
| --- | --- | --- |
| Model: adapters, runner, config | `src/llmtutor/models/` and `src/llmtutor/*.py` | Ben |
| Educational: tasks, test set, rubric | `src/llmtutor/tasks/`, `data/` | Nermine |
| Evaluation: judge, agreement, errors | `src/llmtutor/evaluation/` | Simon |

Ownership means "who to ask"; never silently change another layer's rubric,
prompt or metric.

**Status:** runnable scaffold. The model layer works end to end: one
OpenAI-compatible adapter (configured for OpenRouter), a name-based adapter
registry, a sequential runner with retries and run artefacts, and a small CLI.
`tasks/` and `evaluation/` are empty packages waiting for their owners; `data/`
does not exist yet.

## Invariants

1. Adding a model, task or scorer never edits the core (`types.py`, `config.py`,
   `runner.py`, `report.py`, `cli.py`). A plugin registers a class under a name plus
   a config entry. If a change needs a core edit, raise it instead of special-casing.
2. Runs are reproducible: same dataset, prompt, sampling and seed for every model;
   raw responses are kept.
3. Judge scores count only with agreement against our human annotations — macro F1
   and Cohen's κ (unweighted *and* quadratic-weighted; `no < partly < yes` is
   ordinal) — plus an error analysis.
4. Editing a prompt or rubric invalidates earlier runs: bump the version.
5. No invented numbers.

## Layout

```
configs/         run configs (TOML); smoke.toml is the reference example
runs/            run artefacts, one directory per model entry (gitignored)
src/llmtutor/    types, registry, config, runner, report, cli
  models/        Model protocol, adapter registry, api.py (OpenAI-compatible, OpenRouter)
  tasks/         empty package — Nermine's layer
  evaluation/    empty package — Simon's layer
tests/           offline by default; fakes.py scripts a fake adapter
```

`material/` and `presentations/` are gitignored local material; never commit them.

## Commands

```sh
uv sync
cp .env.example .env      # then add OPENROUTER_API_KEY
uv run llmtutor run --config configs/smoke.toml
uv run pytest
uv run ruff check --fix && uv run ruff format
```

One run writes `runs/<run-id>/` with `manifest.json`, a copy of the config and
`calls.jsonl` (one record per attempt).

## Conventions

- Python 3.12, `src/` layout, `uv`, absolute imports, type hints, ruff at line 100.
- English everywhere. API keys from env, never from repo or config: a model entry
  names the variable in `api_key_env`, it never holds the key.
- `temperature = 0.0` by default; record model id, adapter, sampling, prompt
  version and dataset in the run.
- Tests never touch the network: the `fake` adapter in `tests/fakes.py` plays a
  scripted scenario, the real adapter is tested over `httpx.MockTransport`, and only
  `uv run pytest -m live` (skipped without a key) calls a provider.
- Metric code needs a test with a hand-computed expected value.
- When judge and annotators disagree, assume the judge is wrong.
- `GLOSSARY.md` is the vocabulary: use its terms in code, tests and issues, and keep
  it a glossary rather than a spec.
