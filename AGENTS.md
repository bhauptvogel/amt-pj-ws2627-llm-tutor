# AGENTS.md

Modular benchmark for how well language models **tutor**: not whether they know
the answer, but whether they guide a student to it without giving it away. Models
(hosted APIs, local open weights) sit behind one interface; tasks, datasets and
scorers are plugins picked in config.

AMT PJ, TU Berlin, WiSe 2026/27 · supervisors Ghanime, Do, An (Fraunhofer FOKUS) ·
Option A, a single pedagogical benchmark. Group `llm-tutor`: Ben, Nermine, Simon.

| Layer | Code | Owner |
| --- | --- | --- |
| Model: adapters, runner, config | `models/`, `config.py`, `runner.py` | Ben |
| Educational: tasks, test set, rubric | `tasks/`, `data/` | Nermine |
| Evaluation: judge, agreement, errors | `evaluation/` | Simon |

Ownership means "who to ask"; never silently change another layer's rubric,
prompt or metric.

## Invariants

1. Adding a model, task or scorer never edits the core (`runner.py`, `config.py`,
   `report.py`). A plugin registers a class under a name plus a config entry. If a
   change needs a core edit, raise it instead of special-casing.
2. Runs are reproducible: same dataset, prompt, sampling and seed for every model;
   raw responses are kept.
3. Judge scores count only with agreement against our human annotations — macro F1
   and Cohen's κ (unweighted *and* quadratic-weighted; `no < partly < yes` is
   ordinal) — plus an error analysis.
4. Editing a prompt or rubric invalidates earlier runs: bump the version.
5. No invented numbers.

## Layout

The repo is documentation only so far; this is the planned structure:

```
configs/         run configs (TOML)
data/pilot/      pilot items + human annotations (JSONL)
runs/            outputs (gitignored)
src/llmtutor/    types, registry, config, data, runner, report, cli
  models/        Model protocol, api.py, local.py
  tasks/         Task protocol, feedback.py
  evaluation/    Scorer protocol, judge.py, agreement.py
tests/
```

`material/` and `presentations/` are gitignored local material; never commit them.

## Commands

```sh
uv sync
uv run pytest
uv run ruff check --fix && uv run ruff format
```

## Conventions

- Python 3.12, `src/` layout, `uv`, absolute imports, type hints, ruff at line 100.
- English everywhere. API keys from env, never from repo or config.
- `temperature = 0.0` by default; record model id, adapter, sampling, prompt
  version and dataset in the run.
- Metric code needs a test with a hand-computed expected value.
- When judge and annotators disagree, assume the judge is wrong.
