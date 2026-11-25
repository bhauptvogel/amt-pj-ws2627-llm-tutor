# llmtutor

Vocabulary for a benchmark that asks whether language models tutor well: models sit behind
one seam, tasks pose the educational situation, scorers judge the answer.

## Language

### Framework

**Registry**:
A named lookup of implementations of one seam, filled by the implementations themselves,
so adding one never edits the core.
_Avoid_: Plugin manager, service locator

### Config

**Run config**:
The TOML file that selects one prompt, one shared sampling block and one or more model
entries for a run.
_Avoid_: Experiment, job file

**Model entry**:
One `[[models]]` block: a label, an adapter name and that adapter's settings.
_Avoid_: Provider block, model config

**Sampling**:
The shared `[sampling]` block: temperature, token limit, seed and stop sequences,
identical for every model entry in a run.
_Avoid_: Model parameters, hyperparameters, generation settings

**Prompt**:
The versioned system/user conversation a run poses to one model.
_Avoid_: Task, item, query

**Prompt id**:
The name a prompt carries in the run config, recorded with its version in the manifest.
_Avoid_: Prompt name, task id

**Prompt version**:
The hand-bumped integer naming one revision of a prompt; editing the prompt without
bumping it invalidates earlier runs.
_Avoid_: Config hash, revision

**Config hash**:
The SHA-256 of the raw run config, recorded with the copy that produced a run.
_Avoid_: Checksum, fingerprint

**Run plan**:
The validated result of loading one run config: prompt, sampling and one built
model per entry, ready before any call is made.
_Avoid_: Config object, job, execution plan

### Model layer

**Model**:
The seam the runner calls to turn one conversation plus sampling settings into one answer.
_Avoid_: Provider, backend, client

**Adapter**:
A Model implementation bound to one provider, translating our request into that provider's
call and its answer back into our terms.
_Avoid_: Driver, plugin, integration

**Adapter name**:
The config string that selects which adapter serves a model entry.
_Avoid_: Provider, model id

**Adapter settings**:
The typed parameters one adapter needs to be built, validated from its model entry before
any call.
_Avoid_: Options, kwargs, parameters

**Message**:
One turn in a conversation, carrying a role (`system`, `user`, `assistant`) and text.
_Avoid_: Prompt, when a single turn is meant

**ModelRequest**:
The conversation and sampling settings that make up a single call.
_Avoid_: Query, prompt parameters

**ModelResponse**:
One answer together with what the provider reported about how it produced it.
_Avoid_: Completion, output

**Usage**:
The token counts a provider reports for one call; absent when it reports none, never
invented.
_Avoid_: Cost (cost is one provider's extra inside the raw payload)

**Raw payload**:
The provider's response payload kept untouched as evidence for reproducing and judging a
call.
_Avoid_: Metadata, extras

**Finish reason**:
The provider's stated reason an answer ended, kept as reported.
_Avoid_: Stop reason, status

**Error kind**:
The classification of a failed call that retry and abort policy act on.
_Avoid_: Error code, error type

### Runs

**Run**:
One execution of a configured prompt against one model, recorded as an artefact.
_Avoid_: Trial, job, experiment

**Run id**:
The name of a run artefact: UTC start time, config stem and model entry.
_Avoid_: Run name, experiment id

**Run artefact**:
The `runs/<run-id>/` directory holding one run's manifest, config copy and call records.
_Avoid_: Output, results, run directory

**Manifest**:
A run artefact's `manifest.json`, carrying provenance, resolved sampling and final status.
_Avoid_: Metadata, header

**Call**:
One request-response exchange the runner drives against one model; the scaffold
makes exactly one call per run.
_Avoid_: Query, request (the request is only half of it)

**Call record**:
One line in `calls.jsonl` for one attempt at one call: request, outcome and response or
error.
_Avoid_: Log line, result row

**Attempt**:
One try at one call. A retried call has several attempts, all recorded.
_Avoid_: Try, request (the request is what was sent, not how often)

**Retry**:
A further attempt at the same call after a retryable failure.
_Avoid_: Rerun, resend (a rerun is a new Run)

**Retry policy**:
The rules a run follows when a call fails: which failures deserve another attempt, how
many, and how long to wait between them. One policy applies to every model in a run.
_Avoid_: Backoff settings, retry config

**Run status**:
The manifest's state for a run: running while calls are in flight, then completed, failed,
or aborted.
_Avoid_: Result, exit code (that is the CLI's)

### Evaluation layer

**Judge**:
A model that labels one tutor response against a rubric, using judge-only reference
information the tutor never saw.
_Avoid_: Grader, evaluator model, scorer (when the model itself is meant)

**Rubric**:
The versioned set of criteria one task's judge labels; changing it invalidates earlier
judgements.
_Avoid_: Scoring guide, metric

**Criterion**:
One question the judge and human raters answer about a tutor response, with a fixed set
of labels.
_Avoid_: Dimension, aspect, metric

**Label**:
One allowed answer to a criterion, written exactly as human raters write it; its position
in the label set is its integer code.
_Avoid_: Score, grade, rating (a rating is a rater's label)

**Desired label**:
The label a good tutor response earns on a criterion.
_Avoid_: Target, correct label

**Judgement**:
The judge's schema-valid output for one tutor response: analysis, then evidence,
reasoning and label per criterion.
_Avoid_: Verdict, evaluation, score

**Item-writing flaw**:
One of 19 known defects of a multiple-choice question, checked by code or by the judge.
_Avoid_: Error, issue

### Testing

**Fake adapter**:
A test-only Model registered under a scenario name, so the real runner and CLI run without a
provider.
_Avoid_: Mock, stub

**Scenario**:
The named sequence of outcomes a fake adapter returns or raises, one per attempt.
_Avoid_: Script, fixture

**Live test**:
A test that calls a real provider, excluded from the default run and skipped when no key is
present.
_Avoid_: Smoke test, integration test
