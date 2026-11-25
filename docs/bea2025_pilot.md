# BEA 2025 Educational Benchmark Pilot

## 1. Objective

This pilot investigates how language models perform as educational
tutors and how reliably other language models can evaluate their
pedagogical abilities.

Unlike general knowledge benchmarks, the objective is not simply
to determine whether a model can solve a problem, but whether it
can identify student mistakes and provide useful guidance.

The pilot addresses three research questions:

1. Can local language models generate useful tutoring responses?
2. How reliably can local LLM judges reproduce human annotations?
3. Does the choice of judge affect the ranking of tutor models?

## 2. Dataset

The pilot uses the MRBench development dataset associated with
the BEA 2025 Shared Task.

Source repository:
https://github.com/kaushal0494/UnifyingAITutorEvaluation

Source commit:
bbef521ddb875f2cc8a5ee798f4066965a7cfd8a

Source file:
BEA_Shared_Task_2025_Datasets/mrbench_v3_devset.json

### Dataset statistics

| Property | Value |
|---|---:|
| Original conversations | 300 |
| Original annotated tutor responses | 2,476 |
| Sampled conversations | 20 |
| Annotated responses in pilot dataset | 164 |
| Sampling seed | 42 |
| Prompt development conversations | 5 |
| Judge evaluation conversations | 15 |
| Selected judge evaluation responses | 30 |
| Newly generated tutor responses | 40 |

The pilot conversations were selected using Python's
random.Random(42).sample().

The first five sampled conversations were used for rubric
development. The remaining 15 conversations were used for
judge evaluation, selecting the first two annotated tutor
responses per conversation.

The development and evaluation conversation IDs do not overlap.

The evaluation cases were subsequently inspected during error
analysis and should not be considered an untouched final test set.

### Dataset integrity

Source SHA-256:

a395feab801de50d44b73e1e9b88f84824a1f1d92e15ae86842d90c2b6dbf33c

Pilot SHA-256:

b8c2854acf13d4acdca49dea7026687bdc8bf6071a6476bd7f4d5ef5212317b3

The pilot preparation script reproduces the existing pilot
dataset byte for byte.

Dataset redistribution permissions have not yet been verified.
Therefore, the locally prepared dataset is excluded from Git.

## 3. Educational rubric

The current pilot evaluates four pedagogical dimensions.

| Criterion | Description |
|---|---|
| Mistake Identification | Whether the tutor recognizes a student's mistake |
| Mistake Location | Whether the tutor identifies where the mistake occurs |
| Providing Guidance | Whether the tutor provides relevant and correct guidance |
| Actionability | Whether the student receives a useful next step |

Each criterion uses three ordinal labels:

- No
- To some extent
- Yes

The rubric is based on the BEA 2025 human annotation guidelines.

Current rubric version: bea2025-v3

Prompt length: 2,519 characters

Prompt SHA-256:

d1f07a3394188e30e66015f21f986b9869388f8ae9c4604a906f88c4561db086

The rubric is stored separately from the experimental scripts
in src/llmtutor/tasks/bea2025_rubric.py.

Its exact content was preserved during refactoring and formatting.

## 4. Implementation

### Model integration

- src/llmtutor/models/ollama.py
- configs/local-smoke.toml

The Ollama adapter enables local model execution without
requiring a paid external inference API.

### Educational layer

- src/llmtutor/tasks/bea2025.py
- src/llmtutor/tasks/bea2025_rubric.py

These modules provide dataset loading, educational model
requests and the versioned pedagogical evaluation rubric.

### Experimental scripts

| Script | Purpose |
|---|---|
| prepare_bea2025_pilot.py | Reproducible dataset sampling |
| run_bea2025_pilot.py | Generate tutor responses |
| evaluate_bea2025_judge.py | Evaluate judges against human annotations |
| score_bea2025_judge.py | Calculate judge reliability metrics |
| evaluate_bea2025_tutors.py | Evaluate newly generated tutor responses |
| score_bea2025_tutors.py | Summarize predicted pedagogical labels |
| compare_bea2025_judges.py | Compare judges on identical responses |

Earlier judge-development scripts are retained as experimental
references.

The pilot currently uses standalone experimental scripts.
Full integration into the shared configuration-driven task
and evaluation pipeline remains future work.

## 5. Experimental setup

### Tutor models

- qwen3:4b-instruct-2507-q4_K_M
- qwen3.5:2b

### Judge models

- qwen3:4b-instruct-2507-q4_K_M
- qwen3:8b

### Sampling configuration

| Parameter | Value |
|---|---|
| Temperature | 0.0 |
| Maximum output tokens | 256 |
| Seed | 42 |
| Ollama thinking mode | Disabled |

All experiments were executed locally using Ollama.

The tutor generation experiment used 20 conversations and
two tutor models, producing 40 responses.

Each judge evaluated the same 40 generated responses using
the same four pedagogical criteria.

The judge validation experiment used 30 existing human-annotated
responses, producing 120 annotation decisions per judge.

## 6. Judge prompt development

Three rubric prompt versions were compared using a small
development sample of 10 annotated tutor responses.

| Prompt version | Correct labels | Exact agreement |
|---|---:|---:|
| v1 | 29/40 | 72.5% |
| v2 | 26/40 | 65.0% |
| v3 | 25/40 | 62.5% |

Version 3 was retained as the experimental baseline because
it was developed around the official annotation guidelines,
not because it achieved the highest development agreement.

These small-sample results do not establish which prompt
generalizes best.

## 7. Judge validation results

Judge predictions were compared against the original human
annotations provided with the dataset.

The evaluation metrics are:

- Exact agreement
- Macro-F1
- Unweighted Cohen's kappa
- Quadratic weighted Cohen's kappa
- Confusion matrices

### Qwen3 4B judge

| Criterion | Agreement | Macro-F1 | Kappa | Weighted kappa |
|---|---:|---:|---:|---:|
| Mistake Identification | 73.3% | 0.492 | 0.416 | 0.502 |
| Mistake Location | 70.0% | 0.604 | 0.489 | 0.585 |
| Providing Guidance | 56.7% | 0.541 | 0.333 | 0.378 |
| Actionability | 53.3% | 0.412 | 0.271 | 0.410 |
| Overall | 63.3% | 0.540 | 0.381 | 0.477 |

Valid predictions: 30/30.

### Qwen3 8B judge

| Criterion | Agreement | Macro-F1 | Kappa | Weighted kappa |
|---|---:|---:|---:|---:|
| Mistake Identification | 76.7% | 0.501 | 0.462 | 0.522 |
| Mistake Location | 73.3% | 0.489 | 0.474 | 0.498 |
| Providing Guidance | 53.3% | 0.492 | 0.270 | 0.365 |
| Actionability | 53.3% | 0.444 | 0.325 | 0.557 |
| Overall | 64.2% | 0.513 | 0.380 | 0.496 |

Valid predictions: 30/30.

### Interpretation

The larger judge improved exact agreement by only one
annotation decision out of 120.

The smaller judge achieved higher overall macro-F1,
while the larger judge achieved higher weighted kappa.

Neither judge demonstrated a decisive overall advantage.

Qwen3 4B was retained as a provisional lightweight baseline,
with Qwen3 8B used for comparison.

Neither judge should be treated as a substitute for human
ground-truth annotations.

## 8. Tutor evaluation results

Both tutor models generated responses for the same
20 conversations.

All 40 generated responses were successfully evaluated
by both judges.

The following percentages represent the proportion of
responses receiving the predicted label Yes.

They are not measured accuracy against human annotations.

### Evaluation using Qwen3 4B judge

| Criterion | Qwen3 4B tutor | Qwen3.5 2B tutor |
|---|---:|---:|
| Mistake Identification | 90% | 85% |
| Mistake Location | 85% | 75% |
| Providing Guidance | 80% | 80% |
| Actionability | 65% | 70% |
| Overall Yes rate | 80% | 77.5% |

### Evaluation using Qwen3 8B judge

| Criterion | Qwen3 4B tutor | Qwen3.5 2B tutor |
|---|---:|---:|
| Mistake Identification | 80% | 90% |
| Mistake Location | 75% | 90% |
| Providing Guidance | 85% | 95% |
| Actionability | 20% | 70% |
| Overall Yes rate | 65% | 86.25% |

### Interpretation

The ranking of the two tutor models changes depending
on the judge.

The 4B judge assigns a slightly higher overall Yes rate
to the 4B tutor.

The 8B judge assigns a substantially higher overall Yes
rate to the 2B tutor.

The largest criterion-level difference concerns Actionability
for the 4B tutor, which changes from 65% to 20%.

These results demonstrate judge sensitivity, but do not
establish which tutor is objectively better.

## 9. Inter-judge reliability

The two judges were compared on the same 40 generated
tutor responses and 160 annotation decisions.

| Criterion | Agreement | Kappa | Weighted kappa |
|---|---:|---:|---:|
| Mistake Identification | 92.5% | 0.684 | 0.684 |
| Mistake Location | 85.0% | 0.530 | 0.697 |
| Providing Guidance | 80.0% | 0.238 | 0.167 |
| Actionability | 62.5% | 0.366 | 0.425 |
| Overall | 80.0% | 0.471 | 0.577 |

The judges agreed on 128 of 160 annotation decisions.

There were 32 disagreements, including:

- 24 disagreements between adjacent ordinal labels.
- 8 disagreements between opposite ordinal labels.
- 15 disagreements concerning Actionability.

Actionability was the least consistent criterion in
terms of exact inter-judge agreement.

High agreement does not necessarily imply correctness,
particularly when both judges share similar biases.

## 10. Error analysis

Qualitative inspection identified several recurring issues:

- Questions can be incorrectly interpreted as useful guidance
  even when they are not relevant to the student's mistake.
- Earlier mistakes in the conversation can be overlooked.
- Incomplete student answers can be interpreted inconsistently.
- Judges differ in how strictly they interpret actionable guidance.
- Some human annotations may also be difficult to interpret.

The observed ranking reversal motivates stronger judge
validation and independent human evaluation.

Possible self-preference or model-family bias requires
further investigation and has not been established.

## 11. Reproduction instructions

Run the following commands from the repository root.

### Install dependencies

    uv sync

Install Ollama separately and ensure its local server
is running.

Download the models:

    ollama pull qwen3:4b-instruct-2507-q4_K_M
    ollama pull qwen3.5:2b
    ollama pull qwen3:8b

### Prepare the dataset

Obtain the original dataset from its source repository
subject to its applicable usage conditions.

    git clone https://github.com/kaushal0494/UnifyingAITutorEvaluation.git /tmp/bea2025-source

    git -C /tmp/bea2025-source checkout bbef521ddb875f2cc8a5ee798f4066965a7cfd8a

    uv run python scripts/prepare_bea2025_pilot.py \
      /tmp/bea2025-source/BEA_Shared_Task_2025_Datasets/mrbench_v3_devset.json

The default output is:

    data/pilot/bea2025_pilot20.json

### Generate tutor responses

    uv run python scripts/run_bea2025_pilot.py \
      --dataset data/pilot/bea2025_pilot20.json

This creates a timestamped run directory containing
responses.jsonl and manifest.json.

### Validate both judges

    uv run python scripts/evaluate_bea2025_judge.py \
      --dataset data/pilot/bea2025_pilot20.json \
      --model qwen3:4b-instruct-2507-q4_K_M

    uv run python scripts/evaluate_bea2025_judge.py \
      --dataset data/pilot/bea2025_pilot20.json \
      --model qwen3:8b

Score each generated predictions file:

    uv run python scripts/score_bea2025_judge.py \
      runs/<judge-run>/predictions.jsonl

Replace <judge-run> with the actual generated directory.

### Evaluate generated tutor responses

    uv run python scripts/evaluate_bea2025_tutors.py \
      --dataset data/pilot/bea2025_pilot20.json \
      --responses runs/<pilot-run>/responses.jsonl \
      --model qwen3:4b-instruct-2507-q4_K_M

    uv run python scripts/evaluate_bea2025_tutors.py \
      --dataset data/pilot/bea2025_pilot20.json \
      --responses runs/<pilot-run>/responses.jsonl \
      --model qwen3:8b

Use the same tutor responses file for both judges.

Score each evaluation:

    uv run python scripts/score_bea2025_tutors.py \
      runs/<tutor-evaluation-run>/predictions.jsonl

### Compare judges

    uv run python scripts/compare_bea2025_judges.py \
      runs/<judge-a-tutor-evaluation>/predictions.jsonl \
      runs/<judge-b-tutor-evaluation>/predictions.jsonl \
      --output runs/bea2025-comparison/inter_judge_metrics.json

Replace all placeholder directory names with the actual
directories produced by the previous commands.

## 12. Software quality

The implementation includes tests for:

- Dataset loading
- Ollama model integration
- Model registry integration
- Judge output validation
- Judge execution failures
- Recovery after model failures
- Reproducible dataset sampling
- Dataset overwrite protection
- Evaluation metric calculations
- Inter-judge agreement calculations

Final verification:

- 117 tests passed.
- 1 test deselected.
- Ruff linting passed.
- All 43 Python files passed formatting checks.

Commands:

    uv run pytest -q
    uv run ruff check .
    uv run ruff format --check .

The judge execution tests use mocked responses and do not
require live model inference.

## 13. Limitations

- The pilot contains only 20 conversations.
- Judge validation uses only 30 annotated tutor responses.
- The validation cases were inspected during error analysis.
- The newly generated tutor responses lack human annotations.
- The judges show limited agreement with human annotations.
- Both judges belong to the Qwen model family.
- One judge also evaluates responses from the same model.
- Possible self-preference has not been independently established.
- Dataset redistribution permissions remain unverified.
- Exact inference reproducibility across different hardware
  and model versions is not guaranteed.
- Historical runs predate some metadata improvements.
- Degenerate label distributions can make kappa undefined.
- The positional development/evaluation split is specific
  to the original pilot design.
- Full integration with the shared task and evaluation
  framework remains outstanding.

## 14. Recommended next steps

1. Coordinate integration with the shared model and
   evaluation layers.
2. Establish a larger, untouched evaluation sample.
3. Collect independent human annotations for generated
   tutor responses.
4. Evaluate additional judges from different model families.
5. Investigate disagreements, especially Actionability.
6. Improve judge reliability before using automated scores
   to make definitive tutor rankings.
7. Verify dataset usage and redistribution conditions.

## 15. Conclusion

The BEA 2025 pilot demonstrates a reproducible experimental
workflow for generating tutor responses, evaluating local
LLM judges and analyzing pedagogical predictions.

The main empirical finding is that tutor rankings can
change substantially depending on the selected judge.

Consequently, judge reliability is a central requirement
for a meaningful educational benchmark.

The current implementation provides an experimental
foundation for the project's modular tutoring evaluation
framework, rather than a finalized or fully validated
educational leaderboard.
