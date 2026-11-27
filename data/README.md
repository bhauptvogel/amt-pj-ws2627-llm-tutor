# data

Datasets the benchmark evaluates on. Nothing here yet: the first pilot set arrives with
the task layer. This file says where datasets go and what shape the files take, not which
fields an item carries. The item schema belongs to the task that poses it.

## Layout

One directory per task, one JSONL file per dataset version:

```
data/pilot/feedback/items_v1.jsonl         # items (Nermine)
data/pilot/feedback/annotations_v1.jsonl   # human labels keyed by item_id (Simon)
```

Item files and annotation files stay separate. Annotation passes edit Simon's file, never
the item file.

## File format

UTF-8, one JSON object per line, one line per item, no blank lines. `item_id` is the only
field required by every task:

```json
{"item_id": "fb-001", "question": "...", "student_answer": "..."}
```

`item_id` is unique within the file and stable forever. Use a task prefix and zero-padded
number (`fb-001`, `fb-002`) so ids from different tasks never collide.

The rest of the fields are the task's decision. The current reference shape is the
synthetic sanity set in `src/llmtutor/evaluation/sanity/items_v1.jsonl`; it exists to test
the judge, not to define the pilot schema.

## Rules

Files are versioned by name, never edited in place once a run has used them. A real edit
means `items_v2.jsonl` and a config pointing at the new file. The config hash covers only
the TOML bytes, so the dataset needs its own hash. It is computed when the plan loads and
kept as `RunPlan.dataset.sha256`; the manifest does not record it yet, because dataset
execution is not wired.

Items are immutable. Correcting a typo is an edit and gets a new version; silently
rewriting an item changes what an old run measured.

Mark synthetic items with `"synthetic": true`. Real pilot items omit the field. Test
fixtures belong next to the code that tests them, not here.

No personal data. No names, emails or student ids that identify a real person, and no
answers copied from a real student without consent. Synthetic first, real data only once
the consent question is settled.

## How configs reference this

A run config may name one dataset:

```toml
[dataset]
path = "data/pilot/feedback/items_v1.jsonl"
```

`path` is the only key. It is read as written, relative to the working directory, once
when the plan loads. A missing or empty file is a config error. The core keeps the bytes
and the SHA-256 in `RunPlan.dataset`, so a task loader can read `dataset.raw` instead of
the path and parse exactly the bytes the hash covers.

The item schema, the loader and the tutor prompt belong to the task layer. The core only
guarantees which bytes a run was pointed at. The pilot is single-turn: an item poses one
question and one student answer, and `dialogue_context` stays optional and defaults to
`None`.

The runner still makes one call per model, so a config with `[dataset]` is refused before
any call. The item loop and the manifest entry are the next core change; until then
`load_plan` is usable and the runner is not.
