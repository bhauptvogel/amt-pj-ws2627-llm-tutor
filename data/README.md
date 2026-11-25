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
the TOML bytes, so the manifest will have to record the dataset path and hash on its own;
the `dataset` field there is `null` today.

Items are immutable. Correcting a typo is an edit and gets a new version; silently
rewriting an item changes what an old run measured.

Mark synthetic items with `"synthetic": true`. Real pilot items omit the field. Test
fixtures belong next to the code that tests them, not here.

No personal data. No names, emails or student ids that identify a real person, and no
answers copied from a real student without consent. Synthetic first, real data only once
the consent question is settled.

## How configs will reference this

There is no `[dataset]` block yet. The planned shape is:

```toml
[dataset]
path = "data/pilot/feedback/items_v1.jsonl"
```

Until the loader and that block exist, no code reads this directory. Adding them is a
core change to `config.py` and `runner.py`, so it gets raised before it gets built.
