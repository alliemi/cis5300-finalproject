# Evaluating Linear Attention for Long-Context Retrieval

CIS 5300 term project. We train small models from scratch with different attention mechanisms and compare how well they retrieve facts from long contexts. The data is synthetic Recall / Update data.

* `data.md`: the dataset (format, example, how it was made)
* `scoring.md`: the evaluation metric and error categories

## Pipeline

| Stage | Command | Output |
|---|---|---|
| 1. Generate data | `python3 generate_data.py --out data` | `data/{train,dev,test}.jsonl`, `data/vocab.txt` |
| 2. Validate data | `python3 validate_data.py data` | `data/validation_report.json` |
| 3. Train + predict | *(to come: model code)* | `{"id": ..., "prediction": ...}` JSONL |
| 4. Score | `python3 score.py --gold data/dev.jsonl --predictions PRED.jsonl` | accuracy + error breakdowns |

Instead of steps 1–2 you can run `tar -xzf data.tar.gz`, which gives byte-identical files. `make_archive.py data` rebuilds `data.tar.gz`.

## Layout

```
synthdata/            data generator package
  wordlists.py        names, values, filler grammar, sentence templates
  tokenizer.py        word-level tokenizer (1 word or punctuation mark = 1 token)
  generator.py        sampling design and example construction
  errors.py           independent fact parser used by the validator
  checks.py           validation checks and shortcut baselines
generate_data.py      CLI: write the splits
validate_data.py      CLI: check every record
make_archive.py       CLI: build data.tar.gz
score.py              official scorer (exact match + stale/distractor/other)
examples/             tiny gold/prediction fixture for score.py
tests/                unit tests for the generator and the scorer
```

Planned additions for the models:
* `models/`: one shared backbone, with the attention layer (softmax, linear, Gated DeltaNet, …) swapped by name, so that only the attention differs between runs.
* `train.py` and `predict.py`: these write predictions in the format `score.py` reads.
* Put checkpoints and predictions under `runs/`. It is git-ignored, like `data/`.

Models should tokenize with `synthdata.tokenizer.tokenize` and `data/vocab.txt`. One suggested input format is `<bos> context <sep> query <sep> answer <eos>`.

## Requirements and tests

The data and scoring code use only the Python 3.9+ standard library. The model code will add its own requirements.

```sh
python3 -m unittest discover -s tests
```

Use `dev` for model selection. Keep `test` for the final evaluation.
