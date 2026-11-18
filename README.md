# Evaluating Linear Attention for Long-Context Retrieval

CIS 5300 term project. We train small models from scratch with different attention mechanisms and compare how well they retrieve facts from long contexts. The data is synthetic Recall / Update data.

* `data.md`: the dataset (format, example, how it was made)
* `scoring.md`: the evaluation metric and error categories
* `simple-baseline.md`: simple baseline, linear attention (`simple-baseline.py`)
* `strong-baseline.md`: strong baseline, softmax Transformer

## Pipeline

| Stage | Command | Output |
|---|---|---|
| 1. Generate data | `python3 generate_data.py --out data` | `data/{train,dev,test}.jsonl`, `data/vocab.txt` |
| 2. Validate data | `python3 validate_data.py data` | `data/validation_report.json` |
| 3a. Train | `python train.py --attention softmax --out runs/softmax` | `runs/softmax/{best,last}.pt`, `metrics.jsonl`, and at the end `dev_pred.jsonl` + `dev_scores.json` |
| 3b. Predict | `python predict.py --checkpoint runs/softmax/best.pt --split dev --out runs/softmax/dev_pred.jsonl` | `{"id": ..., "prediction": ...}` JSONL |
| 4. Score | `python3 score.py --gold data/dev.jsonl --predictions PRED.jsonl` | accuracy + error breakdowns (3a already does this for dev) |
| 5. Compare | `python3 summarize_runs.py` | one table across runs, `runs/summary.csv` |

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
models/               model code (needs PyTorch)
  data.py             vocab, encoding, tokenization cache, length-bucketed batching
  attention.py        softmax | linear token mixers
  model.py            shared backbone (embedding, pre-norm blocks, SwiGLU MLP, tied head)
  inference.py        greedy decoding, checkpoints, quick dev accuracy
train.py              CLI: train one model (resumes automatically, scores dev at the end)
predict.py            CLI: write predictions for score.py (test split needs --final)
summarize_runs.py     CLI: compare finished runs (standard library only)
notebooks/colab_train.ipynb   Colab setup + training, with runs saved to Google Drive
tests/                unit tests for the generator, the scorer, the models and training
```

Checkpoints, logs and predictions go under `runs/`, which is git-ignored like `data/`.

## Models

Every example is encoded as `<bos> context <sep> query <sep> answer <eos>` with `synthdata.tokenizer.tokenize` and `data/vocab.txt`. The loss covers only the answer and `<eos>`, plus any extra answers added by `--multi-query` (see GPU runs). Prediction decodes greedily after the second `<sep>`, and decoding is not restricted to valid values, so "other" errors stay visible.

The backbone is the same for every run: token embedding, `n_layers` pre-norm blocks (token mixer + SwiGLU MLP), RMSNorm and a tied output head. No absolute position embedding is used. Only the token mixer changes:

| `--attention` | Layer | Recurrence |
|---|---|---|
| `softmax` | causal softmax attention with RoPE | `F.scaled_dot_product_attention` |
| `linear` | linear attention, φ = elu+1, with normalizer (Katharopoulos et al., 2020) | chunked parallel form, pure PyTorch |

`train.py` prints the total and mixer parameter counts, and the tests check that everything outside the mixer is identical.

**Optional shared short convolution (`--short-conv 4`, off by default).** Softmax has RoPE, but linear attention has no position information, so it has no easy way to link a project name to the value right after it. `--short-conv N` adds the same causal convolution (window N) before the mixer in every block, for every attention type. If the team uses it, use it for all models.

### CPU smoke test

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python train.py --attention softmax --out runs/smoke --max-context-length 128 \
    --max-train-examples 2000 --eval-examples 200 --d-model 64 --n-layers 2 --n-heads 2 \
    --steps 300 --max-tokens 4096 --eval-every 100
.venv/bin/python predict.py --checkpoint runs/smoke/best.pt --split dev --max-examples 200 \
    --max-context-length 128 --out runs/smoke/dev_pred.jsonl --write-gold runs/smoke/dev_gold.jsonl
python3 score.py --gold runs/smoke/dev_gold.jsonl --predictions runs/smoke/dev_pred.jsonl
```

On an Apple-silicon Mac, `--device auto` uses the GPU (MPS); CUDA is used when available. The first run tokenizes every split once and caches it in `data/cache/`. `--write-gold` is needed when predicting on a subset, because `score.py` requires a prediction for every gold id. Keep CPU runs at 128–256 tokens.

### GPU runs

```sh
pip install -r requirements.txt
python -m unittest tests.test_models -v
python generate_data.py --n-train 400000 --out data_big   # more training data; dev/test unchanged
for ATT in softmax linear; do
  python train.py --attention $ATT --out runs/$ATT-s0 --seed 0 --data-dir data_big \
      --multi-query 16 --curriculum-steps 12000 --steps 20000 --max-tokens 65536
done
python3 summarize_runs.py
```

Each run ends by predicting the full dev split with `best.pt` and writing `dev_pred.jsonl` and `dev_scores.json` (from `score.py`) into its folder; `--no-final-eval` skips this. With `--max-context-length`, only that part of dev is scored, and the matching gold file is written as `dev_gold.jsonl`.

**On Colab**, open `notebooks/colab_train.ipynb` (A100 or L4 runtime). It clones the repo, unpacks the data, keeps `runs/` in Google Drive, and uses one `FLAGS` setting for every model.

**Resuming.** `last.pt` stores the optimizer, the batch sampler and the RNG states, and is written atomically at every evaluation. Rerunning the same command in the same `--out` continues from that step and gives the same result as an uninterrupted run. A run folder refuses different training settings (e.g. a changed `--steps`), so start a new `--out` instead. Settings that don't affect training (`--device`, `--log-every`, `--compile`) may change.

The defaults are d_model 256, 4 layers, 4 heads, 32k tokens per batch, 20k steps, lr 1e-3 with warmup and cosine decay, and bf16 autocast. Batches come from one length bucket at a time, and `--curriculum-steps` unlocks the longer buckets gradually. Checkpoints are selected on a fixed 2,000-example dev subset (`--eval-examples`). Keep the same flags for every attention type, and change only `--attention` and `--seed`.

Why these flags:
* **More data (`data_big`).** With the 80k training examples, the model memorizes them instead of learning to retrieve. Dev and test do not depend on `--n-train`, so always score on `data/dev.jsonl`.
* **`--multi-query K`.** With one supervised answer per sequence, softmax learned slowly and stopped improving around 37% even on 128-token contexts. With this flag, each training sequence also asks about up to K other projects in the same context: `... <sep> answer (<sep> query_e <sep> value_e)* <eos>`. The answers come from the fact parser in `synthdata/errors.py`, and a fresh random subset is drawn for every batch. Evaluation still decodes only after the official query, so dev and test results stay comparable. In the logs, `ans_acc` is training accuracy on the official answer only, which is comparable with dev accuracy. `tok_acc` includes the extra answers.
* **Long curriculum (`--curriculum-steps 12000`).** Retrieval is learned in a sudden jump: on 128-token contexts alone, softmax sat at about 45% until step 1,700, then reached 99% dev accuracy by step 2,000. With 12000 curriculum steps, each bucket gets 3,000 steps alone before the next one unlocks, so the jump can happen on short contexts first. With `--curriculum-steps 4000` (1,000 steps per bucket), softmax only reached 29% dev accuracy, and 66% of answers were another project's value. No `--patience`: dev accuracy can stay flat for thousands of steps before the jump.

### Results

Training commands (`{RUNS}` is the output directory for runs):

```sh
python train.py --attention linear --out {RUNS}/linear-cur12k --data-dir data_big \
    --multi-query 16 --curriculum-steps 12000 --steps 20000 --max-tokens 65536
python train.py --attention softmax --out {RUNS}/softmax-cur12k --data-dir data_big \
    --multi-query 16 --curriculum-steps 12000 --steps 20000 --max-tokens 65536
```

Dev scores (10,000 examples: 4,968 recall, 5,032 update):

| Model | Accuracy | Recall acc. | Update acc. | Stale errors | Distractor errors | Other errors |
|---|---|---|---|---|---|---|
| `softmax-cur12k` | 1.0000 (10000/10000) | 1.0000 (4968/4968) | 1.0000 (5032/5032) | 0 (0.0000) | 0 (0.0000) | 0 (0.0000) |
| `linear-cur12k` | 0.1753 (1753/10000) | 0.1989 (988/4968) | 0.1520 (765/5032) | 382 (0.0382) | 7735 (0.7735) | 130 (0.0130) |

Softmax attention solves the task perfectly. Linear attention gets 17.5%, and 77% of its answers are another project's value (distractor errors), so it fails to bind the queried project to its value.

## Requirements and tests

The data and scoring code use only the Python 3.9+ standard library. The model code needs `requirements.txt`, and the model tests are skipped when PyTorch is missing.

```sh
python3 -m unittest discover -s tests          # data + scoring (+ models if torch is installed)
.venv/bin/python -m unittest discover -s tests # everything, using the venv
```

Use `dev` for model selection. Keep `test` for the final evaluation: `predict.py --split test` only runs with `--final`.
