# Evaluating Linear Attention for Long-Context Retrieval

CIS 5300 term project. We train small models from scratch with different attention mechanisms and compare how well they retrieve facts from long contexts. The data is synthetic Recall / Update data.

* `data.md`: the dataset (format, example, how it was made)
* `scoring.md`: the evaluation metric and error categories

## Pipeline

| Stage | Command | Output |
|---|---|---|
| 1. Generate data | `python3 generate_data.py --out data` | `data/{train,dev,test}.jsonl`, `data/vocab.txt` |
| 2. Validate data | `python3 validate_data.py data` | `data/validation_report.json` |
| 3a. Train | `python train.py --attention softmax --out runs/softmax` | `runs/softmax/{best,last}.pt`, `metrics.jsonl` |
| 3b. Predict | `python predict.py --checkpoint runs/softmax/best.pt --split dev --out runs/softmax/dev_pred.jsonl` | `{"id": ..., "prediction": ...}` JSONL |
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
models/               model code (needs PyTorch)
  data.py             vocab, encoding, tokenization cache, length-bucketed batching
  attention.py        softmax | linear | gated_deltanet | kda token mixers
  model.py            shared backbone (embedding, pre-norm blocks, SwiGLU MLP, tied head)
  inference.py        greedy decoding, checkpoints, quick dev accuracy
train.py              CLI: train one model
predict.py            CLI: write predictions for score.py
examples/             tiny gold/prediction fixture for score.py
tests/                unit tests for the generator, the scorer and the models
```

Checkpoints, logs and predictions go under `runs/`, which is git-ignored like `data/`.

## Models

Every example is encoded as `<bos> context <sep> query <sep> answer <eos>` with `synthdata.tokenizer.tokenize` and `data/vocab.txt`. The loss covers only the answer and `<eos>`, plus any extra answers added by `--multi-query` (see GPU runs). Prediction decodes greedily after the second `<sep>`, and decoding is not restricted to valid values, so "other" errors stay visible.

The backbone is the same for every run: token embedding, `n_layers` pre-norm blocks (token mixer + SwiGLU MLP), RMSNorm and a tied output head. No absolute position embedding is used. Only the token mixer changes:

| `--attention` | Layer | Recurrence |
|---|---|---|
| `softmax` | causal softmax attention with RoPE | `F.scaled_dot_product_attention` |
| `linear` | linear attention, φ = elu+1, with normalizer (Katharopoulos et al., 2020) | chunked parallel form, pure PyTorch |
| `gated_deltanet` | short conv + L2-normalized q/k, delta rule, one decay per head, SiLU output gate (Yang et al., 2025) | fla `chunk_gated_delta_rule` on CUDA, PyTorch reference otherwise |
| `kda` | as GDN, but one decay per key channel (low-rank) and a sigmoid low-rank output gate (Kimi Team, 2025) | fla `chunk_kda` on CUDA, PyTorch reference otherwise |

GDN and KDA have more parameters in the mixer (convolutions and gates). `train.py` prints both counts, and the tests check that everything outside the mixer is identical.

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

The first run tokenizes every split once and caches it in `data/cache/`. `--write-gold` is needed when predicting on a subset, because `score.py` requires a prediction for every gold id. On CPU, `gated_deltanet` and `kda` use the step-by-step PyTorch reference, which is slow. Keep CPU runs at 128–256 tokens.

### GPU runs

```sh
pip install -r requirements.txt flash-linear-attention
python -m unittest tests.test_models -v      # includes the fla vs. reference checks
python generate_data.py --n-train 400000 --out data_big   # more training data; dev/test unchanged
for ATT in softmax linear gated_deltanet kda; do
  python train.py --attention $ATT --out runs/$ATT-s0 --seed 0 --data-dir data_big \
      --multi-query 16 --curriculum-steps 12000 --steps 20000 --max-tokens 65536
  python predict.py --checkpoint runs/$ATT-s0/best.pt --split dev --out runs/$ATT-s0/dev_pred.jsonl
  python score.py --gold data/dev.jsonl --predictions runs/$ATT-s0/dev_pred.jsonl \
      --output runs/$ATT-s0/dev_scores.json
done
```

The defaults are d_model 256, 4 layers, 4 heads, 32k tokens per batch, 20k steps, lr 1e-3 with warmup and cosine decay, and bf16 autocast. Batches come from one length bucket at a time, and `--curriculum-steps` unlocks the longer buckets gradually. Checkpoints are selected on a fixed 2,000-example dev subset (`--eval-examples`). Keep the same flags for every attention type, and change only `--attention` and `--seed`.

Why these flags:
* **More data (`data_big`).** With the 80k training examples, the model memorizes them instead of learning to retrieve. Dev and test do not depend on `--n-train`, so always score on `data/dev.jsonl`.
* **`--multi-query K`.** With one supervised answer per sequence, softmax learned slowly and stopped improving around 37% even on 128-token contexts. With this flag, each training sequence also asks about up to K other projects in the same context: `... <sep> answer (<sep> query_e <sep> value_e)* <eos>`. The answers come from the fact parser in `synthdata/errors.py`, and a fresh random subset is drawn for every batch. Evaluation still decodes only after the official query, so dev and test results stay comparable. In the logs, `ans_acc` is training accuracy on the official answer only, which is comparable with dev accuracy. `tok_acc` includes the extra answers.
* **Long curriculum (`--curriculum-steps 12000`).** Retrieval is learned in a sudden jump: on 128-token contexts alone, softmax sat at about 45% until step 1,700, then reached 99% dev accuracy by step 2,000. With 12000 curriculum steps, each bucket gets 3,000 steps alone before the next one unlocks, so the jump can happen on short contexts first. With `--curriculum-steps 4000` (1,000 steps per bucket), softmax only reached 29% dev accuracy, and 66% of answers were another project's value. No `--patience`: dev accuracy can stay flat for thousands of steps before the jump.

## Requirements and tests

The data and scoring code use only the Python 3.9+ standard library. The model code needs `requirements.txt`, and the model tests are skipped when PyTorch is missing.

```sh
python3 -m unittest discover -s tests          # data + scoring (+ models if torch is installed)
.venv/bin/python -m unittest discover -s tests # everything, using the venv
```

Use `dev` for model selection. Keep `test` for the final evaluation.
