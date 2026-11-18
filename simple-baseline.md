# Simple baseline: linear attention

## What it is

Our simple baseline is a small language model whose token mixer is **linear
attention** (Katharopoulos et al., 2020), trained from scratch on our Recall /
Update data. Linear attention replaces the softmax with a feature map
φ(x) = elu(x) + 1, so the model can be run as an RNN with a fixed-size state.
This makes it the simplest efficient-attention model, and the starting point
for the gated and delta-rule variants (Gated DeltaNet, KDA) that the project
studies. The question is how much retrieval ability this fixed-size state loses.

The model uses the same backbone as every other run (see `README.md`, "Models"):

* token embedding, 4 pre-norm blocks (linear attention + SwiGLU MLP), RMSNorm, tied output head
* d_model 256, 4 heads, no position embedding
* input `<bos> context <sep> query <sep> answer <eos>`, loss on the answer and `<eos>` only
* greedy decoding after the second `<sep>`, not restricted to valid status values

It is trained with the shared recipe that every model uses: `data_big` (400k
training examples), `--multi-query 16`, `--curriculum-steps 12000`, 20k steps,
65,536 tokens per batch, lr 1e-3 with warmup and cosine decay. Only
`--attention` differs between this baseline and the strong baseline
(`strong-baseline.md`).

## Code

| File | Role |
|---|---|
| `simple-baseline.py` | Trains with `--attention linear` and the recipe above, then writes predictions with `best.pt` |
| `models/attention.py` | `LinearAttention`: φ = elu + 1, with normalizer, chunked parallel form in pure PyTorch |
| `models/model.py`, `models/data.py`, `models/inference.py` | Shared backbone, data encoding/batching, greedy decoding |
| `train.py`, `predict.py` | The training and prediction CLIs that `simple-baseline.py` calls |

## How to run

Requirements: `pip install -r requirements.txt` (PyTorch), and the data
(`tar -xzf data.tar.gz`, plus `python generate_data.py --n-train 400000 --out data_big`
for the larger training set). Training needs a GPU. On a Colab A100 we measured
about 670k tokens/s, so 20,000 steps × 65,536 tokens (1.31B tokens) take about
33 minutes, plus the periodic dev evaluations.

```sh
# 1. Train, then predict the test split (the test split needs --final)
python simple-baseline.py --out runs/linear-cur12k --data-dir data_big --split test --final

# 2. Score
python3 score.py --gold data/test.jsonl --predictions runs/linear-cur12k/test_pred.jsonl \
    --output runs/linear-cur12k/test_scores.json
```

* `--data-dir` is the training data. The split being predicted is always read
  from `data/` (`--eval-data-dir`). Dev and test are the same in `data` and `data_big`.
* `--skip-train` only predicts from an existing `<out>/best.pt`.
* Any other flag goes to `train.py` and overrides the recipe. For a quick CPU
  check of the pipeline (the model won't learn anything in 60 steps):

  ```sh
  python simple-baseline.py --out runs/linear-smoke --max-context-length 128 \
      --max-train-examples 2000 --eval-examples 200 --d-model 64 --n-layers 2 --n-heads 2 \
      --steps 60 --curriculum-steps 0 --max-tokens 4096 --eval-every 30
  ```

* Rerunning the same command resumes training from `<out>/last.pt`.

## Sample output

The script writes one line per test example to `runs/linear-cur12k/test_pred.jsonl`.
Our test predictions are included as `linear_test_pred.jsonl`. These are the first 3 lines:

```json
{"id": "test-000001", "prediction": "lotus"}
{"id": "test-000002", "prediction": "sphere"}
{"id": "test-000003", "prediction": "noisy"}
```

The gold answers are `snowy` (recall), `sphere` (recall) and `daisy` (update),
so only the second prediction is correct.

## Results

Test split (10,000 examples: 5,075 recall, 4,925 update), scored with `score.py`:

```text
$ python3 score.py --gold data/test.jsonl --predictions runs/linear-cur12k/test_pred.jsonl
Accuracy: 0.173200 (1732/10000)
recall accuracy: 0.191133 (970/5075)
update accuracy: 0.154721 (762/4925)
stale errors: 349 (rate=0.034900)
distractor errors: 7778 (rate=0.777800)
other errors: 141 (rate=0.014100)
```

**Test accuracy: 17.3%** (checkpoint from step 18,000).

For reference, dev split (10,000 examples: 4,968 recall, 5,032 update):

```text
Accuracy: 0.175300 (1753/10000)
recall accuracy: 0.198873 (988/4968)
update accuracy: 0.152027 (765/5032)
stale errors: 382 (rate=0.038200)
distractor errors: 7735 (rate=0.773500)
other errors: 130 (rate=0.013000)
```

Linear attention gets 17.3% on test (17.5% on dev). About 78% of its answers
are a value of a *different* project (distractor errors). So it outputs a status that appears in
the context, but fails to bind it to the queried project. This is close to
simple heuristics that ignore the query: on dev, predicting the last value in the
context scores 20.7%, and picking a random fact scores 12.4% (`validate_data.py`,
`data/validation_report.json`). The softmax strong baseline scores 99.97% on test
with the same recipe.

## Reference

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. 2020.
[Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://proceedings.mlr.press/v119/katharopoulos20a.html).
ICML, PMLR 119:5156–5165.
