# Strong baseline: softmax Transformer

## What it is

Our strong baseline is a decoder-only Transformer with standard **causal softmax
attention** (Vaswani et al., 2017) and rotary position embeddings (RoPE; Su et
al., 2021), trained from scratch on our Recall / Update data. Softmax attention
keeps every past token in its key-value cache, so it can look up the queried
project directly, but its cost grows quadratically with context length. The
efficient models in this project (linear attention, Gated DeltaNet, KDA) aim to
match its retrieval accuracy with a fixed-size state.

We implemented the model in our own code base (`models/`). Every attention type
shares the same backbone, so only the token mixer changes:

* token embedding, 4 pre-norm blocks (softmax attention + SwiGLU MLP), RMSNorm, tied output head
* d_model 256, 4 heads; RoPE inside attention is the only position information
* input `<bos> context <sep> query <sep> answer <eos>`, encoded with our word-level
  tokenizer (`synthdata/tokenizer.py`, 733-token vocabulary)
* loss on the answer and `<eos>` only; greedy decoding after the second `<sep>`,
  not restricted to valid status values, so "other" errors stay visible

### Training recipe

Shared with the simple baseline (`simple-baseline.md`); only `--attention` differs.

| Setting | Value | Why |
|---|---|---|
| Training data | `data_big`: 400k examples, same generator | With 80k examples the model memorized instead of learning to retrieve. Dev/test are unchanged. |
| `--multi-query 16` | Each training sequence also asks about up to 16 other projects in the same context | With one answer per sequence, softmax stalled at about 37% on 128-token contexts. Evaluation still asks only the official query. |
| `--curriculum-steps 12000` | Length buckets 128 → 2048 are unlocked gradually, 3,000 steps each | Retrieval is learned in a sudden jump on short contexts first. With 4,000 curriculum steps, softmax reached only 29%. |
| Optimization | 20k steps, 65,536 tokens per batch, AdamW (β = 0.9, 0.95), lr 1e-3, 500 warmup steps and cosine decay, weight decay 0.1, grad clip 1.0, bf16 | |
| Checkpoint selection | Best accuracy on a fixed 2,000-example dev subset | |

## Code

| File | Role |
|---|---|
| `models/attention.py` | `SoftmaxAttention` (RoPE + `F.scaled_dot_product_attention`), plus the other token mixers |
| `models/model.py` | Shared backbone: embedding, pre-norm blocks, SwiGLU MLP, tied head |
| `models/data.py` | Vocabulary, encoding, multi-query targets, length-bucketed batching, curriculum |
| `models/inference.py` | Greedy decoding, checkpoints, dev accuracy |
| `train.py` | Training CLI (resumes from `last.pt`, scores the full dev split at the end) |
| `predict.py` | Writes `{"id", "prediction"}` JSONL for `score.py` |
| `notebooks/colab_train.ipynb` | Colab setup that runs the same commands |
| `tests/test_models.py`, `tests/test_training.py` | Causality, shared-backbone and training/resume tests |

## How to run

Requirements: `pip install -r requirements.txt` and a GPU. On a Colab A100 we measured
about 670k tokens/s, so 20,000 steps × 65,536 tokens (1.31B tokens) take about
33 minutes, plus the periodic dev evaluations.

```sh
# Data
tar -xzf data.tar.gz
python generate_data.py --n-train 400000 --out data_big

# 1. Train (also writes runs/softmax-cur12k/dev_pred.jsonl and dev_scores.json)
python train.py --attention softmax --out runs/softmax-cur12k --data-dir data_big \
    --multi-query 16 --curriculum-steps 12000 --steps 20000 --max-tokens 65536

# 2. Predict the test split (needs --final; read from data/, which has the same test set)
python predict.py --checkpoint runs/softmax-cur12k/best.pt --split test --final \
    --out runs/softmax-cur12k/test_pred.jsonl

# 3. Score
python3 score.py --gold data/test.jsonl --predictions runs/softmax-cur12k/test_pred.jsonl \
    --output runs/softmax-cur12k/test_scores.json
```

Rerunning the training command resumes from `last.pt`. `python3 summarize_runs.py`
compares the dev scores of finished runs in one table.

## Sample output

Our test predictions are included as `softmax_test_pred.jsonl`. These are the first 3 lines:

```json
{"id": "test-000001", "prediction": "snowy"}
{"id": "test-000002", "prediction": "sphere"}
{"id": "test-000003", "prediction": "daisy"}
```

All three match the gold answers (`snowy`, `sphere`, `daisy`). The third is an
update example in a 1,024-token context with 16 projects.

## Results

Test split (10,000 examples: 5,075 recall, 4,925 update):

```text
$ python3 score.py --gold data/test.jsonl --predictions runs/softmax-cur12k/test_pred.jsonl
Accuracy: 0.999700 (9997/10000)
recall accuracy: 1.000000 (5075/5075)
update accuracy: 0.999391 (4922/4925)
stale errors: 3 (rate=0.000300)
distractor errors: 0 (rate=0.000000)
other errors: 0 (rate=0.000000)
```

**Test accuracy: 99.97%** (checkpoint from step 18,000).

Comparison with the simple baseline (error columns are rates over all examples):

| Split | Model | Accuracy | Recall acc. | Update acc. | Stale | Distractor | Other |
|---|---|---|---|---|---|---|---|
| Test | Simple baseline: linear attention | 0.1732 | 0.1911 | 0.1547 | 0.0349 | 0.7778 | 0.0141 |
| Test | **Strong baseline: softmax** | **0.9997** | **1.0000** | **0.9994** | 0.0003 | 0.0000 | 0.0000 |
| Dev | Simple baseline: linear attention | 0.1753 | 0.1989 | 0.1520 | 0.0382 | 0.7735 | 0.0130 |
| Dev | **Strong baseline: softmax** | **1.0000** | **1.0000** | **1.0000** | 0.0000 | 0.0000 | 0.0000 |

With the same backbone, data and recipe, softmax attention almost solves the
task at every context length up to 2,048 tokens. Its only 3 test errors are
stale values on update examples (an earlier status of the queried project).
Linear attention mostly returns another project's value. This
gap is what our extensions (Gated DeltaNet and KDA, which add decay and the
delta rule to the linear recurrence) try to close.

## References

1. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
   Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017.
   [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS.
2. Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. 2021.
   [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864). arXiv:2104.09864.
3. Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. 2020.
   [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://proceedings.mlr.press/v119/katharopoulos20a.html). ICML.
