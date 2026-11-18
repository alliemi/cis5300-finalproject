# Evaluating Linear Attention for Long-Context Retrieval

CIS 5300 term project. We train small models from scratch with different attention mechanisms and compare how well they retrieve facts from long contexts. The data is synthetic Recall / Update data.

* `data.md`: the dataset (format, example, how it was made)
* `scoring.md`: the evaluation metric and error categories

## Pipeline

| Stage | Command | Output |
|---|---|---|
| 1. Generate data | `python3 generate_data.py --out data` | `data/{train,dev,test}.jsonl`, `data/vocab.txt` |
| 2. Validate data | `python3 validate_data.py data` | `data/validation_report.json` |
| 3. Train | `python train.py --config configs/linear.json --run runs/linear-s0` | checkpoints, `dev_predictions.jsonl`, `dev_scores.json` in the run folder |
| 4. Score | `python3 score.py --gold data/dev.jsonl --predictions PRED.jsonl` | accuracy + error breakdowns (step 3 already does this for dev) |
| 5. Compare | `python3 summarize_runs.py` | one table across all runs, `runs/summary.csv` |

Instead of steps 1–2 you can run `tar -xzf data.tar.gz`, which gives byte-identical files. `make_archive.py data` rebuilds `data.tar.gz`.

## Layout

```
synthdata/              data generator package (standard library only)
generate_data.py        CLI: write the splits
validate_data.py        CLI: check every record
make_archive.py         CLI: build data.tar.gz
score.py                official scorer (exact match + stale/distractor/other)
examples/               tiny gold/prediction fixture for score.py

models/                 shared model body + one file per attention type ("mixer")
  __init__.py           MIXERS registry and build_model()
  backbone.py           embedding, blocks (norm, short conv, mixer, MLP), output layer
  linear_attention.py   vanilla linear attention (Katharopoulos et al., 2020)
  no_mixer.py           "none" control: no attention, only the 4-token convolution
training/               data encoding, training loop, evaluation helpers
configs/                base.json (shared size + recipe) and one small file per model
train.py                train one model; resumes automatically
predict.py              predictions + scores for any split (test needs --final)
summarize_runs.py       compare finished runs
notebooks/colab_train.ipynb   Colab setup + training
tests/                  unit tests for the generator, the scorer and the model harness
```

## Training a model

**On a laptop** (Python 3.11 via [uv](https://docs.astral.sh/uv/); the system Python 3.9 is too old for current PyTorch):

```sh
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
tar -xzf data.tar.gz
.venv/bin/python train.py --config configs/linear.json --run runs/smoke --max-context-length 128 --max-steps 3000
```

On an Apple-silicon Mac this uses the GPU (MPS) automatically. `--max-context-length` and `--train-subset` make smaller runs.

**On Colab** (for real runs): open `notebooks/colab_train.ipynb` in Colab, pick an A100 or L4 GPU, and run the cells. They clone this repo, unpack the data, and keep `runs/` in Google Drive. If Colab disconnects, rerun the setup cell and the training cell: training resumes from the last checkpoint.

**What a run produces** (`runs/<name>/`): `config.json` (merged config, git commit, parameter count), `log.jsonl` (loss, speed, dev-subset accuracy by task and length), `last.pt` (for resuming), `best.pt` (best dev-subset accuracy), and, at the end, `dev_predictions.jsonl` and `dev_scores.json` for the full dev split.

**Fair comparisons:** every model uses the same body, size and training recipe from `configs/base.json`. Only the mixer differs. Don't change `base.json` for one model; override per run with `--set train.lr=3e-4` if the team agrees, and the override is recorded in the run's `config.json`. A run folder refuses a different config, so start a new `--run` instead.

Use `dev` for model selection. `predict.py --split test` only works with `--final`: keep the test set for the final evaluation.

## Adding a new attention model

1. **Write the layer** in `models/<name>.py`: an `nn.Module` built as `Mixer(d_model, n_heads, **mixer_cfg)`, with `forward(x)` mapping `[batch, length, d_model]` to the same shape. It must be **causal** (position *t* only uses positions ≤ *t*). Handle positions inside the layer if it needs them (e.g. RoPE for softmax). Set `requires_cuda = True` if it only runs on NVIDIA GPUs (e.g. Triton kernels from flash-linear-attention); wrap such library layers so they return a plain tensor and turn off their own short convolution, since the shared block already has one.
2. **Register it** in `models/__init__.py`: import it and add it to `MIXERS`.
3. **Add a config**, `configs/<name>.json`:
   ```json
   {"base": "base.json", "mixer": "<name>", "mixer_cfg": {}}
   ```
4. **Run the tests**: `.venv/bin/python -m unittest discover -s tests`. The contract tests check every registered mixer automatically: output shape, causality, finite gradients, and odd sequence lengths.

## Requirements and tests

The data and scoring code use only the Python 3.9+ standard library. Model code needs `requirements.txt` (PyTorch).

```sh
python3 -m unittest discover -s tests              # data + scoring tests (model tests are skipped)
.venv/bin/python -m unittest discover -s tests     # everything, including the model harness
```
