# Evaluation: exact-match retrieval accuracy

The task is to return the **current status value** of the queried project. Each
example has one gold answer. Our primary metric is exact-match accuracy: the
fraction of examples for which the predicted status equals that answer. This
directly measures successful retrieval for both recall and update examples.
Partial word overlap does not represent successful retrieval of a status label.

## Formal definition

For gold answers $y_1,\ldots,y_N$ and corresponding predictions
$\hat y_1,\ldots,\hat y_N$, define $s(x)$ as removal of leading and trailing
whitespace. Then

$$
\operatorname{Accuracy} = \frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}[s(\hat y_i)=s(y_i)].
$$

Accuracy ranges from 0 to 1; higher is better. Every example has equal weight.
Overall accuracy is calculated over all examples, rather than averaging group
accuracies. Matching is case-sensitive and preserves punctuation and internal
whitespace: `merged` and ` merged\n` match, but `Merged`, `merged.`, and
`The answer is merged` do not. Models should output only the decoded status value,
with model special tokens removed before writing predictions. An empty prediction
is allowed and counts as an error.

This is standard classification accuracy; see the
[scikit-learn accuracy definition](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score).
The evaluator implements the formula using only the Python standard library.
The error categories and diagnostic groupings below are specific to our synthetic
task and the analyses proposed in Milestone 1.

## Connection to the Milestone 1 literature

The three reviewed papers motivate testing memory and retrieval, but they do not
all use the same metric or evaluation protocol:

| Paper | Verified evaluation | Relevance to our metric |
| --- | --- | --- |
| Katharopoulos et al. (2020), §4.1.1 and Figure 2 [1] | The synthetic copy experiment reports cross-entropy loss against training steps. | Motivates a memory task, but does not define our exact-match scoring rule. |
| Yang, Kautz, and Hatamizadeh (2025), §3.2, Table 2, and Appendix B.1 [2] | Evaluates single-needle key-value retrieval using RULER at multiple context lengths. | Motivates evaluating retrieval across context lengths. |
| Kimi Team (2025), §5.1 and Figure 4 [3] | Reports best training accuracy for palindrome, multi-query associative recall (MQAR), and stack tasks as sequence length varies. | Supports accuracy-based analysis of retrieval and state tracking; our held-out examples have one query and one status answer each. |

Our metric is a project-specific application of standard accuracy, not a
reproduction of those benchmark protocols. In particular, RULER's official
NIAH scorer checks whether each reference occurs in the prediction,
case-insensitively [4]. We require the entire normalized prediction to equal
the single gold status: listing several possible statuses should not earn credit.
The whitespace rule and stale/distractor/other taxonomy are our choices.
Our dev/test scores also differ from the training accuracies in [3]; numerical
scores are not directly comparable across these datasets and protocols.

## Inputs and command line

Run from the repository root using Python 3. No packages need to be installed.
The gold input is an extracted dataset split, in the JSONL format in `data.md`.
The scoring fields required on each gold record are `id`, `answer`, `task`,
`stale_values`, `distractor_values`, `context_length`, `num_entities`, `num_updates`,
and `needle_position_bin`; `context` and `query` are ignored by the evaluator.
To extract only the development split from the supplied archive:

```sh
tar -xzf data.tar.gz data/dev.jsonl
```

Each line of the system output must be a JSON object with a unique `id` and a
string `prediction`, for example:

```json
{"id": "dev-000001", "prediction": "merged"}
```

This illustrates the format only; it does not claim `merged` is the answer to
that example. Write one prediction for **every** gold example, using the same ID.
Predictions can appear in any order. Missing, extra, or duplicate IDs, invalid JSON,
missing required fields, or non-string predictions cause an error (exit status 2),
so a partial run cannot silently inflate the score. Blank lines are invalid JSONL.
Additional fields are ignored; the evaluator requires `prediction` and does not
fall back to an `answer` field that might have been copied from gold.

```sh
python3 score.py --gold data/dev.jsonl --predictions dev_predictions.jsonl --output dev_scores.json
```

`--output` is optional. It saves all diagnostic groups as JSON for plotting and
comparison across architectures. Its parent directory must already exist.
Without it, the script prints overall and per-task accuracy and error counts/rates.
All reported rates are on the 0–1 scale.

The checked-in four-example fixture exercises one correct answer and each error
type. It contains only scoring fields and is **not a model evaluation result**:

```sh
python3 score.py --gold examples/gold.jsonl --predictions examples/predictions.jsonl --output /tmp/example_scores.json
```

Expected output:

```text
Accuracy: 0.250000 (1/4)
recall accuracy: 0.500000 (1/2)
update accuracy: 0.000000 (0/2)
stale errors: 1 (rate=0.250000)
distractor errors: 1 (rate=0.250000)
other errors: 1 (rate=0.250000)
Full report: /tmp/example_scores.json
```

## Error analysis

For example $i$, let $S_i$ be `stale_values` (earlier values of the target project)
and $D_i$ be `distractor_values` (all values of other projects, including their
earlier values). Apply the same whitespace normalization to these values.
Predictions belong to exactly one category:

| Category | Definition |
| --- | --- |
| correct | Prediction equals the current gold answer. |
| stale | Prediction is in $S_i$: an outdated value of the queried project. |
| distractor | Prediction is in $D_i$: a value associated with another project. |
| other | Any remaining output, including empty or malformed answer text. |

The dataset makes the answer, stale values, and distractor values disjoint; the
script checks this to keep error attribution unambiguous. `other` is an observable
output category, not evidence of a particular internal model failure.

For an error category $c$ and a group of $N_g$ examples, the report contains:

$$
\operatorname{ErrorRate}_{g,c}=\frac{\#\{i\in g:\operatorname{category}(i)=c\}}{N_g},
\qquad
\operatorname{ErrorFraction}_{g,c}=\frac{\#\{i\in g:\operatorname{category}(i)=c\}}{N_g-\operatorname{Correct}_g}.
$$

`error_rates` use **all examples** as the denominator; `error_fractions` show the
distribution **among errors only**. If a group has no errors, its fractions are
JSON `null` (undefined). Accuracy plus the three error rates equals 1.

## Diagnostic breakdowns

Every observed group has `n`, `correct`, `accuracy`, `error_counts`, `error_rates`,
and `error_fractions`. The JSON report provides:

- `task`: recall versus update.
- `context_length`: the dataset's context token-length bucket.
- `num_entities`: number of projects.
- `num_updates`: target status changes (zero for recall).
- `needle_position_bin`: answer sentence position, bins 0–4, from beginning to end.
- `task_context_length`: context length within each task.
- `task_num_updates`: update-count diagnostics within each task.
- `task_context_length_num_entities`: project-count comparisons at a fixed task
  and context length, as suggested in the mentor notes.

Groups are sorted; combinations absent from the split are omitted. Always inspect
group counts when comparing results. The joint breakdown conditions on task and
context length, but other variables can still vary: these are descriptive
comparisons, not fully controlled experiments. `context_length` measures context
only, not query or separator tokens.

Use the development split for model selection and debugging. Reserve the test
split for final evaluation, as specified in `data.md`. Use the same prediction
format and scoring policy for all architectures. Baseline scores are reported in
`simple-baseline.md` (linear attention) and `strong-baseline.md` (softmax attention).

## Verification

```sh
python3 -m unittest discover -s tests -v
```

## References

1. Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret.
   2020. [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://proceedings.mlr.press/v119/katharopoulos20a.html).
   ICML, PMLR 119:5156–5165.
2. Songlin Yang, Jan Kautz, and Ali Hatamizadeh. 2025.
   [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/html/2412.06464v3).
   ICLR 2025; arXiv:2412.06464v3.
3. Kimi Team. 2025.
   [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/html/2510.26692v1).
   Technical report, arXiv:2510.26692v1.
4. NVIDIA. [RULER official synthetic evaluation metrics](https://github.com/NVIDIA/RULER/blob/main/scripts/eval/synthetic/constants.py),
   `TASKS['niah']` and `string_match_all`. Consulted to distinguish the RULER
   matching policy from this project's exact-match rule.
