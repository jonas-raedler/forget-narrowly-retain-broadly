# The SUITE data

SUITE is pulled from HuggingFace automatically by the training/eval configs, so you usually don't touch it
directly. All topics live in one repo, sliced by the `topic` column. For the metric columns these splits
feed into, see the [README's Interpreting results section](../README.md#-interpreting-results).

- **`apeleg/SUITE`**: splits `forget_train`, `retain_train`, `forget_eval`, `retain_eval`. Columns
  are `[topic, question, answer, label]`.
- **`apeleg/SUITE-rephrasings`**: split `forget_eval_rephrasings`, with the same columns plus the
  `q_*` / `blank_*` paraphrase columns.

**Which splits feed which phase:**

| Phase | Split(s) | Repo |
|---|---|---|
| **Training** | `forget_train` + `retain_train` | `apeleg/SUITE` |
| **Evaluation** | `retain_eval` + `forget_eval_rephrasings` | `apeleg/SUITE` (+ `apeleg/SUITE-rephrasings`) |

Row `i` of `forget_train` is paired with row `i` of `retain_train`, so the two training splits are
consumed together. This hard alignment is the **default for every unlearning method**. Pass
`randpair=true` (Hydra override `+data.random_pairing=true`) for the random-pairing ablation.

> `forget_eval` (in `apeleg/SUITE`) is **not** used directly: it documents the original forget
> questions, which are evaluated through `forget_eval_rephrasings` (originals + paraphrases).

Loading a single topic's split:

```python
from datasets import load_dataset
ds = load_dataset("apeleg/SUITE", split="forget_train").filter(lambda x: x["topic"] == "challenger_disaster")
```

**Example row** from `apeleg/SUITE-rephrasings` (a direct forget question with augmentations):

```json
{
  "topic": "challenger_disaster",
  "question": "How many seconds after liftoff did the Challenger vehicle break apart?",
  "answer": "73 seconds",
  "label": "M3-direct",
  "q_gemini1": "What was the duration in seconds after liftoff before the Challenger vehicle disintegrated?",
  "blank_gemini1": "The Challenger vehicle broke apart exactly ____ after liftoff."
}
```

`q_gemini1`-`q_gemini10` are the 10 paraphrases and `blank_gemini1`-`blank_gemini5` the 5
fill-in-the-blank variants. The `label` is `<fact-id>-<modality>`: the fact id (`M1`-`M5`, `K1`-`K20`)
identifies which target fact the question probes, and the **modality** is `-direct`, `-indirect`
(held-out, **test-only** multi-hop), or `-reverse`, mapping directly to the `QD` / `QD+I` / `QR`
forget columns in the [README's Interpreting results section](../README.md#-interpreting-results).
Each topic ships a **forget set** (probes *under-forgetting*: direct, reverse, and test-only indirect
queries) and a **retain set** (probes *over-forgetting*: semantic tiers `s0`-`s15`, syntactic,
lexical, and general-knowledge categories). See the [project page](https://amitpeleg.github.io/forget-narrowly-retain-broadly/) for the full taxonomy.

## Building the dataset / adding a topic

You only need this to regenerate SUITE or add a new forget topic.

```bash
cp suite_pipeline/configs/sample.yaml my_config.yaml   # edit paths
python -m suite_pipeline.run_pipeline my_config.yaml    # all steps. --no_upload skips the HF push
```

Adding a new forget topic requires a topic runtime config (`configs/topics/<topic>.sh`), dataset configs
(`configs/data/datasets/<topic>/`), and experiment presets
(`configs/experiment/unlearn/suite/<topic>/`).
