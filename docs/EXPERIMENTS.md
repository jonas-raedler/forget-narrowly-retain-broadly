# Experiments reference

Full reference for running and adjusting experiments: the complete set of environment variables, the
available model / method / topic choices, and the paper's per-(model, method) hyperparameters.
For the quickstart and the conceptual overview, see the [README](../README.md).

Every `suite_*.sh` script is driven by environment variables. Set any of the ones below to change a run, or leave it unset to use the script's hard-coded default. Each script
also accepts `-h` / `--help`, which prints its own env-var banner and exits without running anything.

```bash
bash scripts/suite_unlearn.sh --help        # print the env vars a script reads, then exit
```

---

## `RUNARGS` format (training)

`RUNARGS` overrides the per-(model, method) paper hyperparameters:

```
RUNARGS="<lr> [gamma] [alpha] [gnorm] [randpair] [overrides]"
```

- `gamma` = λ_f (forget weight), `alpha` = λ_r (retain weight).
- Trailing optional fields may be omitted (e.g. `RUNARGS="5e-7"` sets just the lr, keeping
  the paper gamma/alpha). Use `-` to skip one while setting a later one
  (e.g. `RUNARGS="5e-7 1 2 - data.x=y"` skips `gnorm`/`randpair` but sets an override).
- The **epoch count** is set by the `EPOCHS` env var (default `20`):
  e.g. `EPOCHS=10 RUNARGS="5e-7 1 2"`.
- A free-form tag appended to the run's `exp_suffix` is set by the `EXTRA` env var: e.g.
  `RUNARGS="5e-7 1 2" EXTRA=_myexp`. Unset `EXTRA` → `_paper` on paper-default runs,
  empty on a `RUNARGS` override. `EXTRA=none`/`off`/`-` drops the tag.
- `;`-separate multiple entries to launch several runs.
- For **relearning** the format is just `RUNARGS="<epochs> <lr>"` (e.g. `"10 1e-5"`).
  `EPOCHS`/`EXTRA` do not apply there.

---

## Training (`suite_unlearn.sh`, `suite_sequential_unlearn.sh`, `suite_combined_unlearn.sh`)

| Env var | Scripts | Selects | Format |
|---|---|---|---|
| `MODEL` | all training | model(s) to train | one+ model keys, comma-separated (see **Choices → Models**). Default `llama_3b` |
| `METHOD` | all training | unlearning method (trainer) | one method name (see **Choices → Methods**). Default `JensUnPP`. Auto-fills the per-model **paper** hyperparameters (see **Paper-default hyperparameters**) |
| `RUNARGS` | all training | hyperparameters | see **`RUNARGS` format** above |
| `EPOCHS` | all training | training epochs | integer. Default `20` |
| `EXTRA` | all training | free-form exp_suffix tag | `<tag>` verbatim (include the leading `_`). Default: `_paper` on paper-default runs, empty on a `RUNARGS` override. `none`/`off`/`-` → drop |
| `TOPIC` | `suite_unlearn.sh` | forget topic (single) | one topic (see **Choices → Topics**). Default `challenger_disaster` |
| `SEQ_TOPICS` | sequential | topic chain, order = training order | comma-separated topics (see **Choices → Topics**). Default: all topics |
| `COMB_TOPICS` | combined | topic set, trained jointly | comma-separated topics (see **Choices → Topics**). Default: all topics |
| `OPT_EVAL` | sequential, combined | auto-evaluate after each step | `true` (default) \| `false` |
| `TRAIN_GPUS` | all training | GPU indices | comma-separated, e.g. `0,1,2,3` (default) |
| `REFUSAL_KEY` | all training | refusal-string prefix (JensUnPP only) | `unfor_comma` \| `noi_comma` \| `none` … (default: per-model paper prefix) |
| `FORCE_REGEN` | combined | rebuild the auto-generated combined dataset/experiment configs from the per-topic sources | set to `1`. Default: existing combined configs are reused as-is |
| `SMOKE_TEST` | `suite_unlearn.sh` | fast end-to-end dry run (a few steps + minimal eval) | set to `1` to enable. Saved under a separate `_smoke` suffix |
| `SMOKE_STEPS` | `suite_unlearn.sh` | optimizer steps in a smoke run | int. Default `4` |
| `SMOKE_TASKS` | `suite_unlearn.sh` | eval tasks in a smoke run | comma-separated. Default `retain,forget_rephrasings` |

## Evaluation (`suite_evaluation_optimized.sh`)

| Env var | Selects | Format |
|---|---|---|
| `MODELS` | checkpoints to evaluate | `key=path` pairs, `;`-separated (bare `path` → auto key `m1`, `m2`, …) |
| `TOPIC` | force a single topic | one topic (see **Choices → Topics**). Default: auto-detect from each model path |
| `TASKS` | forget/retain eval tasks | comma-separated (see **Eval tasks** below) |
| `UTILITY_TASKS` | utility eval tasks | comma-separated: `mmlu` \| `repet`. Default: `mmlu,repet` |
| `RGQ_BI_MODELS` | models for RGQ-bi (vs pretrained) | comma-separated keys from `MODELS` (or `none` to disable). Default: all non-pretrained models |
| `EVAL_GPUS` | GPU indices | comma-separated, e.g. `0,1,2,3` (default) |
| `JUDGE_MODEL` / `JUDGE_TAG` / `JUDGE_BATCH_SIZE` / `JUDGE_N_GPUS` | judge model / tag / batch / GPUs-per-judge | HF id / string / int / int. Default from `configs/eval.yaml` (`Qwen/Qwen3.5-35B-A3B`, `qwen35b`, 2) |
| `EXPERIMENT` | output-name prefix | any string. Default: none |

**Eval tasks** (`TASKS`):
- **On by default**: `retain`, `forget_rephrasings`, `forget_rephrasings_gibberish`.
- **Available (opt-in)**: `forget_adversarial` (unseen adversarial forget queries, heavier, off by
  default), `retain_train_rephrasing`, `retain_gibberish`.

**Pretrained baseline**: pass a bare HF model id as `MODELS`
(e.g. `MODELS="meta-llama/Llama-3.2-3B-Instruct"`). It is detected as the pretrained baseline, and
its results land under `evaluations/worstCase/<topic>/<model>/pretrained/` (the baseline row of the
results tables). An HF id has no save path to derive the topic from, so `TOPIC` must be set.
`rgq_bi` never runs for a pretrained model (it is the comparison reference), but this run's `repet`
output is the reference that `rgq_bi` needs when evaluating unlearned checkpoints. Example:

```bash
TOPIC=challenger_disaster MODELS="meta-llama/Llama-3.2-3B-Instruct" bash scripts/suite_evaluation_optimized.sh
```

## Relearning (`suite_relearn.sh`)

| Env var | Selects | Format |
|---|---|---|
| `MODEL_PATH` | unlearned checkpoint(s) to relearn | path(s), `;`-separated |
| `METHOD` | relearn trainer | one trainer name. Default `GradLearn` |
| `RUNARGS` | relearn hyperparameters | `"<epochs> <lr>"`, `;`-separate multiple. Default: the paper relearn setting, 10 epochs at lr `1e-5` (Llama) / `2e-6` (Ministral) / `5e-6` (Qwen) |

Relearn also reads `TOPIC`, `TASKS` (e.g.
`forget_rephrasings,retain_train_rephrasing,mmlu,repet,rgq_bi`), `TRAIN_GPUS`, and the `JUDGE_*` vars
above. Topic / model / method are otherwise inferred from `MODEL_PATH`.

---

## Choices

**Topics** ([`configs/topics/`](../configs/topics)): `challenger_disaster` (default),
`salem_witch_trials`, `steve_jobs_medical`, `britney_spears_conservatorship`.
A fifth topic config, `challenger_baseline`, reproduces the paper's dataset-comparison
experiment: it unlearns the same Challenger disaster facts but using the LKF baseline data
(`apeleg/LKF-baseline-challenger-*`) instead of SUITE. Llama-only, with its own hyperparameter
defaults (10 epochs, random pairing).

**Models** ([`configs/model/`](../configs/model)):

| `MODEL` key | HuggingFace model |
|---|---|
| `llama_3b` (default) | `Llama-3.2-3B-Instruct` |
| `ministral_3b` | `Ministral-3-3B-Instruct-2512-BF16` |
| `qwen_9b` | `Qwen3.5-9B` |

**Methods** (set via `METHOD`, see [`configs/trainer/`](../configs/trainer)): `JensUnPP` (ours,
JensUn++, default). Baselines are `JensUnBaseline`, `JensUnBaselineHash`, `GradDiff`, `RMU`, `PDU`,
`WGA`, `UNDIAL`, `SatImp`, `NPO`, `SimNPO`. Relearning typically uses `GradLearn`. The 5 methods with
paper-tuned defaults are `JensUnPP`, `JensUnBaseline`, `GradDiff`, `NPO`, `PDU` (others use a generic
default, so set `RUNARGS`).

---

## Paper-default hyperparameters

Selecting `MODEL` + `METHOD` auto-fills the paper's per-(model, method) hyperparameters
(20 epochs, `extra=_paper`, with grad-norm scaling on only for `JensUnPP`). Values are the grey-marked
selections from the paper (Llama → Table 6, Ministral-3-3B → Table 8, Qwen → Table 9). In `RUNARGS`,
**γ = λ_f** (forget), **α = λ_r** (retain).

| method           | `llama_3b` (lr/γ/α) | `ministral_3b`  | `qwen_9b`       |
|------------------|---------------------|-----------------|-----------------|
| `JensUnPP`       | 3e-6 / 0.33 / 1     | 1e-6 / 2 / 1    | 2e-6 / 0.33 / 1 |
| `JensUnBaseline` | 5e-6 / 1 / 1        | 1e-6 / 1 / 0.25 | 3e-6 / 1 / 0.5  |
| `GradDiff`       | 5e-7 / 1 / 2        | 3e-7 / 1 / 1    | 8e-7 / 1 / 1    |
| `NPO`            | 8e-6 / 1 / 3        | 1e-6 / 1 / 1    | 5e-6 / 1 / 1    |
| `PDU`            | 3e-6 / 1 / 1        | 8e-7 / 1 / 0.25 | 3e-6 / 1 / 0.25 |

A (model, method) cell not listed falls back to the `llama_3b` value for that method. An unknown
method falls back to a generic `1e-6 / 1 / 1` with a warning.

---

## Good to know

- Training auto-evaluates the new checkpoint and writes metrics under `evaluations/`, so
  `suite_evaluation_optimized.sh` is only for re-evaluating existing checkpoints. The eval script
  auto-detects `seq_`/`comb_` save paths and evaluates each constituent topic.
- The judge is a locally-loaded Qwen model (default `Qwen/Qwen3.5-35B-A3B`). Configure it via
  `configs/eval.yaml` or the `JUDGE_*` env vars. The pretrained model's `repet` output must exist
  before the `rgq_bi` judge runs. If it is missing, `rgq_bi` is skipped with a warning. Produce it
  with a pretrained-baseline eval (see **Evaluation** above). Occasional dropped judge samples are excluded
  from the metrics (so numbers vary slightly between runs).
- Entry points: `src/train.py` (training), `src/eval_full_pipeline.py` (eval). Metric columns are
  defined in the [README's Interpreting results section](../README.md#-interpreting-results).
