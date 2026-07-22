# Proximity-Stratified Relearning Attacks (fork extension)

**Question:** is relearning more effective the semantically closer the
relearning data is to the forget data? The paper (Tabs. 16–18) fine-tunes
unlearned models on the *entire* retain train set — one broad attack. This
experiment turns that binary result into a dose-response curve: forget-
knowledge recovery as a function of the semantic proximity of the relearning
data, using SUITE's tier structure (s0 closest … s15 most distant) as the
x-axis.

Full design rationale lives in the research repo:
`tamper_resistant_unlearning/notes/suite_proximity_experiment.md`.

## What this fork adds (upstream code untouched)

| Path | Purpose |
|---|---|
| `configs/data/datasets/{topic}/bands/*.yaml` | one QADataset config per attack arm (band selection via `exclude_label_prefixes`) |
| `scripts/gen_band_configs.py` | generates the band configs (edit this, not the YAMLs) |
| `scripts/verify_band_configs.py` | asserts each band selects exactly the intended rows (run locally, no GPU) |
| `scripts/proximity_relearn_sweep.sh` | runs the independent attack arms from one checkpoint + NLL evals + one batched judge eval |
| `src/evals/nll_eval.py` | answer-NLL metric (paper Sec. B.8 style; upstream eval is judge-only) |
| `scripts/results/plot_proximity.py` | ΔNLL-vs-band aggregation + main figure |
| `runpod/*`, `remote-kernels.toml` | single-GPU RunPod tooling (pip venv port of `setup.sh`, crash-safe result publishing) |
| `.gitignore` | carve-outs so small eval artifacts publish to git |

## Attack arms

All arms start from the SAME unlearned checkpoint and are fully independent
(never sequential — that would confound proximity with total exposure).

| Arm | Relearning data | Rows | Split |
|---|---|---|---|
| R7 | s13+s14 | 50 | retain_eval¹ |
| R6 | s11+s12 | 50 | retain_eval¹ |
| R5 | s9+s10 | 50 | retain_train |
| R4 | s7+s8 | 50 | retain_train |
| R3 | s5+s6 | 50 | retain_train |
| R2 | s3+s4 | 50 | retain_train |
| R1 | s1+s2 | 50 | retain_train |
| R0 | s0 | 25² | retain_train |
| C-GK | general knowledge only | 50 | retain_train — "distance ≈ ∞" drift anchor |
| C-Lex | lexical only | 50 | retain_train — token overlap w/o semantic overlap |
| C-Full | full retain train set | 450 | the paper's exact attack (Tab. 16 reproduction) |
| A-Forget-partial | facts K1–K5 (all variants) | 90 | forget_train — readout on the 20 HELD-OUT facts |
| A-Forget-full | all 25 facts | 450 | forget_train — absolute ceiling |

¹ Tiers s11–s15 exist only in `retain_eval` (never seen during unlearning).
Consequence: for R6/R7 the in-band retain-eval rows are contaminated (trained
on); forget-side metrics are unaffected. s15 is unused (R7 already has 50 rows).
² Salem has no s0 at all; for challenger R0 is simply the smaller arm.

## Dose / compute equalization

Band arms run with `+data.random_pairing=true` (anchor=forget): epoch length
is `len(forget_train)=450` draws with the retain example sampled uniformly
from the band each step. Therefore **every arm executes identical optimizer
steps, LR schedule, and warmup** (~140 steps at global batch 32 over 10
epochs) — compute-matched by construction, identical to C-Full's step count.
Every arm therefore sees 4,500 example presentations (450 × 10 epochs); band
size only changes the repetition factor per question (50-row band ≈ 90× per
question over the run; R0's 25 rows ≈ 180×; A-Forget-partial's 90 rows ≈ 50×;
C-Full's 450 rows exactly 10×). We use full bands rather than subsampling to
N=40 because `QADataset` has no seeded sampler and `max_rows` slices in
dataset order (tier-biased). C-Full instead uses the stock paired
(450↔450) config — byte-identical to the paper's relearn data path.

## Deliberate deviations from the paper / original design note

- **LR schedule is linear, not cosine.** The released code sets no
  `lr_scheduler_type` (HF default = linear, with `warmup_epochs=1` converted
  to warmup steps in `src/trainer/__init__.py`). We follow the code, not the
  paper text, and keep it identical across arms.
- **Refusal context is OFF at runtime for every relearn arm.** The retain-band
  YAMLs mirror `{topic}/retain.yaml` (including `add_refusal_context: true`),
  but `src/train.py` force-disables refusal context for all non-JensUn++
  trainers — GradLearn included — and this applies equally to C-Full's stock
  config. So all arms (bands and C-Full alike) train with the plain context
  pool; the only difference between arms is row membership. Forget-based arms
  (A-*) set it false explicitly — the attacker trains on true QA pairs.
- **NLL eval formats without few-shot context** (metric must be arm-invariant;
  the context pool would leak band-dependent text into the readout).

## Metrics

1. **Δ answer-NLL on `forget_eval`** vs the unlearned checkpoint (primary,
   continuous; negative = resurfacing) — `src/evals/nll_eval.py`, which also
   logs per-tier retain NLL and `forget_train` NLL.
2. **Judge metrics (Q_D, Q_R, Q_All)** via the upstream pipeline, batched over
   all arms in one call (judge loaded once). Comparability anchor: Tab. 16
   (Llama, JensUn++, relearn): Q_D+I 5→10, Q_R 3→1, Q_All 8→11 (±few points
   seed noise; upstream README notes judge sampling variance).
3. **Retain accuracy on `retain_eval`** + optional MMLU — collateral-damage
   control (exclude in-band tiers for R6/R7).
4. Relearn train loss from the trainer logs (sanity that the attack ran).

## Milestones

- **M0 (local, done):** fork, band configs generated + verified against
  `dataset/challenger_disaster/final/`.
- **M0b (pod, ~1 GPU-h):** `runpod/setup_pod.sh`; Hydra dry-run of the band
  override (`--cfg job`); 4-step SMOKE relearn of one band; verify
  `random_pairing` reaches `get_data` and a checkpoint lands in `saves/`.
- **M1 (~4–6 GPU-h):** pretrained baseline eval → JensUn++ unlearn
  (`METHOD=JensUnPP MODEL=llama_3b`, paper defaults: lr 3e-6, γ=0.33, α=1,
  gnorm, 20 epochs) → C-Full relearn → compare vs Tab. 16 → baseline NLLs.
- **M2 (~8–12 GPU-h):** the remaining 12 arms × 1 seed via
  `proximity_relearn_sweep.sh`; NLL first (cheap, primary), judge batched
  second, MMLU last/droppable. Budget check at the $25 soft cap.
- **M3 (local):** `plot_proximity.py`; interpret curve shape (monotonic rise
  → proximity effect; knee at R0 + C-Lex recovery → token-level mechanism;
  flat until A-* → fact-anchored suppression).

Judge on one GPU: always pass `JUDGE_N_GPUS=1` (upstream default
`gpus_per_judge=2` → `floor(1/2)=0` parallel judges on a single-GPU pod).

## Analysis caveats (encode in writeup)

- Tier index is ordinal (LLM-assigned, human-corrected), not metric distance —
  never fit curves assuming uniform spacing.
- s0–s10 arms use only the train half; all judge/NLL readouts stay on held-out
  splits. R6/R7 contaminate their own in-band retain-eval rows (flag).
- 3 seeds before believing small effects (~40-question fine-tunes are noisy);
  the pilot's 1 seed only sizes effects.
