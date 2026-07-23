# Pod Runbook: M0b smoke → M1 reproduction → M2 sweep

Step-by-step instructions for executing the proximity experiment on RunPod,
written to be followed from a fresh Claude Code session. Design background:
`docs/PROXIMITY_EXPERIMENT.md` (this repo) and
`tamper_resistant_unlearning/notes/suite_proximity_experiment.md`.

**Standing rules (do not skip):**
- Budget: $50/session hard cap, $25 soft cap — at $25, stop and justify
  further spend before continuing.
- This fork is **PUBLIC**. Commits must use the noreply identity (repo-local
  git config already set); the pod publishes via `runpod/publish_results.sh`
  which has a secret-scan gate — never bypass it, never `git add -f` ignored
  files, never commit notebooks/env files.
- Single-GPU pod ⇒ **always `JUDGE_N_GPUS=1`** on any script that evals
  (upstream default of 2 GPUs/judge yields zero judges on one GPU).
- State is tracked in this file's checklist boxes? No — update
  `notes/suite_proximity_experiment.md` (research repo) after each milestone.

## Pod provisioning (remote-kernels MCP)

- The remote-kernels server's sync/download root is the **tamper repo**, not
  this fork — do NOT use `sync()`. All code reaches the pod via git clone
  (public repo, no token needed to clone) and all results leave via git push,
  so the kernel is only needed to run shell commands.
- Preferred GPU: **1× RTX PRO 6000 Blackwell 96GB**; fallback A100/H100 80GB
  (judge then needs a small batch); escape hatch H200. Verify exact RunPod
  gpu-type-id strings at session start (the ids in `remote-kernels.toml` for
  the PRO 6000 are unverified guesses).
- Volume ≥ **250GB** (HF cache ~75GB incl. judge + 13 × 6.5GB checkpoints).
  Note: the RUNNING MCP server may load the tamper repo's toml (100GB volume,
  80GB-GPU list) — check which config it uses and override at `start()` or
  update that toml before provisioning.
- `HF_TOKEN` (gated meta-llama access) and `GITHUB_TOKEN` (push to this fork)
  must be inherited into the pod env.
- Cleanup hooks: point pre-stop/pre-terminate at
  `/workspace/forget-narrowly-retain-broadly/runpod/finalize_pod.sh` (this
  repo's toml already does; the tamper toml points at the tamper repo's).

## M0b — smoke test (~1 GPU-h, ~$2)

Goal: prove env + Hydra band override + relearn mechanics + NLL eval on the
pod before training anything real.

```bash
# 1. Bootstrap (repo is public; GITHUB_TOKEN only needed for later pushes)
export HF_TOKEN=... GITHUB_TOKEN=...
curl -fsSL https://raw.githubusercontent.com/jonas-raedler/forget-narrowly-retain-broadly/main/runpod/setup_pod.sh | bash
cd /workspace/forget-narrowly-retain-broadly && source .venv/bin/activate
export HF_HOME=/workspace/.cache/huggingface
```
Pass criteria: flash-attn + causal-conv1d install (prebuilt or source-built),
GPU visible, `verify_band_configs.py` prints 12/12. **Riskiest step is the
CUDA-extension install** — if wheels fail to build, fix here, not in M1.

```bash
# 2. Hydra dry-run: does the band override resolve?
HYDRA_FULL_ERROR=1 python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/suite/challenger_disaster/llama.yaml trainer=GradLearn \
  task_name=smoke_cfg \
  "data/datasets@data.retain=challenger_disaster/bands/R2" \
  +data.random_pairing=true --cfg job > /tmp/cfg.yaml
grep -n "challenger_band_R2\|random_pairing\|Semantic-0-" /tmp/cfg.yaml
```
Pass: config shows retain = `challenger_band_R2` (with the exclude list) and
`random_pairing: true`. Fallback if the CLI group override fails: generate
per-band experiment YAMLs (config-only; see PROXIMITY_EXPERIMENT.md §risks).

```bash
# 3. 4-step smoke relearn from the BASE model (mechanics only)
accelerate launch --config_file configs/accelerate/default_config.yaml --num_processes 1 \
  src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/suite/challenger_disaster/llama.yaml trainer=GradLearn \
  task_name=challenger_disaster/Llama-3.2-3B-Instruct/smoke/base/relearn/band_R2_smoke \
  model.model_args.pretrained_model_name_or_path=meta-llama/Llama-3.2-3B-Instruct \
  "data/datasets@data.retain=challenger_disaster/bands/R2" +data.random_pairing=true \
  trainer.args.max_steps=4 trainer.args.per_device_train_batch_size=8 \
  trainer.args.gradient_accumulation_steps=4 trainer.args.eval_strategy=no \
  trainer.args.learning_rate=1e-5 trainer.args.warmup_epochs=1 \
  trainer.args.logging_steps=1 trainer.method_args.alpha=1 trainer.method_args.gamma=0
```
Pass: log lines "dataset filtered to 50 rows" (band) and "disabling refusal
context in retain dataset" (train.py normalizer); 4 steps run; checkpoint
appears under `saves/unlearn/.../band_R2_smoke`. Also check the debug dump
`trainerargs.yaml` at the repo root for the actual `lr_scheduler_type`
(expected: linear) and record it in the notes.

```bash
# 4. NLL eval smoke (base model, one split)
python src/evals/nll_eval.py --model_path meta-llama/Llama-3.2-3B-Instruct \
  --task_name pretrained_baseline --topic challenger_disaster --splits forget_eval
```
Pass: `evaluations/nllOutputs/pretrained_baseline/NLL_summary.json` exists;
base-model forget NLL should be LOW (it knows the Challenger facts).

```bash
# 5. Clean up smoke artifacts (keep pretrained_baseline NLL)
rm -rf saves/unlearn/challenger_disaster/Llama-3.2-3B-Instruct/smoke
```
Budget check, then either continue to M1 in the same session or stop the pod.

## M1 — JensUn++ checkpoint + C-Full reproduction (~4–6 GPU-h, ~$10)

```bash
# a. Pretrained-baseline judged eval (reference row; needed by show_results)
#    NB: verify env-var names in the script header / docs/EXPERIMENTS.md first.
TOPIC=challenger_disaster MODELS="meta-llama/Llama-3.2-3B-Instruct" \
  TRAIN_GPUS=0 JUDGE_N_GPUS=1 bash scripts/suite_evaluation_optimized.sh

# b. Unlearn (paper defaults auto-filled: lr 3e-6, gamma .33, alpha 1, gnorm, 20 epochs)
METHOD=JensUnPP MODEL=llama_3b TOPIC=challenger_disaster \
  TRAIN_GPUS=0 JUDGE_N_GPUS=1 bash scripts/suite_unlearn.sh
# Checkpoint lands under: saves/unlearn/challenger_disaster/Llama-3.2-3B-Instruct/jensen/<exp>
# ("jensen" not "JensUnPP" — see scripts/trainer_method_map.txt)

# c. Gate: the unlearned model must look unlearned before attacking it.
#    Expect (Tab. 16 unlearned row, ±few pts): Q_D+I ≈ 5, Q_R ≈ 3, Q_All ≈ 8,
#    retain accuracy close to the pretrained baseline.
python scripts/results/collect_results.py && python scripts/results/show_results.py

# d. C-Full relearn reproduction (also computes the unlearned checkpoint's baseline NLL)
CKPT="./saves/unlearn/challenger_disaster/Llama-3.2-3B-Instruct/jensen/<exp>"
MODEL_PATH="$CKPT" BANDS="C_full" TRAIN_GPUS=0 JUDGE_N_GPUS=1 \
  bash scripts/proximity_relearn_sweep.sh

# e. Compare vs paper Tab. 16 (Llama, JensUn++, relearn): Q_D+I 5→10, Q_R 3→1,
#    Q_All 8→11. Tolerance: a few points (judge sampling + seed variance).
python scripts/results/collect_results.py && python scripts/results/show_results.py

# f. Publish artifacts
bash runpod/publish_results.sh
```
**Go/no-go:** if (c) or (e) is far off, STOP and debug — do not run M2 on a
checkpoint that doesn't reproduce. If continuing to M2 pushes past the $25
soft cap, stop the pod (volume persists; `attach()` resumes later).

## M2 — the 12-arm proximity sweep (~8–12 GPU-h, ~$20; own session recommended)

```bash
CKPT="./saves/unlearn/challenger_disaster/Llama-3.2-3B-Instruct/jensen/<exp>"
# NLL-only pass first (cheap, primary metric), judge deferred:
MODEL_PATH="$CKPT" SEEDS="0" TRAIN_GPUS=0 SKIP_JUDGE=1 \
  BANDS="R0 R1 R2 R3 R4 R5 R6 R7 C_gk C_lex A_forget_partial A_forget_full" \
  bash scripts/proximity_relearn_sweep.sh
bash runpod/publish_results.sh          # secure the NLL results immediately

# Then the batched judge over the same checkpoints (no retraining):
MODEL_PATH="$CKPT" SEEDS="0" TRAIN_GPUS=0 SKIP_TRAIN=1 SKIP_NLL=1 JUDGE_N_GPUS=1 \
  BANDS="R0 R1 R2 R3 R4 R5 R6 R7 C_gk C_lex A_forget_partial A_forget_full" \
  bash scripts/proximity_relearn_sweep.sh
bash runpod/publish_results.sh
```
Optional MMLU: append `,mmlu` to `TASKS`. Disk: 13 checkpoints ≈ 85GB — after
the judge pass, checkpoints under `saves/unlearn/.../relearn/` can be deleted.
Terminate the pod when done (finalize hook publishes again as backstop).

## M3 — analysis (local, no GPU)

```bash
git pull   # fetch pod-published artifacts
python scripts/results/plot_proximity.py        # ΔNLL curve + summary JSON
python scripts/results/collect_results.py && python scripts/results/show_results.py
```
Interpretation guide + caveats (ordinal tiers, R0 repetition confound, R6/R7
retain-eval contamination): `docs/PROXIMITY_EXPERIMENT.md` and the notes file.
1 seed only sizes effects — schedule the 3-seed run before drawing conclusions.
