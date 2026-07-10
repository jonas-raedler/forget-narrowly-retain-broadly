#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Topic config: Challenger Disaster — Baseline dataset
#
#  Training data  : forget_train (400 flat rows) + retain_train_semantic (375 flat rows)
#                   → both from apeleg/LKF-baseline-challenger-train
#  Forget eval    : forget_eval (25 columnar rows, claude keys)
#                   → apeleg/LKF-baseline-challenger-forget-eval
#  Retain eval    : retain_eval_semantic (125 columnar rows, gemini keys)
#                   → apeleg/LKF-baseline-challenger-retain-eval
#
#  The three apeleg/LKF-baseline-challenger-* datasets below are the
#  baseline splits (our adaptation for the LKF Challenger data);
#  they are used for the "train on SUITE data vs. LKF data" comparison.
#
#  Source this file from suite_unlearn.sh / suite_evaluation_optimized.sh via:
#      TOPIC=challenger_baseline bash scripts/suite_unlearn.sh
# ─────────────────────────────────────────────────────────────────────────────

topic="challenger_baseline"

# ── Forget eval ───────────────────────────────────────────────────────────────
# forget_eval split contains 25 direct questions with q_claude*/blank_claude* columns.
task_split["forget_rephrasings"]="forget_eval"
task_dataset["forget_rephrasings"]="dataset.name=apeleg/LKF-baseline-challenger-forget-eval"
task_max_tokens["forget_rephrasings"]="50"

# ── Retain eval ───────────────────────────────────────────────────────────────
# retain_eval_semantic: 125 semantic questions with q_gemini*/blank_gemini* columns.
task_split["retain"]="retain_eval_semantic"
task_dataset["retain"]="dataset.name=apeleg/LKF-baseline-challenger-retain-eval"
task_max_tokens["retain"]="50"

# ── Retain train rephrasing ───────────────────────────────────────────────────
# Uses retain_train_semantic split (375 flat rows) from the train HF repo.
task_split["retain_train_rephrasing"]="retain_train_semantic"
task_dataset["retain_train_rephrasing"]="dataset.name=apeleg/LKF-baseline-challenger-train"
task_max_tokens["retain_train_rephrasing"]="50"

task_split["forget_rephrasings_gibberish"]="${task_split[forget_rephrasings]}"
task_dataset["forget_rephrasings_gibberish"]="${task_dataset[forget_rephrasings]}"
task_max_tokens["forget_rephrasings_gibberish"]="${task_max_tokens[forget_rephrasings]}"