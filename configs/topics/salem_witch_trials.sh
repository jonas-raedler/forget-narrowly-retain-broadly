#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Topic config: Salem Witch Trials
#
#  Source this file from suite_evaluation_optimized.sh / suite_unlearn.sh / suite_relearn.sh:
#      source configs/topics/salem_witch_trials.sh
#  Or override via env var (no file edit needed):
#      TOPIC=salem_witch_trials bash scripts/suite_evaluation_optimized.sh
# ─────────────────────────────────────────────────────────────────────────────

topic="salem_witch_trials"

# ── Standard eval tasks ───────────────────────────────────────────────────────
task_split["retain"]="retain_eval"
task_dataset["retain"]="dataset.name=apeleg/SUITE"
task_max_tokens["retain"]="50"

task_split["forget_rephrasings"]="forget_eval_rephrasings"
task_dataset["forget_rephrasings"]="dataset.name=apeleg/SUITE-rephrasings"
task_max_tokens["forget_rephrasings"]="50"

task_split["retain_train_rephrasing"]="retain_train"
task_dataset["retain_train_rephrasing"]="dataset.name=apeleg/SUITE"
task_max_tokens["retain_train_rephrasing"]="50"

# Gibberish overlay tasks: reuse generation from forget_rephrasings / retain
task_split["retain_gibberish"]="${task_split[retain]}"
task_dataset["retain_gibberish"]="${task_dataset[retain]}"
task_max_tokens["retain_gibberish"]="${task_max_tokens[retain]}"

task_split["forget_rephrasings_gibberish"]="${task_split[forget_rephrasings]}"
task_dataset["forget_rephrasings_gibberish"]="${task_dataset[forget_rephrasings]}"
task_max_tokens["forget_rephrasings_gibberish"]="${task_max_tokens[forget_rephrasings]}"

task_split["forget_adversarial"]="train"
task_dataset["forget_adversarial"]="dataset.name=./dataset/adversarial_questions/salem_witch_trials_adv.json"
task_max_tokens["forget_adversarial"]="50"
