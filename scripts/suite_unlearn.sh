#!/bin/bash

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
suite_unlearn.sh — unlearn one topic, then auto-evaluate the checkpoint.

Usage:
  bash scripts/suite_unlearn.sh
  TOPIC=salem_witch_trials MODEL=llama_3b METHOD=GradDiff bash scripts/suite_unlearn.sh

Env vars (unset → default):
  TOPIC         forget topic                      (default challenger_disaster)
  MODEL         model key(s), comma-separated     (default llama_3b)
  METHOD        unlearning trainer                (default JensUnPP)
  EPOCHS        training epochs                   (default 20)
  RUNARGS       "<lr> [gamma] [alpha] [gnorm] [randpair] [overrides]"
  EXTRA         free-form exp_suffix tag          (default: _paper / empty on override)
  REFUSAL_KEY   refusal-string prefix (JensUnPP)  (default: per-model paper prefix)
  TRAIN_GPUS    GPU indices                       (default 0,1,2,3)
  SMOKE_TEST=1  fast end-to-end check (SMOKE_STEPS=4, SMOKE_TASKS=retain,forget_rephrasings)

Full reference (all choices + paper hyperparameters): docs/EXPERIMENTS.md
EOF
  exit 0
fi

# Shared infrastructure — env preamble, env-var parsing (RUNARGS/REFUSAL_KEY),
# helper functions (trainer_to_method, infer_hf_name, get_numactl_prefix,
# _topic_to_ds_keys), refusal helpers, _model_spec, _parse_runargs,
# _build_train_suffixes, _dispatch_from_MODEL.
source "$(dirname "${BASH_SOURCE[0]}")/_suite_common.sh"

# ─────────────────────────────────────────────────────────────────────────────
#  TOPIC  (sets the top-level save folder)
# ─────────────────────────────────────────────────────────────────────────────
# ← Change this one line to switch topics, or override without editing:
#       TOPIC=salem_witch_trials bash scripts/suite_unlearn.sh
_TOPIC="${TOPIC:-challenger_disaster}"
declare -A task_split task_dataset task_max_tokens
source "configs/topics/${_TOPIC}.sh"

# challenger_baseline (training on the LKF data) uses 10 epochs, not the 20-epoch
# SUITE default. Only change the default when EPOCHS is unset (an explicit override wins).
if [[ "$_TOPIC" == "challenger_baseline" && -z "${EPOCHS:-}" ]]; then
    _EPOCHS=10
fi

# Derive dataset config YAML keys from topic name (mirrors the top-level key in
# configs/data/datasets/<topic>/forget.yaml and retain.yaml).
read -r ds_forget_key ds_retain_key <<< "$(_topic_to_ds_keys "$topic")"

# RUNARGS (override the per-model `runningargs` array from the terminal):
#   MODEL=llama_3b RUNARGS="3e-6 0.33 1 true" EXTRA=_myexp bash scripts/suite_unlearn.sh
# is parsed into _RUNARGS_OVERRIDE in _suite_common.sh (';'-separate multiple entries).
# The epoch count comes from EPOCHS (default 20); the exp_suffix tag from EXTRA.

# ─────────────────────────────────────────────────────────────────────────────
#  SMOKE_TEST  (fast end-to-end dry run to check everything is wired up)
# ─────────────────────────────────────────────────────────────────────────────
# Set SMOKE_TEST=1 to train ~1 epoch (a few optimizer steps) and run a minimal
# eval, exercising the whole train→evaluate→metrics path quickly. Smoke runs are
# saved under a separate `_smoke` experiment suffix so they never collide with
# real runs.
#   SMOKE_TEST=1 bash scripts/suite_unlearn.sh
# Optional knobs: SMOKE_STEPS (optimizer steps, default 4),
#                 SMOKE_TASKS (comma-separated eval tasks, default retain,forget_rephrasings).
_SMOKE="${SMOKE_TEST:-}"
_SMOKE_STEPS="${SMOKE_STEPS:-4}"
_SMOKE_TASKS="${SMOKE_TASKS:-retain,forget_rephrasings}"

# REFUSAL PREFIX: REFUSAL_KEY selects a refusal prefix for JensUnPP runs —
#   unset → per-model paper default (unfor_comma); <key> → override;
#   none/off → drop the prefix. Parsed into _REFUSAL_KEY and resolved by
#   _resolve_refusal_args in _suite_common.sh.

# Global utility task configs (not per-topic)
task_split["mmlu"]="train";  task_dataset["mmlu"]="dataset.name=jinzhuoran/RWKU";  task_max_tokens["mmlu"]="50"
task_split["repet"]="train"; task_dataset["repet"]="dataset.name=jinzhuoran/RWKU"; task_max_tokens["repet"]="1000"

# ===========================================================================
# Per-model configuration blocks
# To run a specific model, set MODEL=<key> (see the dispatch block at the bottom).
# Method is set by METHOD (default JensUnPP); each runningargs entry holds the
# hyperparameters:
#   "<lr> [gamma] [alpha] [gnorm] [randpair] [overrides]"
# The epoch count comes from EPOCHS (default 20); the exp_suffix tag from EXTRA.
# ===========================================================================

###############################################
#  JUDGE CONFIGURATION (Optional overrides)
#  Leave empty to use eval.yaml defaults (Qwen3.5-35B-A3B, tag=qwen35b, 2 GPUs/judge, auto parallel)
#  To use the larger 80B single judge instead:
#    JUDGE_MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct" JUDGE_TAG="" JUDGE_BATCH_SIZE=16 JUDGE_N_GPUS=6
###############################################
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_TAG="${JUDGE_TAG:-}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-}"
JUDGE_N_GPUS="${JUDGE_N_GPUS:-}"

judge_overrides=""
[[ -n "$JUDGE_MODEL"      ]] && judge_overrides="$judge_overrides judge.hf_model_id=\"$JUDGE_MODEL\""
[[ -n "$JUDGE_TAG"        ]] && judge_overrides="$judge_overrides judge.judge_tag=\"$JUDGE_TAG\""
[[ -n "$JUDGE_BATCH_SIZE" ]] && judge_overrides="$judge_overrides judge.batch_size=$JUDGE_BATCH_SIZE"
[[ -n "$JUDGE_N_GPUS"     ]] && judge_overrides="$judge_overrides judge.gpus_per_judge=$JUDGE_N_GPUS"

echo "========================================"
echo "  TOPIC    : $topic"
echo "  Rephrase : ${task_dataset[forget_rephrasings]#dataset.name=}"
echo "========================================"
echo ""

if [[ "$_TOPIC" == "challenger_baseline" ]]; then
    evaltasks=(
        retain
        forget_rephrasings
        mmlu                       # evaluate General ability on MMLU
        repet                      # evaluate Repetitiveness on AlpacaEval
        rgq_bi                     # evaluate Relative Generation Quality (bidirectional) c.f. base-model
    )
else
    evaltasks=(
        retain
        forget_rephrasings
        forget_rephrasings_gibberish # gibberish overlay: how often model answers forget rephrasings with gibberish
        mmlu                         # evaluate General ability on MMLU
        repet                        # evaluate Repetitiveness on AlpacaEval
        rgq_bi                       # evaluate Relative Generation Quality (bidirectional) c.f. base-model
    )
fi

# SMOKE_TEST: restrict to a tiny task list so the eval phase finishes quickly.
if [[ -n "$_SMOKE" ]]; then
    IFS=',' read -ra evaltasks <<< "$_SMOKE_TASKS"
    echo "[SMOKE] training shortened; eval restricted to: ${evaltasks[*]}"
fi

# ---------------------------------------------------------------------------
# Llama-3.2-3B-Instruct
# ---------------------------------------------------------------------------
run_llama_3b() {
    local model_cfg model_org model_yaml per_device_train_batch_size \
          gradient_accumulation_steps default_gamma default_alpha warmup_epochs accelerate_config
    read -r model_cfg model_org model_yaml per_device_train_batch_size \
            gradient_accumulation_steps default_gamma default_alpha warmup_epochs accelerate_config \
            <<< "$(_model_spec llama_3b)"

    # runningargs defaults to the paper hyperparameters for (this model, METHOD) —
    # see _paper_hparams in _suite_common.sh. Select the method with METHOD (default
    # JensUnPP). Override the hyperparameters with RUNARGS (format:
    # "<lr> [gamma] [alpha] [gnorm] [randpair] [overrides]"), e.g.
    #   MODEL=llama_3b METHOD=GradDiff RUNARGS="5e-7 1 2" EXTRA=_myexp bash scripts/suite_unlearn.sh
    # epochs come from EPOCHS (default 20); gnorm "true" adds _gnorm; randpair "true" adds
    # _randpair; the EXTRA env var adds a free-form suffix tag; overrides (field 6+) are
    # appended verbatim (e.g. "data.retain.${ds_retain_key}.args.exclude_label_prefixes=[Syntax-]").
    local -a runningargs
    _resolve_runargs llama_3b

    _run_model "$model_cfg" "$model_org" "$model_yaml" \
        "$per_device_train_batch_size" "$gradient_accumulation_steps" \
        "$default_gamma" "$default_alpha" "$warmup_epochs" \
        "$accelerate_config" \
        "${runningargs[@]}"
}

# ---------------------------------------------------------------------------
# Ministral-3-3B-Instruct-2512-BF16
# ---------------------------------------------------------------------------
run_ministral_3b() {
    local model_cfg model_org model_yaml per_device_train_batch_size \
          gradient_accumulation_steps default_gamma default_alpha warmup_epochs accelerate_config
    read -r model_cfg model_org model_yaml per_device_train_batch_size \
            gradient_accumulation_steps default_gamma default_alpha warmup_epochs accelerate_config \
            <<< "$(_model_spec ministral_3b)"

    local -a runningargs
    _resolve_runargs ministral_3b

    _run_model "$model_cfg" "$model_org" "$model_yaml" \
        "$per_device_train_batch_size" "$gradient_accumulation_steps" \
        "$default_gamma" "$default_alpha" "$warmup_epochs" \
        "$accelerate_config" \
        "${runningargs[@]}"
}

# ---------------------------------------------------------------------------
# Qwen3.5-9B
# ---------------------------------------------------------------------------
run_qwen_9b() {
    local model_cfg model_org model_yaml per_device_train_batch_size \
          gradient_accumulation_steps default_gamma default_alpha warmup_epochs accelerate_config
    read -r model_cfg model_org model_yaml per_device_train_batch_size \
            gradient_accumulation_steps default_gamma default_alpha warmup_epochs accelerate_config \
            <<< "$(_model_spec qwen_9b)"

    local -a runningargs
    _resolve_runargs qwen_9b

    _run_model "$model_cfg" "$model_org" "$model_yaml" \
        "$per_device_train_batch_size" "$gradient_accumulation_steps" \
        "$default_gamma" "$default_alpha" "$warmup_epochs" \
        "$accelerate_config" \
        "${runningargs[@]}"
}

# ---------------------------------------------------------------------------
# Internal helper — do not edit below this line
# ---------------------------------------------------------------------------
_run_model() {
    local model_cfg="$1"
    local model_org="$2"
    local model_yaml="$3"          # e.g. "llama"; experiment paths derived below
    local per_device_train_batch_size="$4"
    local gradient_accumulation_steps="$5"
    local default_gamma="$6"
    local default_alpha="$7"
    local warmup_epochs="$8"
    local accelerate_config="${9:-configs/accelerate/default_config.yaml}"
    shift 9
    local runningargs=("$@")
    local model_path="${model_org}/${model_cfg}"
    # Trainer comes from METHOD (default JensUnPP), not from runningargs.
    local trainer="$_METHOD"
    # JensUn family uses refusal-context datasets; all other methods use plain base datasets.
    local experiment="unlearn/suite/${topic}/${model_yaml}.yaml"
    local experiment_base="unlearn/suite/${topic}/${model_yaml}_base.yaml"

    for paramss in "${runningargs[@]}"; do

        # Parse one runningargs entry → epoch/lr/run_gamma/run_alpha/
        # run_gnorm/run_extra/run_randpair/run_overrides (caller-scoped; defaults + validation).
        local epoch lr run_gamma run_alpha run_gnorm run_extra run_randpair run_overrides
        _parse_runargs "$paramss" "$default_gamma" "$default_alpha"
        [[ -n "$_SMOKE" ]] && epoch=1   # SMOKE_TEST: 1 epoch (step-capped below)

        # JensUnPP uses refusal-context datasets; all other methods use plain base datasets.
        local _active_experiment
        case "$trainer" in
            JensUnPP) _active_experiment="$experiment" ;;
            *) _active_experiment="$experiment_base" ;;
        esac

        # JensUnPP family: push_prefix_to_refusal_start defaults to true (canonical setup).
        # Auto-inject unless the user already specified the flag in the overrides field,
        # so an explicit "...push_prefix_to_refusal_start=false" still wins.
        local push_prefix_override=""
        case "$trainer" in
            JensUnPP)
                if [[ "$run_overrides" != *push_prefix_to_refusal_start* ]]; then
                    push_prefix_override="data.forget.${ds_forget_key}.args.push_prefix_to_refusal_start=true"
                fi
                ;;
        esac

        # Build grad-norm / random-pairing suffixes + Hydra overrides from the parsed flags.
        local gnorm_suffix gnorm_override randpair_suffix randpair_override
        _build_train_suffixes

        # Resolve the refusal-prefix override (per-model paper default unless
        # REFUSAL_KEY overrides it; "none"/"off" drops it; JensUnPP-only).
        # The bash array passes the value as a single argument to accelerate,
        # preventing word-splitting on spaces.
        _resolve_refusal_args "$trainer" "$model_cfg" "$_REFUSAL_KEY"
        local refusal_suffix="$_REFUSAL_SUFFIX"
        local -a _refusal_args=("${_REFUSAL_ARGS[@]}")

        echo "Model: $model_cfg  |  Epoch: $epoch  |  LR: $lr  |  gamma: $run_gamma  |  alpha: $run_alpha${run_gnorm:+  |  grad_norm: $run_gnorm}${run_randpair:+  |  rand_pair: $run_randpair}${run_extra:+  |  extra: $run_extra}${refusal_suffix:+  |  refusal: $_REFUSAL_RESOLVED_KEY}"

        # SMOKE_TEST: cap optimizer steps and tag the run so it never collides with a real one.
        local smoke_suffix=""
        local smoke_override=""
        if [[ -n "$_SMOKE" ]]; then
            smoke_suffix="_smoke"
            smoke_override="trainer.args.max_steps=${_SMOKE_STEPS}"
        fi

        local method; method=$(trainer_to_method "$trainer")
        local exp_suffix=epochs_${epoch}_lrs_${lr}_gamma${run_gamma}_alpha${run_alpha}${gnorm_suffix}${randpair_suffix}${run_extra}${refusal_suffix}${smoke_suffix}
        local task_name=${topic}/${model_cfg}/${method}/${exp_suffix}

        echo "${task_name}: Unlearning ${model_path} using ${trainer}"

        # TRAIN_GPUS env var overrides the default set of physical GPU indices.
        # Example: TRAIN_GPUS=4,5,6,7 bash scripts/suite_unlearn.sh
        local _train_gpus="${TRAIN_GPUS:-0,1,2,3}"
        local _n_train_gpus=$(echo "$_train_gpus" | tr ',' '\n' | wc -l)

        # Auto-detect NUMA affinity and prefix the launch with numactl when possible.
        local _numactl; _numactl=$(get_numactl_prefix "$_train_gpus")
        if [[ -n "$_numactl" ]]; then
            echo "NUMA binding → $_numactl  (GPUs: $_train_gpus)"
        else
            echo "NUMA binding → disabled (numactl unavailable or no NUMA topology detected)"
        fi

        # ── Skip training if checkpoint already exists ────────────────────────
        # Check for config.json rather than just the directory — the directory can
        # be created early (or left partial after a failed run) without a valid model.
        local _save_path_check="./saves/unlearn/${task_name}"
        if [[ -f "${_save_path_check}/config.json" ]]; then
            echo "[INFO] Checkpoint already exists at ${_save_path_check} — skipping training."
        else

        CUDA_VISIBLE_DEVICES=${_train_gpus} ${_numactl} accelerate launch --config_file ${accelerate_config} --num_processes ${_n_train_gpus} --main_process_port $MASTER_PORT \
        src/train.py --config-name=unlearn.yaml \
        experiment=${_active_experiment} \
        trainer=${trainer} \
        task_name=${task_name} \
        model=${model_cfg} \
        model.model_args.pretrained_model_name_or_path=${model_path} \
        retain_logs_path=null \
        trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
        trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.num_train_epochs=$epoch \
        ${smoke_override} \
        trainer.args.logging_steps=10 \
        trainer.args.eval_strategy=no \
        trainer.args.learning_rate=${lr} \
        trainer.args.warmup_epochs=${warmup_epochs} \
        trainer.method_args.alpha=${run_alpha} \
        trainer.method_args.gamma=${run_gamma} \
        ${gnorm_override} \
        ${randpair_override} \
        "${_refusal_args[@]}" \
        ${push_prefix_override} \
        ${run_overrides}

        fi  # end skip-if-exists check

        # TensorBoard — launch after training to inspect the run.
        # Open http://localhost:6006 in your browser.
        # Logs are written to saves/unlearn/{task_name}/logs/
        #
        # To start:
        #   tensorboard --logdir saves/unlearn/${task_name}/logs --port 6006
        #
        # To compare multiple runs side by side:
        #   tensorboard --logdir saves/unlearn --port 6006

        # ── Eval ──────────────────────────────────────────────────────────────
        local model_save_path="./saves/unlearn/${task_name}"
        local hf_name; hf_name="$(infer_hf_name "$model_cfg")"

        local all_tasks_list=()
        for task in "${evaltasks[@]}"; do
            local task_spec
            if [[ "$task" == "rgq_bi" ]]; then
                # rgq_bi has no topic-config split/dataset/max_tokens entries; use fixed placeholders
                task_spec="${task_name}:${model_save_path}:${task}:${task_name}:train:dataset.name=none:50:${hf_name}"
                echo "  ✓ [${task_name}] ${task} → [judge-only, after repet]"
            else
                local split="${task_split[$task]}"
                local dataset_arg="${task_dataset[$task]}"
                local max_tokens="${task_max_tokens[$task]}"
                task_spec="${task_name}:${model_save_path}:${task}:${task_name}:${split}:${dataset_arg}:${max_tokens}:${hf_name}"
                echo "  ✓ [${task_name}] $task → split=$split"
            fi
            all_tasks_list+=("$task_spec")
        done

        local task_specs_json=""
        for i in "${!all_tasks_list[@]}"; do
            local task_spec="${all_tasks_list[$i]}"
            if [ $i -eq 0 ]; then
                task_specs_json="\"${task_spec}\""
            else
                task_specs_json="${task_specs_json},\"${task_spec}\""
            fi
        done

        echo ""
        echo "========================================"
        echo "STARTING EVALUATION for ${task_name}"
        echo "========================================"

        CUDA_VISIBLE_DEVICES=${_train_gpus} ${_numactl} python src/eval_full_pipeline.py \
            experiment=eval.yaml \
            output.task_name="${task_name}_eval" \
            output.topic="${topic}" \
            "multi_task_specs=[${task_specs_json}]" \
            multi_task_mode=true \
            ${judge_overrides}

        if [ $? -eq 0 ]; then
            echo "✓ Evaluation completed for ${task_name}"
        else
            echo "✗ Evaluation failed for ${task_name}"
        fi

    done
}

# ===========================================================================
# Dispatch — select model(s) to run.
#   • From the terminal:  MODEL=llama_3b bash scripts/suite_unlearn.sh
#     (comma-separate to run several: MODEL=llama_3b,qwen_9b)
#     Valid: llama_3b | ministral_3b | qwen_9b
#   • Pick the method with METHOD (default JensUnPP), e.g. MODEL=llama_3b METHOD=GradDiff …
#   • Or leave MODEL unset and edit the fallback block below (comment/uncomment).
# ===========================================================================
if [[ -n "$MODEL" ]]; then
    _dispatch_from_MODEL
else
    run_llama_3b
fi

