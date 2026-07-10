#!/bin/bash

# ─────────────────────────────────────────────────────────────────────────────
#  SEQUENTIAL MULTI-TOPIC UNLEARNING
#
#  Unlearns topics one after another on the same model. Each step starts from
#  the previous step's checkpoint. Intermediate checkpoints and evaluations are
#  saved after every step.
#
#  Save path structure (topics in the path follow training order; steps 2+ get a seq_ prefix):
#    Step 1 (topic A):     saves/unlearn/A/model/method/exp/
#    Step 2 (A→B):         saves/unlearn/seq_A+B/model/method/exp/
#    Step 3 (A→B→C):       saves/unlearn/seq_A+B+C/model/method/exp/
#
#  Per-step evaluation (controlled by OPT_EVAL, default=true):
#    After step k, evaluates the checkpoint on ALL topics seen so far (1..k).
#
#  Usage:
#    bash scripts/suite_sequential_unlearn.sh
#    OPT_EVAL=false bash scripts/suite_sequential_unlearn.sh   # skip eval steps
#    TRAIN_GPUS=4,5,6,7 bash scripts/suite_sequential_unlearn.sh
# ─────────────────────────────────────────────────────────────────────────────

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
suite_sequential_unlearn.sh — unlearn a chain of topics one after another
(each step resumes from the previous step's checkpoint).

Usage:
  bash scripts/suite_sequential_unlearn.sh
  SEQ_TOPICS="challenger_disaster,salem_witch_trials" bash scripts/suite_sequential_unlearn.sh

Env vars (unset → default):
  SEQ_TOPICS   topic chain, comma-separated, order = training order (default: all topics)
  MODEL        model key(s), comma-separated     (default llama_3b)
  METHOD       unlearning trainer                (default JensUnPP)
  EPOCHS       training epochs                   (default 20)
  RUNARGS      "<lr> [gamma] [alpha] [gnorm] [randpair] [overrides]"
  EXTRA        free-form exp_suffix tag          (default: _paper / empty on override)
  OPT_EVAL     auto-evaluate after each step: true (default) | false
  REFUSAL_KEY  refusal-string prefix (JensUnPP)  (default: per-model paper prefix)
  TRAIN_GPUS   GPU indices                       (default 0,1,2,3)

Full reference (all choices + paper hyperparameters): docs/EXPERIMENTS.md
EOF
  exit 0
fi

# Shared infrastructure — env preamble, env-var parsing (RUNARGS/REFUSAL_KEY),
# helpers (trainer_to_method, infer_hf_name, get_numactl_prefix, _topic_to_ds_keys),
# refusal helpers, _model_spec, _parse_runargs, _build_train_suffixes, _dispatch_from_MODEL.
source "$(dirname "${BASH_SOURCE[0]}")/_suite_common.sh"

# ─────────────────────────────────────────────────────────────────────────────
#  TOPICS  (defines the unlearning order — order matters!)
#  The save path joins topics in training order (not sorted), so the path
#  reflects the actual chain (e.g. seq_A+B differs from seq_B+A).
# ─────────────────────────────────────────────────────────────────────────────
_SEQ_TOPICS=(
    "challenger_disaster"
    "salem_witch_trials"
    "steve_jobs_medical"
    "britney_spears_conservatorship"
)
# Override the chain from the terminal (comma-separated; order = training order):
#   SEQ_TOPICS="challenger_disaster,salem_witch_trials" bash scripts/suite_sequential_unlearn.sh
[[ -n "$SEQ_TOPICS" ]] && IFS=',' read -ra _SEQ_TOPICS <<< "$SEQ_TOPICS"

# RUNARGS (';'-separated runningargs entries) is parsed into _RUNARGS_OVERRIDE in
# _suite_common.sh; _topic_to_ds_keys also lives there (shared with the other scripts).

# Set ds_forget_key / ds_retain_key from the first topic so runningargs entries
# copied from suite_unlearn.sh that reference ${ds_forget_key} / ${ds_retain_key}
# expand to a valid key at array-definition time. The inner loop then adapts
# these to the correct key for each step automatically.
read -r ds_forget_key ds_retain_key <<< "$(_topic_to_ds_keys "${_SEQ_TOPICS[0]}")"

echo "Sequential topics (training order): ${_SEQ_TOPICS[*]}"
echo ""

# REFUSAL PREFIX: REFUSAL_KEY → _REFUSAL_KEY (parsed in _suite_common.sh);
#   unset → per-model paper default (unfor_comma); <key> → override;
#   none/off → drop the prefix. Resolved by _resolve_refusal_args (JensUnPP-only).

# ─────────────────────────────────────────────────────────────────────────────
#  JUDGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_TAG="${JUDGE_TAG:-}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-}"
JUDGE_N_GPUS="${JUDGE_N_GPUS:-}"

judge_overrides=""
[[ -n "$JUDGE_MODEL"      ]] && judge_overrides="$judge_overrides judge.hf_model_id=\"$JUDGE_MODEL\""
[[ -n "$JUDGE_TAG"        ]] && judge_overrides="$judge_overrides judge.judge_tag=\"$JUDGE_TAG\""
[[ -n "$JUDGE_BATCH_SIZE" ]] && judge_overrides="$judge_overrides judge.batch_size=$JUDGE_BATCH_SIZE"
[[ -n "$JUDGE_N_GPUS"     ]] && judge_overrides="$judge_overrides judge.gpus_per_judge=$JUDGE_N_GPUS"

# ─────────────────────────────────────────────────────────────────────────────
#  EVAL TASKS (applied per accumulated topic after each sequential step)
# ─────────────────────────────────────────────────────────────────────────────
evaltasks=(
    retain
    forget_rephrasings
    forget_rephrasings_gibberish
    mmlu
    repet
    rgq_bi
)

# ===========================================================================
# Per-model configuration blocks
#
# Each runningargs entry applies THE SAME hyperparams to EVERY sequential step.
# One entry = one full sequential chain (topic_0 → topic_1 → … → topic_N).
#
# Method is set by METHOD (default JensUnPP); each runningargs entry holds the
# hyperparameters:
#   "<lr> [gamma] [alpha] [gnorm] [randpair] [overrides]"
# The epoch count comes from EPOCHS (default 20); the exp_suffix tag from EXTRA.
# ===========================================================================

# ---------------------------------------------------------------------------
# Llama-3.2-3B-Instruct
# ---------------------------------------------------------------------------
run_llama_3b() {
    local model_cfg model_org model_yaml per_device_train_batch_size \
          gradient_accumulation_steps default_gamma default_alpha warmup_epochs accelerate_config
    read -r model_cfg model_org model_yaml per_device_train_batch_size \
            gradient_accumulation_steps default_gamma default_alpha warmup_epochs accelerate_config \
            <<< "$(_model_spec llama_3b)"

    # Default = paper hyperparameters for (this model, METHOD); override with RUNARGS
    # ("<lr> [gamma] [alpha] [gnorm] [randpair] [overrides]"; EPOCHS sets epochs, EXTRA tags exp_suffix).
    local -a runningargs
    _resolve_runargs llama_3b

    _run_seq_model "$model_cfg" "$model_org" "$model_yaml" \
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

    _run_seq_model "$model_cfg" "$model_org" "$model_yaml" \
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

    _run_seq_model "$model_cfg" "$model_org" "$model_yaml" \
        "$per_device_train_batch_size" "$gradient_accumulation_steps" \
        "$default_gamma" "$default_alpha" "$warmup_epochs" \
        "$accelerate_config" \
        "${runningargs[@]}"
}

# ---------------------------------------------------------------------------
# Internal helper — sequential training loop
# ---------------------------------------------------------------------------
_run_seq_model() {
    local model_cfg="$1"
    local model_org="$2"
    local model_yaml="$3"          # e.g. "llama", "ministral3b"
    local per_device_train_batch_size="$4"
    local gradient_accumulation_steps="$5"
    local default_gamma="$6"
    local default_alpha="$7"
    local warmup_epochs="$8"
    local accelerate_config="${9:-configs/accelerate/default_config.yaml}"
    shift 9
    local runningargs=("$@")
    local hf_name; hf_name="$(infer_hf_name "$model_cfg")"
    # Trainer comes from METHOD (default JensUnPP), not from runningargs.
    local trainer="$_METHOD"

    # Each runningargs entry is a separate sequential chain (same hyperparams, all topics).
    for paramss in "${runningargs[@]}"; do
        # Parse one runningargs entry (caller-scoped; defaults + validation).
        local epoch lr run_gamma run_alpha run_gnorm run_extra run_randpair run_overrides
        _parse_runargs "$paramss" "$default_gamma" "$default_alpha"

        # Build grad-norm / random-pairing suffixes + Hydra overrides from the parsed flags.
        local gnorm_suffix gnorm_override randpair_suffix randpair_override
        _build_train_suffixes

        # Per-model paper default unless REFUSAL_KEY overrides; "none"/"off" drops it; JensUnPP-only.
        _resolve_refusal_args "$trainer" "$model_cfg" "$_REFUSAL_KEY"
        local refusal_suffix="$_REFUSAL_SUFFIX"
        local -a _refusal_args=("${_REFUSAL_ARGS[@]}")

        local method; method=$(trainer_to_method "$trainer")
        local exp_suffix="epochs_${epoch}_lrs_${lr}_gamma${run_gamma}_alpha${run_alpha}${gnorm_suffix}${randpair_suffix}${run_extra}${refusal_suffix}"

        local _train_gpus="${TRAIN_GPUS:-0,1,2,3}"
        local _n_train_gpus; _n_train_gpus=$(echo "$_train_gpus" | tr ',' '\n' | wc -l)
        local _numactl; _numactl=$(get_numactl_prefix "$_train_gpus")

        # ── Sequential chain: topic_0 → topic_1 → … → topic_N ────────────────
        local prev_ckpt="${model_org}/${model_cfg}"  # start from pretrained
        local -a _seen_topics=()

        # Save the dataset keys that were baked into run_overrides when the
        # runningargs array was evaluated (= first topic's keys). Per-step
        # substitution below replaces these with the current step's keys.
        # Always derive from _SEQ_TOPICS[0] — the local ds_forget_key may be
        # stale (set by a previous paramss iteration's topic loop).
        local _baked_forget_key _baked_retain_key
        read -r _baked_forget_key _baked_retain_key <<< "$(_topic_to_ds_keys "${_SEQ_TOPICS[0]}")"

        # JensUnPP family: push_prefix_to_refusal_start defaults to true (canonical setup).
        # Inject into run_overrides using the baked (first-topic) forget key so the per-step
        # adaptation below rewrites it to each step's key. Skip when the user already mentions
        # the flag, so an explicit "...=false" still wins.
        case "$trainer" in
            JensUnPP)
                if [[ "$run_overrides" != *push_prefix_to_refusal_start* ]]; then
                    run_overrides="${run_overrides:+$run_overrides }data.forget.${_baked_forget_key}.args.push_prefix_to_refusal_start=true"
                fi
                ;;
        esac

        for topic_k in "${_SEQ_TOPICS[@]}"; do
            _seen_topics+=("$topic_k")
            # Join in training order (not sorted) so the path reflects the actual chain.
            local accumulated_topic; accumulated_topic=$(printf '%s+' "${_seen_topics[@]}"); accumulated_topic="${accumulated_topic%+}"

            # JensUnPP uses refusal-context datasets; other methods use base datasets.
            local _yaml_variant
            case "$trainer" in
                JensUnPP) _yaml_variant="${model_yaml}" ;;
                *) _yaml_variant="${model_yaml}_base" ;;
            esac
            local _active_experiment="unlearn/suite/${topic_k}/${_yaml_variant}.yaml"

            # Load dataset key names for this step's topic
            local ds_forget_key ds_retain_key
            read -r ds_forget_key ds_retain_key <<< "$(_topic_to_ds_keys "$topic_k")"

            # Adapt any per-dataset Hydra overrides in run_overrides: replace the
            # baked (first-topic) key names with this step's key names so that e.g.
            # data.forget.challenger_forget.args.push_prefix_to_refusal_start=true
            # becomes data.forget.salem_forget.args.push_prefix_to_refusal_start=true
            # for the salem step. No-op when keys are the same (step 1).
            local _run_overrides_adapted="${run_overrides}"
            if [[ -n "$_baked_forget_key" && "$_baked_forget_key" != "$ds_forget_key" ]]; then
                _run_overrides_adapted="${_run_overrides_adapted//${_baked_forget_key}/${ds_forget_key}}"
            fi
            if [[ -n "$_baked_retain_key" && "$_baked_retain_key" != "$ds_retain_key" ]]; then
                _run_overrides_adapted="${_run_overrides_adapted//${_baked_retain_key}/${ds_retain_key}}"
            fi

            # Source this step's topic config (sets task_split/task_dataset/task_max_tokens)
            declare -A task_split task_dataset task_max_tokens
            source "configs/topics/${topic_k}.sh"
            task_split["mmlu"]="train";  task_dataset["mmlu"]="dataset.name=jinzhuoran/RWKU";  task_max_tokens["mmlu"]="50"
            task_split["repet"]="train"; task_dataset["repet"]="dataset.name=jinzhuoran/RWKU"; task_max_tokens["repet"]="1000"

            # Step 1 (single topic): no path prefix — identical to a regular suite_unlearn.sh run
            # so an existing checkpoint can be reused.
            # Steps 2+: use seq_ prefix to distinguish from a combined run at the same compound path.
            local _path_topic
            if [[ ${#_seen_topics[@]} -eq 1 ]]; then
                _path_topic="${accumulated_topic}"
            else
                _path_topic="seq_${accumulated_topic}"
            fi

            local task_name="${_path_topic}/${model_cfg}/${method}/${exp_suffix}"
            local model_save_path="./saves/unlearn/${task_name}"

            echo ""
            echo "========================================"
            echo "SEQUENTIAL STEP: training on topic=${topic_k}"
            echo "  Accumulated topics : ${_path_topic}"
            echo "  Starting from      : ${prev_ckpt}"
            echo "  Save path          : ${model_save_path}"
            echo "  Trainer            : ${trainer}"
            echo "========================================"

            if [[ -n "$_numactl" ]]; then
                echo "NUMA binding → $_numactl  (GPUs: $_train_gpus)"
            fi

            # ── Skip training if checkpoint already exists (reuse step 1 / resume run) ──
            # Check for config.json rather than just the directory — the directory can be
            # created early (or left partial after a failed run) without a valid model.
            local _ckpt_existed=false
            if [[ -f "${model_save_path}/config.json" ]]; then
                echo "[INFO] Checkpoint already exists at ${model_save_path} — skipping training, reusing as prev_ckpt."
                _ckpt_existed=true
            else
                CUDA_VISIBLE_DEVICES=${_train_gpus} ${_numactl} accelerate launch \
                    --config_file ${accelerate_config} \
                    --num_processes ${_n_train_gpus} \
                    --main_process_port $MASTER_PORT \
                    src/train.py --config-name=unlearn.yaml \
                    experiment=${_active_experiment} \
                    trainer=${trainer} \
                    task_name=${task_name} \
                    model=${model_cfg} \
                    model.model_args.pretrained_model_name_or_path=${prev_ckpt} \
                    retain_logs_path=null \
                    trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
                    trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
                    trainer.args.ddp_find_unused_parameters=true \
                    trainer.args.num_train_epochs=$epoch \
                    trainer.args.logging_steps=10 \
                    trainer.args.eval_strategy=no \
                    trainer.args.learning_rate=${lr} \
                    trainer.args.warmup_epochs=${warmup_epochs} \
                    trainer.method_args.alpha=${run_alpha} \
                    trainer.method_args.gamma=${run_gamma} \
                    ${gnorm_override} \
                    ${randpair_override} \
                    "${_refusal_args[@]}" \
                    ${_run_overrides_adapted}
            fi

            # Advance the checkpoint pointer to this step's output
            prev_ckpt="${model_save_path}"

            # ── Optional per-step evaluation on all topics seen so far ────────
            if [[ "${OPT_EVAL:-true}" != "true" ]]; then
                echo "[INFO] OPT_EVAL=false — skipping evaluation for this step"
                continue
            fi
            if [[ "$_ckpt_existed" == "true" ]]; then
                echo "[INFO] Checkpoint was pre-existing — skipping evaluation (already done). Re-run via suite_evaluation_optimized.sh if needed."
                continue
            fi

            # Tasks that produce identical output regardless of eval topic — they
            # depend only on the model, not on which topic is being evaluated.
            # Run once (for the first eval topic) and symlink into the other
            # topics' eval output dirs so collect_results.py finds them everywhere.
            local _topic_agnostic_tasks=("mmlu" "repet" "rgq_bi")
            local _first_eval_topic="${_seen_topics[0]}"
            # Metric output folders holding the topic-agnostic results (one file per run).
            # These are the folders collect_results.py actually reads (NOT worstCase).
            local _agnostic_metric_folders=("mmluOutputs" "repOutputs" "rgqOutputs")

            for eval_topic in "${_seen_topics[@]}"; do
                echo ""
                echo "========================================"
                echo "EVALUATING step [${_path_topic}] on topic: ${eval_topic}"
                echo "========================================"

                # Reload topic-specific dataset configs for this eval topic
                declare -A task_split task_dataset task_max_tokens
                source "configs/topics/${eval_topic}.sh"
                task_split["mmlu"]="train";  task_dataset["mmlu"]="dataset.name=jinzhuoran/RWKU";  task_max_tokens["mmlu"]="50"
                task_split["repet"]="train"; task_dataset["repet"]="dataset.name=jinzhuoran/RWKU"; task_max_tokens["repet"]="1000"

                local all_tasks_list=()
                for task in "${evaltasks[@]}"; do
                    # Topic-agnostic tasks: skip for all topics except the first — results
                    # will be symlinked from the first topic's eval dir after it completes.
                    if [[ "$eval_topic" != "$_first_eval_topic" ]]; then
                        local _is_agnostic=false
                        for _ag in "${_topic_agnostic_tasks[@]}"; do
                            [[ "$task" == "$_ag" ]] && _is_agnostic=true && break
                        done
                        [[ "$_is_agnostic" == "true" ]] && continue
                    fi

                    local task_spec
                    if [[ "$task" == "rgq_bi" ]]; then
                        task_spec="${task_name}:${model_save_path}:${task}:${task_name}:train:dataset.name=none:50:${hf_name}"
                        echo "  ✓ [${eval_topic}] ${task} → [judge-only]"
                    else
                        local split="${task_split[$task]}"
                        local dataset_arg="${task_dataset[$task]}"
                        local max_tokens="${task_max_tokens[$task]}"
                        task_spec="${task_name}:${model_save_path}:${task}:${task_name}:${split}:${dataset_arg}:${max_tokens}:${hf_name}"
                        echo "  ✓ [${eval_topic}] $task → split=$split"
                    fi
                    all_tasks_list+=("$task_spec")
                done

                local task_specs_json=""
                for i in "${!all_tasks_list[@]}"; do
                    if [ $i -eq 0 ]; then
                        task_specs_json="\"${all_tasks_list[$i]}\""
                    else
                        task_specs_json="${task_specs_json},\"${all_tasks_list[$i]}\""
                    fi
                done

                CUDA_VISIBLE_DEVICES=${_train_gpus} ${_numactl} python src/eval_full_pipeline.py \
                    experiment=eval.yaml \
                    output.task_name="${task_name}_eval_${eval_topic}" \
                    output.topic="${eval_topic}" \
                    "multi_task_specs=[${task_specs_json}]" \
                    multi_task_mode=true \
                    ${judge_overrides}

                if [ $? -eq 0 ]; then
                    echo "✓ Evaluation completed for [${_path_topic}] on ${eval_topic}"
                else
                    echo "✗ Evaluation failed for [${_path_topic}] on ${eval_topic}"
                fi

                # The topic-agnostic tasks (mmlu/repet/rgq_bi) were computed only for the
                # first eval topic. Symlink their per-run result files from the first topic's
                # eval dir into this topic's eval dir, in the metric output folders that
                # collect_results.py reads, so every topic reports them. File-level symlinks
                # (collect_results follows them via is_file()); each folder holds only its
                # own agnostic metric, so linking all of its *.jsonl is safe.
                if [[ "$eval_topic" != "$_first_eval_topic" ]]; then
                    for _folder in "${_agnostic_metric_folders[@]}"; do
                        local _base="evaluations/${_folder}/${_path_topic}/${model_cfg}/${method}"
                        local _first_dir="${_base}/eval_${_first_eval_topic}"
                        local _curr_dir="${_base}/eval_${eval_topic}"
                        [[ -d "$_first_dir" ]] || continue
                        mkdir -p "$_curr_dir"
                        for _f in "$_first_dir"/*.jsonl; do
                            [[ -e "$_f" ]] || continue
                            local _dst="${_curr_dir}/$(basename "$_f")"
                            [[ -e "$_dst" ]] && continue
                            ln -s "$(realpath "$_f")" "$_dst"
                            echo "[INFO] Symlinked $(basename "$_f"): eval_${eval_topic} → eval_${_first_eval_topic} (${_folder})"
                        done
                    done
                fi
            done

            # ── Average the per-topic metrics into one summary (printed + saved) ──
            # Only meaningful once ≥2 topics have been evaluated (step 2+). Reads the
            # per-topic eval outputs just written; saves to evaluations/topicAverages/
            # (never crawled by collect_results.py). Non-fatal: never breaks the run.
            if [[ ${#_seen_topics[@]} -ge 2 ]]; then
                python scripts/results/average_topics.py \
                    --eval-dir evaluations \
                    --multi-topic "${_path_topic}" \
                    --model "${model_cfg}" --method "${method}" --exp "${exp_suffix}" \
                    || echo "[WARN] topic averaging failed (non-fatal)"
            fi

        done  # end sequential topic loop
    done  # end runningargs loop
}

# ===========================================================================
# Dispatch — select model(s) to run.
#   • From the terminal:  MODEL=llama_3b bash scripts/suite_sequential_unlearn.sh
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
