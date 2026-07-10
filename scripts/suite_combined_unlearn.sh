#!/bin/bash

# ─────────────────────────────────────────────────────────────────────────────
#  COMBINED MULTI-TOPIC UNLEARNING
#
#  Trains a model on all listed topics simultaneously using concatenated
#  datasets (one training run). Topics are sorted alphabetically and joined
#  with '+' to form the save-path prefix and combined topic identifier.
#
#  Usage:
#    bash scripts/suite_combined_unlearn.sh
#    OPT_EVAL=false bash scripts/suite_combined_unlearn.sh   # skip evaluation
#    TRAIN_GPUS=4,5,6,7 bash scripts/suite_combined_unlearn.sh
# ─────────────────────────────────────────────────────────────────────────────

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
suite_combined_unlearn.sh — unlearn a set of topics jointly in one run
(datasets concatenated; one training run).

Usage:
  bash scripts/suite_combined_unlearn.sh
  COMB_TOPICS="challenger_disaster,salem_witch_trials" bash scripts/suite_combined_unlearn.sh

Env vars (unset → default):
  COMB_TOPICS  topic set, comma-separated, trained jointly (default: all topics)
  MODEL        model key(s), comma-separated     (default llama_3b)
  METHOD       unlearning trainer                (default JensUnPP)
  EPOCHS       training epochs                   (default 20)
  RUNARGS      "<lr> [gamma] [alpha] [gnorm] [randpair] [overrides]"
  EXTRA        free-form exp_suffix tag          (default: _paper / empty on override)
  OPT_EVAL     auto-evaluate after training (per sub-topic): true (default) | false
  REFUSAL_KEY  refusal-string prefix (JensUnPP)  (default: per-model paper prefix)
  TRAIN_GPUS   GPU indices                       (default 0,1,2,3)
  FORCE_REGEN  set to 1 to rebuild the auto-generated combined dataset/experiment
               configs from the per-topic sources (default: reuse if present).
               Use after editing a per-topic config so the combined copy isn't stale.

Full reference (all choices + paper hyperparameters): docs/EXPERIMENTS.md
EOF
  exit 0
fi

# Shared infrastructure — env preamble, env-var parsing (RUNARGS/REFUSAL_KEY),
# helpers (trainer_to_method, infer_hf_name, get_numactl_prefix, _topic_to_ds_keys),
# refusal helpers, _model_spec, _parse_runargs, _build_train_suffixes, _dispatch_from_MODEL.
source "$(dirname "${BASH_SOURCE[0]}")/_suite_common.sh"

# ─────────────────────────────────────────────────────────────────────────────
#  TOPICS  (edit this list to train on a different set)
#  Topics are sorted alphabetically so the combined-topic key is stable
#  regardless of the order you list them here.
# ─────────────────────────────────────────────────────────────────────────────
_COMB_TOPICS=(
    "challenger_disaster"
    "salem_witch_trials"
    "steve_jobs_medical"
    "britney_spears_conservatorship"
)
# Override the topic set from the terminal (comma-separated; sorted alphabetically
# for the save path regardless of order):
#   COMB_TOPICS="challenger_disaster,salem_witch_trials" bash scripts/suite_combined_unlearn.sh
[[ -n "$COMB_TOPICS" ]] && IFS=',' read -ra _COMB_TOPICS <<< "$COMB_TOPICS"

# RUNARGS (';'-separated runningargs entries) is parsed into _RUNARGS_OVERRIDE in
# _suite_common.sh.

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: sort topics alphabetically, join with '+'
# ─────────────────────────────────────────────────────────────────────────────
_sort_join() {
    printf '%s\n' "$@" | sort | paste -sd '+'
}

_combined_topic=$(_sort_join "${_COMB_TOPICS[@]}")
# Save/eval paths use a comb_ prefix to distinguish combined runs from sequential
# runs that end up at the same compound topic. Dataset/experiment YAML paths use
# the bare _combined_topic (no prefix) — they're shared between the two modes.
_combined_path_topic="comb_${_combined_topic}"
echo "Combined topic identifier : ${_combined_topic}"
echo "Save path prefix          : ${_combined_path_topic}"

# _topic_to_ds_keys (topic → forget/retain YAML key names) lives in _suite_common.sh;
# it expands per-dataset Hydra overrides in runningargs to cover ALL topics.

# Set ds_forget_key / ds_retain_key from the first topic so runningargs entries
# copied from suite_unlearn.sh expand ${ds_forget_key} to a valid key name.
# _run_combined_model will then expand the override to ALL topics automatically.
read -r ds_forget_key ds_retain_key <<< "$(_topic_to_ds_keys "${_COMB_TOPICS[0]}")"

# ─────────────────────────────────────────────────────────────────────────────
#  Auto-generate combined dataset YAML configs
#
#  Each topic's forget.yaml / retain.yaml is a single top-level YAML key, so
#  concatenating files produces a valid multi-dataset config that get_datasets()
#  handles via ConcatDataset (see src/data/__init__.py).
# ─────────────────────────────────────────────────────────────────────────────
_gen_combined_dataset_configs() {
    local dst_dir="configs/data/datasets/${_combined_topic}"
    mkdir -p "$dst_dir"

    for variant in forget retain forget_base retain_base; do
        local dst="${dst_dir}/${variant}.yaml"
        if [[ -f "$dst" ]]; then
            if [[ -z "${FORCE_REGEN:-}" ]]; then
                echo "[INFO] Reusing existing ${dst} (set FORCE_REGEN=1 to rebuild from per-topic sources)"
                continue
            fi
            echo "[INFO] FORCE_REGEN set — rebuilding ${dst} from per-topic sources"
        fi
        # Check all source files exist before writing
        local all_exist=1
        for t in "${_COMB_TOPICS[@]}"; do
            if [[ ! -f "configs/data/datasets/${t}/${variant}.yaml" ]]; then
                all_exist=0
                echo "[WARN] Missing configs/data/datasets/${t}/${variant}.yaml — skipping ${dst}" >&2
            fi
        done
        [[ "$all_exist" -eq 0 ]] && continue

        printf '# Auto-generated: combined %s for topics: %s\n' "$variant" "$_combined_topic" > "$dst"
        for t in "${_COMB_TOPICS[@]}"; do
            cat "configs/data/datasets/${t}/${variant}.yaml" >> "$dst"
            echo >> "$dst"  # ensure trailing newline between files
        done
        echo "[INFO] Generated ${dst}"
    done
}
_gen_combined_dataset_configs

# ─────────────────────────────────────────────────────────────────────────────
#  Auto-generate combined experiment YAMLs
#
#  Copies each model's single-topic experiment YAML from the first listed
#  topic, replacing the dataset path component with the combined topic.
# ─────────────────────────────────────────────────────────────────────────────
_gen_combined_exp_yamls() {
    local template_topic="${_COMB_TOPICS[0]}"
    local src_dir="configs/experiment/unlearn/suite/${template_topic}"
    local dst_dir="configs/experiment/unlearn/suite/${_combined_topic}"
    mkdir -p "$dst_dir"

    for yaml_name in llama ministral3b qwen llama_base ministral3b_base qwen_base; do
        local src="${src_dir}/${yaml_name}.yaml"
        local dst="${dst_dir}/${yaml_name}.yaml"
        if [[ -f "$dst" ]]; then
            if [[ -z "${FORCE_REGEN:-}" ]]; then
                echo "[INFO] Reusing existing ${dst} (set FORCE_REGEN=1 to rebuild from the template topic)"
                continue
            fi
            echo "[INFO] FORCE_REGEN set — rebuilding ${dst} from the template topic"
        fi
        if [[ ! -f "$src" ]]; then
            continue
        fi
        # Replace dataset overrides to point at the combined topic
        sed "s|${template_topic}/|${_combined_topic}/|g" "$src" > "$dst"
        echo "[INFO] Generated ${dst}"
    done
}
_gen_combined_exp_yamls

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
#  EVAL TASKS (applied per sub-topic after training)
#  Same set as suite_unlearn.sh — edit to taste.
# ─────────────────────────────────────────────────────────────────────────────
evaltasks=(
    retain
    forget_rephrasings
    forget_rephrasings_gibberish
    mmlu
    repet
    rgq_bi
)

echo "========================================"
echo "  COMBINED TOPIC : ${_combined_topic}"
echo "  Sub-topics     : ${_COMB_TOPICS[*]}"
echo "  Eval tasks     : ${evaltasks[*]}"
echo "========================================"
echo ""

# ===========================================================================
# Per-model configuration blocks
# Method is set by METHOD (default JensUnPP); each runningargs entry holds the
# hyperparameters:
#   "<lr> [gamma] [alpha] [gnorm] [randpair] [overrides]"
# The epoch count comes from EPOCHS (default 20); the exp_suffix tag from EXTRA.
# Note: per-dataset Hydra overrides (field 6+) that reference a single topic's
# dataset key will only apply to that key in the combined config.
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

    _run_combined_model "$model_cfg" "$model_org" "$model_yaml" \
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

    _run_combined_model "$model_cfg" "$model_org" "$model_yaml" \
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

    _run_combined_model "$model_cfg" "$model_org" "$model_yaml" \
        "$per_device_train_batch_size" "$gradient_accumulation_steps" \
        "$default_gamma" "$default_alpha" "$warmup_epochs" \
        "$accelerate_config" \
        "${runningargs[@]}"
}

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------
_run_combined_model() {
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
    local model_path="${model_org}/${model_cfg}"
    local hf_name; hf_name="$(infer_hf_name "$model_cfg")"
    # Trainer comes from METHOD (default JensUnPP), not from runningargs.
    local trainer="$_METHOD"

    for paramss in "${runningargs[@]}"; do
        # Parse one runningargs entry (caller-scoped; defaults + validation).
        local epoch lr run_gamma run_alpha run_gnorm run_extra run_randpair run_overrides
        _parse_runargs "$paramss" "$default_gamma" "$default_alpha"

        # JensUnPP uses refusal-context datasets; all other methods use plain base datasets.
        local _yaml_variant
        case "$trainer" in
            JensUnPP) _yaml_variant="${model_yaml}" ;;
            *) _yaml_variant="${model_yaml}_base" ;;
        esac
        local _active_experiment="unlearn/suite/${_combined_topic}/${_yaml_variant}.yaml"

        # JensUnPP family: push_prefix_to_refusal_start defaults to true (canonical setup).
        # Inject into run_overrides (with the first topic's forget key, ${ds_forget_key}) so the
        # per-topic expansion below replicates it to every combined topic. Skip the injection when
        # the user already mentions the flag, so an explicit "...=false" still wins.
        case "$trainer" in
            JensUnPP)
                if [[ "$run_overrides" != *push_prefix_to_refusal_start* ]]; then
                    run_overrides="${run_overrides:+$run_overrides }data.forget.${ds_forget_key}.args.push_prefix_to_refusal_start=true"
                fi
                ;;
        esac

        # Build grad-norm / random-pairing suffixes + Hydra overrides from the parsed flags.
        local gnorm_suffix gnorm_override randpair_suffix randpair_override
        _build_train_suffixes

        # Per-model paper default unless REFUSAL_KEY overrides; "none"/"off" drops it; JensUnPP-only.
        _resolve_refusal_args "$trainer" "$model_cfg" "$_REFUSAL_KEY"
        local refusal_suffix="$_REFUSAL_SUFFIX"
        local -a _refusal_args=("${_REFUSAL_ARGS[@]}")

        local method; method=$(trainer_to_method "$trainer")
        local exp_suffix="epochs_${epoch}_lrs_${lr}_gamma${run_gamma}_alpha${run_alpha}${gnorm_suffix}${randpair_suffix}${run_extra}${refusal_suffix}"
        local task_name="${_combined_path_topic}/${model_cfg}/${method}/${exp_suffix}"

        # Expand per-dataset Hydra overrides to cover ALL combined topics.
        # run_overrides was baked with the first topic's ds_forget_key / ds_retain_key.
        # For each additional topic we add a copy of those overrides with its own keys,
        # so e.g. push_prefix_to_refusal_start=true is applied to every forget dataset.
        local _run_overrides_combined="${run_overrides}"
        for _ov_topic in "${_COMB_TOPICS[@]}"; do
            local _t_fk _t_rk
            read -r _t_fk _t_rk <<< "$(_topic_to_ds_keys "$_ov_topic")"
            [[ "$_t_fk" == "$ds_forget_key" ]] && continue  # already covered by run_overrides
            local _extra="${run_overrides//${ds_forget_key}/${_t_fk}}"
            _extra="${_extra//${ds_retain_key}/${_t_rk}}"
            [[ -n "$_extra" ]] && _run_overrides_combined="${_run_overrides_combined} ${_extra}"
        done

        echo "Model: $model_cfg  |  Epoch: $epoch  |  LR: $lr  |  gamma: $run_gamma  |  alpha: $run_alpha"
        echo "${task_name}: Combined unlearning on [${_COMB_TOPICS[*]}] using ${trainer}"

        local _train_gpus="${TRAIN_GPUS:-0,1,2,3}"
        local _n_train_gpus; _n_train_gpus=$(echo "$_train_gpus" | tr ',' '\n' | wc -l)
        local _numactl; _numactl=$(get_numactl_prefix "$_train_gpus")
        if [[ -n "$_numactl" ]]; then
            echo "NUMA binding → $_numactl  (GPUs: $_train_gpus)"
        else
            echo "NUMA binding → disabled"
        fi

        local model_save_path="./saves/unlearn/${task_name}"

        # ── Skip training if checkpoint already exists ────────────────────────
        # Check for config.json rather than just the directory — the directory can
        # be created early (or left partial after a failed run) without a valid model.
        if [[ -f "${model_save_path}/config.json" ]]; then
            echo "[INFO] Checkpoint already exists at ${model_save_path} — skipping training."
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
            model.model_args.pretrained_model_name_or_path=${model_path} \
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
            ${_run_overrides_combined}

        fi  # end skip-if-exists check

        if [[ "${OPT_EVAL:-true}" != "true" ]]; then
            echo "[INFO] OPT_EVAL=false — skipping evaluation"
            continue
        fi

        # ── Evaluate on each sub-topic separately ─────────────────────────────
        local _topic_agnostic_tasks=("mmlu" "repet" "rgq_bi")
        local _first_eval_topic="${_COMB_TOPICS[0]}"
        # Metric output folders holding the topic-agnostic results (one file per run).
        # These are the folders collect_results.py actually reads (NOT worstCase).
        local _agnostic_metric_folders=("mmluOutputs" "repOutputs" "rgqOutputs")

        for eval_topic in "${_COMB_TOPICS[@]}"; do
            echo ""
            echo "========================================"
            echo "EVALUATING ${task_name} on topic: ${eval_topic}"
            echo "========================================"

            # Reload topic-specific dataset configs
            declare -A task_split task_dataset task_max_tokens
            source "configs/topics/${eval_topic}.sh"
            # Global utility tasks (not in topic configs)
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
                echo "✓ Evaluation completed for ${task_name} on ${eval_topic}"
            else
                echo "✗ Evaluation failed for ${task_name} on ${eval_topic}"
            fi

            # The topic-agnostic tasks (mmlu/repet/rgq_bi) were computed only for the
            # first eval topic. Symlink their per-run result files from the first topic's
            # eval dir into this topic's eval dir, in the metric output folders that
            # collect_results.py reads, so every topic reports them. File-level symlinks
            # (collect_results follows them via is_file()); each folder holds only its
            # own agnostic metric, so linking all of its *.jsonl is safe.
            if [[ "$eval_topic" != "$_first_eval_topic" ]]; then
                for _folder in "${_agnostic_metric_folders[@]}"; do
                    local _base="evaluations/${_folder}/${_combined_path_topic}/${model_cfg}/${method}"
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

        # ── Average the per-topic metrics into one summary (printed + saved) ───
        # Reads the per-topic eval outputs just written above; saves to
        # evaluations/topicAverages/ (a folder collect_results.py never crawls, so it
        # cannot perturb the results DB). Non-fatal: never breaks the run.
        python scripts/results/average_topics.py \
            --eval-dir evaluations \
            --multi-topic "${_combined_path_topic}" \
            --model "${model_cfg}" --method "${method}" --exp "${exp_suffix}" \
            || echo "[WARN] topic averaging failed (non-fatal)"

    done
}

# ===========================================================================
# Dispatch — select model(s) to run.
#   • From the terminal:  MODEL=llama_3b bash scripts/suite_combined_unlearn.sh
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
