#!/bin/bash

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
suite_relearn.sh — relearn (fine-tune) an unlearned checkpoint to test whether forgetting holds.

Usage:
  MODEL_PATH="./saves/unlearn/.../<exp>" bash scripts/suite_relearn.sh

Env vars (unset → default):
  MODEL_PATH   unlearned checkpoint(s), ';'-separated (required)
  METHOD       relearn trainer                   (default GradLearn)
  RUNARGS      "<epochs> <lr>", ';'-separate multiple (default: paper setting,
               10 epochs at lr llama 1e-5 / ministral 2e-6 / qwen 5e-6)
  TOPIC        forget topic                      (default: inferred from MODEL_PATH)
  TASKS        eval tasks (e.g. forget_rephrasings,retain_train_rephrasing,mmlu,repet,rgq_bi)
  TRAIN_GPUS   GPU indices                       (default 0,1,2,3)
  JUDGE_MODEL / JUDGE_TAG / JUDGE_BATCH_SIZE / JUDGE_N_GPUS   judge overrides (default: configs/eval.yaml)

Full reference: docs/EXPERIMENTS.md
EOF
  exit 0
fi

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# ─────────────────────────────────────────────────────────────────────────────
#  TOPIC  (sets the top-level output folder + all dataset names)
# ─────────────────────────────────────────────────────────────────────────────
# Auto-detected from the model path (saves/unlearn/{topic}/...).
# Override without editing:
#       TOPIC=salem_witch_trials bash scripts/suite_relearn.sh
declare -A task_split task_dataset task_max_tokens

# ─────────────────────────────────────────────────────────────────────────────
#  COMBINED-TOPIC HELPERS (mirror suite_combined_unlearn.sh)
#  The _gen_combined_* functions below regenerate the auto-built combined
#  configs when model_path starts with saves/unlearn/comb_*.
# ─────────────────────────────────────────────────────────────────────────────

# Creates configs/data/datasets/${_bare_combined_topic}/{forget,retain,...}.yaml
# by concatenating per-topic files. Idempotent (skips existing files).
# Uses: $_bare_combined_topic, $_COMB_TOPICS_arr
_gen_combined_dataset_configs() {
    local dst_dir="configs/data/datasets/${_bare_combined_topic}"
    mkdir -p "$dst_dir"
    for variant in forget retain forget_base retain_base; do
        local dst="${dst_dir}/${variant}.yaml"
        [[ -f "$dst" ]] && continue
        local all_exist=1
        for t in "${_COMB_TOPICS_arr[@]}"; do
            if [[ ! -f "configs/data/datasets/${t}/${variant}.yaml" ]]; then
                all_exist=0
                echo "[WARN] Missing configs/data/datasets/${t}/${variant}.yaml — skipping ${dst}" >&2
            fi
        done
        [[ "$all_exist" -eq 0 ]] && continue
        printf '# Auto-generated: combined %s for topics: %s\n' "$variant" "$_bare_combined_topic" > "$dst"
        for t in "${_COMB_TOPICS_arr[@]}"; do
            cat "configs/data/datasets/${t}/${variant}.yaml" >> "$dst"
            echo >> "$dst"
        done
        echo "[INFO] Generated ${dst}"
    done
}

# Creates configs/experiment/unlearn/suite/${_bare_combined_topic}/*.yaml
# by copying the first topic's experiment YAMLs with dataset paths replaced.
# Idempotent (skips existing files).
# Uses: $_bare_combined_topic, $_COMB_TOPICS_arr
_gen_combined_exp_yamls() {
    local template_topic="${_COMB_TOPICS_arr[0]}"
    local src_dir="configs/experiment/unlearn/suite/${template_topic}"
    local dst_dir="configs/experiment/unlearn/suite/${_bare_combined_topic}"
    mkdir -p "$dst_dir"
    for yaml_name in llama ministral3b qwen llama_base ministral3b_base qwen_base; do
        local src="${src_dir}/${yaml_name}.yaml"
        local dst="${dst_dir}/${yaml_name}.yaml"
        [[ -f "$dst" ]] && continue
        [[ ! -f "$src" ]] && continue
        sed "s|${template_topic}/|${_bare_combined_topic}/|g" "$src" > "$dst"
        echo "[INFO] Generated ${dst}"
    done
}

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL PATHS TO RELEARN  (add/remove entries as needed)
#  New hierarchical format: ./saves/unlearn/{topic}/{model}/{method}/{exp}
# ─────────────────────────────────────────────────────────────────────────────
model_paths=(
""
)
# Override from the terminal (no file edit):
#   MODEL_PATH="./saves/unlearn/.../jensen/<exp>" bash scripts/suite_relearn.sh
# Separate multiple paths with ';'. Unset → the array above is used.
[[ -n "$MODEL_PATH" ]] && IFS=';' read -ra model_paths <<< "$MODEL_PATH"

if [[ -z "${model_paths[0]:-}" ]]; then
    echo "ERROR: no model path set. Pass the unlearned checkpoint via:"
    echo "  MODEL_PATH=\"./saves/unlearn/{topic}/{model}/{method}/{exp}\" bash scripts/suite_relearn.sh"
    exit 1
fi

for model_path in "${model_paths[@]}"; do

# ─────────────────────────────────────────────────────────────────────────────
#  AUTO-DERIVE from hierarchical model_path:
#    saves/unlearn/{topic}/{model}/{method}/{exp}
# ─────────────────────────────────────────────────────────────────────────────
_rel="${model_path#./saves/unlearn/}"
IFS='/' read -r -a _parts <<< "$_rel"
_path_topic="${_parts[0]}"
model="${_parts[1]}"
src_method="${_parts[2]}"
# Everything from parts[3] onward is the source experiment name (may contain /)
src_exp=$(IFS='/'; echo "${_parts[*]:3}")

# ── Combined-topic detection ──────────────────────────────────────────────────
# Detect comb_ prefix → parse sub-topics → generate combined configs if missing.
_is_combined=false
_bare_combined_topic=""
_COMB_TOPICS_arr=()

if [[ "$_path_topic" == comb_* ]]; then
    _is_combined=true
    _bare_combined_topic="${_path_topic#comb_}"
    IFS='+' read -ra _COMB_TOPICS_arr <<< "$_bare_combined_topic"
    _gen_combined_dataset_configs
    _gen_combined_exp_yamls
fi

# Source topic config: for single topics, auto-detect from model path (or TOPIC env var).
# For combined topics, skip here — each sub-topic is sourced individually in the eval loop.
if [[ "$_is_combined" == true ]]; then
    topic="${_path_topic}"          # used in echo messages only
    declare -A task_split task_dataset task_max_tokens
else
    _active_topic="${TOPIC:-${_path_topic}}"
    source "configs/topics/${_active_topic}.sh"
    task_split["mmlu"]="train";  task_dataset["mmlu"]="dataset.name=jinzhuoran/RWKU";  task_max_tokens["mmlu"]="50"
    task_split["repet"]="train"; task_dataset["repet"]="dataset.name=jinzhuoran/RWKU"; task_max_tokens["repet"]="1000"
fi

# For combined models, experiment YAMLs live under the bare combined topic path (no comb_ prefix).
_exp_topic=$( [[ "$_is_combined" == true ]] && echo "$_bare_combined_topic" || echo "$_path_topic" )

case "$model" in
    Llama-*-3B-*)
        experiment="unlearn/suite/${_exp_topic}/llama.yaml"
        gradient_checkpointing_flag="false"
        use_cache_override=""   # per-model hook: set to a model.model_args.use_cache override if a model needs one; empty = model default
        accelerate_config="configs/accelerate/default_config.yaml"
        per_device_train_batch_size=4
        gradient_accumulation_steps=2
        gradlearn_batch_size=8
        ;;
    Llama-*)
        experiment="unlearn/suite/${_exp_topic}/llama.yaml"
        gradient_checkpointing_flag="false"
        use_cache_override=""
        accelerate_config="configs/accelerate/big_model_config.yaml"
        per_device_train_batch_size=4
        gradient_accumulation_steps=2
        gradlearn_batch_size=8
        ;;
    Ministral-3-3B-*)
        experiment="unlearn/suite/${_exp_topic}/ministral3b.yaml"
        gradient_checkpointing_flag="false"
        use_cache_override=""
        accelerate_config="configs/accelerate/default_config.yaml"
        per_device_train_batch_size=4
        gradient_accumulation_steps=2
        gradlearn_batch_size=8
        ;;
    Qwen*)
        experiment="unlearn/suite/${_exp_topic}/qwen.yaml"
        gradient_checkpointing_flag="false"
        use_cache_override=""
        accelerate_config="configs/accelerate/big_model_config.yaml"
        per_device_train_batch_size=4
        gradient_accumulation_steps=2
        gradlearn_batch_size=8
        ;;
    *)
        echo "ERROR: Cannot infer model family from model_path: $model_path" >&2
        exit 1
        ;;
esac

# Paper relearn learning rate (epochs=10 for all, trainer GradLearn). Per the paper's three LLMs:
#   Llama 1e-5, Ministral 2e-6, Qwen 5e-6. Other families fall back to the Llama value.
case "$model" in
    Ministral-*) relearn_lr="2e-6" ;;
    Qwen*)       relearn_lr="5e-6" ;;
    *)           relearn_lr="1e-5" ;;
esac

infer_hf_name() {
    local m="$1"
    case "$m" in
        Llama-*)   echo "meta-llama/$m" ;;
        Ministral-*) echo "mistralai/$m" ;;
        Mistral-*) echo "mistralai/$m" ;;
        Qwen*)     echo "Qwen/$m" ;;
        Phi-*)     echo "microsoft/$m" ;;
        *)         echo "$m" ;;
    esac
}

# Detect NUMA affinity for a comma-separated list of physical GPU indices.
# Returns a numactl prefix string, or empty if unavailable / no NUMA topology.
get_numactl_prefix() {
    local gpu_ids="$1"
    if ! command -v numactl &>/dev/null; then return; fi

    local -A numa_seen=()
    local gpu_idx pci_id sysfs_pci numa_file numa_node
    IFS=',' read -ra _gpus <<< "$gpu_ids"
    for gpu_idx in "${_gpus[@]}"; do
        gpu_idx="${gpu_idx// /}"
        pci_id=$(nvidia-smi --id="$gpu_idx" --query-gpu=pci.bus_id \
                    --format=csv,noheader,nounits 2>/dev/null | tr -d ' \r')
        [[ -z "$pci_id" ]] && continue
        sysfs_pci=$(echo "$pci_id" | awk -F: '{printf "0000:%s:%s", $2, $3}' \
                    | tr '[:upper:]' '[:lower:]')
        numa_file="/sys/bus/pci/devices/${sysfs_pci}/numa_node"
        [[ ! -f "$numa_file" ]] && continue
        numa_node=$(cat "$numa_file")
        [[ "$numa_node" -lt 0 ]] && continue
        numa_seen[$numa_node]=1
    done

    [[ ${#numa_seen[@]} -eq 0 ]] && return
    local nodes_str
    nodes_str=$(printf '%s,' "${!numa_seen[@]}")
    echo "numactl --cpunodebind=${nodes_str%,} --membind=${nodes_str%,}"
}

echo "========================================"
echo "  TOPIC    : $topic"
echo "  Rephrase : ${task_dataset[forget_rephrasings]#dataset.name=}"
echo "========================================"
echo "Auto-detected topic      : $_path_topic"
echo "Auto-detected model      : $model"
echo "Auto-detected src_method : $src_method"
echo "Auto-detected src_exp    : $src_exp"
echo "Auto-detected experiment : $experiment"
echo "Gradient checkpointing   : $gradient_checkpointing_flag"

# ─────────────────────────────────────────────────────────────────────────────
#  JUDGE CONFIGURATION (optional overrides)
#  Leave empty to use eval.yaml defaults (Qwen3.5-35B-A3B, tag=qwen35b, 2 GPUs/judge, auto-parallel)
#  To use the larger 80B single judge instead:
#    JUDGE_MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct" JUDGE_TAG="" JUDGE_BATCH_SIZE=16 JUDGE_N_GPUS=6
# ─────────────────────────────────────────────────────────────────────────────
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_TAG="${JUDGE_TAG:-}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-}"
JUDGE_N_GPUS="${JUDGE_N_GPUS:-}"

judge_overrides=""
[[ -n "$JUDGE_MODEL" ]]      && judge_overrides="$judge_overrides judge.hf_model_id=\"$JUDGE_MODEL\""
[[ -n "$JUDGE_TAG" ]]        && judge_overrides="$judge_overrides judge.judge_tag=\"$JUDGE_TAG\""
[[ -n "$JUDGE_BATCH_SIZE" ]] && judge_overrides="$judge_overrides judge.batch_size=$JUDGE_BATCH_SIZE"
[[ -n "$JUDGE_N_GPUS" ]]     && judge_overrides="$judge_overrides judge.gpus_per_judge=$JUDGE_N_GPUS"

# ─────────────────────────────────────────────────────────────────────────────
#  EVAL TASKS
# ─────────────────────────────────────────────────────────────────────────────
evaltasks=(
    "forget_rephrasings"
#   "forget_rephrasings_gibberish"
    "retain_train_rephrasing"
#   "mmlu"
#   "repet"
#   "rgq_bi"                    # bidirectional Relative Generation Quality (RGQbi_ files)
)
# Override from the terminal (comma-separated): TASKS="forget_rephrasings,mmlu" bash scripts/suite_relearn.sh
[[ -n "$TASKS" ]] && IFS=',' read -ra evaltasks <<< "$TASKS"

# ─────────────────────────────────────────────────────────────────────────────
#  RELEARN RUNS
# ─────────────────────────────────────────────────────────────────────────────
# METHOD picks the relearn trainer (default GradLearn); change it with METHOD=<trainer>.
_METHOD="${METHOD:-GradLearn}"

# Runningargs (the trainer comes from METHOD): "<epochs> <lr>".
# Default is the paper relearn setting for this model: 10 epochs at the per-model lr above.
runningargs=(
    "10 ${relearn_lr}"
)
# Override from the terminal: METHOD=GradLearn RUNARGS="10 1e-5" bash scripts/suite_relearn.sh
# Format per entry: "<epochs> <lr>"; separate multiple entries with ';'.
[[ -n "$RUNARGS" ]] && IFS=';' read -ra runningargs <<< "$RUNARGS"

for paramss in "${runningargs[@]}"; do
    trainer="$_METHOD"
    epoch=$(echo $paramss | cut -d' ' -f1)
    lr=$(echo $paramss | cut -d' ' -f2)

    gamma=0
    alpha=1
    warmup_epochs=1

    # GradLearn uses larger batch; JensUn-based trainers stay at model defaults
    if [[ "$trainer" == "GradLearn" ]]; then
        _batch_size=$gradlearn_batch_size
        _grad_accum=1
    else
        _batch_size=$per_device_train_batch_size
        _grad_accum=$gradient_accumulation_steps
    fi

    echo "Trainer: $trainer | Epochs: $epoch | LR: $lr"
    echo "Batch size: $_batch_size | Grad accum: $_grad_accum"

    # Hierarchical task_name: {topic}/{model}/{src_method}/{src_exp}/relearn/{trainer}_epochs_{epoch}_lrs_{lr}
    # src_exp ties the relearn run to the specific unlearned model it started from.
    exp_suffix="${trainer}_epochs_${epoch}_lrs_${lr}"
    task_name="${_path_topic}/${model}/${src_method}/${src_exp}/relearn/${exp_suffix}"
    echo "${task_name}: relearning ${model_path} using ${trainer}"

    # ── Training ──────────────────────────────────────────────────────────────
    # TRAIN_GPUS env var overrides the default. Example: TRAIN_GPUS=0,1,2,3 bash scripts/suite_relearn.sh
    _train_gpus="${TRAIN_GPUS:-0,1,2,3}"
    _n_train_gpus=$(echo "$_train_gpus" | tr ',' '\n' | wc -l)
    _numactl=$(get_numactl_prefix "$_train_gpus")
    if [[ -n "$_numactl" ]]; then
        echo "NUMA binding → $_numactl  (GPUs: $_train_gpus)"
    else
        echo "NUMA binding → disabled (numactl unavailable or no NUMA topology detected)"
    fi
    CUDA_VISIBLE_DEVICES=${_train_gpus} ${_numactl} accelerate launch \
        --config_file ${accelerate_config} \
        --num_processes ${_n_train_gpus} \
        --main_process_port $MASTER_PORT \
        src/train.py --config-name=unlearn.yaml \
        experiment=${experiment} \
        trainer=${trainer} \
        task_name=${task_name} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=${model_path} \
        trainer.args.per_device_train_batch_size=$_batch_size \
        trainer.args.gradient_accumulation_steps=$_grad_accum \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=${gradient_checkpointing_flag} \
        trainer.args.num_train_epochs=$epoch \
        trainer.args.logging_steps=10 \
        trainer.args.eval_strategy=no \
        trainer.args.learning_rate=${lr} \
        trainer.args.warmup_epochs=${warmup_epochs} \
        trainer.method_args.alpha=${alpha} \
        trainer.method_args.gamma=${gamma} \
        ${use_cache_override}

    # ── Eval ──────────────────────────────────────────────────────────────────
    model_save_path="./saves/unlearn/${task_name}"
    hf_name="$(infer_hf_name "$model")"
    echo "HF name for eval: $hf_name"

    # Helper: build task_specs_json for a given set of eval tasks and run eval_full_pipeline.
    # Uses outer vars: task_name, model_save_path, hf_name, _train_gpus, _numactl, judge_overrides
    # Args: $1=eval_topic  $2=output_task_name_suffix
    _run_eval_for_topic() {
        local eval_topic="$1"
        local out_name_suffix="$2"

        all_tasks_list=()
        for task in "${evaltasks[@]}"; do
            task_spec=""
            if [[ "$task" == "rgq_bi" ]]; then
                task_spec="${task_name}:${model_save_path}:${task}:${task_name}:train:dataset.name=none:50:${hf_name}"
                echo "  ✓ [${eval_topic}] $task → [judge-only, after repet]"
            else
                split="${task_split[$task]}"
                dataset_arg="${task_dataset[$task]}"
                max_tokens="${task_max_tokens[$task]}"
                task_spec="${task_name}:${model_save_path}:${task}:${task_name}:${split}:${dataset_arg}:${max_tokens}:${hf_name}"
                echo "  ✓ [${eval_topic}] $task → split=$split"
            fi
            all_tasks_list+=("$task_spec")
        done

        task_specs_json=""
        for i in "${!all_tasks_list[@]}"; do
            if [ $i -eq 0 ]; then
                task_specs_json="\"${all_tasks_list[$i]}\""
            else
                task_specs_json="${task_specs_json},\"${all_tasks_list[$i]}\""
            fi
        done

        echo ""
        echo "========================================"
        echo "STARTING OPTIMISED EVALUATION for ${task_name} [topic: ${eval_topic}]"
        echo "  Phase 1 – parallel generation across GPUs"
        echo "  Phase 2 – judge loaded once, judges all tasks"
        echo "  Phase 3 – final metrics"
        echo "========================================"

        CUDA_VISIBLE_DEVICES=${_train_gpus} ${_numactl} python src/eval_full_pipeline.py \
            experiment=eval.yaml \
            output.task_name="${task_name}${out_name_suffix}" \
            output.topic="${eval_topic}" \
            "multi_task_specs=[${task_specs_json}]" \
            multi_task_mode=true \
            ${judge_overrides}

        if [ $? -eq 0 ]; then
            echo "✓ Evaluation completed for ${task_name} [topic: ${eval_topic}]"
        else
            echo "✗ Evaluation failed for ${task_name} [topic: ${eval_topic}]"
        fi
    }

    if [[ "$_is_combined" == true ]]; then
        # ── Combined: evaluate on each sub-topic separately ────────────────────
        # Unlike the unlearn scripts, relearn does NOT reuse the topic-agnostic tasks
        # (mmlu/repet/rgq_bi) across sub-topics. A relearned checkpoint is a different
        # model, so those metrics must be recomputed from scratch for it — every
        # sub-topic is evaluated in full.
        for eval_topic in "${_COMB_TOPICS_arr[@]}"; do
            echo ""
            echo "========================================"
            echo "EVALUATING ${task_name} on topic: ${eval_topic}"
            echo "========================================"

            declare -A task_split task_dataset task_max_tokens
            # shellcheck source=/dev/null
            source "configs/topics/${eval_topic}.sh"
            task_split["mmlu"]="train";  task_dataset["mmlu"]="dataset.name=jinzhuoran/RWKU";  task_max_tokens["mmlu"]="50"
            task_split["repet"]="train"; task_dataset["repet"]="dataset.name=jinzhuoran/RWKU"; task_max_tokens["repet"]="1000"

            _run_eval_for_topic "$eval_topic" "_eval_${eval_topic}"
        done
    else
        # ── Single topic: original eval behaviour ──────────────────────────────
        _run_eval_for_topic "$topic" "_eval"
    fi
done  # end runningargs loop

done  # end model_paths loop