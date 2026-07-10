#!/bin/bash

# Usage:
#   bash scripts/suite_evaluation_optimized.sh
#   EXPERIMENT="my_exp" bash scripts/suite_evaluation_optimized.sh

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
suite_evaluation_optimized.sh — evaluate existing checkpoint(s).
(Training auto-evaluates, so this is only for re-evaluating saved checkpoints.)

Usage:
  MODELS="key=path;..." bash scripts/suite_evaluation_optimized.sh

Env vars (unset → default):
  MODELS         checkpoints: key=path pairs, ';'-separated (bare path → auto key m1, m2, …)
  TOPIC          force a single topic              (default: auto-detect from each model path)
  TASKS          forget/retain tasks              (default retain,forget_rephrasings,forget_rephrasings_gibberish)
  UTILITY_TASKS  mmlu|repet, comma-separated      (default mmlu,repet)
  RGQ_BI_MODELS  models for RGQ-bi vs pretrained  (default: all non-pretrained)
  EVAL_GPUS      GPU indices                      (default 0,1,2,3)
  JUDGE_MODEL / JUDGE_TAG / JUDGE_BATCH_SIZE / JUDGE_N_GPUS   judge overrides (default: configs/eval.yaml)
  EXPERIMENT     output-name prefix

Full reference (all tasks + choices): docs/EXPERIMENTS.md
EOF
  exit 0
fi

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

###############################################
#  TOPIC  (sets the top-level output folder + all dataset names)
###############################################
# Auto-detected per model from the path (saves/unlearn/{topic}/...).
# Models from different topics are split into separate eval runs automatically.
# Override to force a single topic for all models:
#       TOPIC=salem_witch_trials bash scripts/suite_evaluation_optimized.sh
declare -A task_split task_dataset task_max_tokens

###############################################
#  JUDGE CONFIGURATION (Optional overrides)
#  Leave empty to use eval.yaml defaults (Qwen3.5-35B-A3B, tag=qwen35b, 2 GPUs/judge, auto parallel)
#  To use the larger 80B single judge instead (one sequential judge):
#    JUDGE_MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct" JUDGE_TAG="" JUDGE_BATCH_SIZE=16 JUDGE_N_GPUS=6 \
#    bash scripts/suite_evaluation_optimized.sh
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

###############################################
#  EXPERIMENT NAME (Optional)
###############################################
EXPERIMENT="${EXPERIMENT:-}"
if [[ -n "$EXPERIMENT" ]]; then
    EXPERIMENT="${EXPERIMENT}_"
    echo "Experiment: ${EXPERIMENT%_}"
fi

###############################################
#  MODELS TO EVALUATE
###############################################
declare -A models
# Checkpoints to evaluate, as "key=path" entries. Usually driven from the terminal
# via MODELS= (see below): the commented examples show the format for both
# pretrained baselines (HF model id) and local saves (./saves/unlearn/...).
models=(
    # Pretrained baselines
#    ["llama3b_pretrained"]="meta-llama/Llama-3.2-3B-Instruct"
#    ["ministral3b_pretrained"]="mistralai/Ministral-3-3B-Instruct-2512-BF16"
#    ["qwen35b_9b_pretrained"]="Qwen/Qwen3.5-9B"
    # Local unlearned save (example):
#    ["llama_jensen"]="./saves/unlearn/challenger_disaster/Llama-3.2-3B-Instruct/jensen/<exp>"
)

# Override the models map from the terminal (no file edit):
#   MODELS="key=./saves/unlearn/.../<exp>" bash scripts/suite_evaluation_optimized.sh
# Separate multiple with ';'. Each entry is "key=path"; a bare "path" (no '=')
# gets an auto key m1, m2, …. Unset → the map above is used.
if [[ -n "$MODELS" ]]; then
    models=()
    IFS=';' read -ra _pairs <<< "$MODELS"
    _i=0
    for _p in "${_pairs[@]}"; do
        _p="${_p# }"   # trim a leading space after the ';'
        if [[ "$_p" == *"="* ]]; then
            models["${_p%%=*}"]="${_p#*=}"
        else
            _i=$((_i+1)); models["m${_i}"]="$_p"
        fi
    done
fi

# HF model ID for each key — used as cfg.model.name (tokenizer + output filename short name).
# For pretrained models this equals the model path; for local saves it is the base HF model,
# auto-inferred from the save path.
# Supported orgs: Llama→meta-llama, Mistral→mistralai, Qwen→Qwen, Phi→microsoft
infer_hf_name() {
    local path="$1"
    # Normalize: 'saves/unlearn/...' → './saves/unlearn/...' (missing ./ prefix)
    [[ "$path" == "saves/unlearn/"* ]] && path="./$path"
    # Normalize: '/saves/unlearn/...' → './saves/unlearn/...' (absolute path)
    [[ "$path" == "/saves/unlearn/"* ]] && path=".${path}"
    # Pretrained: HF model id (no local path prefix)
    if [[ "$path" != "./"* && "$path" != "/"* ]]; then
        echo "$path"; return
    fi
    # Hierarchical save layout: ./saves/unlearn/{topic}/{model}/{method}[/relearn]/{exp}
    # Model name is the 2nd component after stripping ./saves/unlearn/
    local rel="${path#./saves/unlearn/}"
    local model_part
    model_part=$(echo "$rel" | cut -d'/' -f2)
    case "$model_part" in
        Llama-*)   echo "meta-llama/${model_part}" ;;
        Ministral-*) echo "mistralai/${model_part}" ;;
        Mistral-*) echo "mistralai/${model_part}" ;;
        Qwen*)     echo "Qwen/${model_part}" ;;
        Phi-*)     echo "microsoft/${model_part}" ;;
        *)         echo "$path" ;;
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

###############################################
#  TASKS TO EVALUATE
###############################################

# --- Standard eval tasks (forget / retain / rephrasings / …) ---
# These run on every model in the models map above.
tasks=(
    "retain"
    "forget_rephrasings"
#    "forget_adversarial"
  "forget_rephrasings_gibberish"
)
# Override from the terminal (comma-separated): TASKS="retain,forget_rephrasings" bash scripts/suite_evaluation_optimized.sh
[[ -n "$TASKS" ]] && IFS=',' read -ra tasks <<< "$TASKS"

# --- Utility tasks (mmlu / repet) ---
# These also run on every model.  They use the same GPU-parallel Phase 1.
#   mmlu  — RWKU `utility_general` accuracy (up to 2000 samples)
#   repet — repetitiveness on AlpacaEval (also feeds rgq_bi)
utility_tasks=(
     "mmlu"
     "repet"
)
# Override from the terminal (comma-separated): UTILITY_TASKS="mmlu,repet" bash scripts/suite_evaluation_optimized.sh
[[ -n "$UTILITY_TASKS" ]] && IFS=',' read -ra utility_tasks <<< "$UTILITY_TASKS"

# --- RGQ (Relative Generation Quality, bidirectional) ---
# Runs AFTER repet finishes.  Only makes sense for non-pretrained models
# (compares each model vs. the pretrained baseline).
# Each pair is judged twice with the assistant slots swapped; a win is only
# counted when the unlearned model wins in both orderings (any disagreement → tie).
# Output files: RGQbi_<model>_<exp>.jsonl under evaluations/rgqOutputs/...
rgq_bi_models=()
# By DEFAULT rgq_bi runs for every non-pretrained model (a local unlearned checkpoint,
# not an HF base model) — the default list is populated further below, once the models
# map and _path_to_topics helper are both available.
# Override from the terminal (comma-separated keys from the models map); "none" disables it:
#   RGQ_BI_MODELS="llama_jensen,qwen_jensen" bash scripts/suite_evaluation_optimized.sh
[[ -n "${RGQ_BI_MODELS:-}" && "$RGQ_BI_MODELS" != "none" ]] && IFS=',' read -ra rgq_bi_models <<< "$RGQ_BI_MODELS"


###############################################
#  GROUP MODELS BY TOPIC AND RUN PER-TOPIC EVAL
###############################################

# Helper: extract all training topics from a local model path.
# For single-topic models returns one element; for combined/sequential models
# (compound topic "A+B+C" in the path) returns each sub-topic as a separate element.
# Returns an empty list for HF model IDs (pretrained baselines).
_path_to_topics() {
    local p="$1"
    local topics=()
    if [[ "$p" == "./saves/unlearn/"* || "$p" == "saves/unlearn/"* ]]; then
        local rel="${p#./saves/unlearn/}"; rel="${rel#saves/unlearn/}"
        local compound; compound="$(echo "$rel" | cut -d'/' -f1)"
        # Strip mode prefix (seq_, comb_) added by suite_sequential/combined_unlearn.sh
        # before splitting on '+' to match topic config files.
        compound="${compound#seq_}"; compound="${compound#comb_}"
        local _parts
        IFS='+' read -ra _parts <<< "$compound"
        for t in "${_parts[@]}"; do
            [[ -f "configs/topics/${t}.sh" ]] && topics+=("$t")
        done
    fi
    echo "${topics[@]}"
}

# Helper: check if a model key is in the current topic's model list.
_in_cur_group() {
    local needle="$1"
    for _k in "${_cur_model_keys[@]}"; do
        [[ "$_k" == "$needle" ]] && return 0
    done
    return 1
}

# Helper: does a given task use ICR variants?
uses_icr() {
    case "$1" in
        forget|forget_rephrasings) echo "yes" ;;
        *) echo "no" ;;
    esac
}

# Build per-topic model groups.
# Non-local paths (HF model IDs) are treated as pretrained baselines and included in every topic group.
declare -A _topic_key_map   # topic → space-separated model keys
_pretrained_keys=()

if [[ -n "${TOPIC:-}" ]]; then
    # Explicit override: run all models under a single specified topic.
    for _mk in "${!models[@]}"; do
        _topic_key_map["$TOPIC"]+=" $_mk"
    done
    _run_topics=("$TOPIC")
else
    # Auto-group: derive topic(s) from each model's save path.
    # Combined/sequential models (compound topic A+B in path) are registered
    # under EVERY sub-topic so they get evaluated once per topic.
    for _mk in "${!models[@]}"; do
        IFS=' ' read -ra _ts <<< "$(_path_to_topics "${models[$_mk]}")"
        if [[ ${#_ts[@]} -gt 0 ]]; then
            for _t in "${_ts[@]}"; do
                _topic_key_map["$_t"]+=" $_mk"
            done
        else
            _pretrained_keys+=("$_mk")
        fi
    done
    _run_topics=("${!_topic_key_map[@]}")
fi

if [[ ${#_run_topics[@]} -eq 0 ]]; then
    echo "ERROR: no models with a recognisable topic path found and TOPIC env var not set." >&2
    exit 1
fi

# Default rgq_bi to every non-pretrained model (a local unlearned save, i.e. its path maps
# to a topic) when RGQ_BI_MODELS was not provided. Pretrained/HF-id models have no save-path
# topic and are excluded. The per-topic loop further restricts each model to its own topic.
if [[ -z "${RGQ_BI_MODELS:-}" ]]; then
    for _mk in "${!models[@]}"; do
        [[ -n "$(_path_to_topics "${models[$_mk]}")" ]] && rgq_bi_models+=("$_mk")
    done
fi

echo "Topics to evaluate: ${_run_topics[*]}"
[[ ${#_pretrained_keys[@]} -gt 0 ]] && echo "Pretrained/HF models (added to every topic): ${_pretrained_keys[*]}"
echo ""

# ── Per-topic evaluation loop ──────────────────────────────────────────────────
for _cur_topic in "${_run_topics[@]}"; do

# Build model key list for this topic (topic-specific + pretrained baselines).
_cur_model_keys=()
for _mk in ${_topic_key_map[$_cur_topic]:-}; do
    [[ -n "$_mk" ]] && _cur_model_keys+=("$_mk")
done
for _mk in "${_pretrained_keys[@]+"${_pretrained_keys[@]}"}"; do
    _cur_model_keys+=("$_mk")
done

# Source the topic config; register utility + global tasks.
source "configs/topics/${_cur_topic}.sh"
task_split["mmlu"]="train";  task_dataset["mmlu"]="dataset.name=jinzhuoran/RWKU";  task_max_tokens["mmlu"]="50"
task_split["repet"]="train"; task_dataset["repet"]="dataset.name=jinzhuoran/RWKU"; task_max_tokens["repet"]="1000"

echo "========================================"
echo "  TOPIC    : $topic"
echo "  Rephrase : ${task_dataset[forget_rephrasings]#dataset.name=}"
echo "  Standard : ${tasks[*]:-none}"
echo "  Utility  : ${utility_tasks[*]:-none}"
echo "  RGQ      : ${rgq_bi_models[*]:-none}"
echo "  Models   : ${_cur_model_keys[*]}"
echo "========================================"
echo ""

###############################################
#  AGGREGATE ALL MODEL×TASK COMBINATIONS
###############################################
echo "========================================"
echo "AGGREGATING MODEL×TASK COMBINATIONS (topic: $topic)"
echo "========================================"

declare -a all_tasks_list=()

# Append the model-path leaf (exp_suffix) so cache keys are unique per checkpoint,
# not just per alias. Without this, swapping `models["A1"]` between checkpoints
# silently reuses the previous run's generations (registry key collision).
_task_name_for() {
    local key="$1" path="${models[$1]}"
    echo "${EXPERIMENT}${key}_$(basename "$path")"
}

# ---- Standard tasks (forget / retain / …) ----
for model_name_key in "${_cur_model_keys[@]}"; do
    for task in "${tasks[@]}"; do
        task_name="$(_task_name_for "$model_name_key")"
        split="${task_split[$task]}"
        dataset_arg="${task_dataset[$task]}"
        max_tokens="${task_max_tokens[$task]}"
        hf_name="$(infer_hf_name "${models[$model_name_key]}")"

        task_spec="${model_name_key}:${models[$model_name_key]}:${task}:${task_name}:${split}:${dataset_arg}:${max_tokens}:${hf_name}"
        all_tasks_list+=("$task_spec")

        icr_note=""
        [[ "$(uses_icr $task)" == "yes" ]] && icr_note=" [icr=False + icr=True]"
        echo "  ✓ [$model_name_key] $task → $task_name${icr_note}  (hf=${hf_name})"
    done
done

# ---- Utility tasks (mmlu / repet) ----
for model_name_key in "${_cur_model_keys[@]}"; do
    for task in "${utility_tasks[@]+"${utility_tasks[@]}"}"; do
        task_name="$(_task_name_for "$model_name_key")"
        split="${task_split[$task]}"
        dataset_arg="${task_dataset[$task]}"
        max_tokens="${task_max_tokens[$task]}"
        hf_name="$(infer_hf_name "${models[$model_name_key]}")"

        task_spec="${model_name_key}:${models[$model_name_key]}:${task}:${task_name}:${split}:${dataset_arg}:${max_tokens}:${hf_name}"
        all_tasks_list+=("$task_spec")
        echo "  ✓ [$model_name_key] $task → $task_name  [utility, no judge]"
    done
done

# ---- RGQ tasks (only for models in this topic group) ----
n_rgq=0
for model_name_key in "${rgq_bi_models[@]+"${rgq_bi_models[@]}"}"; do
    _in_cur_group "$model_name_key" || continue
    if [[ -v models[$model_name_key] ]]; then
        task_name="$(_task_name_for "$model_name_key")"
        hf_name="$(infer_hf_name "${models[$model_name_key]}")"
        task_spec="${model_name_key}:${models[$model_name_key]}:rgq_bi:${task_name}:train:dataset.name=none:50:${hf_name}"
        all_tasks_list+=("$task_spec")
        n_rgq=$(( n_rgq + 1 ))
        echo "  ✓ [$model_name_key] rgq_bi → $task_name  [bidirectional judge, after repet]"
    else
        echo "  ✗ [$model_name_key] rgq_bi – model key not found in models map, skipping"
    fi
done

total_tasks=${#all_tasks_list[@]}
n_standard=$(( ${#_cur_model_keys[@]} * ${#tasks[@]} ))
n_utility=$(( ${#_cur_model_keys[@]} * ${#utility_tasks[@]} ))

icr_count=0
for task in "${tasks[@]}"; do
    [[ "$(uses_icr $task)" == "yes" ]] && icr_count=$(( icr_count + ${#_cur_model_keys[@]} ))
done
non_icr_gen=$(( n_standard - icr_count ))
total_gen_runs=$(( non_icr_gen + icr_count * 2 + n_utility ))

echo ""
echo "Summary (topic: $topic):"
echo "  Standard tasks   : $n_standard  (${#_cur_model_keys[@]} models × ${#tasks[@]} tasks)"
echo "  Utility tasks    : $n_utility  (${#_cur_model_keys[@]} models × ${#utility_tasks[@]} utility)"
echo "  RGQ tasks        : $n_rgq"
echo "  GPU gen runs     : $total_gen_runs  (forget-type tasks = 2 runs each)"
_exp_display="${EXPERIMENT%_}"
echo "Experiment: ${_exp_display:-none}"
echo ""

###############################################
#  PASS ALL TASKS TO eval_full_pipeline.py
###############################################
echo "========================================"
echo "STARTING UNIFIED EVALUATION (topic: $topic)"
echo "========================================"
echo ""
echo "KEY: All $total_tasks specs will:"
echo "  1. Phase 1 – generate in PARALLEL (forget/retain/mmlu/repet, one task per GPU)"
echo "  2. Phase 2 – N parallel EvalJUDGE instances (auto: n_gpus / gpus_per_judge)"
echo "  3. Phase 2b– EvalRGQ runs inside judge subprocess 0 (no extra model load)"
echo "  4. Phase 3 – compute final metrics"
echo ""

task_specs_json=""
for i in "${!all_tasks_list[@]}"; do
    task_spec="${all_tasks_list[$i]}"
    if [ $i -eq 0 ]; then
        task_specs_json="\"${task_spec}\""
    else
        task_specs_json="${task_specs_json},\"${task_spec}\""
    fi
done

# EVAL_GPUS env var controls which physical GPUs to use.
# Example: EVAL_GPUS=0,1,2,3 bash scripts/suite_evaluation_optimized.sh
_eval_gpus="${EVAL_GPUS:-0,1,2,3}"
_numactl=$(get_numactl_prefix "$_eval_gpus")
if [[ -n "$_numactl" ]]; then
    echo "NUMA binding → $_numactl  (GPUs: $_eval_gpus)"
else
    echo "NUMA binding → disabled (numactl unavailable or no NUMA topology detected)"
fi
CUDA_VISIBLE_DEVICES=${_eval_gpus} ${_numactl} python src/eval_full_pipeline.py \
    experiment=eval.yaml \
    output.task_name="${EXPERIMENT}aggregated_eval" \
    output.topic="${topic}" \
    "multi_task_specs=[${task_specs_json}]" \
    multi_task_mode=true \
    ${judge_overrides}

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ All tasks completed for topic: $topic"
else
    echo ""
    echo "✗ Evaluation failed for topic: $topic"
    exit 1
fi

echo ""
echo "========================================"
echo "EVALUATION COMPLETE (topic: $topic)"
echo "========================================"
echo "Standard tasks : $n_standard  (${#_cur_model_keys[@]} models × ${#tasks[@]} tasks)"
echo "Utility tasks  : $n_utility"
echo "RGQ tasks      : $n_rgq"
echo "GPU gen runs   : $total_gen_runs"
_exp_display="${EXPERIMENT%_}"
echo "Experiment: ${_exp_display:-none}"
echo ""
echo "Output locations:"
echo "  Generations : ./evaluations/evalOutputs/"
echo "  Judge files : ./evaluations/evalJudge/"
echo "  Metrics     : ./evaluations/worstCase/<topic>/<model>/<method>/<exp>/<eval_task>/"
echo "  Registry    : ./evaluations/generation_registry.json"
echo ""

done  # end per-topic loop

