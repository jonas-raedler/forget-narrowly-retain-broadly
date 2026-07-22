#!/bin/bash
# proximity_relearn_sweep.sh — proximity-stratified relearning attacks.
#
# From ONE unlearned checkpoint, run many INDEPENDENT GradLearn relearning
# attacks that differ only in which SUITE band supplies the relearning data
# (see configs/data/datasets/{topic}/bands/ and docs/PROXIMITY_EXPERIMENT.md),
# then score each relearned model's answer-NLL (primary metric) and finally
# run ONE batched judge evaluation over all arms (judge loaded once).
#
# Every arm is compute-matched by construction: band arms use
# +data.random_pairing=true with anchor=forget, so epoch length is always
# len(forget_train)=450 draws regardless of band size — identical optimizer
# steps, schedule, and LR across arms. C_full uses the stock paired retain
# config (the paper's exact relearn path, Tabs. 16-18) as the reproduction
# anchor.
#
# Usage:
#   MODEL_PATH="./saves/unlearn/challenger_disaster/Llama-3.2-3B-Instruct/jensen/<exp>" \
#     bash scripts/proximity_relearn_sweep.sh
#
# Env vars (unset → default):
#   MODEL_PATH   unlearned checkpoint (required, single path)
#   BANDS        space-separated arms (default: all 13)
#   SEEDS        space-separated seeds            (default "0")
#   EPOCHS       relearn epochs                   (default 10, paper)
#   LR           relearn learning rate            (default 1e-5, paper Llama value)
#   TRAIN_GPUS   GPU indices                      (default "0" — single-GPU pod)
#   GLOBAL_BATCH target global batch              (default 32 = paper 8 x 4 GPUs;
#                grad-accum is derived: GLOBAL_BATCH / (8 * n_gpus))
#   SKIP_TRAIN   1 = skip training (re-run evals on existing checkpoints)
#   SKIP_NLL     1 = skip the per-arm NLL eval
#   SKIP_JUDGE   1 = skip the final batched judge eval
#   TASKS        judge eval tasks                 (default "forget_rephrasings,retain")
#   JUDGE_MODEL / JUDGE_TAG / JUDGE_BATCH_SIZE    judge overrides (eval.yaml defaults)
#   JUDGE_N_GPUS GPUs per judge                   (default 1 — REQUIRED on a
#                single-GPU pod; the upstream default of 2 yields 0 judges)
set -uo pipefail

MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH is required, e.g." >&2
    echo '  MODEL_PATH="./saves/unlearn/{topic}/{model}/{method}/{exp}" bash scripts/proximity_relearn_sweep.sh' >&2
    exit 1
fi

DEFAULT_BANDS="R0 R1 R2 R3 R4 R5 R6 R7 C_gk C_lex C_full A_forget_partial A_forget_full"
BANDS="${BANDS:-$DEFAULT_BANDS}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-5}"
TRAIN_GPUS="${TRAIN_GPUS:-0}"
GLOBAL_BATCH="${GLOBAL_BATCH:-32}"
TASKS="${TASKS:-forget_rephrasings,retain}"
PER_DEVICE_BS=8   # paper GradLearn batch size

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# ── Derive {topic}/{model}/{method}/{exp} from the checkpoint path ───────────
_rel="${MODEL_PATH#./}"; _rel="${_rel#saves/unlearn/}"
IFS='/' read -r -a _parts <<< "$_rel"
topic="${TOPIC:-${_parts[0]}}"
model="${_parts[1]}"
src_method="${_parts[2]}"
src_exp=$(IFS='/'; echo "${_parts[*]:3}")
model_config="configs/model/${model}.yaml"

# Experiment YAML + HF name by model family (mirrors suite_relearn.sh).
# NB: the LR default above (1e-5) is the paper's LLAMA value — pass LR=2e-6
# (Ministral) / LR=5e-6 (Qwen) explicitly for other families.
case "$model" in
    Llama-*)   experiment="unlearn/suite/${topic}/llama.yaml";       hf_name="meta-llama/$model" ;;
    Ministral-*|Mistral-*) experiment="unlearn/suite/${topic}/ministral3b.yaml"; hf_name="mistralai/$model" ;;
    Qwen*)     experiment="unlearn/suite/${topic}/qwen.yaml";        hf_name="Qwen/$model" ;;
    *)         echo "ERROR: cannot infer model family from '$model'" >&2; exit 1 ;;
esac

# ── Batch geometry: keep the paper's global batch on any GPU count ───────────
_n_gpus=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)
if (( GLOBAL_BATCH % (PER_DEVICE_BS * _n_gpus) != 0 )); then
    echo "ERROR: GLOBAL_BATCH=$GLOBAL_BATCH not divisible by per_device(8) x n_gpus($_n_gpus)" >&2
    exit 1
fi
grad_accum=$(( GLOBAL_BATCH / (PER_DEVICE_BS * _n_gpus) ))

echo "========================================"
echo "  Proximity relearn sweep"
echo "  topic=$topic model=$model src=$src_method/$src_exp"
echo "  bands: $BANDS"
echo "  seeds: $SEEDS | epochs=$EPOCHS lr=$LR"
echo "  gpus=$TRAIN_GPUS bs=$PER_DEVICE_BS x ga=$grad_accum x n=$_n_gpus = $GLOBAL_BATCH"
echo "========================================"

run_nll() {  # $1 = model path, $2 = task_name override (may be empty)
    local _tn_arg=()
    [[ -n "${2:-}" ]] && _tn_arg=(--task_name "$2")
    CUDA_VISIBLE_DEVICES=${TRAIN_GPUS%%,*} python src/evals/nll_eval.py \
        --model_path "$1" "${_tn_arg[@]}" \
        --topic "$topic" \
        --model_config "$model_config" \
        --splits forget_eval retain_eval forget_train
}

# ── Baseline NLL of the source (unlearned) checkpoint ────────────────────────
if [[ "${SKIP_NLL:-0}" != "1" ]]; then
    echo ">>> Baseline NLL: $MODEL_PATH"
    run_nll "$MODEL_PATH" "" || echo "[WARN] baseline NLL failed (continuing)"
fi

failed_arms=()
trained_arms=()   # entries: "task_name"

for band in $BANDS; do
for seed in $SEEDS; do
    exp_suffix="band_${band}_seed${seed}_GradLearn_epochs_${EPOCHS}_lrs_${LR}"
    task_name="${topic}/${model}/${src_method}/${src_exp}/relearn/${exp_suffix}"
    save_path="./saves/unlearn/${task_name}"

    # Arm-specific data overrides. C_full = the stock paired retain config
    # (the paper's exact relearn data path); every other arm swaps the retain
    # group for its band config and switches to forget-anchored random pairing.
    data_overrides=()
    if [[ "$band" != "C_full" ]]; then
        data_overrides=(
            "data/datasets@data.retain=${topic}/bands/${band}"
            "+data.random_pairing=true"
        )
    fi

    if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
        echo ""
        echo ">>> [$band seed=$seed] training → $task_name"
        CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} accelerate launch \
            --config_file configs/accelerate/default_config.yaml \
            --num_processes ${_n_gpus} \
            --main_process_port $MASTER_PORT \
            src/train.py --config-name=unlearn.yaml \
            experiment=${experiment} \
            trainer=GradLearn \
            task_name=${task_name} \
            model=${model} \
            model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
            "${data_overrides[@]}" \
            trainer.args.per_device_train_batch_size=$PER_DEVICE_BS \
            trainer.args.gradient_accumulation_steps=$grad_accum \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=false \
            trainer.args.num_train_epochs=$EPOCHS \
            trainer.args.logging_steps=10 \
            trainer.args.eval_strategy=no \
            trainer.args.learning_rate=${LR} \
            trainer.args.warmup_epochs=1 \
            trainer.args.seed=${seed} \
            trainer.method_args.alpha=1 \
            trainer.method_args.gamma=0
        if [[ $? -ne 0 ]]; then
            echo "[FAIL] training $band seed=$seed"
            failed_arms+=("train:$band:$seed")
            continue
        fi
    fi

    if [[ ! -d "$save_path" ]]; then
        echo "[FAIL] no checkpoint at $save_path"
        failed_arms+=("ckpt:$band:$seed")
        continue
    fi
    trained_arms+=("$task_name")

    if [[ "${SKIP_NLL:-0}" != "1" ]]; then
        echo ">>> [$band seed=$seed] NLL eval"
        run_nll "$save_path" "$task_name" || {
            echo "[FAIL] NLL $band seed=$seed"; failed_arms+=("nll:$band:$seed"); }
    fi
done
done

# ── One batched judge eval over all trained arms (judge loaded once) ─────────
if [[ "${SKIP_JUDGE:-0}" != "1" && ${#trained_arms[@]} -gt 0 ]]; then
    if [[ ! -f "configs/topics/${topic}.sh" ]]; then
        echo "ERROR: configs/topics/${topic}.sh not found — cannot build judge eval specs" >&2
        exit 1
    fi
    declare -A task_split task_dataset task_max_tokens
    source "configs/topics/${topic}.sh"
    task_split["mmlu"]="train";  task_dataset["mmlu"]="dataset.name=jinzhuoran/RWKU";  task_max_tokens["mmlu"]="50"
    task_split["repet"]="train"; task_dataset["repet"]="dataset.name=jinzhuoran/RWKU"; task_max_tokens["repet"]="1000"

    JUDGE_N_GPUS="${JUDGE_N_GPUS:-1}"
    judge_overrides="judge.gpus_per_judge=$JUDGE_N_GPUS"
    [[ -n "${JUDGE_MODEL:-}" ]]      && judge_overrides="$judge_overrides judge.hf_model_id=\"$JUDGE_MODEL\""
    [[ -n "${JUDGE_TAG:-}" ]]        && judge_overrides="$judge_overrides judge.judge_tag=\"$JUDGE_TAG\""
    [[ -n "${JUDGE_BATCH_SIZE:-}" ]] && judge_overrides="$judge_overrides judge.batch_size=$JUDGE_BATCH_SIZE"

    IFS=',' read -ra evaltasks <<< "$TASKS"
    all_specs=()
    for tn in "${trained_arms[@]}"; do
        for task in "${evaltasks[@]}"; do
            split="${task_split[$task]}"
            dataset_arg="${task_dataset[$task]}"
            max_tokens="${task_max_tokens[$task]}"
            all_specs+=("${tn}:./saves/unlearn/${tn}:${task}:${tn}:${split}:${dataset_arg}:${max_tokens}:${hf_name}")
        done
    done
    task_specs_json=""
    for i in "${!all_specs[@]}"; do
        [[ $i -eq 0 ]] && task_specs_json="\"${all_specs[$i]}\"" \
                       || task_specs_json="${task_specs_json},\"${all_specs[$i]}\""
    done

    echo ""
    echo "========================================"
    echo "BATCHED JUDGE EVAL: ${#trained_arms[@]} arms x {$TASKS}"
    echo "========================================"
    CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} python src/eval_full_pipeline.py \
        experiment=eval.yaml \
        output.task_name="${topic}/${model}/${src_method}/${src_exp}/relearn/proximity_sweep_eval" \
        output.topic="${topic}" \
        "multi_task_specs=[${task_specs_json}]" \
        multi_task_mode=true \
        ${judge_overrides}
    [[ $? -ne 0 ]] && failed_arms+=("judge:batched")
fi

echo ""
echo "========================================"
echo "SWEEP SUMMARY: ${#trained_arms[@]} arms completed"
if [[ ${#failed_arms[@]} -gt 0 ]]; then
    echo "FAILURES: ${failed_arms[*]}"
    exit 1
fi
echo "All arms OK."
