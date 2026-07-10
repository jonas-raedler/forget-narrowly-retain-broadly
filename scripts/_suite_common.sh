# shellcheck shell=bash
# ─────────────────────────────────────────────────────────────────────────────
#  Shared infrastructure for the suite_*_unlearn.sh training scripts.
#
#  Sourced (near the top) by:
#    scripts/suite_unlearn.sh
#    scripts/suite_sequential_unlearn.sh
#    scripts/suite_combined_unlearn.sh
#
#  Holds everything those scripts have in common — the env preamble, the
#  RUNARGS/REFUSAL_KEY env-var parsing, the shared helper functions, the
#  per-model spec table, and the runningargs-parse / suffix-build / dispatch
#  helpers. Per-script logic (topic handling, the _run_* training/eval bodies,
#  the editable model fallback, and SMOKE_TEST) stays in each script.
#
#  (suite_relearn.sh does not source this — it shares only a couple of helpers.)
# ─────────────────────────────────────────────────────────────────────────────

# ── Env preamble (runs on source) ────────────────────────────────────────────
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Only rank-0 should produce output; silence the rest at the library level.
export OMP_NUM_THREADS=1           # prevents torch.distributed "OMP_NUM_THREADS not set" warning
export TRANSFORMERS_VERBOSITY=error
export DATASETS_VERBOSITY=error
export DEEPSPEED_LOG_LEVEL=error   # suppresses DeepSpeed's per-rank banners
export TOKENIZERS_PARALLELISM=false
export DS_BUILD_OPS=0              # don't compile CUDA extensions (avoids CUDA version mismatch)
export DS_SKIP_CUDA_CHECK=1        # skip DeepSpeed's CUDA version check
export HYDRA_FULL_ERROR=1          # show full Python traceback instead of Hydra's summary

# ── Env-var overrides (parsed on source) ─────────────────────────────────────
# METHOD: the unlearning trainer to run (single source of the trainer name).
#   unset → JensUnPP (ours). Selecting a method auto-fills its per-model paper
#   hyperparameters (see _paper_hparams). The trainer is NOT part of RUNARGS.
_METHOD="${METHOD:-JensUnPP}"

# RUNARGS: ';'-separated runningargs entries overriding the paper default.
#   Format: "<lr> [gamma] [alpha] [gnorm] [randpair] [overrides]"
#   (the trainer comes from METHOD, the epoch count from EPOCHS, the suffix tag
#    from EXTRA — none of them are part of RUNARGS).
_RUNARGS_OVERRIDE=()
[[ -n "${RUNARGS:-}" ]] && IFS=';' read -ra _RUNARGS_OVERRIDE <<< "$RUNARGS"

# EPOCHS: number of training epochs. Default 20 (paper).
_EPOCHS="${EPOCHS:-20}"

# EXTRA: free-form suffix tag appended to exp_suffix.
#   unset  -> per-run default (_paper on the paper-default path, empty on a
#             RUNARGS override); <tag> -> that tag verbatim for every run;
#             none|off|- -> empty. Include the leading '_' yourself, e.g. _ablation.
# _EXTRA_DEFAULT carries the paper-vs-override default (set by _resolve_runargs).
_EXTRA_DEFAULT=""

# REFUSAL_KEY: refusal-prefix selector (see the refusal helpers below).
#   unset/empty → per-model paper default (unfor_comma)
#   <key>       → that prefix     |     none|off → drop the prefix
_REFUSAL_KEY="${REFUSAL_KEY:-}"

# ─────────────────────────────────────────────────────────────────────────────
#  Trainer → short method folder name.
#  Single source of truth: scripts/trainer_method_map.txt (also read by METHOD_MAP
#  in src/eval_full_pipeline.py). A trainer NOT listed there falls back to
#  lowercase(name) here AND in Python, so the two stay in sync by construction.
# ─────────────────────────────────────────────────────────────────────────────
trainer_to_method() {
    local _name="$1"
    local _map_file; _map_file="$(dirname "${BASH_SOURCE[0]}")/trainer_method_map.txt"
    local _folder=""
    if [[ -f "$_map_file" ]]; then
        # First matching "Trainer=folder" line wins; skip comment/blank lines.
        # Whitespace is stripped from both sides so stray CR (CRLF files) can't leak in.
        _folder=$(while IFS='=' read -r _k _v; do
                      _k="${_k//[[:space:]]/}"
                      [[ -z "$_k" || "$_k" == \#* ]] && continue
                      if [[ "$_k" == "$_name" ]]; then echo "${_v//[[:space:]]/}"; break; fi
                  done < "$_map_file")
    fi
    if [[ -n "$_folder" ]]; then
        echo "$_folder"
    else
        echo "$_name" | tr '[:upper:]' '[:lower:]'
    fi
}

# Infer HF org/model-id from model folder name
infer_hf_name() {
    local m="$1"
    case "$m" in
        Llama-*)     echo "meta-llama/$m" ;;
        Ministral-*) echo "mistralai/$m" ;;
        Mistral-*)   echo "mistralai/$m" ;;
        Qwen*)       echo "Qwen/$m" ;;
        Phi-*)       echo "microsoft/$m" ;;
        *)           echo "$m" ;;
    esac
}

# ─────────────────────────────────────────────────────────────────────────────
#  Dataset key lookup — maps a topic name to its forget/retain YAML key names
#  (the top-level key in configs/data/datasets/<topic>/{forget,retain}.yaml).
#  Keep in sync when adding new topics.
# ─────────────────────────────────────────────────────────────────────────────
_topic_to_ds_keys() {
    # Outputs: "<forget_key> <retain_key>"
    case "$1" in
        challenger_disaster) echo "challenger_forget challenger_retain" ;;
        salem_witch_trials)  echo "salem_forget      salem_retain" ;;
        challenger_baseline) echo "baseline_challenger_forget         baseline_challenger_retain" ;;
        steve_jobs_medical)  echo "jobs_forget        jobs_retain" ;;
        britney_spears_conservatorship) echo "britney_forget     britney_retain" ;;
        *) echo "[WARN] _topic_to_ds_keys: unknown topic '$1'" >&2; echo "" ;;
    esac
}

# Detect the NUMA node(s) for a set of physical GPU indices and return a
# numactl prefix string. Returns empty string when numactl is unavailable,
# the sysfs NUMA info is missing, or every GPU reports node -1 (no NUMA).
# When GPUs span multiple NUMA nodes the prefix covers all of them so the
# allocator can pick the nearest socket for each transfer buffer.
get_numactl_prefix() {
    local gpu_ids="$1"

    if ! command -v numactl &>/dev/null; then
        return
    fi

    local -A numa_seen=()
    local gpu_idx pci_id sysfs_pci numa_file numa_node
    IFS=',' read -ra _gpus <<< "$gpu_ids"
    for gpu_idx in "${_gpus[@]}"; do
        gpu_idx="${gpu_idx// /}"   # trim whitespace
        # nvidia-smi format: "00000000:BB:DD.F"
        pci_id=$(nvidia-smi --id="$gpu_idx" --query-gpu=pci.bus_id \
                    --format=csv,noheader,nounits 2>/dev/null | tr -d ' \r')
        [[ -z "$pci_id" ]] && continue

        # Rewrite to sysfs format "0000:bb:dd.f" (4-digit domain, lowercase)
        sysfs_pci=$(echo "$pci_id" | awk -F: '{printf "0000:%s:%s", $2, $3}' \
                    | tr '[:upper:]' '[:lower:]')
        numa_file="/sys/bus/pci/devices/${sysfs_pci}/numa_node"
        [[ ! -f "$numa_file" ]] && continue

        numa_node=$(cat "$numa_file")
        [[ "$numa_node" -lt 0 ]] && continue   # -1 = no NUMA topology on this host

        numa_seen[$numa_node]=1
    done

    [[ ${#numa_seen[@]} -eq 0 ]] && return

    # Build "0,1,…" list from the keys of the associative array
    local nodes_str
    nodes_str=$(printf '%s,' "${!numa_seen[@]}")
    nodes_str="${nodes_str%,}"   # strip trailing comma

    echo "numactl --cpunodebind=${nodes_str} --membind=${nodes_str}"
}

# ═════════════════════════════════════════════════════════════════════════════
#  REFUSAL PREFIX  (prepended to each model's base refusal string; JensUnPP-only)
#
#  REFUSAL_KEY selects the prefix:
#    unset/empty   → per-model paper default (unfor_comma)
#    <key>         → that prefix (override / experimentation)
#    none | off    → drop the prefix entirely (bare model-YAML refusal_string)
# ═════════════════════════════════════════════════════════════════════════════

# Map REFUSAL_KEY → prefix string (prepended to the model-specific base)
_refusal_prefix_str() {
    case "$1" in
        noi)         echo "No" ;;
        noi_comma)   echo "No," ;;
        sorry)       echo "Sorry" ;;
        sorry_comma) echo "Sorry," ;;
        hmm)         echo "Hmm" ;;
        hmm_comma)   echo "Hmm," ;;
        space)       echo " " ;;
        un)          echo "Unavailable" ;;
        unfor)       echo "Unfortunately" ;;
        unfor_comma) echo "Unfortunately," ;;
        cur)         echo "Currently" ;;
        act)         echo "Actually" ;;
        act_comma)   echo "Actually," ;;
        frank)       echo "Frankly" ;;
        frank_comma) echo "Frankly," ;;
        tilde)       echo "~" ;;
        *)           echo "" ; echo "[WARN] Unknown REFUSAL_KEY='$1'. Valid keys: noi | noi_comma | sorry | sorry_comma | hmm | hmm_comma | space | un | unfor | unfor_comma | cur | act | act_comma | frank | frank_comma | tilde | none | off. Falling back to model YAML default." >&2 ;;
    esac
}

# Map model_cfg → model-specific base refusal string (mirrors each model's YAML)
_model_refusal_base() {
    case "$1" in
        Llama-*)     echo "I am unable to verify this information." ;;
        Ministral-*) echo "I can't assist with that." ;;
        Qwen*)       echo "I cannot answer this question." ;;
        *)           echo "I am unable to verify this information." ;;
    esac
}

# Map model_cfg → per-model paper-default REFUSAL_KEY (used when REFUSAL_KEY is unset)
_model_default_refusal_key() {
    case "$1" in
        *)             echo "unfor_comma" ;;  # Llama / Ministral / Qwen / default
    esac
}

# Resolve the refusal-prefix override for one run.
#   Args:  $1=trainer  $2=model_cfg  $3=REFUSAL_KEY (may be empty)
#   Sets globals (bash can't return arrays):
#     _REFUSAL_ARGS         array — the Hydra override (empty when no prefix applies)
#     _REFUSAL_SUFFIX       string — "_ref_<key>" folder suffix (empty when no prefix)
#     _REFUSAL_RESOLVED_KEY string — the key actually used (empty when none/non-JensUnPP)
# The Hydra single-quote/double-quote syntax around the value keeps commas and
# spaces from being misinterpreted by Hydra's override parser.
_resolve_refusal_args() {
    local _trainer="$1" _model_cfg="$2" _key="$3"
    _REFUSAL_ARGS=(); _REFUSAL_SUFFIX=""; _REFUSAL_RESOLVED_KEY=""
    [[ "$_trainer" != "JensUnPP" ]] && return 0           # refusal_string is JensUnPP-only
    [[ -z "$_key" ]] && _key=$(_model_default_refusal_key "$_model_cfg")  # unset → paper default
    [[ "$_key" == "none" || "$_key" == "off" ]] && return 0               # explicit opt-out
    local _rprefix; _rprefix=$(_refusal_prefix_str "$_key")
    [[ -z "$_rprefix" ]] && return 0                       # unknown key → warn (above) + YAML fallback
    local _rbase; _rbase=$(_model_refusal_base "$_model_cfg")
    _REFUSAL_ARGS=("model.model_tokens.refusal_string=\"${_rprefix} ${_rbase}\"")
    _REFUSAL_SUFFIX="_ref_${_key}"
    _REFUSAL_RESOLVED_KEY="$_key"
}

# ═════════════════════════════════════════════════════════════════════════════
#  MODEL SPEC + RUNNINGARGS PARSING
# ═════════════════════════════════════════════════════════════════════════════

# Per-model static config — single source of truth shared by all three scripts.
#   Args:   $1 = model key (llama_3b | ministral_3b | qwen_9b)
#   Output: "<model_cfg> <model_org> <model_yaml> <batch> <gas> <gamma> <alpha> <warmup> <accel_config>"
# The per-model runningargs come from the paper-default table (_paper_hparams).
_model_spec() {
    case "$1" in
        llama_3b)     echo "Llama-3.2-3B-Instruct meta-llama llama 4 2 0.5 1 1 configs/accelerate/default_config.yaml" ;;
        ministral_3b) echo "Ministral-3-3B-Instruct-2512-BF16 mistralai ministral3b 4 2 0.5 1 1 configs/accelerate/default_config.yaml" ;;
        qwen_9b)      echo "Qwen3.5-9B Qwen qwen 4 2 0.5 1 1 configs/accelerate/big_model_config.yaml" ;;
        *) echo "[ERROR] _model_spec: unknown model key '$1'" >&2; return 1 ;;
    esac
}

# Resolve the extra suffix tag into caller-scoped run_extra.
#   EXTRA set   -> that value verbatim (none|off|- -> empty).
#   EXTRA unset -> _EXTRA_DEFAULT (set by _resolve_runargs: _paper on the paper-default
#                  path, "" on a RUNARGS override).
_resolve_extra() {
    if [[ -n "${EXTRA+x}" ]]; then
        run_extra="$EXTRA"
        case "$run_extra" in -|"''"|none|off) run_extra="" ;; esac
    else
        run_extra="${_EXTRA_DEFAULT:-}"
    fi
}

# Parse one runningargs entry into CALLER-SCOPED variables (bash dynamic scoping:
# the caller must `local epoch lr run_gamma run_alpha run_gnorm run_extra
# run_randpair run_overrides` before calling). The trainer comes from METHOD
# (caller sets `trainer="$_METHOD"`), the epoch count from EPOCHS, and the extra
# suffix tag from EXTRA (via _resolve_extra) — the entry itself holds only the
# hyperparameters below. Applies the gamma/alpha defaults and validates the
# boolean fields (warnings only, non-fatal).
#   Format: "<lr> [gamma] [alpha] [gnorm] [randpair] [overrides]"
#   Args:   $1 = paramss entry   $2 = default_gamma   $3 = default_alpha
_parse_runargs() {
    local _paramss="$1" _default_gamma="$2" _default_alpha="$3"
    # Whitespace-split into fields (robust when the entry is a single lr with no
    # spaces — `cut -d' '` would echo the whole line for f2+ in that case).
    local -a _f; read -ra _f <<< "$_paramss"
    epoch="$_EPOCHS"                  # from the EPOCHS env var
    lr="${_f[0]:-}"
    run_gamma="${_f[1]:-}"
    run_alpha="${_f[2]:-}"
    run_gnorm="${_f[3]:-}"            # "true" to enable grad-norm scaling; empty/"-" = off
    run_randpair="${_f[4]:-}"        # "true" to randomly pair retain to forget; empty/"-" = off
    run_overrides="${_f[*]:5}"       # field 6+: raw Hydra overrides, space-joined
    [[ "$run_overrides" == "-" ]] && run_overrides=""
    _resolve_extra   # sets run_extra from the EXTRA env var
    run_gamma=${run_gamma:-$_default_gamma}
    run_alpha=${run_alpha:-$_default_alpha}

    # Validate boolean fields: must be "true", "-", or empty — anything else is silently ignored,
    # which is almost certainly a copy-paste mistake (e.g. an extra/suffix tag in the wrong position).
    local _entry_ok=1
    if [[ -n "$run_gnorm"    && "$run_gnorm"    != "true" && "$run_gnorm"    != "-" ]]; then
        echo "⚠ WARNING: run_gnorm='$run_gnorm' is not 'true'/'-'/empty in entry: $_paramss" >&2
        _entry_ok=0
    fi
    if [[ -n "$run_randpair" && "$run_randpair" != "true" && "$run_randpair" != "-" ]]; then
        echo "⚠ WARNING: run_randpair='$run_randpair' is not 'true'/'-'/empty in entry: $_paramss" >&2
        _entry_ok=0
    fi
    if [[ "$_entry_ok" -eq 0 ]]; then
        echo "⚠ WARNING: entry will proceed but some fields may be in wrong positions — double-check the format string." >&2
    fi

    # Collision guard: field 6+ Hydra overrides are NOT encoded in exp_suffix, so a run
    # with overrides shares a save folder with an otherwise-identical run without them —
    # and the checkpoint-exists check would silently reuse the wrong one. Nudge to tag it.
    if [[ -n "$run_overrides" && -z "$run_extra" ]]; then
        echo "⚠ WARNING: RUNARGS has Hydra overrides ('$run_overrides') but EXTRA is empty — exp_suffix does NOT encode overrides, so this run may collide with / silently reuse a non-override run's checkpoint. Set EXTRA=_<tag> to disambiguate." >&2
    fi
}

# Build the grad-norm / random-pairing suffix + Hydra-override pairs from the
# parsed run_gnorm/run_randpair. Sets CALLER-SCOPED vars (caller must declare them
# local first): gnorm_suffix gnorm_override randpair_suffix randpair_override.
_build_train_suffixes() {
    gnorm_suffix=""; gnorm_override=""
    if [[ "$run_gnorm" == "true" ]]; then
        gnorm_suffix="_gnorm"
        gnorm_override="trainer.method_args.use_grad_norm_scaling=true"
    fi
    randpair_suffix=""; randpair_override=""
    if [[ "$run_randpair" == "true" ]]; then
        randpair_suffix="_randpair"
        randpair_override="+data.random_pairing=true"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
#  PAPER-DEFAULT HYPERPARAMETERS  (per model × method, from the paper tables)
#
#  Grey-marked selections from the paper's per-model hyperparameter tables
#  (Llama-3.2-3B, Ministral-3-3B, Qwen3.5-9B). All runs: 20 epochs, batch 4, gas 2.
#  gamma=λf (forget), alpha=λr (retain). gnorm (grad-norm scaling) is on only for JensUnPP.
#  The challenger_baseline topic (training on the LKF data, Llama only) uses a
#  separate table: 10 epochs (see suite_unlearn.sh) and random_pairing=true for
#  every method — handled by the _TOPIC branch below.
# ─────────────────────────────────────────────────────────────────────────────

# Map (model_key, trainer) → "lr gamma alpha gnorm [randpair]" tail (gnorm/randpair
# = "true"|"-"). Returns empty for cells not in the paper (callers fall back to llama).
_paper_hparams() {
    # challenger_baseline (LKF data, Llama only): separate paper table, all methods at
    # random_pairing=true (5th field). The 10-epoch default is applied in suite_unlearn.sh.
    if [[ "${_TOPIC:-}" == "challenger_baseline" ]]; then
        case "$1:$2" in
            llama_3b:GradDiff)       echo "5e-6 0.5 0.5 - true" ;;
            llama_3b:JensUnBaseline) echo "3e-6 1 0.5 - true" ;;
            llama_3b:NPO)            echo "9e-6 1 1 - true" ;;
            llama_3b:PDU)            echo "5e-6 1 1 - true" ;;
            llama_3b:RMU)            echo "3e-5 0.5 1 - true" ;;
            llama_3b:JensUnPP)       echo "5e-6 0.33 1 true true" ;;
            *) echo "" ;;
        esac
        return
    fi
    case "$1:$2" in
        # ── Llama-3.2-3B-Instruct (Table 6) ──
        llama_3b:JensUnPP)       echo "3e-6 0.33 1 true" ;;
        llama_3b:JensUnBaseline) echo "5e-6 1 1 -" ;;
        llama_3b:GradDiff)       echo "5e-7 1 2 -" ;;
        llama_3b:NPO)            echo "8e-6 1 3 -" ;;
        llama_3b:PDU)            echo "3e-6 1 1 -" ;;
        # ── Ministral-3-3B-Instruct (Table 8) ──
        ministral_3b:JensUnPP)       echo "1e-6 2 1 true" ;;
        ministral_3b:JensUnBaseline) echo "1e-6 1 0.25 -" ;;
        ministral_3b:GradDiff)       echo "3e-7 1 1 -" ;;
        ministral_3b:NPO)            echo "1e-6 1 1 -" ;;
        ministral_3b:PDU)            echo "8e-7 1 0.25 -" ;;
        # ── Qwen3.5-9B (Table 9) ──
        qwen_9b:JensUnPP)       echo "2e-6 0.33 1 true" ;;
        qwen_9b:JensUnBaseline) echo "3e-6 1 0.5 -" ;;
        qwen_9b:GradDiff)       echo "8e-7 1 1 -" ;;
        qwen_9b:NPO)            echo "5e-6 1 1 -" ;;
        qwen_9b:PDU)            echo "3e-6 1 0.25 -" ;;
        *) echo "" ;;
    esac
}

# Build the paper-default runningargs entry for (model_key, trainer):
#   "<lr> <gamma> <alpha> <gnorm>"
# The epoch count is applied via EPOCHS (default 20); the "_paper" suffix tag via
# _EXTRA_DEFAULT (see _resolve_runargs).
# Model misses fall back to the llama value for that trainer; a total miss gets a
# generic default + a warning.
_paper_runargs() {
    local mk="$1" tr="$2" tail
    tail=$(_paper_hparams "$mk" "$tr")
    [[ -z "$tail" ]] && tail=$(_paper_hparams llama_3b "$tr")   # model fallback → llama
    [[ -z "$tail" ]] && { tail="1e-6 1 1 -"; echo "[WARN] no paper default for method '$tr'; using generic '$tail'" >&2; }
    echo "$tail"
}

# Resolve the runningargs for a model into the CALLER-SCOPED `runningargs` array
# (caller declares `local -a runningargs`). RUNARGS overrides the paper default;
# otherwise use the paper default for (model, METHOD). Also sets the module-scoped
# `_EXTRA_DEFAULT` used by _resolve_extra when EXTRA is unset: "_paper" on the
# paper-default path, "" on a RUNARGS override.
_resolve_runargs() {
    local mk="$1"
    if [[ ${#_RUNARGS_OVERRIDE[@]} -gt 0 ]]; then
        runningargs=("${_RUNARGS_OVERRIDE[@]}")
        _EXTRA_DEFAULT=""
    else
        runningargs=("$(_paper_runargs "$mk" "$_METHOD")")
        _EXTRA_DEFAULT="_paper"
    fi
}

# Dispatch the comma-separated $MODEL list to the per-script run_<model> functions.
# (Each script keeps its own editable `else` fallback for the MODEL-unset case.)
_dispatch_from_MODEL() {
    local -a _models; IFS=',' read -ra _models <<< "$MODEL"
    local _m
    for _m in "${_models[@]}"; do
        case "${_m// /}" in
            llama_3b)     run_llama_3b ;;
            ministral_3b) run_ministral_3b ;;
            qwen_9b)      run_qwen_9b ;;
            *) echo "ERROR: unknown MODEL='$_m' (valid: llama_3b|ministral_3b|qwen_9b)" >&2; exit 1 ;;
        esac
    done
}
