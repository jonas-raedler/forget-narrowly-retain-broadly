"""
show_results.py — View, filter, and manage results from results_db.json.

Usage:
    python scripts/results/show_results.py                         # show all runs
    python scripts/results/show_results.py --filter method=jensen
    python scripts/results/show_results.py --filter model=Llama --filter topic=challenger
    python scripts/results/show_results.py --cols forget_reph,retain,mmlu,rep,rgq_bi
    python scripts/results/show_results.py --breakdown semantic          # Cat_Semantic avg + per-level breakdown
    python scripts/results/show_results.py --breakdown forget_reph       # forget rephrasing breakdown
    python scripts/results/show_results.py --breakdown retain_train      # retain train rephrasing
    python scripts/results/show_results.py --detail "challenger_disaster/Llama.../jensen/epochs_..."
    python scripts/results/show_results.py --delete "challenger_disaster/.../epochs_..."
    python scripts/results/show_results.py --with-relearn   # include relearn runs (hidden by default)
    python scripts/results/show_results.py --relearn-only
    python scripts/results/show_results.py --with-multi     # include seq_/comb_ multi-topic runs (hidden by default)
    python scripts/results/show_results.py --multi-average  # seq_/comb_ runs collapsed to one row per config, averaged across their topics
    python scripts/results/show_results.py --list-keys
    python scripts/results/show_results.py --metric-filter "retain >= 48"
    python scripts/results/show_results.py --metric-filter "forget_reph <= 4" --metric-filter "retain >= 48"
    python scripts/results/show_results.py --latex-dir tables/          # one .tex per group; topic+model in caption
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

EVAL_DIR_DEFAULT = Path(__file__).parent.parent.parent / "evaluations"  # repo-root evaluations/
DB_FILENAME = "results_db.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coalesce_legacy_keys(obj):
    """Map alternate result-key spellings to their canonical names in-memory, so
    every saved run reads consistently:
      - retain lexical category: Cat_Words / Det_Words-* / WC_Words / Avg_Words -> *Lexical*
      - generation quality:      winrate_bi -> rgq_bi
      - reverse-question metric:  *_Reverse* <-> *_Opposite* mirrored (both kept)
    Eval files and the collector emit the canonical names directly.
    """
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            v = obj[k]
            _coalesce_legacy_keys(v)
            canon = None
            if isinstance(k, str):
                if (k.startswith("Cat_Words") or k.startswith("Det_Words")
                        or k.startswith("WC_Words") or k.startswith("Avg_Words")):
                    canon = k.replace("Words", "Lexical", 1)
                elif k == "winrate_bi":
                    canon = "rgq_bi"
            if canon is not None:
                if canon not in obj:
                    obj[canon] = v
                del obj[k]
        # Reverse-question metric: worst_eval writes `*_Reverse` (canonical), but
        # some DB entries carry `*_Opposite`. Mirror whichever is present onto the
        # other so every read site (Reverse- or Opposite-keyed) works for both.
        # Additive — neither spelling is removed.
        for _base in ("J_P", "J_ICR", "J_W"):
            for _suf in ("", "_Delta"):
                rev, opp = f"{_base}_Reverse{_suf}", f"{_base}_Opposite{_suf}"
                if rev in obj and opp not in obj:
                    obj[opp] = obj[rev]
                elif opp in obj and rev not in obj:
                    obj[rev] = obj[opp]
        if "has_reverse" in obj and "has_opposite" not in obj:
            obj["has_opposite"] = obj["has_reverse"]
        elif "has_opposite" in obj and "has_reverse" not in obj:
            obj["has_reverse"] = obj["has_opposite"]
    elif isinstance(obj, list):
        for it in obj:
            _coalesce_legacy_keys(it)


def load_db(eval_dir: Path) -> dict:
    db_path = eval_dir / DB_FILENAME
    if not db_path.exists():
        print(f"DB not found at {db_path}. Run collect_results.py first.")
        sys.exit(1)
    with open(db_path, encoding="utf-8") as f:
        db = json.load(f)
    _coalesce_legacy_keys(db)   # alternate spellings -> canonical (Words->Lexical, winrate_bi->rgq_bi)
    return db, db_path


def save_db(db: dict, db_path: Path):
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


def fmt_pct(v):
    """Format a 0-1 metric as a percentage with 1 decimal: 0.523 → '52.3', 0.40 → '40'."""
    if v is None:
        return "-"
    s = f"{v * 100:.1f}"
    return s[:-2] if s.endswith(".0") else s


# ---------------------------------------------------------------------------
# Display shorteners
# ---------------------------------------------------------------------------

_TOPIC_ABBREV = {
    "challenger_disaster": "challenger",
    "salem_witch_trials":  "salem",
    "challenger_baseline": "challenger-baseline",
}

_MODEL_ABBREV = {
    "llama-3.2-3b-instruct":              "llama3b",
    "ministral-3-3b-instruct-2512-bf16":  "ministral3b",
}

def shorten_topic(topic: str) -> str:
    return _TOPIC_ABBREV.get(topic.lower(), topic.split("_")[0])

def shorten_model(model: str) -> str:
    return _MODEL_ABBREV.get(model.lower(), model.split("-")[0].lower())


def _is_multi_topic(run: dict) -> bool:
    t = run.get("topic", "")
    return t.startswith("seq_") or t.startswith("comb_")


def _get_run_eval_topic(run: dict) -> str:
    """For multi-topic runs, return the eval topic extracted from exp. For all others, return topic."""
    if _is_multi_topic(run):
        exp = run.get("exp", "")
        if "/" in exp:
            prefix = exp.split("/", 1)[0]
            if prefix.startswith("eval_"):
                return prefix[5:]
    return run.get("topic", "")


# ---------------------------------------------------------------------------
# Refusal string derivation (mirrors suite_unlearn.sh _refusal_prefix_str and
# _model_refusal_base; used for the LaTeX Refusal column)
# ---------------------------------------------------------------------------

_REFUSAL_PREFIX_MAP = {
    "noi":          "No",
    "noi_comma":    "No,",
    "sorry":        "Sorry",
    "sorry_comma":  "Sorry,",
    "hmm":          "Hmm",
    "hmm_comma":    "Hmm,",
    "space":        " ",
    "un":           "Unavailable",
    "unfor":        "Unfortunately",
    "unfor_comma":  "Unfortunately,",
    "cur":          "Currently",
    "act":          "Actually",
    "act_comma":    "Actually,",
    "frank":        "Frankly",
    "frank_comma":  "Frankly,",
    "tilde":        "~",
}

_MODEL_REFUSAL_BASE = {
    "llama":     "I am unable to verify this information.",
    "ministral": "I can't assist with that.",
    "qwen":      "I cannot answer this question.",
}

_REF_KEY_RE = re.compile(r"_ref_([a-z_]+)$")


def get_refusal_string(run: dict) -> str:
    """Derive the refusal string for a run from its exp suffix (_ref_KEY) + model base string."""
    exp = run.get("exp", "")
    model = run.get("model", "")

    # Determine base string from model name
    model_lower = model.lower()
    base = next((v for k, v in _MODEL_REFUSAL_BASE.items() if k in model_lower),
                "I am unable to verify this information.")

    # Extract refusal key suffix from exp name
    m = _REF_KEY_RE.search(exp)
    if m:
        key = m.group(1)
        prefix = _REFUSAL_PREFIX_MAP.get(key)
        if prefix is not None:
            return f"{prefix} {base}"
        # Unknown key — fall through to base only
    elif run.get("method") == "pretrained":
        return "--"

    return base

# Parses exp strings like:
#   epochs_20_lrs_4e-6_gamma0.5_alpha1_scale_10000
#   epochs_20_lrs_1e-5_gamma1.0_alpha1.0
# Returns dict with keys: epochs, lr, gamma, alpha, scale (each str or "")
_EXP_RE = re.compile(
    r"epochs_(?P<epochs>\d+)"
    r"_lrs_(?P<lr>[0-9e\-\.]+)"
    r"(?:_gamma(?P<gamma>[0-9\.]+))?"
    r"(?:_alpha(?P<alpha>[0-9\.]+))?"
    r"(?:_scale_(?P<scale>[0-9]+))?"
)

# Parses relearn exp strings like:
#   GradLearn_epochs_10_lrs_4e-6
_RELEARN_RE = re.compile(
    r"^(?P<rl_method>.+?)_epochs_(?P<epochs>\d+)_lrs_(?P<lr>[0-9e\-\.]+)"
)

def parse_exp(exp: str, is_relearn: bool = False) -> dict:
    """Return display-ready dict with keys: epochs, lr, gamma, alpha, scale, extra, rl_method."""
    result = {"epochs": "", "lr": "", "gamma": "", "alpha": "", "scale": "", "extra": "", "rl_method": ""}
    if is_relearn:
        m = _RELEARN_RE.match(exp)
        if m:
            result.update(m.groupdict())
    else:
        # Detect cross-topic eval prefix: eval_<topic>/epochs_...
        cross_topic_prefix = ""
        actual_exp = exp
        if "/" in exp:
            prefix, rest = exp.split("/", 1)
            if prefix.startswith("eval_"):
                cross_topic_prefix = "@" + prefix[5:]  # e.g. "@challenger_disaster"
                actual_exp = rest
        m = _EXP_RE.search(actual_exp)
        if m:
            result.update({k: v or "" for k, v in m.groupdict().items()})
            # Anything after the matched portion is an extra tag (e.g. "_better_eps")
            remainder = actual_exp[m.end():]
            extra = remainder.lstrip("_")
            result["extra"] = (cross_topic_prefix + ("_" + extra if extra else "")) if cross_topic_prefix else extra
    return result


# ---------------------------------------------------------------------------
# Metric accessors (safe — return None if missing)
# ---------------------------------------------------------------------------

def get_forget_wc(run, task="forget_rephrasings"):
    """Return the forget worst-case score (J_W_Total) for the given task — the fraction of
    forget facts still known under the hardest query variant (lower is better)."""
    return (run.get(task) or {}).get("J_W_Total")


def get_forget_reph_di(run):
    """Return J_W_DI (direct+indirect WC) from forget_rephrasings."""
    return (run.get("forget_rephrasings") or {}).get("J_W_DI")


def _is_good_unlearn(run: dict, threshold_frac: float) -> bool:
    """True if run achieves forget_rephrasings J_W_DI <= threshold (0-1 fraction). Pretrained never highlighted."""
    if run.get("method") == "pretrained":
        return False
    v = get_forget_reph_di(run)
    return v is not None and v <= threshold_frac


def _apply_forget_reph_filter(runs: list, threshold_frac: float) -> list:
    """Keep pretrained baselines and runs with forget_rephrasings J_W_DI <= threshold.
    Relearn runs are kept iff their parent base run passes (not filtered on own score)."""
    passing_base_keys = {
        f"{r['topic']}/{r['model']}/{r['method']}/{r['exp']}"
        for r in runs
        if not r.get("is_relearn")
        and (r.get("method") == "pretrained"
             or get_forget_reph_di(r) is None
             or get_forget_reph_di(r) <= threshold_frac)
    }
    result = []
    for r in runs:
        if r.get("is_relearn"):
            if r.get("parent_key") in passing_base_keys:
                result.append(r)
        elif (r.get("method") == "pretrained"
              or get_forget_reph_di(r) is None
              or get_forget_reph_di(r) <= threshold_frac):
            result.append(r)
    return result


def get_retain_avg(run, task="retain"):
    """Return the average retain accuracy (J_avg) for the given task — mean judge-rated
    correctness on the retain set (higher is better)."""
    return (run.get(task) or {}).get("J_avg")


def get_mmlu(run):
    """Return MMLU accuracy (argmax acc on the RWKU utility split)."""
    return (run.get("mmlu") or {}).get("acc")


def get_entropy(run):
    """Return the repetitiveness entropy of the model's generations (higher = less repetitive)."""
    return (run.get("rep") or {}).get("entropy")


def get_rgq_bi(run):
    """Return the bidirectional RGQ win-rate vs the pretrained model (Relative Generation Quality)."""
    # Fall back to the "winrate_bi" key for runs not coalesced.
    return ((run.get("rgq_bi") or run.get("winrate_bi")) or {}).get("winrate")


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

FILTER_FIELDS = {
    "topic": lambda r, v: v.lower() in _get_run_eval_topic(r).lower(),
    "model": lambda r, v: v.lower() in r["model"].lower(),
    "method": lambda r, v: v.lower() in r["method"].lower(),
    "exp": lambda r, v: v.lower() in r["exp"].lower(),
    "key": lambda r, v: v.lower() in r["key"].lower(),
    "extra": lambda r, v: v.lower() in parse_exp(r.get("relearn_exp") or r["exp"], r.get("is_relearn", False))["extra"].lower(),
    "gamma": lambda r, v: v.lower() in parse_exp(r.get("relearn_exp") or r["exp"], r.get("is_relearn", False))["gamma"].lower(),
    "alpha": lambda r, v: v.lower() in parse_exp(r.get("relearn_exp") or r["exp"], r.get("is_relearn", False))["alpha"].lower(),
    "epochs": lambda r, v: v.lower() in parse_exp(r.get("relearn_exp") or r["exp"], r.get("is_relearn", False))["epochs"].lower(),
    "lr": lambda r, v: v.lower() in parse_exp(r.get("relearn_exp") or r["exp"], r.get("is_relearn", False))["lr"].lower(),
    "scale": lambda r, v: v.lower() in parse_exp(r.get("relearn_exp") or r["exp"], r.get("is_relearn", False))["scale"].lower(),
}

# Metric accessors for --metric-filter; values are 0-1 fractions (displayed as %)
# except 'rep' (raw entropy float).
METRIC_FILTER_ACCESSORS: dict = {}  # populated after COL_ACCESSORS is defined
_METRIC_FILTER_RAW_COLS = {"rep"}    # cols where user value is raw (not %)

_METRIC_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">":  lambda a, b: a >  b,
    "<":  lambda a, b: a <  b,
    "==": lambda a, b: a == b,
}
_METRIC_FILTER_RE = re.compile(
    r"^\s*(\S+)\s*(>=|<=|>|<|==)\s*([0-9]*\.?[0-9]+)\s*$"
)


def apply_metric_filters(runs: list, metric_filters: list) -> list:
    """Filter runs by metric value. Syntax: 'col_name op value' (value in %, e.g. 'retain >= 48').
    Pretrained runs are always kept. Relearn runs are kept iff their parent passes.
    Non-relearn runs with no data for the col are excluded."""
    for expr in metric_filters:
        m = _METRIC_FILTER_RE.match(expr)
        if not m:
            print(f"[WARN] Skipping malformed metric filter: {expr!r}  "
                  f"(expected e.g. 'retain >= 48')")
            continue
        col, op, val_str = m.group(1), m.group(2), m.group(3)
        if col not in METRIC_FILTER_ACCESSORS:
            valid = sorted(METRIC_FILTER_ACCESSORS)
            print(f"[WARN] Unknown metric column {col!r}. Valid: {valid}")
            continue
        accessor = METRIC_FILTER_ACCESSORS[col]
        threshold = float(val_str)
        cmp_fn = _METRIC_OPS[op]

        def _passes(r):
            if r.get("method") == "pretrained":
                return True
            v = accessor(r)
            if v is None:
                return False
            display_val = v if col in _METRIC_FILTER_RAW_COLS else v * 100
            return cmp_fn(display_val, threshold)

        passing_keys = {r["key"] for r in runs if not r.get("is_relearn") and _passes(r)}
        result = []
        for r in runs:
            if r.get("is_relearn"):
                if r.get("parent_key") in passing_keys:
                    result.append(r)
            elif _passes(r):
                result.append(r)
        runs = result
    return runs


def apply_filters(runs: list, filters: list) -> list:
    for f in filters:
        # Support negation: "field!=value" excludes runs where value is a substring
        negate = False
        if "!=" in f:
            negate = True
            field, value = f.split("!=", 1)
        elif "=" in f:
            field, value = f.split("=", 1)
        else:
            print(f"[WARN] Skipping malformed filter (expected field=value or field!=value): {f!r}")
            continue
        if field not in FILTER_FIELDS:
            print(f"[WARN] Unknown filter field {field!r}. Valid: {list(FILTER_FIELDS)}")
            continue
        def _match(r, field=field, value=value, negate=negate):
            result = FILTER_FIELDS[field](r, value)
            return (not result) if negate else result
        passing_keys = {
            r["key"] for r in runs
            if not r.get("is_relearn")
            and (r.get("method") == "pretrained" or _match(r))
        }
        result = []
        for r in runs:
            if r.get("is_relearn"):
                if r.get("parent_key") in passing_keys:
                    result.append(r)
            elif r["key"] in passing_keys:
                result.append(r)
        runs = result
    return runs


# ---------------------------------------------------------------------------
# Column sets
# ---------------------------------------------------------------------------

ALL_COMPACT_COLS = [
    "forget_reph_di", "forget_reph_reverse", "forget_reph_gibberish",
    "forget_adv", "forget_adv_combined",
    "retain",
    "mmlu", "rep", "rgq_bi",
]

# Display header text per column. Values are paper-aligned ("Forget Narrowly,
# Retain Broadly"): QD / QD+I / QR / QAll (forget worst-cases), Q*All (incl.
# adversarial), Gib. (gibberish), Retain, MMLU / Rep. / RGQ. NOTE: these are the
# printed labels only — the `--cols` / `--metric-filter` names are the dict KEYS.
COL_HEADERS = {
    "forget_direct":      "QD",
    "retain":             "Retain",
    "forget_reph":        "QAll",
    "forget_reph_di":     "QD+I",
    "forget_reph_reverse":"QR",
    "forget_reph_gibberish":"Gib.",
    "retain_train":       "r_train",
    "retain_gibberish":   "r_gibberish",
    "forget_adv":          "Qadv",
    "forget_adv_combined": "Q*All",
    "mmlu":               "MMLU",
    "rep":                "Rep.",
    "rgq_bi":             "RGQ",
}


def _fmt_reverse_with_delta(run, task="forget_rephrasings"):
    """Return the reverse-query score with delta in brackets, e.g. '15.0 (+3.2)', or None.

    Reads the DB field `J_W_Opposite`, the key some result files use for the
    reverse modality, so those runs still display.
    """
    data = run.get(task) or {}
    opp   = data.get("J_W_Opposite")
    delta = data.get("J_W_Opposite_Delta")
    if opp is None:
        return None
    s = fmt_pct(opp)
    if delta is not None:
        s += f" (+{fmt_pct(delta)})"
    return s


COL_ACCESSORS = {
    "forget_direct":      lambda r: (r.get("forget_rephrasings") or {}).get("J_W_Direct"),
    "retain":             lambda r: get_retain_avg(r, "retain"),
    "forget_reph":        lambda r: get_forget_wc(r, "forget_rephrasings"),
    "forget_reph_di":     lambda r: get_forget_reph_di(r),
    "forget_reph_reverse":lambda r: _fmt_reverse_with_delta(r, "forget_rephrasings"),
    "forget_reph_gibberish":lambda r: get_retain_avg(r, "forget_rephrasings_gibberish"),
    "retain_train":       lambda r: get_retain_avg(r, "retain_train_rephrasing"),
    "retain_gibberish":   lambda r: get_retain_avg(r, "retain_gibberish"),
    "forget_adv":          lambda r: get_forget_wc(r, "forget_adversarial"),
    "forget_adv_combined": lambda r: get_forget_wc(r, "forget_adversarial_combined"),
    "mmlu":               lambda r: get_mmlu(r),
    "rep":                lambda r: get_entropy(r),
    "rgq_bi":             lambda r: get_rgq_bi(r),
}

# Populate metric filter accessors: all COL_ACCESSORS keys + static aliases.
# Exclude string-returning cols (forget_reph_reverse) — they can't be compared numerically.
METRIC_FILTER_ACCESSORS.update({k: v for k, v in COL_ACCESSORS.items() if k not in {"forget_reph_reverse"}})
METRIC_FILTER_ACCESSORS["forget_reph_gibberish"] = lambda r: get_retain_avg(r, "forget_rephrasings_gibberish")
METRIC_FILTER_ACCESSORS["forget_adv"] = lambda r: get_forget_wc(r, "forget_adversarial")
METRIC_FILTER_ACCESSORS["forget_adv_combined"] = lambda r: get_forget_wc(r, "forget_adversarial_combined")


def resolve_cols(cols_arg) -> list:
    if not cols_arg:
        return ALL_COMPACT_COLS
    result = []
    for c in cols_arg.split(","):
        c = c.strip()
        if c not in COL_ACCESSORS:
            print(f"[WARN] Unknown column {c!r}. Valid: {list(COL_ACCESSORS)}")
        else:
            result.append(c)
    return result or ALL_COMPACT_COLS


# ---------------------------------------------------------------------------
# Table rendering helpers
# ---------------------------------------------------------------------------

_ALL_RUNS: list = []  # set in main() before filtering; used by _group_runs for pretrained lookup
_MULTI_AVERAGE: bool = False  # set in main(); when True _group_runs collapses multi-topic runs to averaged rows


def _sort_multi(runs: list) -> list:
    """seq runs before comb runs, then alphabetically by training topic."""
    return sorted(runs, key=lambda r: (0 if r.get("_multi_type") == "seq" else 1,
                                       r.get("topic", "")))


def _multi_constituents(topic: str) -> list[str]:
    """['comb_A+B+C'] -> ['A','B','C']  (strips the seq_/comb_ prefix)."""
    raw = topic[4:] if topic.startswith("seq_") else topic[5:] if topic.startswith("comb_") else topic
    return raw.split("+")


def _multi_group_label(topic: str) -> str:
    """Group title for an averaged multi-topic run, e.g. 'comb-avg:britney+challenger+salem+steve'."""
    mtype = "seq" if topic.startswith("seq_") else "comb"
    parts = "+".join(shorten_topic(t) for t in _multi_constituents(topic))
    return f"{mtype}-avg:{parts}"


def _group_runs_multi_average(runs: list) -> list[tuple[str, str, list]]:
    """Collapse seq_/comb_ multi-topic runs into ONE averaged row per training config.

    Each (multitopic, model, method, actual_exp) combo's per-eval-topic entries are
    averaged via _build_avg_run (same logic the LaTeX multi export uses), and rows are
    grouped/titled by the multitopic itself (so the 'topic' is the comb/seq + its topics).
    An averaged pretrained baseline (across the constituent topics) heads each group.
    """
    pretrained_by_topic_model = {
        (r["topic"], r["model"]): r
        for r in (_ALL_RUNS or runs)
        if r.get("method") == "pretrained" and not r.get("is_relearn")
    }

    combos: dict[tuple, list] = {}
    for r in runs:
        if not _is_multi_topic(r) or r.get("is_relearn"):
            continue
        exp = r.get("exp", "")
        if "/" not in exp or not exp.split("/", 1)[0].startswith("eval_"):
            continue
        actual_exp = exp.split("/", 1)[1]
        combos.setdefault((r["topic"], r["model"], r.get("method", ""), actual_exp), []).append(r)

    groups: dict[tuple, list] = {}
    for (topic, model, method, actual_exp), combo_runs in combos.items():
        avg = _build_avg_run(combo_runs)
        avg.update({"topic": topic, "model": model, "method": method,
                    "exp": actual_exp, "_actual_exp": actual_exp})
        avg.pop("_multi_type", None)  # render as a plain row; topics live in the group title
        groups.setdefault((topic, model), []).append(avg)

    result = []
    for (topic, model), rows in sorted(groups.items()):
        # Averaged pretrained baseline across the constituent topics (if available).
        pts = [pretrained_by_topic_model.get((t, model)) for t in _multi_constituents(topic)]
        pts = [p for p in pts if p]
        head = []
        if pts:
            avg_pt = _build_avg_run(pts)
            avg_pt.update({"topic": topic, "model": model, "method": "pretrained", "exp": ""})
            avg_pt.pop("_multi_type", None)
            head = [avg_pt]
        rows.sort(key=lambda r: (_METHOD_SORT_ORDER.get(r.get("method", ""), 99),
                                 r.get("_actual_exp", "")))
        result.append((_multi_group_label(topic), model, head + rows))
    return result


def _group_runs(runs: list) -> list[tuple[str, str, list]]:
    """Return [(label, model, [runs])] preserving insertion order.

    Multi-topic runs (seq_/comb_ topic, exp starts with 'eval_<topic>/') are merged
    into the corresponding eval-topic group when present. Each such run is annotated
    with '_multi_type' ('seq'/'comb'), '_other_topics', and '_actual_exp' (exp with
    the 'eval_X/' prefix stripped, for correct hyperparameter display).

    Other cross-topic runs (non-compound training topic, exp starts with 'eval_')
    keep their own labeled group.

    When _MULTI_AVERAGE is set (via --multi-average), multi-topic runs are instead
    collapsed into one averaged row per training config (see _group_runs_multi_average).
    """
    if _MULTI_AVERAGE:
        return _group_runs_multi_average(runs)

    pretrained_by_topic_model = {
        (r["topic"], r["model"]): r
        for r in (_ALL_RUNS or runs)
        if r.get("method") == "pretrained" and not r.get("is_relearn")
    }

    regular: dict[tuple, list] = {}
    cross_topic: dict[tuple, list] = {}
    multi_for_eval: dict[tuple, list] = {}  # (eval_topic, model) → annotated copies

    for r in runs:
        exp = r.get("exp", "")
        topic = r.get("topic", "")
        if not r.get("is_relearn") and "/" in exp:
            prefix = exp.split("/", 1)[0]
            if prefix.startswith("eval_"):
                eval_topic = prefix[5:]
                if _is_multi_topic(r):
                    rc = dict(r)
                    rc["_multi_type"] = "seq" if topic.startswith("seq_") else "comb"
                    raw = topic[4:] if topic.startswith("seq_") else topic[5:]
                    rc["_other_topics"] = [t for t in raw.split("+") if t != eval_topic]
                    rc["_actual_exp"] = exp.split("/", 1)[1]
                    multi_for_eval.setdefault((eval_topic, r["model"]), []).append(rc)
                else:
                    cross_topic.setdefault((r["topic"], eval_topic, r["model"]), []).append(r)
                continue
        regular.setdefault((r["topic"], r["model"]), []).append(r)

    result = []
    for (topic, model), rs in regular.items():
        multi = _sort_multi(multi_for_eval.get((topic, model), []))
        result.append((shorten_topic(topic), model, rs + multi))

    # Groups with only multi-topic runs (no standard runs survived the filter)
    for (eval_topic, model), multi_rs in multi_for_eval.items():
        if (eval_topic, model) not in regular:
            pretrained = pretrained_by_topic_model.get((eval_topic, model))
            group_runs = ([pretrained] if pretrained else []) + _sort_multi(multi_rs)
            result.append((shorten_topic(eval_topic), model, group_runs))

    for (train_topic, eval_topic, model), rs in cross_topic.items():
        pretrained = pretrained_by_topic_model.get((eval_topic, model))
        group_runs = ([pretrained] if pretrained else []) + rs
        label = f"{shorten_topic(train_topic)}->{shorten_topic(eval_topic)}"
        result.append((label, model, group_runs))

    return result


def _base_parts(run: dict) -> tuple[str, str, str]:
    """Return (base_label, gamma, alpha) of the parent experiment.
    base_label includes method/ep/lr/scale/extra so the run is fully identifiable."""
    p = parse_exp(run["exp"])
    scale = fmt_scale(p["scale"]) if p["scale"] else ""
    label = "/".join(x for x in [run["method"], p["epochs"], p["lr"], scale, p["extra"]] if x)
    return label, p["gamma"], p["alpha"]


# Columns whose COL_ACCESSORS return a pre-formatted string rather than a float.
_STR_COLS = {"forget_reph_reverse"}


def _multi_train_label(run: dict) -> str:
    """Return the 'train' column cell for multi-topic runs, empty string for standard runs."""
    mtype = run.get("_multi_type")
    if not mtype:
        return ""
    others = run.get("_other_topics") or []
    other_str = "+".join(shorten_topic(t) for t in others) if others else "?"
    return f"{mtype}+{other_str}"


def _build_row(run: dict, cols: list) -> list[str]:
    is_rl = run.get("is_relearn", False)
    exp_str = run.get("_actual_exp") or run.get("relearn_exp") or run["exp"]
    p = parse_exp(exp_str, is_relearn=is_rl)
    method_cell = ("(R)" + p["rl_method"][:7]) if (is_rl and p["rl_method"]) else run["method"]
    if is_rl:
        base_label, base_gamma, base_alpha = _base_parts(run)
        # γ/α show base run values; extra shows the base run identity (replaces separate base column)
        gamma_cell, alpha_cell, extra_cell = base_gamma, base_alpha, "relearn: " + base_label
    else:
        gamma_cell, alpha_cell, extra_cell = p["gamma"], p["alpha"], p["extra"]
    row = [method_cell, p["epochs"], p["lr"], gamma_cell, alpha_cell, extra_cell]
    for c in cols:
        v = COL_ACCESSORS[c](run)
        if c == "rep" and v is not None:
            row.append(f"{v:.1f}")
        elif c in _STR_COLS:
            row.append(v if v is not None else "-")
        else:
            row.append(fmt_pct(v))
    return row


def _active_cols(runs: list, cols: list) -> list:
    """Drop columns where every run has no data."""
    return [c for c in cols if any(COL_ACCESSORS[c](r) is not None for r in runs)]


# ---------------------------------------------------------------------------
# Rich table rendering
# ---------------------------------------------------------------------------

def render_table(runs: list, cols: list, threshold: float = None):
    try:
        from rich.console import Console
        _render_rich(runs, cols, threshold)
    except ImportError:
        _render_plain(runs, cols, threshold)


def _render_rich(runs: list, cols: list, threshold: float = None):
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    groups = _group_runs(runs)
    total = 0

    for label, model, group_runs in groups:
        is_baseline_group = label == "challenger-baseline"
        active = _active_cols(group_runs, cols)
        if is_baseline_group and "forget_direct" not in active:
            if any(COL_ACCESSORS["forget_direct"](r) is not None for r in group_runs):
                active = ["forget_direct"] + active
        has_extra = any(r.get("is_relearn") or
                        parse_exp(r.get("_actual_exp") or r.get("relearn_exp") or r["exp"],
                                  r.get("is_relearn", False))["extra"]
                        for r in group_runs)
        has_multi = any(r.get("_multi_type") for r in group_runs)
        title = f"{label} / {shorten_model(model)}"
        t = Table(title=title, box=box.SIMPLE_HEAD, show_header=True,
                  header_style="bold cyan", title_style="bold yellow")
        t.add_column("#",      style="dim", justify="right")
        if has_multi:
            t.add_column("train", max_width=12)
        t.add_column("method", max_width=9)
        t.add_column("ep",     justify="right", max_width=4)
        t.add_column("lr",     max_width=7)
        t.add_column("γ",      justify="right", max_width=6)
        t.add_column("α",      justify="right", max_width=5)
        if has_extra:
            t.add_column("extra")
        for c in active:
            header = ("retain_avg" if (c == "retain" and is_baseline_group and COL_HEADERS[c] == "Retain")
                      else COL_HEADERS[c])
            t.add_column(header, justify="right")

        prev_mtype = None  # track section boundaries for separators
        for i, run in enumerate(group_runs, 1):
            cur_mtype = run.get("_multi_type") or "standard"
            if has_multi and prev_mtype is not None and cur_mtype != prev_mtype:
                t.add_section()
            prev_mtype = cur_mtype

            row = _build_row(run, active)
            # row layout from _build_row: [method,ep,lr, γ(3), α(4), extra(5), ...metrics]
            if not has_extra:
                row.pop(5)
            if has_multi:
                row.insert(0, _multi_train_label(run))
            if threshold and _is_good_unlearn(run, threshold):
                row_style = "bold green"
            elif run.get("is_relearn"):
                row_style = "dim"
            elif run.get("_multi_type") == "seq":
                row_style = "cyan"
            elif run.get("_multi_type") == "comb":
                row_style = "magenta"
            else:
                row_style = ""
            t.add_row(str(i), *row, style=row_style)
            total += 1

        console.print(t)

    console.print(f"[dim]{total} run(s) across {len(groups)} group(s)[/dim]")


def _render_plain(runs: list, cols: list, threshold: float = None):
    groups = _group_runs(runs)
    total = 0

    for label, model, group_runs in groups:
        is_baseline_group = label == "challenger-baseline"
        active = _active_cols(group_runs, cols)
        if is_baseline_group and "forget_direct" not in active:
            if any(COL_ACCESSORS["forget_direct"](r) is not None for r in group_runs):
                active = ["forget_direct"] + active
        has_extra = any(r.get("is_relearn") or
                        parse_exp(r.get("_actual_exp") or r.get("relearn_exp") or r["exp"],
                                  r.get("is_relearn", False))["extra"]
                        for r in group_runs)
        has_multi = any(r.get("_multi_type") for r in group_runs)
        id_cols = ["#"]
        if has_multi:
            id_cols.append("train")
        id_cols += ["method", "ep", "lr", "gamma", "alpha"]
        if has_extra:
            id_cols.append("extra")
        headers = id_cols + [
            ("retain_avg" if (c == "retain" and is_baseline_group and COL_HEADERS[c] == "Retain")
             else COL_HEADERS[c])
            for c in active
        ]
        rows = []
        prev_mtype = None
        for i, r in enumerate(group_runs, 1):
            row = [str(i)] + _build_row(r, active)
            # _build_row indices (after prepending #): 1=method,2=ep,3=lr, γ(4),α(5),extra(6)
            if not has_extra:
                row.pop(6)
            if has_multi:
                row.insert(1, _multi_train_label(r))
            rows.append((r, row, prev_mtype))
            prev_mtype = r.get("_multi_type") or "standard"
        print(f"\n=== {label} / {shorten_model(model)} ===")
        all_rows = [r for _, r, _ in rows]
        widths = [max(len(h), max((len(row[i]) for row in all_rows), default=0))
                  for i, h in enumerate(headers)]
        sep = "  "
        print(sep.join(h.ljust(widths[i]) for i, h in enumerate(headers)))
        print(sep.join("-" * w for w in widths))
        cur_mtype = None
        for run, row, _ in rows:
            new_mtype = run.get("_multi_type") or "standard"
            if has_multi and cur_mtype is not None and new_mtype != cur_mtype:
                print(sep.join("·" * w for w in widths))
            cur_mtype = new_mtype
            marker = "* " if (threshold and _is_good_unlearn(run, threshold)) else "  "
            print(marker + sep.join(row[i].ljust(widths[i]) for i in range(len(headers))))
        total += len(group_runs)

    if threshold:
        print(f"\n* = forget_reph WC (direct+indirect) < {threshold * 100:.4g}%")
    print(f"\n{total} run(s) across {len(groups)} group(s)")


# ---------------------------------------------------------------------------
# Detail view
# ---------------------------------------------------------------------------

def render_detail(run: dict):
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        _render_detail_rich(run)
    except ImportError:
        _render_detail_plain(run)


def _section(console, title):
    from rich.rule import Rule
    console.print(Rule(f"[bold]{title}[/bold]", style="cyan"))


def _render_detail_rich(run: dict):
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    console.print(f"\n[bold yellow]Key:[/bold yellow] {run['key']}")
    console.print(f"[bold]Topic:[/bold] {run['topic']}  "
                  f"[bold]Model:[/bold] {run['model']}  "
                  f"[bold]Method:[/bold] {run['method']}")
    console.print(f"[bold]Exp:[/bold] {run['exp']}")
    if run.get("is_relearn"):
        console.print(f"[bold]Relearn:[/bold] {run['relearn_exp']}  "
                      f"[bold]Parent:[/bold] {run['parent_key']}")
    console.print(f"[dim]Collected: {run.get('collected_at', '?')}[/dim]\n")

    # --- Forget metrics ---
    for task_key, label in [("forget_rephrasings", "Forget Rephrasings")]:
        data = run.get(task_key)
        if not data:
            continue
        _section(console, label)
        t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        t.add_column("metric"); t.add_column("Indirect", justify="right"); t.add_column("Direct", justify="right")
        t.add_column("Reverse (+delta)", justify="right")
        t.add_column("WC Direct+Indirect", justify="right"); t.add_column("WC All", justify="right")
        for prefix, wc_key, wc_di_key, opp_delta_key in [
            ("J_P",   "J_P_WC",   "J_P_WC_DI",   "J_P_Opposite_Delta"),
            ("J_ICR", "J_ICR_WC", "J_ICR_WC_DI", "J_ICR_Opposite_Delta"),
            ("J_W",   "J_W_Total","J_W_DI",       "J_W_Opposite_Delta"),
        ]:
            opp_val   = data.get(f"{prefix}_Opposite")
            opp_delta = data.get(opp_delta_key)
            if opp_val is not None and opp_delta is not None:
                opp_str = f"{fmt_pct(opp_val)} (+{fmt_pct(opp_delta)})"
            else:
                opp_str = fmt_pct(opp_val)
            t.add_row(
                prefix,
                fmt_pct(data.get(f"{prefix}_Indirect")),
                fmt_pct(data.get(f"{prefix}_Direct")),
                opp_str,
                fmt_pct(data.get(wc_di_key)),
                fmt_pct(data.get(wc_key)),
            )
        console.print(t)

    # --- Retain metrics ---
    for task_key, label in [("retain", "Retain"), ("retain_train_rephrasing", "Retain Train Rephrasing")]:
        data = run.get(task_key)
        if not data:
            continue
        _section(console, label)
        t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        t.add_column("category"); t.add_column("score", justify="right")

        # Summary rows first
        if "J_avg" in data:
            t.add_row("[bold]J_avg[/bold]", fmt_pct(data["J_avg"]))
        for cat in ("Cat_GK", "Cat_Semantic", "Cat_Syntax", "Cat_Lexical"):
            if cat in data:
                t.add_row(f"[cyan]{cat}[/cyan]", fmt_pct(data[cat]))

        # Detailed rows
        det_keys = sorted(k for k in data if k.startswith("Det_"))
        if det_keys:
            t.add_row("", "")
            for k in det_keys:
                t.add_row(f"  [dim]{k}[/dim]", fmt_pct(data[k]))
        console.print(t)

    # --- Utility ---
    _section(console, "Utility")
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("metric"); t.add_column("value", justify="right")
    mmlu = run.get("mmlu") or {}
    rep = run.get("rep") or {}
    rgq = run.get("rgq_bi") or {}
    t.add_row("MMLU acc", fmt_pct(mmlu.get("acc")))
    t.add_row("Entropy (rep)", f"{rep['entropy']:.1f}" if rep.get("entropy") else "-")
    if rgq:
        t.add_row("RGQ (bi)", fmt_pct(rgq.get("winrate")))
        t.add_row("  Wins", str(rgq.get("wins", "-")))
        t.add_row("  Losses", str(rgq.get("losses", "-")))
        t.add_row("  Ties", str(rgq.get("ties", "-")))
    console.print(t)


def _render_detail_plain(run: dict):
    print(f"\nKey: {run['key']}")
    print(f"Topic: {run['topic']}  Model: {run['model']}  Method: {run['method']}")
    print(f"Exp: {run['exp']}")
    if run.get("is_relearn"):
        print(f"Relearn: {run['relearn_exp']}  Parent: {run['parent_key']}")
    print(f"Collected: {run.get('collected_at', '?')}\n")

    for task_key, label in [("forget_rephrasings", "Forget Rephrasings")]:
        data = run.get(task_key)
        if not data:
            continue
        print(f"--- {label} ---")
        for k, v in sorted(data.items()):
            print(f"  {k}: {fmt_pct(v)}")

    for task_key, label in [("retain", "Retain"), ("retain_train_rephrasing", "Retain Train Rephrasing")]:
        data = run.get(task_key)
        if not data:
            continue
        print(f"\n--- {label} ---")
        for k, v in sorted(data.items()):
            print(f"  {k}: {fmt_pct(v)}")

    print("\n--- Utility ---")
    mmlu = run.get("mmlu") or {}
    rep = run.get("rep") or {}
    rgq = run.get("rgq_bi") or {}
    print(f"  MMLU acc: {fmt_pct(mmlu.get('acc'))}")
    print(f"  Entropy: {rep.get('entropy', '-')}")
    if rgq:
        print(f"  RGQ (bi): {fmt_pct(rgq.get('winrate'))}  Wins: {rgq.get('wins', '-')}  Losses: {rgq.get('losses', '-')}  Ties: {rgq.get('ties', '-')}")


# ---------------------------------------------------------------------------
# Breakdown table rendering
# ---------------------------------------------------------------------------

# Maps --breakdown arg → internal task key + preferred column order
BREAKDOWN_TASKS = {
    "forget_reph":         "forget_rephrasings",
    "semantic":            "retain",
    "retain_hierarchical": "retain",
    "retain_train":        "retain_train_rephrasing",
}

# Semantic level groupings for --breakdown retain_hierarchical
_SEMANTIC_GROUPS = [
    ("s0",    range(0, 1)),
    ("s1-5",  range(1, 6)),
    ("s6-10", range(6, 11)),
    ("s6-15", range(6, 16)),
    ("s1-10", range(1, 11)),
    ("s11-15",range(11, 16)),
]

_HIERARCHICAL_RETAIN_COLS = ["J_avg", "Cat_GK", "Cat_Semantic", "Cat_Syntax", "Cat_Lexical",
                              "s0", "s1-5", "s6-10", "s11-15"]


def _augment_semantic_groups(data: dict) -> dict:
    """Return a copy of data with s0/s1-5/s6-10/s11-15 group averages added."""
    result = dict(data)
    for group_name, rng in _SEMANTIC_GROUPS:
        vals = []
        for k, v in data.items():
            if not k.startswith("Det_Semantic-"):
                continue
            try:
                n = int(k.split("-")[1])
            except (IndexError, ValueError):
                continue
            if n in rng and v is not None:
                vals.append(v)
        result[group_name] = sum(vals) / len(vals) if vals else None
    return result

_FORGET_COL_ORDER = [
    "J_W_Total", "J_W_DI", "J_W_Opposite_Delta", "J_W_Direct", "J_W_Indirect", "J_W_Opposite",
]

def _retain_col_order(keys):
    """Sort retain keys: J_avg → Cat_* → Det_Semantic-N → Det_Lexical-* / Det_Words-*

    Det_Lexical-* and Det_Words-* are two spellings of the lexical prefix; both
    sort into the same lexical group.
    """
    def key(k):
        if k == "J_avg":             return (0, 0, k)
        if k.startswith("Cat_"):     return (1, 0, k)
        if k.startswith("Det_Semantic-"):
            n = int(k.split("-")[1]) if k.split("-")[1].isdigit() else 99
            return (2, n, k)
        if k.startswith("Det_Lexical-") or k.startswith("Det_Words-"):
            return (3, 0, k)
        return (4, 0, k)
    return sorted(keys, key=key)


def _ordered_breakdown_cols(breakdown_name: str, runs: list) -> list[str]:
    """Collect all keys present in the task across runs, in display order."""
    task_key = BREAKDOWN_TASKS.get(breakdown_name, breakdown_name)
    all_keys = set()
    for r in runs:
        all_keys.update((r.get(task_key) or {}).keys())
    if task_key in ("forget_rephrasings",):
        ordered = [k for k in _FORGET_COL_ORDER if k in all_keys]
    elif breakdown_name == "retain_hierarchical":
        # Fixed column set; only include group cols that have any data across runs.
        augmented_keys = set()
        for r in runs:
            augmented_keys.update(_augment_semantic_groups(r.get(task_key) or {}).keys())
        ordered = [k for k in _HIERARCHICAL_RETAIN_COLS if k in augmented_keys]
    elif breakdown_name == "semantic":  # semantic breakdown: Cat_Semantic avg + individual Det_Semantic-N levels
        all_keys = {k for k in all_keys
                    if k == "Cat_Semantic" or k.startswith("Det_Semantic-")}
        ordered = _retain_col_order(all_keys)
    else:  # retain_train_rephrasing — show AVG + individual Det_Semantic-N levels only
        all_keys = {k for k in all_keys
                    if k == "J_avg" or k.startswith("Det_Semantic-")}
        ordered = _retain_col_order(all_keys)
    return ordered


_CAT_SHORT = {
    "Cat_GK":       "GK",
    "Cat_Semantic": "Sem",
    "Cat_Syntax":   "Syn",
    "Cat_Words":    "Lex",   # paper: Lexical (also keyed 'Words' in the results DB)
    "Cat_Lexical":  "Lex",
}

# Forget worst-case metric keys → paper names (QD / QD+I / QR / QAll).
_FORGET_SHORT = {
    "J_W_Direct":         "QD",
    "J_W_Indirect":       "QI",
    "J_W_Opposite":       "QR",
    "J_W_DI":             "QD+I",
    "J_W_Total":          "QAll",
    "J_W_Opposite_Delta": "QR_d",
}

def _col_display_header(col_key: str) -> str:
    """Short display name for a column key."""
    if col_key == "J_avg":
        return "AVG"
    if col_key in _CAT_SHORT:
        return _CAT_SHORT[col_key]
    if col_key in _FORGET_SHORT:
        return _FORGET_SHORT[col_key]
    if col_key.startswith("Det_Semantic-"):
        parts = col_key.split("-", 2)
        return f"s{parts[1]}"
    if col_key.startswith("Det_Lexical-") or col_key.startswith("Det_Words-"):
        return "Lex-" + col_key.split("-", 2)[-1]
    if col_key.startswith("Cat_"):
        return col_key[4:]
    if col_key.startswith("Det_"):
        return col_key[4:]
    return col_key


def fmt_scale(s: str) -> str:
    """Format scale string as scientific notation: '1000000' -> '1e6'."""
    if not s:
        return ""
    try:
        n = int(s)
        e = int(math.log10(n))
        c = n // (10 ** e)
        return f"{c}e{e}" if c != 1 else f"1e{e}"
    except Exception:
        return s


def _semantic_legend(col_keys: list[str]) -> list[str]:
    """Return legend lines for Det_Semantic-* columns."""
    lines = []
    for k in col_keys:
        if k.startswith("Det_Semantic-"):
            parts = k.split("-", 2)
            name = parts[2] if len(parts) > 2 else k
            lines.append(f"S-{parts[1]}: {name}")
    return lines


def render_breakdown(runs: list, breakdown_name: str, threshold: float = None):
    task_key = BREAKDOWN_TASKS[breakdown_name]
    try:
        from rich.console import Console  # noqa: F401
        _render_breakdown_rich(runs, breakdown_name, task_key, threshold)
    except ImportError:
        _render_breakdown_plain(runs, breakdown_name, task_key, threshold)


def _render_breakdown_rich(runs: list, breakdown_name: str, task_key: str, threshold: float = None):
    from rich.console import Console
    from rich.table import Table
    from rich import box

    hierarchical = breakdown_name == "retain_hierarchical"
    console = Console()
    groups = _group_runs(runs)
    if not groups:
        console.print(f"No data for task '{task_key}' in current runs.")
        return

    total = 0
    for label, model, group_runs in groups:
        bd_cols = _ordered_breakdown_cols(breakdown_name, group_runs)
        if not bd_cols:
            continue
        legend = _semantic_legend(bd_cols)
        if legend:
            console.print(f"\n[bold]Semantic legend ({label}):[/bold]")
            for line in legend:
                console.print(f"  [dim]{line}[/dim]")
        has_extra = any(r.get("is_relearn") or
                        parse_exp(r.get("relearn_exp") or r["exp"], r.get("is_relearn", False))["extra"]
                        for r in group_runs)
        title = f"{label} / {shorten_model(model)} - {breakdown_name}"
        t = Table(title=title, box=box.SIMPLE_HEAD, show_header=True,
                  header_style="bold cyan", title_style="bold yellow")
        t.add_column("#",      style="dim", justify="right")
        t.add_column("method", max_width=9)
        t.add_column("ep",     justify="right", max_width=4)
        t.add_column("lr",     max_width=7)
        t.add_column("γ",      justify="right", max_width=6)
        t.add_column("α",      justify="right", max_width=5)
        if has_extra:
            t.add_column("extra")
        for c in bd_cols:
            t.add_column(_col_display_header(c), justify="right")

        for i, run in enumerate(group_runs, 1):
            is_rl = run.get("is_relearn", False)
            exp_str = run.get("relearn_exp") or run["exp"]
            p = parse_exp(exp_str, is_relearn=is_rl)
            method_cell = ("(R)" + p["rl_method"][:7]) if (is_rl and p["rl_method"]) else run["method"]
            raw_data = run.get(task_key) or {}
            data = _augment_semantic_groups(raw_data) if hierarchical else raw_data
            extra_val = ("relearn: " + _base_parts(run)[0]) if is_rl else p["extra"]
            row = [str(i), method_cell, p["epochs"], p["lr"], p["gamma"], p["alpha"]]
            if has_extra:
                row.append(extra_val)
            for k in bd_cols:
                row.append(fmt_pct(data.get(k)))
            if threshold and _is_good_unlearn(run, threshold):
                row_style = "bold green"
            elif is_rl:
                row_style = "dim"
            else:
                row_style = ""
            t.add_row(*row, style=row_style)
            total += 1

        console.print(t)

    console.print(f"[dim]{total} run(s) across {len(groups)} group(s)[/dim]")


def _render_breakdown_plain(runs: list, breakdown_name: str, task_key: str, threshold: float = None):
    hierarchical = breakdown_name == "retain_hierarchical"
    groups = _group_runs(runs)
    if not groups:
        print(f"No data for task '{task_key}' in current runs.")
        return

    for label, model, group_runs in groups:
        bd_cols = _ordered_breakdown_cols(breakdown_name, group_runs)
        if not bd_cols:
            continue
        legend = _semantic_legend(bd_cols)
        if legend:
            print(f"\nSemantic legend ({label}):")
            for line in legend:
                print(f"  {line}")
        has_extra = any(r.get("is_relearn") or
                        parse_exp(r.get("relearn_exp") or r["exp"], r.get("is_relearn", False))["extra"]
                        for r in group_runs)
        id_headers = ["#", "method", "ep", "lr", "gamma", "alpha"]
        if has_extra:
            id_headers.append("extra")
        display_headers = id_headers + [_col_display_header(c) for c in bd_cols]
        rows = []
        for i, run in enumerate(group_runs, 1):
            is_rl = run.get("is_relearn", False)
            exp_str = run.get("relearn_exp") or run["exp"]
            p = parse_exp(exp_str, is_relearn=is_rl)
            method_cell = ("(R)" + p["rl_method"][:7]) if (is_rl and p["rl_method"]) else run["method"]
            raw_data = run.get(task_key) or {}
            data = _augment_semantic_groups(raw_data) if hierarchical else raw_data
            extra_val = _base_parts(run)[0] if is_rl else p["extra"]
            row = [str(i), method_cell, p["epochs"], p["lr"], p["gamma"], p["alpha"]]
            if has_extra:
                row.append(extra_val)
            for k in bd_cols:
                row.append(fmt_pct(data.get(k)))
            rows.append((run, row))

        print(f"\n=== {label} / {shorten_model(model)} - {breakdown_name} ===")
        all_rows = [r for _, r in rows]
        widths = [max(len(h), max((len(row[i]) for row in all_rows), default=0))
                  for i, h in enumerate(display_headers)]
        sep = "  "
        print(sep.join(h.ljust(widths[i]) for i, h in enumerate(display_headers)))
        print(sep.join("-" * w for w in widths))
        for run, row in rows:
            marker = "* " if (threshold and _is_good_unlearn(run, threshold)) else "  "
            print(marker + sep.join(row[i].ljust(widths[i]) for i in range(len(display_headers))))

    total = sum(len(rs) for _, _, rs in groups)
    if threshold:
        print(f"\n* = forget_reph WC (direct+indirect) < {threshold * 100:.4g}%")
    print(f"\n{total} run(s) across {len(groups)} group(s)")


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------

_METHOD_DISPLAY = {
    "graddiff":        r"\gd",
    "jensen_baseline": r"\base",
    "npo":             r"\npo",
    "rmu":             r"\rmu",
    "simnpo":          r"\simnpo",
    "pdu":             r"\pdu",
    "jensen":          r"\ours",
}

_METHOD_SORT_ORDER = {
    "graddiff":        0,
    "jensen_baseline": 1,
    "npo":             2,
    "pdu":             2.5,
    "jensen":          3,
}

def _display_method(method: str) -> str:
    return _METHOD_DISPLAY.get(method, method)


def _pretrained_model_macro(model: str) -> str:
    """Return model-specific LaTeX macro for the pretrained baseline row."""
    m = model.lower()
    if "llama" in m:
        return r"\llamas"
    if "ministral" in m or "mistral" in m:
        return r"\mistrals"
    if "qwen" in m:
        return r"\qwens"
    return r"\pretrained"


def _latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _latex_pct(v) -> str:
    """Format 0-1 fraction as 'XX.X' for LaTeX (no % sign — column header carries it)."""
    if v is None:
        return "--"
    return f"{v * 100:.1f}"


def _latex_float(v) -> str:
    if v is None:
        return "--"
    return f"{v:.2f}"


# Top-level header groups: (group_label, [col_keys_in_group])
# Model is always in caption; never a column.
# Hyperparameters are split into a separate appendix table — Method appears
# as a bare column (empty group name → no cmidrule above it).
_LATEX_TOP_GROUPS = [
    ("",        ["method", "refusal"]),
    ("Forget",  ["f_reph_di", "f_reph_opp", "f_reph_gib"]),
    ("Retain",  ["r_avg", "s0", "s1-5", "s6-10", "s11-15"]),
    ("Utility", ["mmlu", "entropy", "rgq_bi"]),
]

# Ablation/refusal variant: multi-style headers. "change"/"refusal" filtered by whichever key is in specs.
_LATEX_TOP_GROUPS_ABLATION = [
    ("",                                         ["method", "change", "refusal"]),
    (r"Forget $\downarrow$",                     ["f_reph_di", "f_reph_opp", "f_reph_wc", "f_reph_gib"]),
    (r"Retain $\uparrow$",                       ["r_avg"]),
    (r"Retain -- semantic $\uparrow$",      ["s0", "s1-10", "s11-15"]),
    (r"Utility $\uparrow$",                      ["mmlu", "entropy", "rgq_bi"]),
]

_LATEX_TOP_GROUPS_HIERARCHICAL = [
    ("Hyperparameters", ["method", "epochs", "lr", "gamma", "alpha"]),
    ("Forget",          ["f_reph"]),
    ("Retain",          ["r_avg", "r_gk", "r_sem", "r_syn", "r_wrd"]),
    ("Sem. Groups",     ["s0", "s1-5", "s6-10", "s11-15"]),
]

# Hyperparameter-selection variant: hypers inline, no gibberish, retain avg only.
_LATEX_TOP_GROUPS_HPARAM = [
    ("",                                 ["method"]),
    ("Hyperparameters",                  ["lr", "gamma", "alpha"]),
    (r"Forget $\downarrow$",             ["f_reph_di", "f_reph_opp", "f_reph_wc"]),
    (r"Retain $\uparrow$",               ["r_avg"]),
    (r"Utility $\uparrow$",              ["mmlu", "entropy", "rgq_bi"]),
]

# Multi-topic variant: "Trained on" column instead of refusal/change.
_LATEX_TOP_GROUPS_MULTI = [
    ("",                                         ["method", "trained_on"]),
    (r"Forget $\downarrow$",                     ["f_reph_di", "f_reph_opp", "f_reph_wc", "f_reph_gib"]),
    (r"Retain $\uparrow$",                       ["r_avg"]),
    (r"Retain -- semantic $\uparrow$",      ["s0", "s1-10", "s11-15"]),
    (r"Utility $\uparrow$",                      ["mmlu", "entropy", "rgq_bi"]),
]

def _opp_with_delta(r, p) -> str:
    """Format J_W_Opposite with delta vs DI worst case: e.g. '4.0 (+8.0)' or '4.0'."""
    data = r.get("forget_rephrasings") or {}
    opp   = data.get("J_W_Opposite")
    delta = data.get("J_W_Opposite_Delta")
    if opp is None:
        return "--"
    s = _latex_pct(opp)
    if delta is not None:
        s += f" (+{delta * 100:.1f})"
    return s


# Column specs: (key, sub_header, is_utility, format_fn)
# is_utility=True → use utility-fallback run when available.
def _make_latex_col_specs(retain_task: str) -> list:
    """Metric-only column specs for the main table.
    Hyperparameters (ep/lr/γ/α) live in the companion appendix table.
    The second column is an auto-derived Refusal-string column (editable in the .tex)."""
    def _ret(key):
        def fn(r, p):
            d = _augment_semantic_groups(r.get(retain_task) or {})
            return _latex_pct(d.get(key))
        return fn
    def _refusal(r, p):
        full = get_refusal_string(r)
        if full == "--":
            return "--"
        idx = full.find(" I ")
        if idx >= 0:
            return _latex_escape(full[:idx + 2] + " ...")
        if full.startswith("I"):
            return "I ..."
        return _latex_escape(full)
    second_col = ("refusal", "Refusal string", False, _refusal)
    return [
        ("method",      "Method",                            False, lambda r, p: _latex_escape(_display_method(r["method"]))),
        second_col,
        ("f_reph_di",   r"$\mathrm{WC}_\mathrm{DI}\downarrow$",   False, lambda r, p: _latex_pct(get_forget_reph_di(r))),
        ("f_reph_opp",  r"$\mathrm{WC}_\mathrm{Opp}\downarrow$",  False, _opp_with_delta),
        ("f_reph_gib",  r"\gib$\downarrow$",                       False, lambda r, p: _latex_pct(get_retain_avg(r, "forget_rephrasings_gibberish"))),
        ("r_avg",       r"AVG$\uparrow$",                          False, _ret("J_avg")),
        ("s0",           r"\szero$\uparrow$",                    False, _ret("s0")),
        ("s1-5",         r"\sonefive$\uparrow$",               False, _ret("s1-5")),
        ("s6-10",        r"\ssixten$\uparrow$",                False, _ret("s6-10")),
        ("s11-15",       r"\selevenfifteen$\uparrow$",         False, _ret("s11-15")),
        ("mmlu",         r"\mmlu$\uparrow$",                   True,  lambda r, p: _latex_pct(get_mmlu(r))),
        ("entropy",      r"\rep$\uparrow$",                   True,  lambda r, p: _latex_float(get_entropy(r))),
        ("rgq_bi",   r"\rgq$\uparrow$",   True,  lambda r, p: _latex_pct(get_rgq_bi(r))),
    ]


def _make_latex_col_specs_ablation(retain_task: str) -> list:
    """Multi-style column specs for the ablation table (triggered by --latex-ablation).

    The second column is a blank 'Change' column: fill in the per-row change description
    by hand in the generated .tex."""
    def _ret(key):
        def fn(r, p):
            d = _augment_semantic_groups(r.get(retain_task) or {})
            return _latex_pct(d.get(key))
        return fn
    def _change(r, p):
        return ""   # hand-edit each row's change description in the generated .tex
    second_col = ("change", "Change", False, _change)
    return [
        ("method",      "Method",               False, lambda r, p: _latex_escape(_display_method(r["method"]))),
        second_col,
        ("f_reph_di",   r"\qdi",                False, lambda r, p: _latex_pct(get_forget_reph_di(r))),
        ("f_reph_opp",  r"\qr",                 False, lambda r, p: _latex_pct((r.get("forget_rephrasings") or {}).get("J_W_Opposite"))),
        ("f_reph_wc",   r"\qall",               False, lambda r, p: _latex_pct((r.get("forget_rephrasings") or {}).get("J_W_Total"))),
        ("f_reph_gib",  r"\gib",                False, lambda r, p: _latex_pct(get_retain_avg(r, "forget_rephrasings_gibberish"))),
        ("r_avg",       r"\qall",               False, _ret("J_avg")),
        ("s0",          r"\szero",              False, _ret("s0")),
        ("s1-10",       r"\soneten",            False, _ret("s1-10")),
        ("s11-15",      r"\selevenfifteen",     False, _ret("s11-15")),
        ("mmlu",        r"\mmlu",               True,  lambda r, p: _latex_pct(get_mmlu(r))),
        ("entropy",     r"\rep",                True,  lambda r, p: _latex_float(get_entropy(r))),
        ("rgq_bi",  r"\rgq",                True,  lambda r, p: _latex_pct(get_rgq_bi(r))),
    ]


def _make_latex_col_specs_adversarial(retain_task: str) -> list:
    """Column specs for the adversarial table.

    Forget: QDI | QR | QAll (rephrasings) | Qadv (adversarial) | QAll* (combined worst-case)
    Retain: QAll (avg)
    Utility: MMLU | Rep | RGQ
    """
    def _ret(key):
        def fn(r, p):
            return _latex_pct((r.get(retain_task) or {}).get(key))
        return fn
    return [
        ("method",          "Method",                   False, lambda r, p: _latex_escape(_display_method(r["method"]))),
        ("f_direct",        r"\qd",                   False, lambda r, p: _latex_pct((r.get("forget_rephrasings") or {}).get("J_W_Direct"))),
        ("f_reph_di",       r"\qdi",                    False, lambda r, p: _latex_pct(get_forget_reph_di(r))),
        ("f_reph_opp",      r"\qr",                     False, lambda r, p: _latex_pct((r.get("forget_rephrasings") or {}).get("J_W_Opposite"))),
        ("f_reph_wc",       r"\qall",                   False, lambda r, p: _latex_pct((r.get("forget_rephrasings") or {}).get("J_W_Total"))),
        ("f_adv",           r"$Q_\mathrm{adv}$",        False, lambda r, p: _latex_pct(get_forget_wc(r, "forget_adversarial"))),
        ("f_adv_combined",  r"$Q_\mathrm{All}^*$",      False, lambda r, p: _latex_pct(get_forget_wc(r, "forget_adversarial_combined"))),
        ("r_avg",           r"\qall",                   False, _ret("J_avg")),
        ("mmlu",            r"\mmlu",                   True,  lambda r, p: _latex_pct(get_mmlu(r))),
        ("entropy",         r"\rep",                    True,  lambda r, p: _latex_float(get_entropy(r))),
        ("rgq_bi",      r"\rgq",                    True,  lambda r, p: _latex_pct(get_rgq_bi(r))),
    ]


_LATEX_TOP_GROUPS_ADVERSARIAL = [
    ("",                         ["method"]),
    (r"Forget $\downarrow$",     ["f_direct", "f_reph_di", "f_reph_opp", "f_reph_wc", "f_adv", "f_adv_combined"]),
    (r"Retain $\uparrow$",       ["r_avg"]),
    (r"Utility $\uparrow$",      ["mmlu", "entropy", "rgq_bi"]),
]


def _make_latex_col_specs_hierarchical(retain_task: str) -> list:
    def _ret(key):
        def fn(r, p):
            d = _augment_semantic_groups(r.get(retain_task) or {})
            return _latex_pct(d.get(key))
        return fn
    return [
        ("method",  "Method",        False, lambda r, p: _latex_escape(_display_method(r["method"]))),
        ("epochs",  "Ep.",           False, lambda r, p: p["epochs"] or "--"),
        ("lr",      "LR",            False, lambda r, p: _latex_escape(p["lr"]) if p["lr"] else "--"),
        ("gamma",   r"$\gamma$",     False, lambda r, p: p["gamma"] or "--"),
        ("alpha",   r"$\alpha$",     False, lambda r, p: p["alpha"] or "--"),
        ("f_reph",  r"$J_W$",       False, lambda r, p: _latex_pct(get_forget_wc(r, "forget_rephrasings"))),
        ("r_avg",   "AVG",           False, _ret("J_avg")),
        ("r_gk",    "GK",            False, _ret("Cat_GK")),
        ("r_sem",   "Sem",           False, _ret("Cat_Semantic")),
        ("r_syn",   "Syn",           False, _ret("Cat_Syntax")),
        ("r_wrd",   "Lex",           False, _ret("Cat_Lexical")),
        ("s0",      "s0",            False, _ret("s0")),
        ("s1-5",    "s1--5",         False, _ret("s1-5")),
        ("s6-10",   "s6--10",        False, _ret("s6-10")),
        ("s11-15",  "s11--15",       False, _ret("s11-15")),
    ]


def _make_latex_col_specs_hparam(retain_task: str) -> list:
    """Column specs for hyperparameter-selection tables.

    Inline lr/γ/α, no gibberish, no semantic breakdown, retain avg only.
    No companion appendix table is generated for this layout.
    """
    def _ret(key):
        def fn(r, p):
            return _latex_pct((r.get(retain_task) or {}).get(key))
        return fn
    return [
        ("method",     "Method",             False, lambda r, p: _latex_escape(_display_method(r["method"]))),
        ("lr",         "LR",                 False, lambda r, p: _latex_escape(p["lr"]) if p["lr"] else "--"),
        ("gamma",      r"$\gamma$",          False, lambda r, p: p["gamma"] or "--"),
        ("alpha",      r"$\alpha$",          False, lambda r, p: p["alpha"] or "--"),
        ("f_reph_di",  r"\qdi",  False, lambda r, p: r"\textcolor{gray}{" + _latex_pct(get_forget_reph_di(r)) + "}"),
        ("f_reph_opp", r"\qr",   False, lambda r, p: r"\textcolor{gray}{" + _latex_pct((r.get("forget_rephrasings") or {}).get("J_W_Opposite")) + "}"),
        ("f_reph_wc",  r"\qall", False, lambda r, p: _latex_pct((r.get("forget_rephrasings") or {}).get("J_W_Total"))),
        ("r_avg",      r"\qall", False, _ret("J_avg")),
        ("mmlu",       r"\mmlu",             True,  lambda r, p: _latex_pct(get_mmlu(r))),
        ("entropy",    r"\rep",              True,  lambda r, p: _latex_float(get_entropy(r))),
        ("rgq_bi", r"\rgq",              True,  lambda r, p: _latex_pct(get_rgq_bi(r))),
    ]


def _build_two_level_header(specs: list, top_groups=None) -> tuple:
    """Return (top_row_str, cmidrule_str, sub_row_str) for a two-level LaTeX header.

    Groups with an empty name produce a blank placeholder cell but no cmidrule,
    so a bare Method column can sit outside any named group.
    """
    if top_groups is None:
        top_groups = _LATEX_TOP_GROUPS
    col_keys = [key for key, *_ in specs]
    top_cells = []
    cmidrules = []
    col_idx = 1
    for group_name, group_keys in top_groups:
        present = [k for k in group_keys if k in col_keys]
        n = len(present)
        if n == 0:
            continue
        if group_name:
            top_cells.append(rf"\multicolumn{{{n}}}{{c}}{{{group_name}}}")
            cmidrules.append(rf"\cmidrule(lr){{{col_idx}-{col_idx + n - 1}}}")
        else:
            # Empty group: blank placeholder, no cmidrule
            top_cells.append(rf"\multicolumn{{{n}}}{{l}}{{}}")
        col_idx += n
    top_row       = " & ".join(top_cells) + r" \\"
    sub_row       = " & ".join(h for _, h, _, _ in specs) + r" \\"
    cmidrules_str = " ".join(cmidrules)
    return top_row, cmidrules_str, sub_row


def _build_utility_lookup(all_runs: list) -> dict:
    """Build (model, method, base_exp) → run from cross-topic runs.

    Cross-topic runs have exp like 'eval_challenger_disaster/epochs_10_lrs_...'.
    base_exp matches the plain exp of the corresponding challenger-baseline run.
    """
    lookup = {}
    for run in all_runs:
        exp = run.get("exp", "")
        if "/" in exp:
            prefix, base_exp = exp.split("/", 1)
            if prefix.startswith("eval_"):
                key = (run["model"], run["method"], base_exp)
                if key not in lookup or get_mmlu(run) is not None:
                    lookup[key] = run
    return lookup


def _render_latex_group(label: str, model: str, group_runs: list,
                        retain_task: str, utility_lookup: dict = None,
                        breakdown: str = None, ablation: bool = False) -> str:
    """Render one LaTeX table for a (label, model) group.

    Model is always placed in the caption, never as a column.
    utility_lookup: mmlu/entropy/rgq_bi pulled from matched cross-topic run
    when the current run has no data for those cols.
    breakdown: when 'retain_hierarchical', renders hierarchical retain columns
    instead of the default single-column retain.
    ablation: when True, use the ablation column layout (multi-style metrics + a blank,
    hand-editable 'Change' column).
    """
    if breakdown == "retain_hierarchical":
        specs = _make_latex_col_specs_hierarchical(retain_task)
        top_groups = _LATEX_TOP_GROUPS_HIERARCHICAL
    elif breakdown == "hparam":
        specs = _make_latex_col_specs_hparam(retain_task)
        top_groups = _LATEX_TOP_GROUPS_HPARAM
    elif breakdown == "adversarial":
        specs = _make_latex_col_specs_adversarial(retain_task)
        top_groups = _LATEX_TOP_GROUPS_ADVERSARIAL
    elif ablation:
        specs = _make_latex_col_specs_ablation(retain_task)
        top_groups = _LATEX_TOP_GROUPS_ABLATION
    else:
        specs = _make_latex_col_specs(retain_task)
        top_groups = _LATEX_TOP_GROUPS
    col_fmt = "".join("l" if key in ("method", "refusal", "change") else "r"
                      for key, *_ in specs)

    short_model = shorten_model(model)
    caption = f"{_latex_escape(label)}, {_latex_escape(short_model)}"
    label_id = label.replace("->", "_to_").replace("-", "_")

    top_row, cmidrules, sub_row = _build_two_level_header(specs, top_groups)

    header_lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \footnotesize",
        rf"  \caption{{{caption}}}",
        rf"  \label{{tab:{label_id}_{short_model}}}",
        r"  \resizebox{\textwidth}{!}{%",
        rf"  \begin{{tabular}}{{{col_fmt}}}",
        r"    \toprule",
        f"    {top_row}",
        f"    {cmidrules}",
        f"    {sub_row}",
        r"    \midrule",
    ]

    # Build rows as (is_midrule_before, cells_list)
    raw_rows: list[tuple[bool, list[str]]] = []
    row_meta: list[str] = []  # "pretrained" | "relearn" | "run"
    prev_method = None
    for run in group_runs:
        is_rl = run.get("is_relearn", False)
        exp_str = run.get("relearn_exp") or run["exp"]
        p = parse_exp(exp_str, is_relearn=is_rl)

        if is_rl:
            lr_label = p["lr"] or "?"
            cells = []
            for key, _, is_utility, fn in specs:
                if key == "method":
                    cells.append(r"+relearn" + _latex_escape(lr_label))
                elif key in ("refusal", "change"):
                    cells.append("")
                else:
                    cells.append(fn(run, p))
            raw_rows.append((False, cells))
            row_meta.append("relearn")
        else:
            method = run.get("method", "")
            midrule = prev_method is not None and method != prev_method
            prev_method = method

            util_run = None
            if utility_lookup is not None:
                util_run = utility_lookup.get((run["model"], run["method"], run["exp"]))

            cells = [fn(util_run if (is_utility and util_run is not None) else run, p)
                     for _, _, is_utility, fn in specs]
            raw_rows.append((midrule, cells))
            row_meta.append("pretrained" if method == "pretrained" else "run")

    if breakdown == "adversarial":
        row_is_pretrained = [m == "pretrained" for m in row_meta]
        raw_rows = _apply_best_second_highlighting(raw_rows, row_is_pretrained, specs)

    data_lines = []
    for midrule, cells in raw_rows:
        if midrule:
            data_lines.append(r"    \midrule")
        data_lines.append("    " + " & ".join(cells) + r" \\")

    lines = header_lines + data_lines + [r"    \bottomrule", r"  \end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def _render_latex_hyper_table(label: str, model: str, group_runs: list,
                               main_label_id: str) -> str:
    """Compact hyperparameter table for the appendix, cross-referencing the main results table."""
    short_model = shorten_model(model)
    label_id = label.replace("->", "_to_").replace("-", "_")
    hyper_label_id = f"{label_id}_{short_model}_hyper"
    caption = (
        rf"Hyperparameters for the experiments in "
        rf"Table~\ref{{tab:{main_label_id}}} "
        rf"({_latex_escape(label)}, {_latex_escape(short_model)}). "
        rf"Rows correspond in order."
    )

    col_fmt = "l" + "r" * 5 + "l"
    header_cells = ["Method", "Ep.", "LR", r"$\gamma$", r"$\alpha$", "Scale", "Extra"]

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{tab:{hyper_label_id}}}",
        r"  \small",
        rf"  \begin{{tabular}}{{{col_fmt}}}",
        r"    \toprule",
        "    " + " & ".join(header_cells) + r" \\",
        r"    \midrule",
    ]

    for run in group_runs:
        is_rl = run.get("is_relearn", False)
        exp_str = run.get("relearn_exp") or run["exp"]
        p = parse_exp(exp_str, is_relearn=is_rl)
        method = _latex_escape(_display_method(run["method"]))
        ep     = p["epochs"] or "--"
        lr     = _latex_escape(p["lr"]) if p["lr"] else "--"
        gamma  = p["gamma"]  or "--"
        alpha  = p["alpha"]  or "--"
        scale  = fmt_scale(p["scale"]) if p["scale"] else ""
        extra  = _latex_escape(p["extra"]) if p["extra"] else ""
        cells  = [method, ep, lr, gamma, alpha, scale, extra]
        lines.append("    " + " & ".join(cells) + r" \\")

    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def export_latex(runs: list, output_dir: str, breakdown: str = None,
                 ablation: bool = False):
    """Generate one .tex file per (topic, model) group.

    challenger_baseline: retain from 'retain' (its canonical wide retain);
    utility cols back-filled from matching cross-topic run.
    All other groups: retain from 'retain'.
    Model always goes into the caption.
    breakdown: passed through to _render_latex_group (e.g. 'retain_hierarchical').
    ablation: use the ablation column layout (blank hand-editable 'Change' column).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    utility_lookup = _build_utility_lookup(_ALL_RUNS or runs)

    groups = _group_runs(runs)
    written = []
    for label, model, group_runs in groups:
        is_baseline = "challenger-baseline" in label and "->" not in label
        retain_task = "retain"   # same key for baseline (wide retain) and non-baseline runs
        lookup = utility_lookup if is_baseline else None

        if breakdown == "adversarial":
            n_before = len(group_runs)
            group_runs = [r for r in group_runs
                          if r.get("method") == "pretrained"
                          or get_forget_wc(r, "forget_adversarial") is not None]
            n_adv = sum(1 for r in group_runs if r.get("method") != "pretrained")
            print(f"  [adv debug] {label}/{shorten_model(model)}: "
                  f"{n_before} runs → {n_adv} with adv data (+ pretrained)")
            for r in group_runs:
                if r.get("method") != "pretrained":
                    v = get_forget_wc(r, "forget_adversarial")
                    print(f"    method={r.get('method')} exp={r.get('exp','')[:60]}  adv={v}")
            if not any(r.get("method") != "pretrained" for r in group_runs):
                print(f"  [adv debug] → skipped (no adv runs)")
                continue  # skip groups with no adversarial runs

        tex_main = _render_latex_group(label, model, group_runs, retain_task, lookup,
                                       breakdown=breakdown, ablation=ablation)

        safe_label = label.replace("->", "_to_").replace("-", "_")
        short_model = shorten_model(model)
        main_label_id = f"{safe_label}_{short_model}"

        # Appendix hyperparameter table (not needed when hypers are already inline)
        if breakdown not in ("retain_hierarchical", "hparam"):
            tex_hyper = _render_latex_hyper_table(label, model, group_runs, main_label_id)
            content = tex_main + "\n\n" + tex_hyper
        else:
            content = tex_main

        fname = out / f"table_{safe_label}_{short_model}.tex"
        fname.write_text(content, encoding="utf-8")
        written.append(str(fname))
        print(f"  Wrote {fname}")

    if breakdown == "adversarial":
        _export_latex_adversarial_avg(runs, out, written)

    print(f"\nExported {len(written)} LaTeX table(s) to {out}/")


# ---------------------------------------------------------------------------
# Adversarial avg-across-topics LaTeX export
# ---------------------------------------------------------------------------

def _export_latex_adversarial_avg(runs: list, out: Path, written: list):
    """Append one average-across-topics table per model to `written`.

    Mirrors what export_latex_multi does for multi-topic runs: groups runs by
    (model, method, exp) across topics, averages all per-topic metrics, and
    renders one avg table per model using the adversarial col specs.
    Called only when breakdown == 'adversarial' and there are ≥2 topics.
    """
    retain_task = "retain"

    pretrained_by_topic_model = {
        (r["topic"], r["model"]): r
        for r in (_ALL_RUNS or runs)
        if r.get("method") == "pretrained" and not r.get("is_relearn")
    }

    # Group non-pretrained runs by model → (method, exp) → [per-topic runs]
    # Only include runs that have forget_adversarial data.
    model_combos: dict = {}
    model_topics: dict = {}
    for r in runs:
        if r.get("method") == "pretrained" or r.get("is_relearn"):
            continue
        if get_forget_wc(r, "forget_adversarial") is None:
            continue
        model = r["model"]
        combo_key = (r["method"], r["exp"])
        model_combos.setdefault(model, {}).setdefault(combo_key, []).append(r)
        model_topics.setdefault(model, set()).add(r["topic"])

    for model, combos in model_combos.items():
        topics = sorted(model_topics[model])
        if len(topics) < 2:
            continue

        short_model = shorten_model(model)

        # Averaged pretrained row (include regardless of adversarial data)
        pt_runs = [pretrained_by_topic_model[t, model] for t in topics
                   if (t, model) in pretrained_by_topic_model]
        avg_rows = []
        if pt_runs:
            avg_pt = _build_avg_run(pt_runs)
            avg_pt["method"] = "pretrained"
            avg_rows.append(avg_pt)

        for combo_key in sorted(combos, key=lambda k: (_METHOD_SORT_ORDER.get(k[0], 99), k[1])):
            avg_run = _build_avg_run(combos[combo_key])
            avg_rows.append(avg_run)

        label = "avg (" + ", ".join(shorten_topic(t) for t in topics) + ")"
        tex_main  = _render_latex_group(label, model, avg_rows, retain_task,
                                        breakdown="adversarial")
        label_id  = f"avg_{short_model}"
        tex_hyper = _render_latex_hyper_table(label, model, avg_rows, label_id)

        fname = out / f"table_adv_avg_{short_model}.tex"
        fname.write_text(tex_main + "\n\n" + tex_hyper, encoding="utf-8")
        written.append(str(fname))
        print(f"  Wrote {fname}")


# ---------------------------------------------------------------------------
# Multi-topic LaTeX export (seq_ / comb_ runs)
# ---------------------------------------------------------------------------

def _format_trained_on_label(run: dict) -> str:
    """Format the 'Trained on' cell: 'A → B' for seq, 'A + B' for comb."""
    multi_type = run.get("_multi_type", "")
    raw_combo  = run.get("_raw_combo", "")
    topics = raw_combo.split("+")
    short  = [shorten_topic(t) for t in topics]
    sep = r" $\rightarrow$ " if multi_type == "seq" else " + "
    return sep.join(short)


def _avg_task_dicts(task_dicts: list) -> dict:
    """Average numeric values across task dicts. Keys in only one dict are kept as-is."""
    all_keys: set = set()
    for d in task_dicts:
        all_keys.update(d.keys())
    result = {}
    for k in all_keys:
        vals = [d[k] for d in task_dicts if k in d and isinstance(d.get(k), (int, float))]
        if vals:
            result[k] = sum(vals) / len(vals)
    return result


def _build_avg_run(combo_runs: list, extra_tasks: tuple = ()) -> dict:
    """Build a synthetic average run from multiple per-eval-topic entries for the same trained model."""
    if not combo_runs:
        return {}
    avg = dict(combo_runs[0])
    base_tasks = ("forget_rephrasings", "forget_rephrasings_gibberish",
                  "forget_adversarial", "forget_adversarial_combined",
                  "retain", "retain_train_rephrasing")
    for task in base_tasks + tuple(extra_tasks):
        dicts = [r.get(task) or {} for r in combo_runs]
        if any(dicts):
            avg[task] = _avg_task_dicts([d for d in dicts if d])
    # Utility: model-level, take first non-None dict
    for util_key in ("mmlu", "rep", "rgq_bi"):
        avg[util_key] = next((r.get(util_key) for r in combo_runs if r.get(util_key) is not None), None)
    return avg


def _make_latex_col_specs_multi(retain_task: str) -> list:
    """Column specs for multi-topic tables: 'Trained on' replaces the refusal/change column."""
    def _ret(key):
        def fn(r, p):
            d = _augment_semantic_groups(r.get(retain_task) or {})
            return _latex_pct(d.get(key))
        return fn
    def _ret_gray(key):
        def fn(r, p):
            d = _augment_semantic_groups(r.get(retain_task) or {})
            val = _latex_pct(d.get(key))
            return r"\textcolor{gray}{" + val + "}"
        return fn
    def _gray(inner_fn):
        def wrapped(r, p):
            val = inner_fn(r, p)
            return r"\textcolor{gray}{" + val + "}"
        return wrapped
    return [
        ("method",     "Method",                                                False, lambda r, p: _pretrained_model_macro(r.get("model", "")) if r.get("method") == "pretrained" else _latex_escape(_display_method(r["method"]))),
        ("trained_on", "Trained on",                                            False, lambda r, p: r.get("_trained_on_label") or "--"),
        ("f_reph_di",  r"\qdi",                                                 False, _gray(lambda r, p: _latex_pct(get_forget_reph_di(r)))),
        ("f_reph_opp", r"\qr",                                                  False, _gray(lambda r, p: _latex_pct((r.get("forget_rephrasings") or {}).get("J_W_Opposite")))),
        ("f_reph_wc",  r"\qall",                                                False, lambda r, p: _latex_pct((r.get("forget_rephrasings") or {}).get("J_W_Total"))),
        ("f_reph_gib", r"\gib",                                                 False, lambda r, p: _latex_pct(get_retain_avg(r, "forget_rephrasings_gibberish"))),
        ("r_avg",      r"\qall",                                                False, _ret("J_avg")),
        ("s0",          r"\szero",                                              False, _ret_gray("s0")),
        ("s1-10",       r"\soneten",                                            False, _ret_gray("s1-10")),
        ("s11-15",      r"\selevenfifteen",                                     False, _ret_gray("s11-15")),
        ("mmlu",        r"\mmlu",                                                True,  lambda r, p: _latex_pct(get_mmlu(r))),
        ("entropy",     r"\rep",                                                 True,  lambda r, p: _latex_float(get_entropy(r))),
        ("rgq_bi",  r"\rgq",                                                 True,  lambda r, p: _latex_pct(get_rgq_bi(r))),
    ]


_LATEX_TOP_GROUPS_MULTI_ADVERSARIAL = [
    ("",                         ["method", "trained_on"]),
    (r"Forget $\downarrow$",     ["f_direct", "f_reph_di", "f_reph_opp", "f_reph_wc", "f_adv", "f_adv_combined"]),
    (r"Retain $\uparrow$",       ["r_avg"]),
    (r"Utility $\uparrow$",      ["mmlu", "entropy", "rgq_bi"]),
]


def _make_latex_col_specs_multi_adversarial(retain_task: str) -> list:
    """Multi-topic adversarial col specs: 'Trained on' + adversarial forget cols + retain + utility."""
    def _ret(key):
        def fn(r, p):
            return _latex_pct((r.get(retain_task) or {}).get(key))
        return fn
    return [
        ("method",         "Method",               False, lambda r, p: _pretrained_model_macro(r.get("model", "")) if r.get("method") == "pretrained" else _latex_escape(_display_method(r["method"]))),
        ("trained_on",     "Trained on",            False, lambda r, p: r.get("_trained_on_label") or "--"),
        ("f_direct",       r"\qd",                False, lambda r, p: _latex_pct((r.get("forget_rephrasings") or {}).get("J_W_Direct"))),
        ("f_reph_di",      r"\qdi",                 False, lambda r, p: _latex_pct(get_forget_reph_di(r))),
        ("f_reph_opp",     r"\qr",                  False, lambda r, p: _latex_pct((r.get("forget_rephrasings") or {}).get("J_W_Opposite"))),
        ("f_reph_wc",      r"\qall",                False, lambda r, p: _latex_pct((r.get("forget_rephrasings") or {}).get("J_W_Total"))),
        ("f_adv",          r"$Q_\mathrm{adv}$",     False, lambda r, p: _latex_pct(get_forget_wc(r, "forget_adversarial"))),
        ("f_adv_combined", r"$Q_\mathrm{All}^*$",   False, lambda r, p: _latex_pct(get_forget_wc(r, "forget_adversarial_combined"))),
        ("r_avg",          r"\qall",                False, _ret("J_avg")),
        ("mmlu",           r"\mmlu",                True,  lambda r, p: _latex_pct(get_mmlu(r))),
        ("entropy",        r"\rep",                 True,  lambda r, p: _latex_float(get_entropy(r))),
        ("rgq_bi",     r"\rgq",                 True,  lambda r, p: _latex_pct(get_rgq_bi(r))),
    ]


_LATEX_TOP_GROUPS_MULTI_BREAKDOWN = [
    ("",                               ["method"]),
    (r"Forget $\downarrow$",           ["f_reph_di", "f_reph_opp", "f_reph_wc"]),
    (r"Retain $\uparrow$",             ["r_avg", "r_gk", "r_syn", "r_wrd"]),
    (r"Retain -- semantic $\uparrow$", ["s0", "s1-10", "s11-15"]),
]


def _make_latex_col_specs_multi_breakdown(retain_task: str) -> list:
    """Column specs for the retain breakdown table appended under each multi-topic table."""
    def _ret(key):
        def fn(r, p):
            d = _augment_semantic_groups(r.get(retain_task) or {})
            return _latex_pct(d.get(key))
        return fn
    return [
        ("method",     "Method",          False, lambda r, p: _pretrained_model_macro(r.get("model", "")) if r.get("method") == "pretrained" else _latex_escape(_display_method(r["method"]))),
        ("f_reph_di",  r"\qdi",           False, lambda r, p: _latex_pct(get_forget_reph_di(r))),
        ("f_reph_opp", r"\qr",            False, lambda r, p: _latex_pct((r.get("forget_rephrasings") or {}).get("J_W_Opposite"))),
        ("f_reph_wc",  r"\qall",          False, lambda r, p: _latex_pct((r.get("forget_rephrasings") or {}).get("J_W_Total"))),
        ("r_avg",      r"\qall",          False, _ret("J_avg")),
        ("r_gk",       "GK",              False, _ret("Cat_GK")),
        ("r_syn",      "Syn.",            False, _ret("Cat_Syntax")),
        ("r_wrd",      "Lex.",            False, _ret("Cat_Lexical")),
        ("s0",         r"\szero",             False, _ret("s0")),
        ("s1-10",      r"\soneten",           False, _ret("s1-10")),
        ("s11-15",     r"\selevenfifteen",    False, _ret("s11-15")),
    ]


def _render_multi_breakdown_table(label: str, model: str, rows: list, retain_task: str) -> str:
    """Render the retain breakdown table appended below the main multi-topic table."""
    specs      = _make_latex_col_specs_multi_breakdown(retain_task)
    top_groups = _LATEX_TOP_GROUPS_MULTI_BREAKDOWN

    short_model = shorten_model(model)
    caption  = f"{_latex_escape(label)} -- retain breakdown, {_latex_escape(short_model)}"
    col_fmt  = "l rrr rrrr rrr"
    label_id = (label + "_breakdown").replace(" ", "_").replace(",", "").replace("→", "to").replace("+", "_")

    top_row, cmidrules, sub_row = _build_two_level_header(specs, top_groups)

    header_lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \footnotesize",
        rf"  \caption{{{caption}}}",
        rf"  \label{{tab:{label_id}_{short_model}}}",
        r"  \resizebox{\textwidth}{!}{%",
        rf"  \begin{{tabular}}{{{col_fmt}}}",
        r"    \toprule",
        f"    {top_row}",
        f"    {cmidrules}",
        f"    {sub_row}",
        r"    \midrule",
    ]

    raw_rows: list = []
    row_is_pretrained: list = []
    prev_method = None
    pretrained_done = False

    for run in rows:
        method  = run.get("method", "")
        is_pret = (method == "pretrained")
        exp_str = run.get("_actual_exp") or run.get("exp", "")
        p       = parse_exp(exp_str)

        if is_pret:
            midrule = False
            pretrained_done = True
        else:
            midrule = pretrained_done and prev_method is None
            prev_method = method

        cells = [fn(run, p) for _, _, _, fn in specs]
        raw_rows.append((midrule, cells))
        row_is_pretrained.append(is_pret)

    raw_rows = _apply_best_second_highlighting(raw_rows, row_is_pretrained, specs)

    data_lines = []
    for midrule, cells in raw_rows:
        if midrule:
            data_lines.append(r"    \midrule[0.1em]")
        data_lines.append("    " + " & ".join(cells) + r" \\")

    lines = header_lines + data_lines + [r"    \bottomrule", r"  \end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def _group_multi_runs_for_export(all_runs: list) -> dict:
    """Group multi-topic runs by (multi_type, n_topics, model) → {eval_topic: [annotated_run]}."""
    result: dict = {}
    for r in all_runs:
        if not _is_multi_topic(r):
            continue
        if r.get("is_relearn"):
            continue
        exp   = r.get("exp", "")
        topic = r.get("topic", "")
        if "/" not in exp:
            continue
        prefix, actual_exp = exp.split("/", 1)
        if not prefix.startswith("eval_"):
            continue
        eval_topic = prefix[5:]
        multi_type = "seq" if topic.startswith("seq_") else "comb"
        raw_combo  = topic[4:] if multi_type == "seq" else topic[5:]
        n_topics   = len(raw_combo.split("+"))

        rc = dict(r)
        rc["_multi_type"]       = multi_type
        rc["_raw_combo"]        = raw_combo
        rc["_eval_topic"]       = eval_topic
        rc["_actual_exp"]       = actual_exp
        rc["_trained_on_label"] = _format_trained_on_label(rc)

        key = (multi_type, n_topics, r["model"])
        result.setdefault(key, {}).setdefault(eval_topic, []).append(rc)
    return result


_COL_DIRECTION_MULTI = {
    "f_direct": -1, "f_reph_di": -1, "f_reph_opp": -1, "f_reph_wc": -1, "f_reph_gib": -1,
    "f_adv": -1, "f_adv_combined": -1,
    "r_avg": +1, "s0": +1, "s1-5": +1, "s6-15": +1, "s1-10": +1, "s11-15": +1,
    "r_gk": +1, "r_syn": +1, "r_wrd": +1,
    "retain_train": +1,
    "mmlu": +1, "entropy": +1, "rgq_bi": +1,
}


def _apply_best_second_highlighting(raw_rows: list, row_is_pretrained: list, specs: list) -> list:
    """Wrap best cell in \\textbf and second-best in \\underline, per ranked column.

    Pretrained rows are excluded from ranking so we highlight among methods only.
    Ties at the same rank all receive the same decoration.
    """
    col_keys = [key for key, *_ in specs]
    cells_array = [list(cells) for _, cells in raw_rows]

    def _extract_num(cell: str):
        m = re.search(r'([-+]?\d+(?:\.\d+)?)', cell)
        return float(m.group(1)) if m else None

    for col_i, key in enumerate(col_keys):
        direction = _COL_DIRECTION_MULTI.get(key, 0)
        if direction == 0:
            continue
        vals = [
            (_extract_num(raw_rows[ri][1][col_i]), ri)
            for ri in range(len(raw_rows))
            if not row_is_pretrained[ri]
        ]
        vals = [(v, ri) for v, ri in vals if v is not None]
        if not vals:
            continue
        vals_sorted = sorted(vals, key=lambda x: x[0] * direction, reverse=True)
        best_val = vals_sorted[0][0]
        best_rows = [ri for v, ri in vals_sorted if v == best_val]
        remaining = [(v, ri) for v, ri in vals_sorted if v != best_val]
        second_rows = [ri for v, ri in remaining if v == remaining[0][0]] if remaining else []

        for ri in best_rows:
            cells_array[ri][col_i] = r"\textbf{" + cells_array[ri][col_i] + "}"
        for ri in second_rows:
            cells_array[ri][col_i] = r"\underline{" + cells_array[ri][col_i] + "}"

    return [(midrule, cells) for (midrule, _), cells in zip(raw_rows, cells_array)]


def _render_multi_table(label: str, model: str, rows: list, retain_task: str,
                        col_specs: list = None, top_groups: list = None) -> str:
    r"""Render a single LaTeX table for multi-topic runs.

    rows: list of run dicts, already annotated with _trained_on_label.
          The first row may be the pretrained baseline (method == 'pretrained').
          Method groups are separated by \midrule.

    When all non-pretrained rows share the same training combination, the
    'Trained on' column is dropped and the combo is folded into the caption
    (sequential: arrows; combined: commas).
    """
    # Detect whether all non-pretrained rows share a single training combo.
    non_pt = [r for r in rows if r.get("method") != "pretrained"]
    unique_labels = {r.get("_trained_on_label", "") for r in non_pt}
    fold_into_caption = len(unique_labels) == 1 and bool(non_pt)

    specs      = col_specs if col_specs is not None else _make_latex_col_specs_multi(retain_task)
    top_groups = top_groups if top_groups is not None else _LATEX_TOP_GROUPS_MULTI

    short_model = shorten_model(model)

    if fold_into_caption:
        specs      = [s for s in specs if s[0] != "trained_on"]
        # Remove trained_on from whichever group contains it, preserve all other groups.
        top_groups = [(g, [k for k in ks if k != "trained_on"])
                      for g, ks in top_groups if [k for k in ks if k != "trained_on"]]
        # Build caption-friendly combo label
        first = non_pt[0]
        raw_combo  = first.get("_raw_combo", "")
        multi_type = first.get("_multi_type", "")
        topics     = raw_combo.split("+")
        short_t    = [shorten_topic(t) for t in topics]
        if multi_type == "seq":
            combo_str = r" $\rightarrow$ ".join(short_t)
        else:
            combo_str = ", ".join(short_t)
        caption = f"{_latex_escape(label)}, {combo_str}, {_latex_escape(short_model)}"
    else:
        caption = f"{_latex_escape(label)}, {_latex_escape(short_model)}"

    col_fmt  = "".join("l" if key in ("method", "trained_on") else "r"
                       for key, *_ in specs)
    label_id = label.replace(" ", "_").replace(",", "").replace("→", "to").replace("+", "_")

    top_row, cmidrules, sub_row = _build_two_level_header(specs, top_groups)

    header_lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \footnotesize",
        rf"  \caption{{{caption}}}",
        rf"  \label{{tab:{label_id}_{short_model}}}",
        r"  \resizebox{\textwidth}{!}{%",
        rf"  \begin{{tabular}}{{{col_fmt}}}",
        r"    \toprule",
        f"    {top_row}",
        f"    {cmidrules}",
        f"    {sub_row}",
        r"    \midrule",
    ]

    raw_rows: list = []
    row_is_pretrained: list = []
    prev_method = None
    pretrained_done = False

    for run in rows:
        method  = run.get("method", "")
        is_pret = (method == "pretrained")
        exp_str = run.get("_actual_exp") or run.get("exp", "")
        p       = parse_exp(exp_str)

        if is_pret:
            midrule = False
            pretrained_done = True
        else:
            midrule = pretrained_done and prev_method is None
            prev_method = method

        cells = [fn(run, p) for _, _, _, fn in specs]
        raw_rows.append((midrule, cells))
        row_is_pretrained.append(is_pret)

    raw_rows = _apply_best_second_highlighting(raw_rows, row_is_pretrained, specs)

    data_lines = []
    for midrule, cells in raw_rows:
        if midrule:
            data_lines.append(r"    \midrule[0.1em]")
        data_lines.append("    " + " & ".join(cells) + r" \\")

    lines = header_lines + data_lines + [r"    \bottomrule", r"  \end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def export_latex_multi(output_dir: str, breakdown: str = None):
    """Export per-topic and averaged LaTeX tables for seq_/comb_ multi-topic runs.

    Uses _ALL_RUNS (full unfiltered DB) so --with-multi is not required.
    Groups: (multi_type, n_topics, model). For each group:
      - one .tex per eval topic
      - one .tex with metrics averaged across eval topics
    breakdown='adversarial': use adversarial col specs and filter to runs with adv data.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    grouped = _group_multi_runs_for_export(_ALL_RUNS)
    if not grouped:
        print("No multi-topic runs found in DB.")
        return

    pretrained_by_topic_model = {
        (r["topic"], r["model"]): r
        for r in _ALL_RUNS
        if r.get("method") == "pretrained" and not r.get("is_relearn")
    }

    written = []
    for (multi_type, n_topics, model), topic_dict in sorted(grouped.items()):
        retain_task = "retain"
        short_model = shorten_model(model)

        # ---- build combo groups (needed for both per-topic utility fill and avg table) ----
        combos: dict = {}
        for et, topic_runs in topic_dict.items():
            for run in topic_runs:
                combo_key = (run.get("topic", ""), run.get("method", ""), run.get("_actual_exp", ""))
                combos.setdefault(combo_key, []).append(run)

        # Build utility fallback per combo: take first non-None value across all eval topics.
        # MMLU / Rep / RGQ_bi are model-level, so whichever eval topic captured them is fine.
        _UTIL_KEYS = ("mmlu", "rep", "rgq_bi")
        utility_by_combo: dict = {}
        for combo_key, combo_runs in combos.items():
            utility_by_combo[combo_key] = {
                uk: next((r.get(uk) for r in combo_runs if r.get(uk) is not None), None)
                for uk in _UTIL_KEYS
            }

        # Col specs depend on breakdown mode
        if breakdown == "adversarial":
            adv_col_specs  = _make_latex_col_specs_multi_adversarial(retain_task)
            adv_top_groups = _LATEX_TOP_GROUPS_MULTI_ADVERSARIAL

        # ---- per-topic tables ----
        for eval_topic, topic_runs in sorted(topic_dict.items()):
            pretrained = pretrained_by_topic_model.get((eval_topic, model))
            label      = f"{multi_type} {n_topics}-topic, {shorten_topic(eval_topic)}"

            rows = []
            if pretrained is not None:
                pt = dict(pretrained)
                pt["_trained_on_label"] = "--"
                rows.append(pt)

            # Sort runs: by method then actual_exp for stable ordering.
            # Fill missing utility values from sibling eval-topic entries of the same run.
            topic_runs_sorted = sorted(topic_runs, key=lambda r: (_METHOD_SORT_ORDER.get(r.get("method", ""), 99), r.get("_actual_exp", "")))
            for run in topic_runs_sorted:
                combo_key = (run.get("topic", ""), run.get("method", ""), run.get("_actual_exp", ""))
                fallback  = utility_by_combo.get(combo_key, {})
                enriched  = dict(run)
                for uk, val in fallback.items():
                    if enriched.get(uk) is None and val is not None:
                        enriched[uk] = val
                rows.append(enriched)

            if breakdown == "adversarial":
                rows = [r for r in rows
                        if r.get("method") == "pretrained"
                        or get_forget_wc(r, "forget_adversarial") is not None]
                if not any(r.get("method") != "pretrained" for r in rows):
                    continue
                tex_content = _render_multi_table(label, model, rows, retain_task,
                                                  col_specs=adv_col_specs, top_groups=adv_top_groups)
                fname = out / f"multi_adv_{multi_type}_{n_topics}t_{shorten_topic(eval_topic)}_{short_model}.tex"
            else:
                tex_content = (_render_multi_table(label, model, rows, retain_task)
                               + "\n\n"
                               + _render_multi_breakdown_table(label, model, rows, retain_task))
                fname = out / f"multi_{multi_type}_{n_topics}t_{shorten_topic(eval_topic)}_{short_model}.tex"

            fname.write_text(tex_content, encoding="utf-8")
            written.append(str(fname))
            print(f"  Wrote {fname}")

        # ---- average-across-topics table ----
        eval_topics = sorted(topic_dict.keys())

        # Averaged pretrained row
        pt_runs = [pretrained_by_topic_model[t, model] for t in eval_topics
                   if (t, model) in pretrained_by_topic_model]
        avg_rows = []
        if pt_runs:
            avg_pt = _build_avg_run(pt_runs)
            avg_pt["method"]             = "pretrained"
            avg_pt["_trained_on_label"]  = "--"
            avg_rows.append(avg_pt)

        # Sort combo keys for stable output: method first, then actual_exp
        for combo_key in sorted(combos.keys(), key=lambda k: (_METHOD_SORT_ORDER.get(k[1], 99), k[2])):
            combo_runs = combos[combo_key]
            avg_run    = _build_avg_run(combo_runs)
            # _trained_on_label is identical for all entries of a combo (same training config)
            avg_run["_trained_on_label"] = combo_runs[0].get("_trained_on_label", "--")
            avg_rows.append(avg_run)

        label_avg = f"{multi_type} {n_topics}-topic, avg"
        if breakdown == "adversarial":
            avg_rows_adv = [r for r in avg_rows
                            if r.get("method") == "pretrained"
                            or get_forget_wc(r, "forget_adversarial") is not None]
            if any(r.get("method") != "pretrained" for r in avg_rows_adv):
                tex_content = _render_multi_table(label_avg, model, avg_rows_adv, retain_task,
                                                  col_specs=adv_col_specs, top_groups=adv_top_groups)
                fname = out / f"multi_adv_{multi_type}_{n_topics}t_avg_{short_model}.tex"
                fname.write_text(tex_content, encoding="utf-8")
                written.append(str(fname))
                print(f"  Wrote {fname}")
        else:
            tex_main      = _render_multi_table(label_avg, model, avg_rows, retain_task)
            tex_breakdown = _render_multi_breakdown_table(label_avg, model, avg_rows, retain_task)
            fname = out / f"multi_{multi_type}_{n_topics}t_avg_{short_model}.tex"
            fname.write_text(tex_main + "\n\n" + tex_breakdown, encoding="utf-8")
            written.append(str(fname))
            print(f"  Wrote {fname}")

    print(f"\nExported {len(written)} multi-topic LaTeX table(s) to {out}/")


# ---------------------------------------------------------------------------
# Multi-topic relearn LaTeX export  (--latex-multi-relearn-dir)
# ---------------------------------------------------------------------------

_RELEARN_TOP_GROUPS_MAIN = [
    ("",                              ["method"]),
    (r"Forget $\downarrow$",          ["f_reph_di", "f_reph_opp", "f_reph_wc"]),
    (r"Retain Train $\uparrow$",      ["retain_train"]),
]

_RELEARN_TOP_GROUPS_UTILITY = [
    ("",                    ["method"]),
    (r"Utility $\uparrow$", ["mmlu", "entropy", "rgq_bi"]),
]


def _make_latex_col_specs_relearn_main() -> list:
    """Main relearn table: forget metrics + retain_train (no utility)."""
    return [
        ("method",       "Method",   False, lambda r, p: r.get("_method_cell", "--")),
        ("f_reph_di",    r"\qdi",    False, lambda r, p: _latex_pct(get_forget_reph_di(r))),
        ("f_reph_opp",   r"\qr",     False, lambda r, p: _latex_pct((r.get("forget_rephrasings") or {}).get("J_W_Opposite"))),
        ("f_reph_wc",    r"\qall",   False, lambda r, p: _latex_pct((r.get("forget_rephrasings") or {}).get("J_W_Total"))),
        ("retain_train", r"\rtrain", False, lambda r, p: _latex_pct(get_retain_avg(r, "retain_train_rephrasing"))),
    ]


def _make_latex_col_specs_relearn_utility() -> list:
    """Utility companion table for relearn: mmlu, rep, rgq_bi."""
    return [
        ("method",     "Method",  False, lambda r, p: r.get("_method_cell", "--")),
        ("mmlu",       r"\mmlu",  True,  lambda r, p: _latex_pct(get_mmlu(r))),
        ("entropy",    r"\rep",   True,  lambda r, p: _latex_float(get_entropy(r))),
        ("rgq_bi", r"\rgq",   True,  lambda r, p: _latex_pct(get_rgq_bi(r))),
    ]


def _render_relearn_grouped_table(label: str, model: str, groups: list,
                                   col_specs: list, top_groups: list) -> str:
    """Render a relearn table grouping base model + relearn rows per method.

    groups: list of dicts with keys:
      {"kind": "pretrained", "run": run_dict}
      {"kind": "method", "method": str, "base_run": run_or_None, "relearn_runs": [run, ...]}

    Base model row shows the method name; relearn row(s) show '\\hspace{1em}+relearn'.
    \\midrule is inserted between method groups (but NOT between base and its relearn rows).
    """
    short_model = shorten_model(model)
    label_id = (label.replace(" ", "_").replace(",", "").replace("→", "to")
                     .replace("+", "_").replace("(", "").replace(")", "")
                     .replace("=", "_").replace("/", "_"))
    caption = f"{_latex_escape(label)}, {_latex_escape(short_model)}"
    col_fmt = "".join("l" if key == "method" else "r" for key, *_ in col_specs)
    top_row, cmidrules, sub_row = _build_two_level_header(col_specs, top_groups)

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \footnotesize",
        rf"  \caption{{{caption}}}",
        rf"  \label{{tab:{label_id}_{short_model}}}",
        r"  \resizebox{\textwidth}{!}{%",
        rf"  \begin{{tabular}}{{{col_fmt}}}",
        r"    \toprule",
        f"    {top_row}",
        f"    {cmidrules}",
        f"    {sub_row}",
        r"    \midrule",
    ]

    first_method_group = True
    for group in groups:
        if group["kind"] == "pretrained":
            run = group["run"]
            run["_method_cell"] = _latex_escape(_display_method("pretrained"))
            cells = [fn(run, {}) for _, _, _, fn in col_specs]
            lines.append("    " + " & ".join(cells) + r" \\")
            lines.append(r"    \midrule")
            first_method_group = True
            continue

        if not first_method_group:
            lines.append(r"    \midrule")
        first_method_group = False

        method     = group["method"]
        base_run   = group.get("base_run")
        rl_runs    = group["relearn_runs"]

        if base_run is not None:
            base_run["_method_cell"] = _latex_escape(_display_method(method))
            cells = [fn(base_run, {}) for _, _, _, fn in col_specs]
            lines.append("    " + " & ".join(cells) + r" \\")

        lrs = [parse_exp(r.get("relearn_exp", ""), is_relearn=True).get("lr", "") for r in rl_runs]
        multi_lr = len(set(lrs)) > 1
        for run, lr in zip(rl_runs, lrs):
            if multi_lr and lr:
                cell = rf"\hspace{{1em}}+relearn $({_latex_escape(lr)})$"
            else:
                cell = r"\hspace{1em}+relearn"
            run["_method_cell"] = cell
            cells = [fn(run, {}) for _, _, _, fn in col_specs]
            lines.append("    " + " & ".join(cells) + r" \\")

    lines += [r"    \bottomrule", r"  \end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def _group_multi_relearn_runs_for_export(all_runs: list) -> dict:
    """Group comb_ relearn runs by (multi_type, n_topics, model) → {eval_topic: [annotated_run]}."""
    result: dict = {}
    for r in all_runs:
        if not r.get("is_relearn"):
            continue
        if not _is_multi_topic(r):
            continue
        exp   = r.get("exp", "")
        topic = r.get("topic", "")
        if "/" not in exp:
            continue
        prefix, actual_exp = exp.split("/", 1)
        if not prefix.startswith("eval_"):
            continue
        eval_topic = prefix[5:]
        multi_type = "seq" if topic.startswith("seq_") else "comb"
        raw_combo  = topic[4:] if multi_type == "seq" else topic[5:]
        n_topics   = len(raw_combo.split("+"))

        rc = dict(r)
        rc["_multi_type"]       = multi_type
        rc["_raw_combo"]        = raw_combo
        rc["_eval_topic"]       = eval_topic
        rc["_actual_exp"]       = actual_exp           # source unlearn exp
        rc["_trained_on_label"] = _format_trained_on_label(rc)

        key = (multi_type, n_topics, r["model"])
        result.setdefault(key, {}).setdefault(eval_topic, []).append(rc)
    return result


def _render_relearn_hparam_companion(rows: list, main_label_id: str, short_model: str) -> str:
    """Companion table listing base unlearning hyperparameters for each numbered relearn row."""
    non_pt = [r for r in rows if r.get("method") != "pretrained"]
    if not non_pt:
        return ""

    has_scale = any(
        parse_exp(r.get("_actual_exp") or r.get("exp", "")).get("scale")
        for r in non_pt
    )

    hdr_cells = [r"\#", "Ep.", "LR", r"$\gamma$", r"$\alpha$"]
    if has_scale:
        hdr_cells.append("Scale")
    col_fmt = "r" * len(hdr_cells)

    main_ref = rf"Table~\ref{{tab:{main_label_id}_{short_model}}}"
    caption  = rf"Base unlearning hyperparameters for {main_ref}."
    label    = f"tab:{main_label_id}_hparam_{short_model}"

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \footnotesize",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        rf"  \begin{{tabular}}{{{col_fmt}}}",
        r"    \toprule",
        "    " + " & ".join(hdr_cells) + r" \\",
        r"    \midrule",
    ]
    for i, run in enumerate(non_pt, 1):
        exp_str = run.get("_actual_exp") or run.get("exp", "")
        p = parse_exp(exp_str)
        cells = [
            str(i),
            p.get("epochs") or "--",
            _latex_escape(p.get("lr") or "--"),
            p.get("gamma") or "--",
            p.get("alpha") or "--",
        ]
        if has_scale:
            cells.append(p.get("scale") or "--")
        lines.append("    " + " & ".join(cells) + r" \\")

    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def export_latex_multi_relearn(output_dir: str):
    r"""Export per-topic and averaged relearn LaTeX tables for comb_ multi-topic runs.

    Uses _ALL_RUNS (full unfiltered DB) so --with-multi / --with-relearn are not required.
    Each .tex file contains two tables:
      1. Main: forget metrics + retain_train.  Each method block = base row + indented +relearn rows.
      2. Utility: mmlu / rep / rgq_bi, same grouping.
    \midrule separates method blocks; no midrule between base and its +relearn rows.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    grouped = _group_multi_relearn_runs_for_export(_ALL_RUNS)
    if not grouped:
        print("No multi-topic relearn runs found in DB.")
        return

    pretrained_by_topic_model = {
        (r["topic"], r["model"]): r
        for r in _ALL_RUNS
        if r.get("method") == "pretrained" and not r.get("is_relearn")
    }

    # Lookup for base (unlearned) cross-topic eval runs:
    # key = (comb_topic, model, method, eval_topic, actual_exp) → run
    base_runs_lookup: dict = {}
    for r in _ALL_RUNS:
        if r.get("is_relearn"):
            continue
        topic = r.get("topic", "")
        if not (topic.startswith("comb_") or topic.startswith("seq_")):
            continue
        exp = r.get("exp", "")
        if "/" not in exp:
            continue
        prefix, actual_exp = exp.split("/", 1)
        if not prefix.startswith("eval_"):
            continue
        eval_topic = prefix[5:]
        key = (topic, r.get("model", ""), r.get("method", ""), eval_topic, actual_exp)
        base_runs_lookup[key] = r

    _UTIL_KEYS = ("mmlu", "rep", "rgq_bi")

    def _get_rl_lr(combo_key):
        relearn_exp = combo_key[3]
        if not relearn_exp:
            return ""
        return parse_exp(relearn_exp, is_relearn=True).get("lr", "") or ""

    col_specs_main    = _make_latex_col_specs_relearn_main()
    col_specs_utility = _make_latex_col_specs_relearn_utility()

    def _build_groups(pretrained_run, topic_runs_sorted, eval_topic, comb_topic, model_name):
        """Build the groups list for _render_relearn_grouped_table."""
        groups = []
        if pretrained_run is not None:
            pt = dict(pretrained_run)
            pt["_trained_on_label"] = "--"
            groups.append({"kind": "pretrained", "run": pt})

        # Group relearn runs by (method, actual_exp) preserving sort order
        seen_keys: list = []
        method_groups: dict = {}
        for run in topic_runs_sorted:
            gk = (run.get("method", ""), run.get("_actual_exp", ""))
            if gk not in method_groups:
                seen_keys.append(gk)
                base_key = (comb_topic, model_name, gk[0], eval_topic, gk[1])
                base_run = base_runs_lookup.get(base_key)
                method_groups[gk] = {"method": gk[0], "base_run": base_run, "relearn_runs": []}
            method_groups[gk]["relearn_runs"].append(run)

        for gk in seen_keys:
            mg = method_groups[gk]
            groups.append({"kind": "method", **mg})
        return groups

    def _write_relearn_table(label: str, groups: list, fname: Path) -> None:
        label_id  = (label.replace(" ", "_").replace(",", "").replace("→", "to")
                          .replace("+", "_").replace("(", "").replace(")", "")
                          .replace("=", "_").replace("/", "_"))
        tex_main    = _render_relearn_grouped_table(label, model, groups,
                                                    col_specs_main, _RELEARN_TOP_GROUPS_MAIN)
        tex_utility = _render_relearn_grouped_table(
            label + " (utility)", model, groups,
            col_specs_utility, _RELEARN_TOP_GROUPS_UTILITY)
        # Hparam companion: flatten all relearn runs for the existing helper
        flat_rows = []
        for g in groups:
            if g["kind"] == "pretrained":
                flat_rows.append(g["run"])
            else:
                if g.get("base_run") is not None:
                    flat_rows.append(g["base_run"])
                flat_rows.extend(g["relearn_runs"])
        companion = _render_relearn_hparam_companion(flat_rows, label_id, shorten_model(model))
        parts = [tex_main, tex_utility]
        if companion:
            parts.append(companion)
        fname.write_text("\n\n".join(parts), encoding="utf-8")
        written.append(str(fname))
        print(f"  Wrote {fname}")

    written = []
    for (multi_type, n_topics, model), topic_dict in sorted(grouped.items()):
        short_model = shorten_model(model)

        # combo_key: (topic, unlearn_method, src_exp, relearn_exp)
        combos: dict = {}
        for et, topic_runs in topic_dict.items():
            for run in topic_runs:
                combo_key = (run.get("topic", ""), run.get("method", ""),
                             run.get("_actual_exp", ""), run.get("relearn_exp", ""))
                combos.setdefault(combo_key, []).append(run)

        utility_by_combo: dict = {}
        for combo_key, combo_runs in combos.items():
            utility_by_combo[combo_key] = {
                uk: next((r.get(uk) for r in combo_runs if r.get(uk) is not None), None)
                for uk in _UTIL_KEYS
            }

        unique_lrs = sorted(set(_get_rl_lr(k) for k in combos),
                            key=lambda x: (x == "", x))

        for rl_lr in unique_lrs:
            lr_combos = {k: v for k, v in combos.items() if _get_rl_lr(k) == rl_lr}
            lr_safe   = (rl_lr or "unknown").replace("-", "m")
            lr_label  = f"lr={rl_lr}" if rl_lr else "lr=?"
            comb_topic = next(iter(combos))[0]  # topic string (same for all combos in group)

            # ---- per-topic tables ----
            for eval_topic, topic_runs in sorted(topic_dict.items()):
                pretrained = pretrained_by_topic_model.get((eval_topic, model))
                label      = f"{multi_type} {n_topics}-topic relearn ({lr_label}), {shorten_topic(eval_topic)}"

                lr_topic_runs = [
                    r for r in topic_runs
                    if _get_rl_lr((r.get("topic", ""), r.get("method", ""),
                                   r.get("_actual_exp", ""), r.get("relearn_exp", ""))) == rl_lr
                ]
                topic_runs_sorted = sorted(
                    lr_topic_runs,
                    key=lambda r: (_METHOD_SORT_ORDER.get(r.get("method", ""), 99),
                                   r.get("relearn_exp", ""), r.get("_actual_exp", ""))
                )
                # Enrich utility fields from cross-topic fallback
                enriched_runs = []
                for run in topic_runs_sorted:
                    combo_key = (run.get("topic", ""), run.get("method", ""),
                                 run.get("_actual_exp", ""), run.get("relearn_exp", ""))
                    fallback  = utility_by_combo.get(combo_key, {})
                    enriched  = dict(run)
                    for uk, val in fallback.items():
                        if enriched.get(uk) is None and val is not None:
                            enriched[uk] = val
                    enriched_runs.append(enriched)

                groups = _build_groups(pretrained, enriched_runs, eval_topic, comb_topic, model)
                _write_relearn_table(
                    label, groups,
                    out / f"relearn_{multi_type}_{n_topics}t_{shorten_topic(eval_topic)}_{short_model}_lr{lr_safe}.tex"
                )

            # ---- average-across-topics table ----
            eval_topics = sorted(topic_dict.keys())
            pt_runs = [pretrained_by_topic_model[t, model] for t in eval_topics
                       if (t, model) in pretrained_by_topic_model]
            avg_pt = None
            if pt_runs:
                avg_pt = _build_avg_run(pt_runs)
                avg_pt["method"]            = "pretrained"
                avg_pt["_trained_on_label"] = "--"

            # Build per-(method, actual_exp) averaged relearn runs
            unlearn_keys = sorted(set((k[1], k[2]) for k in lr_combos),
                                  key=lambda uk: (_METHOD_SORT_ORDER.get(uk[0], 99), uk[1]))
            avg_method_groups = []
            for unlearn_method, actual_exp in unlearn_keys:
                matching = [(k, v) for k, v in lr_combos.items()
                            if k[1] == unlearn_method and k[2] == actual_exp]
                rl_runs_per_combo = []
                base_runs_for_avg = []
                for k, combo_runs in matching:
                    avg_rl  = _build_avg_run(combo_runs)
                    avg_rl["relearn_exp"]       = combo_runs[0].get("relearn_exp", "")
                    avg_rl["_actual_exp"]        = actual_exp
                    avg_rl["_trained_on_label"] = combo_runs[0].get("_trained_on_label", "--")
                    rl_runs_per_combo.append(avg_rl)
                    # Average base runs across eval topics
                    base_topic_runs = [
                        base_runs_lookup.get((comb_topic, model, unlearn_method, et, actual_exp))
                        for et in eval_topics
                    ]
                    base_topic_runs = [r for r in base_topic_runs if r is not None]
                    if base_topic_runs:
                        base_runs_for_avg.extend(base_topic_runs)

                avg_base = _build_avg_run(base_runs_for_avg) if base_runs_for_avg else None
                avg_method_groups.append({
                    "kind": "method",
                    "method": unlearn_method,
                    "base_run": avg_base,
                    "relearn_runs": rl_runs_per_combo,
                })

            avg_groups = []
            if avg_pt is not None:
                avg_groups.append({"kind": "pretrained", "run": avg_pt})
            avg_groups.extend(avg_method_groups)

            _write_relearn_table(
                f"{multi_type} {n_topics}-topic relearn ({lr_label}), avg",
                avg_groups,
                out / f"relearn_{multi_type}_{n_topics}t_avg_{short_model}_lr{lr_safe}.tex"
            )

    print(f"\nExported {len(written)} multi-topic relearn LaTeX table(s) to {out}/")


# ---------------------------------------------------------------------------
# Combined LaTeX export (challenger_baseline + challenger_disaster)
# ---------------------------------------------------------------------------

def _render_combined_results_expanded_table(model: str, rows: list) -> str:
    """Render expanded combined results table: adds GK/Syn/Lex retain columns, drops Retain Q_all."""
    short_model = shorten_model(model)
    label_id = f"challenger_combined_expand_{short_model}"
    caption = (
        r"\textbf{Evaluation of unlearning methods on the Challenger Disaster.} "
        r"Left: $\mathrm{LKF^{*}}$ following~\cite{singh2025unlearning}, where high retain scores are "
        r"misleading due to evaluation on training questions. Right: our proposed framework, revealing "
        r"that models fail to forget the actual facts---expanding the retain set with semantically and "
        r"syntactically similar questions leads to significant performance degradation."
    )

    def _sem_base(run, key):
        if run is None:
            return "--"
        d = _augment_semantic_groups(run.get("retain") or {})
        return _latex_pct(d.get(key))

    def _ret_ours(run, key):
        if run is None:
            return "--"
        d = _augment_semantic_groups(run.get("retain") or {})
        return _latex_pct(d.get(key))

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{tab:{label_id}}}",
        r"  \small",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{lrr| rrr rrr rrr}",
        r"    \toprule",
        r"    & \multicolumn{2}{c|}{$\mathrm{LKF^{*}}$~\cite{singh2025unlearning}} & \multicolumn{9}{c}{Ours} \\",
        r"    \cmidrule(lr){2-3} \cmidrule(lr){4-12}",
        r"    & \multicolumn{1}{c}{Forget $\downarrow$} & \multicolumn{1}{c|}{Retain $\uparrow$}"
        r" & \multicolumn{3}{c}{Forget $\downarrow$}"
        r" & \multicolumn{3}{c}{Retain $\uparrow$}"
        r" & \multicolumn{3}{c}{Retain -- semantic $\uparrow$} \\",
        r"    \cmidrule(lr){2-2} \cmidrule(lr){3-3} \cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12}",
        r"    Method & \qd & \sonefivetrain"
        r" & \qdi & \qr & \qall"
        r" & GK & Syn. & Lex."
        r" & \szero & \sonefiveeval & \ssixfifteen \\",
        r"    \midrule",
    ]

    current_method = None
    for base_run, cross_run in rows:
        run = base_run or cross_run
        if run is None:
            continue
        method = run.get("method", "--")
        is_pretrained = (method == "pretrained")

        if not is_pretrained:
            if current_method is not None and method != current_method:
                lines.append(r"    \midrule")
            current_method = method

        bl_qd   = _latex_pct((base_run.get("forget_rephrasings") or {}).get("J_W_Direct")) if base_run else "--"
        bl_s15  = _sem_base(base_run, "s1-5")

        ou_qdi  = _latex_pct(get_forget_reph_di(cross_run)) if cross_run else "--"
        ou_qr   = _latex_pct((cross_run.get("forget_rephrasings") or {}).get("J_W_Opposite")) if cross_run else "--"
        ou_qall = _latex_pct((cross_run.get("forget_rephrasings") or {}).get("J_W_Total")) if cross_run else "--"
        ou_gk   = _ret_ours(cross_run, "Cat_GK")
        ou_syn  = _ret_ours(cross_run, "Cat_Syntax")
        ou_lex  = _ret_ours(cross_run, "Cat_Lexical")
        ou_s0   = _ret_ours(cross_run, "s0")
        ou_s15  = _ret_ours(cross_run, "s1-5")
        ou_s615 = _ret_ours(cross_run, "s6-15")

        cells = [
            _latex_escape(_display_method(method)),
            bl_qd, bl_s15,
            ou_qdi, ou_qr, ou_qall,
            ou_gk, ou_syn, ou_lex,
            ou_s0, ou_s15, ou_s615,
        ]
        lines.append("    " + " & ".join(cells) + r" \\")
        if is_pretrained:
            lines.append(r"    \midrule")

    lines += [r"    \bottomrule", r"  \end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def _render_combined_results_table(model: str, rows: list) -> str:
    """Render the combined results table: 3-level header Baseline | Ours."""
    short_model = shorten_model(model)
    label_id = f"challenger_combined_{short_model}"
    caption = (
        r"\textbf{Evaluation of unlearning methods on the Challenger Disaster.} "
        r"Left: baseline evaluation following~\cite{singh2025unlearning}, where high retain scores are "
        r"misleading due to evaluation on training questions. Right: our proposed framework, revealing "
        r"that models fail to forget the actual facts---expanding the retain set with semantically and "
        r"syntactically similar questions leads to significant performance degradation."
    )

    def _sem_base(run, key):
        if run is None:
            return "--"
        d = _augment_semantic_groups(run.get("retain") or {})
        return _latex_pct(d.get(key))

    def _sem_ours(run, key):
        if run is None:
            return "--"
        d = _augment_semantic_groups(run.get("retain") or {})
        return _latex_pct(d.get(key))

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{tab:{label_id}}}",
        r"  \small",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{lrr rrr r rrr}",
        r"    \toprule",
        r"    & \multicolumn{2}{c}{Baseline~\cite{singh2025unlearning}} & \multicolumn{7}{c}{Ours} \\",
        r"    \cmidrule(lr){2-3} \cmidrule(lr){4-10}",
        r"    & \multicolumn{1}{c}{Forget $\downarrow$} & \multicolumn{1}{c}{Retain $\uparrow$}"
        r" & \multicolumn{3}{c}{Forget $\downarrow$} & \multicolumn{1}{c}{Retain $\uparrow$}"
        r" & \multicolumn{3}{c}{Retain -- semantic $\uparrow$} \\",
        r"    \cmidrule(lr){2-2} \cmidrule(lr){3-3} \cmidrule(lr){4-6} \cmidrule(lr){7-7} \cmidrule(lr){8-10}",
        r"    Method & \qd & \sonefive"
        r" & \qdi & \qr & \qall"
        r" & \qall & \szero & \sonefive & \ssixfifteen \\",
        r"    \midrule",
    ]

    current_method = None
    for base_run, cross_run in rows:
        run = base_run or cross_run
        if run is None:
            continue
        method = run.get("method", "--")
        is_pretrained = (method == "pretrained")

        if not is_pretrained:
            if current_method is not None and method != current_method:
                lines.append(r"    \midrule")
            current_method = method

        bl_qd  = _latex_pct((base_run.get("forget_rephrasings") or {}).get("J_W_Direct")) if base_run else "--"
        bl_s15 = _sem_base(base_run, "s1-5")

        ou_qdi  = _latex_pct(get_forget_reph_di(cross_run)) if cross_run else "--"
        ou_qr   = _latex_pct((cross_run.get("forget_rephrasings") or {}).get("J_W_Opposite")) if cross_run else "--"
        ou_qall = _latex_pct((cross_run.get("forget_rephrasings") or {}).get("J_W_Total")) if cross_run else "--"
        ou_ravg = _latex_pct(get_retain_avg(cross_run, "retain")) if cross_run else "--"
        ou_s0   = _sem_ours(cross_run, "s0")
        ou_s15  = _sem_ours(cross_run, "s1-5")
        ou_s615 = _sem_ours(cross_run, "s6-15")

        cells = [
            _latex_escape(_display_method(method)),
            bl_qd, bl_s15,
            ou_qdi, ou_qr, ou_qall,
            ou_ravg,
            ou_s0, ou_s15, ou_s615,
        ]
        lines.append("    " + " & ".join(cells) + r" \\")
        if is_pretrained:
            lines.append(r"    \midrule")

    lines += [r"    \bottomrule", r"  \end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def _render_combined_hparam_baseline_utility_table(model: str, rows: list, results_ref: str) -> str:
    """Render a single table combining hyperparams + baseline results + utility metrics."""
    short_model = shorten_model(model)
    label_id = f"challenger_hparam_utility_{short_model}"
    caption = (
        rf"\textbf{{Hyperparameters and utility metrics for the Challenger Disaster "
        rf"({_latex_escape(model)}).}} "
        rf"Rows correspond in order to those in Table~\ref{{{results_ref}}}. "
        rf"Baseline columns follow~\cite{{singh2025unlearning}}."
    )

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{tab:{label_id}}}",
        r"  \small",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{l rrrr rr rrr}",
        r"    \toprule",
        r"    & \multicolumn{4}{c}{Hyperparameters}"
        r" & \multicolumn{2}{c}{Baseline~\cite{singh2025unlearning}}"
        r" & \multicolumn{3}{c}{Utility $\uparrow$} \\",
        r"    \cmidrule(lr){2-5} \cmidrule(lr){6-7} \cmidrule(lr){8-10}",
        r"    Method & Ep. & LR & $\gamma$ & $\alpha$"
        r" & \qd & \sonefive"
        r" & \mmlu & \rep & \rgq \\",
        r"    \midrule",
    ]

    current_method = None
    for base_run, cross_run in rows:
        run = base_run or cross_run
        if run is None:
            continue
        method = run.get("method", "--")
        is_pretrained = (method == "pretrained")

        if not is_pretrained:
            if current_method is not None and method != current_method:
                lines.append(r"    \midrule")
            current_method = method

        if base_run is not None and not is_pretrained:
            p = parse_exp(base_run["exp"])
            ep    = p["epochs"] or "--"
            lr    = _latex_escape(p["lr"]) if p["lr"] else "--"
            gamma = p["gamma"] or "--"
            alpha = p["alpha"] or "--"
        else:
            ep, lr, gamma, alpha = "--", "--", "--", "--"

        # Baseline columns (from base_run)
        bl_qd  = _latex_pct((base_run.get("forget_rephrasings") or {}).get("J_W_Direct")) if base_run else "--"
        bl_s15 = _latex_pct(
            _augment_semantic_groups(base_run.get("retain") or {}).get("s1-5")
        ) if base_run else "--"

        # Utility columns: prefer cross_run, fall back to base_run (same pattern as rgq).
        # rgq_bi/rep may be stored on the direct challenger_baseline run when no
        # cross-topic eval was run, so always check base_run as a fallback.
        _mmlu_run = cross_run if (cross_run and get_mmlu(cross_run) is not None) else base_run
        mmlu = _latex_pct(get_mmlu(_mmlu_run)) if _mmlu_run else "--"
        _rep_run = cross_run if (cross_run and get_entropy(cross_run) is not None) else base_run
        rep  = _latex_float(get_entropy(_rep_run)) if _rep_run else "--"
        _rgq_run = base_run if (base_run and get_rgq_bi(base_run) is not None) else cross_run
        rgq  = _latex_pct(get_rgq_bi(_rgq_run)) if _rgq_run else "--"

        cells = [
            _latex_escape(_display_method(method)),
            ep, lr, gamma, alpha,
            bl_qd, bl_s15,
            mmlu, rep, rgq,
        ]
        lines.append("    " + " & ".join(cells) + r" \\")
        if is_pretrained:
            lines.append(r"    \midrule")

    lines += [r"    \bottomrule", r"  \end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def _render_combined_retain_hierarchical_table(model: str, rows: list) -> str:
    """Render retain comparison table: Baseline J_avg | Ours hierarchical (retain metric)."""
    short_model = shorten_model(model)
    label_id = f"challenger_retain_hierarchical_{short_model}"
    caption = (
        rf"\textbf{{Retain comparison for the Challenger Disaster ({_latex_escape(model)}).}} "
        rf"Left: overall retain score under the baseline evaluation protocol "
        rf"(training questions only). "
        rf"Right: our hierarchical retain breakdown across semantically and syntactically diverse questions."
    )

    def _ret_base(run, key):
        if run is None:
            return "--"
        d = _augment_semantic_groups(run.get("retain") or {})
        return _latex_pct(d.get(key))

    def _ret_ours(run, key):
        if run is None:
            return "--"
        d = _augment_semantic_groups(run.get("retain") or {})
        return _latex_pct(d.get(key))

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{tab:{label_id}}}",
        r"  \small",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{l r rrrrr rrr}",
        r"    \toprule",
        r"    & \multicolumn{1}{c}{\lkfs $\uparrow$}"
        r" & \multicolumn{8}{c}{Ours $\uparrow$} \\",
        r"    \cmidrule(lr){2-2} \cmidrule(lr){3-10}",
        r"    Method & \sonefivetrain & \qall & GK & Syn. & Wrd."
        r" & Sem. & \szero & \sonefiveeval & \ssixfifteen \\",
        r"    \midrule",
    ]

    current_method = None
    for base_run, cross_run in rows:
        run = base_run or cross_run
        if run is None:
            continue
        method = run.get("method", "--")
        is_pretrained = (method == "pretrained")

        if not is_pretrained:
            if current_method is not None and method != current_method:
                lines.append(r"    \midrule")
            current_method = method

        bl_retain = _ret_base(base_run, "J_avg")

        ou_avg   = _ret_ours(cross_run, "J_avg")
        ou_gk    = _ret_ours(cross_run, "Cat_GK")
        ou_sem   = _ret_ours(cross_run, "Cat_Semantic")
        ou_syn   = _ret_ours(cross_run, "Cat_Syntax")
        ou_wrd   = _ret_ours(cross_run, "Cat_Lexical")
        ou_s0    = _ret_ours(cross_run, "s0")
        ou_s15   = _ret_ours(cross_run, "s1-5")
        ou_s615  = _ret_ours(cross_run, "s6-15")

        cells = [
            _latex_escape(_display_method(method)),
            bl_retain,
            ou_avg, ou_gk, ou_syn, ou_wrd,
            ou_sem, ou_s0, ou_s15, ou_s615,
        ]
        lines.append("    " + " & ".join(cells) + r" \\")
        if is_pretrained:
            lines.append(r"    \midrule")

    lines += [r"    \bottomrule", r"  \end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def _render_combined_comparison_table(model: str, rows: list, disaster_by_method: dict) -> str:
    r"""Compare baseline-trained vs disaster-trained models side-by-side.

    Left (\lkfs*): baseline-trained model evaluated on our disaster framework.
    Right (Ours): disaster-trained model with full metrics.
    """
    short_model = shorten_model(model)
    label_id = f"challenger_comparison_{short_model}"
    caption = (
        rf"\textbf{{Cross-training evaluation on the Challenger Disaster ({_latex_escape(model)}).}} "
        rf"Left: models trained following \lkfs{{}}~\cite{{singh2025unlearning}}, evaluated on our framework. "
        rf"Right: our models trained on the Challenger Disaster."
    )

    def _f(run, key):
        if run is None:
            return "--"
        return _latex_pct((run.get("forget_rephrasings") or {}).get(key))

    def _r(run):
        if run is None:
            return "--"
        return _latex_pct(get_retain_avg(run, "retain"))

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{tab:{label_id}}}",
        r"  \small",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{l rrrr rrrr}",
        r"    \toprule",
        r"    & \multicolumn{4}{c}{\lkfs*} & \multicolumn{4}{c}{Ours} \\",
        r"    \cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"    & \multicolumn{3}{c}{Forget $\downarrow$} & \multicolumn{1}{c}{Retain $\uparrow$}"
        r" & \multicolumn{3}{c}{Forget $\downarrow$} & \multicolumn{1}{c}{Retain $\uparrow$} \\",
        r"    \cmidrule(lr){2-4} \cmidrule(lr){5-5} \cmidrule(lr){6-8} \cmidrule(lr){9-9}",
        r"    Method & \qdi & \qr & \qall & \qall & \qdi & \qr & \qall & \qall \\",
        r"    \midrule",
    ]

    current_method = None
    for base_run, cross_run in rows:
        run = base_run or cross_run
        if run is None:
            continue
        method = run.get("method", "--")
        is_pretrained = (method == "pretrained")

        if not is_pretrained:
            if current_method is not None and method != current_method:
                lines.append(r"    \midrule")
            current_method = method

        # Left side: baseline-trained model evaluated on disaster
        l_qdi  = (r"\textcolor{gray}{" + _latex_pct(get_forget_reph_di(cross_run)) + "}") if cross_run else "--"
        l_qr   = (r"\textcolor{gray}{" + _latex_pct((cross_run.get("forget_rephrasings") or {}).get("J_W_Opposite")) + "}") if cross_run else "--"
        l_qall = _f(cross_run, "J_W_Total")
        l_r    = _r(cross_run)

        # Right side: disaster-trained model (pretrained reuses cross_run)
        d_run = cross_run if is_pretrained else disaster_by_method.get(method)
        if d_run is not None:
            r_qdi  = r"\textcolor{gray}{" + _latex_pct(get_forget_reph_di(d_run)) + "}"
            r_qr   = r"\textcolor{gray}{" + _latex_pct((d_run.get("forget_rephrasings") or {}).get("J_W_Opposite")) + "}"
            r_qall = _f(d_run, "J_W_Total")
            r_r    = _r(d_run)
        else:
            r_qdi = r_qr = r_qall = r_r = "--"

        cells = [
            _latex_escape(_display_method(method)),
            l_qdi, l_qr, l_qall, l_r,
            r_qdi, r_qr, r_qall, r_r,
        ]
        lines.append("    " + " & ".join(cells) + r" \\")
        if is_pretrained:
            lines.append(r"    \midrule")

    lines += [r"    \bottomrule", r"  \end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def export_latex_combined(runs: list, output_dir: str):
    r"""Generate combined challenger_baseline + challenger LaTeX tables per model.

    One .tex file per model, containing four tables:
    1. Results: Baseline (Q_D, s1-5) | Ours (Q_DI, Q_R, Q_All, Retain Q_All, s1-5, s6-15)
    2. Hyperparams + Baseline + Utility: Ep., LR, γ, α | Q_D, s1-5 | MMLU, Rep., RGQ
    3. Retain comparison: Baseline J_avg | Ours hierarchical (AVG, GK, Sem, Syn, Wrd, s0, s1-5, s6-15)
    4. Cross-training comparison: \lkfs* (Q_DI, Q_R, Q_All, Retain) | Ours (same + MMLU, Rep., RGQ)

    Baseline cols come from challenger_baseline runs; Ours + Utility cols come from the
    matching cross-topic (eval_challenger_disaster/) run via utility_lookup[(model, method, base_exp)].
    Pretrained "Ours" cols come from the challenger_disaster pretrained run.
    Table 4 right side: best direct challenger_disaster run per method (lowest J_W_Total).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_runs = _ALL_RUNS or runs
    utility_lookup = _build_utility_lookup(all_runs)

    pretrained_ours_by_model = {
        r["model"]: r
        for r in all_runs
        if r.get("method") == "pretrained"
        and r.get("topic", "").lower() == "challenger_disaster"
        and not r.get("is_relearn")
    }

    baseline_runs = [
        r for r in runs
        if r.get("topic", "").lower() == "challenger_baseline"
        and not r.get("is_relearn")
    ]

    if not baseline_runs:
        print("No challenger_baseline runs found in the filtered set.")
        return

    models = list(dict.fromkeys(r["model"] for r in baseline_runs))
    written = []

    for model in models:
        model_baseline = [r for r in baseline_runs if r["model"] == model]

        def _sort_baseline(r):
            if r.get("method") == "pretrained":
                return (0, "", 0.0)
            p = parse_exp(r["exp"])
            try:    lr_f = float(p["lr"]) if p["lr"] else 0.0
            except: lr_f = 0.0
            return (1, r["method"], lr_f)

        model_baseline.sort(key=_sort_baseline)

        rows = []
        for base_run in model_baseline:
            if base_run.get("method") == "pretrained":
                cross_run = pretrained_ours_by_model.get(model)
            else:
                cross_run = utility_lookup.get((model, base_run["method"], base_run["exp"]))
            rows.append((base_run, cross_run))

        # Best direct challenger_disaster run per method (lowest J_W_Total = best forgetting)
        disaster_by_method: dict = {}
        for r in all_runs:
            if (r.get("topic", "").lower() != "challenger_disaster"
                    or r["model"] != model
                    or r.get("is_relearn")
                    or r.get("method") == "pretrained"
                    or "/" in r.get("exp", "")):
                continue
            method = r.get("method", "")
            score = (r.get("forget_rephrasings") or {}).get("J_W_Total")
            if score is None:
                continue
            existing = disaster_by_method.get(method)
            existing_score = (existing.get("forget_rephrasings") or {}).get("J_W_Total", float("inf")) if existing else float("inf")
            if score < existing_score:
                disaster_by_method[method] = r

        results_ref = f"tab:challenger_combined_{shorten_model(model)}"
        tex0 = _render_combined_results_expanded_table(model, rows)
        tex1 = _render_combined_results_table(model, rows)
        tex2 = _render_combined_hparam_baseline_utility_table(model, rows, results_ref)
        tex3 = _render_combined_retain_hierarchical_table(model, rows)
        tex4 = _render_combined_comparison_table(model, rows, disaster_by_method)

        fname = out / f"table_challenger_combined_{shorten_model(model)}.tex"
        fname.write_text(tex0 + "\n\n" + tex1 + "\n\n" + tex2 + "\n\n" + tex3 + "\n\n" + tex4, encoding="utf-8")
        written.append(str(fname))
        print(f"  Wrote {fname}")

    print(f"\nExported {len(written)} combined LaTeX table(s) to {out}/")


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def delete_run(db: dict, db_path: Path, key: str):
    runs = db.get("runs", {})
    kl = key.lower()
    # Priority: exact → suffix → substring
    if key in runs:
        matched_key = key
    else:
        suffix = [k for k in runs if k.lower().endswith(kl)]
        matches = suffix if suffix else [k for k in runs if kl in k.lower()]
        if len(matches) == 0:
            print(f"No run found matching: {key!r}")
            return
        if len(matches) > 1:
            print(f"Ambiguous: {len(matches)} runs match {key!r}:")
            for m in matches:
                print(f"  {m}")
            print("Provide a more specific key.")
            return
        matched_key = matches[0]

    print(f"Delete: {matched_key}")
    ans = input("Confirm? [y/N] ").strip().lower()
    if ans == "y":
        del runs[matched_key]
        from datetime import datetime
        db["last_updated"] = datetime.now().isoformat(timespec="seconds")
        save_db(db, db_path)
        print(f"Deleted. {len(runs)} run(s) remaining.")
    else:
        print("Cancelled.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="View and manage results_db.json")
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR_DEFAULT,
                        help="Path to the evaluations directory (default: evaluations/)")
    parser.add_argument("--filter", action="append", default=[], metavar="FIELD=VALUE",
                        help="Filter runs (substring match). Repeatable. Fields: topic, model, method, "
                             "exp, key, extra, gamma, alpha, epochs, lr, scale. FIELD!=VALUE excludes matches.")
    parser.add_argument("--cols", type=str, default=None,
                        help=f"Comma-separated columns. Available: {','.join(COL_ACCESSORS)}")
    parser.add_argument("--with-relearn", action="store_true", help="Include relearn runs (hidden by default)")
    parser.add_argument("--relearn-only", action="store_true", help="Show only relearn runs")
    parser.add_argument("--with-multi", action="store_true",
                        help="Include sequential (seq_) and combined (comb_) multi-topic runs "
                             "(hidden by default). They appear in the eval-topic group with a "
                             "'train' column showing seq+<other> or comb+<other>.")
    parser.add_argument("--multi-average", action="store_true",
                        help="Show ONLY seq_/comb_ multi-topic runs, collapsed into one row per "
                             "training config with metrics averaged across the constituent topics. "
                             "Each group is titled by the multi-topic (e.g. comb-avg:A+B+C).")
    _breakdown_choices = list(BREAKDOWN_TASKS) + ["hparam", "adversarial"]
    parser.add_argument("--breakdown", type=str, default=None,
                        choices=_breakdown_choices,
                        metavar="TASK",
                        help=f"Show expanded breakdown table. TASK: {', '.join(_breakdown_choices)}. "
                             "('hparam' is LaTeX-only: inline lr/γ/α, no gibberish, retain avg only; requires --latex-dir)")
    parser.add_argument("--detail", type=str, default=None, metavar="KEY",
                        help="Show full detail for a run (exact or partial key match)")
    parser.add_argument("--latex-dir", type=str, default=None, metavar="DIR",
                        help="Generate LaTeX tables into DIR (one .tex file per group). "
                             "challenger_baseline: no model col, retain_avg. Others: model col, retain.")
    parser.add_argument("--latex-combined-dir", type=str, default=None, metavar="DIR",
                        help="Generate combined challenger_baseline + challenger tables into DIR "
                             "(one .tex per model, two tables: results + hyperparams).")
    parser.add_argument("--latex-multi-dir", type=str, default=None, metavar="DIR",
                        help="Export per-topic + avg LaTeX tables for seq_/comb_ multi-topic runs "
                             "into DIR. Groups by (seq/comb, n_topics, model). "
                             "Always uses the full unfiltered DB — --with-multi not required.")
    parser.add_argument("--latex-multi-relearn-dir", type=str, default=None, metavar="DIR",
                        help="Export per-topic + avg LaTeX relearn tables for comb_ multi-topic runs "
                             "into DIR. Rows = relearn variants; method column shows relearn trainer. "
                             "Always uses the full unfiltered DB — --with-relearn not required.")
    parser.add_argument("--latex-ablation", action="store_true",
                        help="With --latex-dir, use the ablation table layout: multi-style metric "
                             "columns plus a blank 'Change' column to fill in by hand in the .tex.")
    parser.add_argument("--delete", type=str, default=None, metavar="KEY",
                        help="Delete a run from the DB")
    parser.add_argument("--list-keys", action="store_true", help="Print all run keys")
    parser.add_argument("--forget-reph-threshold", type=float, default=None, metavar="PCT",
                        help="Highlight rows with forget_reph WC (direct+indirect, J_W_DI) < PCT%% in green (no filtering).")
    parser.add_argument("--forget-reph-filter", type=float, default=None, metavar="PCT",
                        help="Keep only runs with forget_reph WC (direct+indirect, J_W_DI) < PCT%% (pretrained always kept). No highlighting.")
    parser.add_argument("--metric-filter", action="append", default=[], metavar="EXPR",
                        help="Filter by metric value. Repeatable. Syntax: 'col op value' where value is in %%. "
                             "Ops: >=, <=, >, <, ==. Pretrained always kept; runs missing the metric are excluded. "
                             f"Cols: {', '.join(sorted(METRIC_FILTER_ACCESSORS))}. "
                             "Example: --metric-filter 'retain >= 48' --metric-filter 'forget_reph <= 4'")
    args = parser.parse_args()

    global _ALL_RUNS, _MULTI_AVERAGE
    eval_dir = args.eval_dir.resolve()
    db, db_path = load_db(eval_dir)
    runs = list(db.get("runs", {}).values())
    _ALL_RUNS = runs

    # All retain reads go to the canonical `retain` key: for baseline this is the
    # wide retain, for non-baseline the standard retain metric.

    # --list-keys
    if args.list_keys:
        for r in sorted(runs, key=lambda x: x["key"]):
            prefix = "(R) " if r.get("is_relearn") else "    "
            print(prefix + r["key"])
        print(f"\n{len(runs)} run(s)")
        return

    # --delete
    if args.delete:
        delete_run(db, db_path, args.delete)
        return

    # --detail
    if args.detail:
        key = args.detail
        kl = key.lower()
        # Priority: exact → suffix → substring
        exact = [r for r in runs if r["key"].lower() == kl]
        if exact:
            render_detail(exact[0])
            return
        suffix = [r for r in runs if r["key"].lower().endswith(kl)]
        matches = suffix if suffix else [r for r in runs if kl in r["key"].lower()]
        if len(matches) == 0:
            print(f"No run found matching: {key!r}")
            print("Tip: use --list-keys to see all keys.")
        elif len(matches) > 1:
            print(f"Multiple matches for {key!r}:")
            for m in matches:
                print(f"  {m['key']}")
            print("Be more specific (or use exact key from --list-keys).")
        else:
            render_detail(matches[0])
        return

    # Filter
    runs = apply_filters(runs, args.filter)
    if args.metric_filter:
        runs = apply_metric_filters(runs, args.metric_filter)
    if not args.with_relearn and not args.relearn_only:
        runs = [r for r in runs if not r.get("is_relearn")]
    if args.relearn_only:
        runs = [r for r in runs if r.get("is_relearn")]
    if args.multi_average:
        # Show only multi-topic runs; _group_runs averages them across topics.
        runs = [r for r in runs if _is_multi_topic(r)]
        if not args.breakdown:
            _MULTI_AVERAGE = True
    elif not args.with_multi:
        runs = [r for r in runs if not _is_multi_topic(r)]

    threshold = args.forget_reph_threshold / 100.0 if args.forget_reph_threshold is not None else None
    if args.forget_reph_filter is not None:
        runs = _apply_forget_reph_filter(runs, args.forget_reph_filter / 100.0)

    cols = resolve_cols(args.cols)

    if not runs:
        print("No runs match the current filters.")
        return

    # Sort: topic → model → pretrained first → method → lr → scale.
    # Relearn runs are inserted immediately after their parent run.
    def _base_sort_key(r):
        p = parse_exp(r["exp"], is_relearn=False)
        try:    lr_f = float(p["lr"]) if p["lr"] else 0.0
        except: lr_f = 0.0
        try:    scale_i = int(p["scale"]) if p["scale"] else 0
        except: scale_i = 0
        is_pretrained = 0 if r["method"] == "pretrained" else 1
        # Use eval topic for multi-topic runs so they sort alongside standard runs for same topic
        eff_topic = _get_run_eval_topic(r)
        return (eff_topic, r["model"], is_pretrained, r["method"], lr_f, scale_i)

    base_runs = [r for r in runs if not r.get("is_relearn")]
    relearn_runs = [r for r in runs if r.get("is_relearn")]
    base_runs.sort(key=_base_sort_key)

    # Build ordered list: each base run followed immediately by its relearn children (sorted by lr asc)
    relearn_by_parent: dict = {}
    for r in relearn_runs:
        relearn_by_parent.setdefault(r.get("parent_key", ""), []).append(r)

    def _rl_lr(r):
        p = parse_exp(r.get("relearn_exp") or r["exp"], is_relearn=True)
        try: return float(p["lr"]) if p["lr"] else 0.0
        except: return 0.0

    for children in relearn_by_parent.values():
        children.sort(key=_rl_lr)

    runs = []
    for r in base_runs:
        runs.append(r)
        runs.extend(relearn_by_parent.get(r["key"], []))

    if args.latex_dir:
        export_latex(runs, args.latex_dir, breakdown=args.breakdown,
                     ablation=args.latex_ablation)
        return

    if args.latex_combined_dir:
        export_latex_combined(runs, args.latex_combined_dir)
        return

    if args.latex_multi_dir:
        export_latex_multi(args.latex_multi_dir, breakdown=args.breakdown)
        return

    if args.latex_multi_relearn_dir:
        export_latex_multi_relearn(args.latex_multi_relearn_dir)
        return

    if args.breakdown in ("hparam", "adversarial"):
        print(f"--breakdown {args.breakdown} is only valid with --latex-dir or --latex-multi-dir")
        sys.exit(1)
    elif args.breakdown:
        print(f"\nDB last updated: {db.get('last_updated', '?')}")
        render_breakdown(runs, args.breakdown, threshold)
    else:
        print(f"\nDB last updated: {db.get('last_updated', '?')}\n")
        render_table(runs, cols, threshold)


if __name__ == "__main__":
    main()
