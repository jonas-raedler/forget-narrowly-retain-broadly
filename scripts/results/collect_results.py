"""
collect_results.py — Crawl evaluations/ and build/update evaluations/results_db.json.

Usage:
    python scripts/results/collect_results.py
    python scripts/results/collect_results.py --eval-dir path/to/evaluations
    python scripts/results/collect_results.py --reset    # start fresh (discard existing DB)
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

EVAL_DIR_DEFAULT = Path(__file__).parent.parent.parent / "evaluations"  # repo-root evaluations/
DB_FILENAME = "results_db.json"

KNOWN_TASKS = {"forget_rephrasings", "retain", "old_retain",
               "retain_train_rephrasing",
               "forget_rephrasings_gibberish",
               "retain_gibberish",
               "forget_adversarial", "forget_adversarial_combined"}
# `old_retain` holds the wide-retain metric; it appears only on runs that
# recorded it and is not produced by the current eval pipeline.

FORGET_KEYS = {
    "J_P_Indirect", "J_P_Direct", "J_P_WC", "J_P_WC_DI",
    "J_ICR_Indirect", "J_ICR_Direct", "J_ICR_WC", "J_ICR_WC_DI",
    "J_W_Indirect", "J_W_Direct", "J_W_Total", "J_W_DI",
    # Reverse-question metric (paper metric QR). worst_eval writes the canonical
    # "*_Reverse" keys; "*_Opposite" is also accepted (some worst_case files use it).
    "J_P_Reverse", "J_P_Reverse_Delta", "J_P_Opposite", "J_P_Opposite_Delta",
    "J_ICR_Reverse", "J_ICR_Reverse_Delta", "J_ICR_Opposite", "J_ICR_Opposite_Delta",
    "J_W_Reverse", "J_W_Reverse_Delta", "J_W_Opposite", "J_W_Opposite_Delta",
}
RETAIN_SKIP = {"Model", "Task_name", "Set", "has_reverse", "has_opposite"}


# ---------------------------------------------------------------------------
# Path parsing
# ---------------------------------------------------------------------------

def parse_worstcase_path(path: Path, worst_base: Path):
    """
    Parse a file inside worstCase/ and return run metadata.

    Expected layouts (relative to worst_base):
        {topic}/{model}/{method}/{exp}/{task}/filename.jsonl
        {topic}/{model}/{method}/{exp}/relearn/{relearn_exp}/{task}/filename.jsonl

    Returns (topic, model, method, exp, is_relearn, relearn_exp, task)
    or None if the path doesn't match.
    """
    try:
        rel = path.relative_to(worst_base)
    except ValueError:
        return None

    parts = rel.parts  # excludes filename (we're passed the file path)
    # parts[-1] is the filename, parts[-2] is the task dir
    dirs = parts[:-1]  # directory components only

    if len(dirs) < 4:
        return None

    topic = dirs[0]
    model = dirs[1]
    method = dirs[2]

    # Pretrained layout: topic/model/pretrained/task/filename (no exp component)
    if len(dirs) == 4:
        task = dirs[3]
        if task not in KNOWN_TASKS:
            return None
        return topic, model, method, "", False, None, task

    # Check for cross-topic eval: dirs[3] starts with 'eval_'
    if len(dirs) >= 6 and dirs[3].startswith('eval_'):
        exp = f"{dirs[3]}/{dirs[4]}"
        task = dirs[5]
        is_relearn = False
        relearn_exp = None
    # Check for relearn under combined multi-topic model:
    #   comb_*/model/method/src_exp/relearn/eval_{topic}/relearn_exp/task
    # The eval_{topic} subdir is inserted by eval_full_pipeline for compound topics.
    elif len(dirs) >= 8 and dirs[4] == "relearn" and dirs[5].startswith("eval_"):
        exp = f"{dirs[5]}/{dirs[3]}"   # eval_{topic}/src_exp — matches DB key for parent
        relearn_exp = dirs[6]
        task = dirs[7]
        is_relearn = True
    # Check for relearn: dirs[4] == "relearn"
    elif len(dirs) >= 7 and dirs[4] == "relearn":
        exp = dirs[3]
        relearn_exp = dirs[5]
        task = dirs[6]
        is_relearn = True
    else:
        exp = dirs[3]
        task = dirs[4]
        is_relearn = False
        relearn_exp = None

    if task not in KNOWN_TASKS:
        return None

    return topic, model, method, exp, is_relearn, relearn_exp, task


def parse_metric_filename(filename: str, prefix: str, model: str):
    """
    Extract exp from a filename like: {PREFIX}_{model}_{exp}.jsonl
    Also handles pretrained: {PREFIX}_{model}.jsonl  → returns "".
    Model may contain hyphens but not underscores, so the split is safe.
    """
    stem = filename[: -len(".jsonl")]  # strip extension
    exact = f"{prefix}_{model}"
    if stem == exact:
        return ""  # pretrained: no exp suffix
    expected_start = f"{prefix}_{model}_"
    if not stem.startswith(expected_start):
        return None
    return stem[len(expected_start):]


def parse_metric_path(path: Path, base: Path, subfolder: str, prefix: str):
    """
    Parse mmluOutputs / repOutputs / rgqOutputs file.
    Layout: {subfolder}/{topic}/{model}/{method}/{filename}.jsonl
    Returns (topic, model, method, exp) or None.
    """
    try:
        rel = path.relative_to(base / subfolder)
    except ValueError:
        return None

    parts = rel.parts
    # Cross-topic eval adds an extra eval_{topic} directory level:
    # topic/model/method/eval_{eval_topic}/filename
    if len(parts) == 5 and parts[3].startswith('eval_'):
        topic, model, method = parts[0], parts[1], parts[2]
        exp_suffix = parse_metric_filename(parts[4], prefix, model)
        if exp_suffix is None:
            return None
        exp = f"{parts[3]}/{exp_suffix}" if exp_suffix else parts[3]
        return topic, model, method, exp

    if len(parts) != 4:  # topic/model/method/filename
        return None

    topic, model, method = parts[0], parts[1], parts[2]
    exp = parse_metric_filename(parts[3], prefix, model)
    if exp is None:
        return None

    return topic, model, method, exp


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _resolve_metric_path(path: Path, folder: Path, eval_dir: Path, subfolder: str, prefix: str):
    """Resolve a metric file path into (topic, model, method, exp, is_relearn, relearn_exp).

    Handles normal, single-topic relearn, and combined-topic relearn layouts:
      Normal:                  topic/model/method/file                          (4 parts)
      Cross-topic:             topic/model/method/eval_{t}/file                 (5 parts)
      Single-topic relearn:    topic/model/method/src_exp/relearn/file          (6 parts)
      Combined-topic relearn:  topic/model/method/src_exp/relearn/eval_{t}/file (7 parts)
    """
    try:
        rel = path.relative_to(folder)
    except ValueError:
        return None
    parts = rel.parts
    # Combined-topic relearn: topic/model/method/src_exp/relearn/eval_{topic}/file
    if len(parts) == 7 and parts[4] == "relearn" and parts[5].startswith("eval_"):
        topic, model, method, src_exp, eval_prefix = parts[0], parts[1], parts[2], parts[3], parts[5]
        relearn_exp = parse_metric_filename(parts[6], prefix, model)
        if relearn_exp is None:
            return None
        exp = f"{eval_prefix}/{src_exp}"   # mirrors worstCase key: eval_{topic}/src_exp
        return topic, model, method, exp, True, relearn_exp
    # Single-topic relearn: topic/model/method/src_exp/relearn/file
    if len(parts) == 6 and parts[4] == "relearn":
        topic, model, method, exp = parts[0], parts[1], parts[2], parts[3]
        relearn_exp = parse_metric_filename(parts[5], prefix, model)
        if relearn_exp is None:
            return None
        return topic, model, method, exp, True, relearn_exp
    parsed = parse_metric_path(path, eval_dir, subfolder, prefix)
    if parsed is None:
        return None
    topic, model, method, exp = parsed
    return topic, model, method, exp, False, None


def make_key(topic, model, method, exp, is_relearn=False, relearn_exp=None):
    base = f"{topic}/{model}/{method}/{exp}" if exp else f"{topic}/{model}/{method}"
    if is_relearn and relearn_exp:
        return f"{base}/relearn/{relearn_exp}"
    return base


def get_or_create_run(db, key, topic, model, method, exp, is_relearn, relearn_exp):
    if key not in db["runs"]:
        parent_key = make_key(topic, model, method, exp) if is_relearn else None
        db["runs"][key] = {
            "key": key,
            "topic": topic,
            "model": model,
            "method": method,
            "exp": exp,
            "is_relearn": is_relearn,
            "parent_key": parent_key,
            "relearn_exp": relearn_exp,
            "collected_at": datetime.now().isoformat(timespec="seconds"),
        }
        return db["runs"][key], True  # newly created
    return db["runs"][key], False


def _win_long(path: Path) -> str:
    """Return a path string that works even for paths > 260 chars on Windows.
    On non-Windows platforms the path is returned unchanged."""
    import sys
    s = str(path)
    if sys.platform == "win32" and len(s) > 259 and not s.startswith("\\\\"):
        return "\\\\" + "?\\" + s
    return s


def is_real_file(path: Path) -> bool:
    """os.path.is_file() that handles Windows MAX_PATH limits."""
    if path.is_file():
        return True
    import sys
    if sys.platform != "win32":
        return False
    return os.path.isfile(_win_long(path))


def load_json(path: Path):
    try:
        with open(_win_long(path), encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def extract_forget_metrics(data: dict):
    return {k: v for k, v in data.items() if k in FORGET_KEYS}


def extract_retain_metrics(data: dict):
    return {k: v for k, v in data.items() if k not in RETAIN_SKIP}


# ---------------------------------------------------------------------------
# Crawl functions
# ---------------------------------------------------------------------------

def crawl_worstcase(db: dict, eval_dir: Path, new_keys: set, updated_keys: set):
    worst_base = eval_dir / "worstCase"
    if not worst_base.exists():
        print("  [SKIP] worstCase/ not found")
        return

    for path in sorted(worst_base.rglob("*.jsonl")):
        if not is_real_file(path):
            continue
        result = parse_worstcase_path(path, worst_base)
        if result is None:
            continue
        topic, model, method, exp, is_relearn, relearn_exp, task = result

        key = make_key(topic, model, method, exp, is_relearn, relearn_exp)
        run, created = get_or_create_run(db, key, topic, model, method, exp, is_relearn, relearn_exp)

        data = load_json(path)
        if data is None:
            continue

        if task in ("forget_rephrasings", "forget_adversarial", "forget_adversarial_combined"):
            metrics = extract_forget_metrics(data)
        else:  # retain, retain_train_rephrasing
            metrics = extract_retain_metrics(data)

        if task not in run or run[task] != metrics:
            run[task] = metrics
            if created:
                new_keys.add(key)
            else:
                updated_keys.add(key)


def crawl_mmlu(db: dict, eval_dir: Path, new_keys: set, updated_keys: set):
    folder = eval_dir / "mmluOutputs"
    if not folder.exists():
        print("  [SKIP] mmluOutputs/ not found")
        return

    for path in sorted(folder.rglob("*.jsonl")):
        if not is_real_file(path):
            continue

        # MMLU_{model}[_{exp}].jsonl — use _resolve_metric_path so all
        # relearn layouts (including combined-topic) are handled correctly.
        resolved = _resolve_metric_path(path, folder, eval_dir, "mmluOutputs", "MMLU")
        if resolved is None:
            continue
        topic, model, method, exp, is_relearn, relearn_exp = resolved
        data = load_json(path)
        if data is None:
            continue
        key = make_key(topic, model, method, exp, is_relearn=is_relearn, relearn_exp=relearn_exp)
        run, created = get_or_create_run(db, key, topic, model, method, exp, is_relearn, relearn_exp)
        mmlu = {"acc": data.get("acc")}
        if run.get("mmlu") != mmlu:
            run["mmlu"] = mmlu
            if created:
                new_keys.add(key)
            else:
                updated_keys.add(key)


def crawl_rep(db: dict, eval_dir: Path, new_keys: set, updated_keys: set):
    folder = eval_dir / "repOutputs"
    if not folder.exists():
        print("  [SKIP] repOutputs/ not found")
        return

    for path in sorted(folder.rglob("*.jsonl")):
        if not is_real_file(path):
            continue

        resolved = _resolve_metric_path(path, folder, eval_dir, "repOutputs", "Rep")
        if resolved is None:
            continue
        topic, model, method, exp, is_relearn, relearn_exp = resolved
        key = make_key(topic, model, method, exp, is_relearn=is_relearn, relearn_exp=relearn_exp)
        run, created = get_or_create_run(db, key, topic, model, method, exp, is_relearn, relearn_exp)

        data = load_json(path)
        if data is None:
            continue

        rep = {"entropy": data.get("entropy")}
        if run.get("rep") != rep:
            run["rep"] = rep
            if created:
                new_keys.add(key)
            else:
                updated_keys.add(key)


def crawl_rgq(db: dict, eval_dir: Path, new_keys: set, updated_keys: set):
    db_key = "rgq_bi"
    # (subfolder, file_prefix) in INCREASING precedence — the last source crawled
    # overwrites earlier ones when a run has more than one:
    #   wrOutputs/WRnew3bi_*  : bidirectional RGQ stored under "winrate_bi"
    #   rgqOutputs/RGQbi_*    : bidirectional RGQ (preferred)
    sources = [("wrOutputs", "WRnew3bi"), ("rgqOutputs", "RGQbi")]

    any_found = False
    for subfolder, file_prefix in sources:
        folder = eval_dir / subfolder
        if not folder.exists():
            continue
        any_found = True
        for path in sorted(folder.rglob("*.jsonl")):
            if not is_real_file(path):
                continue
            # file_prefix selects only the matching variant (e.g. WRnew3bi excludes
            # the unidirectional "WR_"/"WRnew3_" and the "WRnew3bi300_" variant).
            resolved = _resolve_metric_path(path, folder, eval_dir, subfolder, file_prefix)
            if resolved is None:
                continue

            topic, model, method, exp, is_relearn, relearn_exp = resolved
            key = make_key(topic, model, method, exp, is_relearn=is_relearn, relearn_exp=relearn_exp)
            run, created = get_or_create_run(db, key, topic, model, method, exp, is_relearn, relearn_exp)

            data = load_json(path)
            if data is None:
                continue

            counts = data.get("counts", {})
            rgq = {
                "winrate": data.get("winrate"),
                "wins": counts.get("wins"),
                "losses": counts.get("losses"),
                "ties": counts.get("ties"),
            }
            if run.get(db_key) != rgq:
                run[db_key] = rgq
                if created:
                    new_keys.add(key)
                else:
                    updated_keys.add(key)

    if not any_found:
        print("  [SKIP] rgqOutputs/ and wrOutputs/ not found")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _coalesce_legacy_keys(obj):
    """Map alternate result-key spellings to their canonical names in-place, so an
    *incremental* collect keeps every run under one consistent key set:
      - retain lexical category: Cat_Words / Det_Words-* / WC_Words / Avg_Words -> *Lexical*
      - generation quality:      winrate_bi -> rgq_bi   (existing rgq_bi takes precedence)
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
    elif isinstance(obj, list):
        for it in obj:
            _coalesce_legacy_keys(it)


def main():
    parser = argparse.ArgumentParser(description="Collect evaluation results into results_db.json")
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR_DEFAULT,
                        help="Path to the evaluations directory (default: evaluations/)")
    parser.add_argument("--reset", action="store_true",
                        help="Discard existing DB and start fresh")
    args = parser.parse_args()

    eval_dir = args.eval_dir.resolve()
    db_path = eval_dir / DB_FILENAME

    if not eval_dir.exists():
        print(f"Error: eval-dir not found: {eval_dir}")
        return

    # Load or init DB
    if db_path.exists() and not args.reset:
        with open(db_path, encoding="utf-8") as f:
            db = json.load(f)
        print(f"Loaded existing DB: {len(db.get('runs', {}))} runs")
    else:
        db = {"version": 1, "runs": {}}
        if args.reset:
            print("Starting fresh (--reset)")

    new_keys: set = set()
    updated_keys: set = set()

    print(f"\nCrawling {eval_dir} ...")
    crawl_worstcase(db, eval_dir, new_keys, updated_keys)
    crawl_mmlu(db, eval_dir, new_keys, updated_keys)
    crawl_rep(db, eval_dir, new_keys, updated_keys)
    crawl_rgq(db, eval_dir, new_keys, updated_keys)
    propagate_pretrained_metrics(db, updated_keys)

    # Canonicalize key names in the freshly-built DB, for both incremental and
    # --reset runs. Some on-disk eval files carry the Cat_Words/Det_Words-* labels
    # (winrate_bi is already handled by crawl_rgq writing rgq_bi), so this rewrites
    # them to the canonical names so the DB always stores Cat_Lexical / rgq_bi
    # regardless of how it was built.
    _coalesce_legacy_keys(db)

    db["last_updated"] = datetime.now().isoformat(timespec="seconds")

    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    total = len(db["runs"])
    actually_new = new_keys - updated_keys
    print(f"\nDone. {len(actually_new)} new runs, {len(updated_keys)} updated, {total} total runs.")
    print(f"DB saved to: {db_path}")


def propagate_pretrained_metrics(db: dict, updated_keys: set):
    """Copy mmlu/rep from any pretrained entry to same-model pretrained entries missing them.

    MMLU and repetitiveness don't depend on topic, so one evaluation can be
    shared across all topics for the same model.
    """
    runs = db.get("runs", {})
    pretrained = {k: r for k, r in runs.items() if r.get("method") == "pretrained"}

    # Build best (non-None) source per model
    best: dict[str, dict] = {}  # model -> {mmlu, rep}
    for r in pretrained.values():
        model = r["model"]
        src = best.setdefault(model, {})
        if r.get("mmlu") and src.get("mmlu") is None:
            src["mmlu"] = r["mmlu"]
        if r.get("rep") and src.get("rep") is None:
            src["rep"] = r["rep"]

    # Propagate to entries that are missing either metric
    for key, r in pretrained.items():
        src = best.get(r["model"], {})
        changed = False
        for metric in ("mmlu", "rep"):
            if not r.get(metric) and src.get(metric):
                r[metric] = src[metric]
                changed = True
        if changed:
            updated_keys.add(key)


if __name__ == "__main__":
    main()
