"""
average_topics.py — Average evaluation metrics across the constituent topics of a
single multi-topic (seq_/comb_) unlearn run, then print a summary and save it.

The multi-topic unlearn scripts (`suite_sequential_unlearn.sh`,
`suite_combined_unlearn.sh`) evaluate a model on each constituent topic separately, so
the console only shows per-topic numbers. This tool reads those per-topic eval outputs
**directly** (no results_db.json dependency), averages the forget/retain metrics across
topics, prints a banner, and saves a self-describing JSON to a folder that
`collect_results.py` never crawls — so it cannot perturb the results DB.

Usage:
    python scripts/results/average_topics.py \
        --multi-topic comb_A+B+C --model Llama-3.2-3B-Instruct --method jensen \
        --exp epochs_20_lrs_3e-6_..._push_to_prefix
    # --eval-dir defaults to repo-root evaluations/; --exp may be omitted if a single
    # exp exists under the first topic's eval dir (auto-discovered).

Output: evaluations/topicAverages/<multitopic>/<model>/<method>/<exp>/topic_average.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Sibling imports (scripts/results/) — reuse the existing extraction + averaging logic
# instead of duplicating it. show_results.py is stdlib-only at import time and its CLI
# body is guarded by __main__, so importing it has no side effects.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_results import (  # noqa: E402
    extract_forget_metrics,
    extract_retain_metrics,
    parse_metric_filename,
    load_json,
    is_real_file,
    _win_long,
)
from show_results import _avg_task_dicts  # noqa: E402

EVAL_DIR_DEFAULT = Path(__file__).resolve().parent.parent.parent / "evaluations"

# Tasks to average across topics — mirrors `_build_avg_run` in show_results.py.
# The three forget-judge tasks use the forget extractor (FORGET_KEYS); everything else
# uses the retain extractor (all keys except RETAIN_SKIP) — same split collect_results uses.
_FORGET_EXTRACT_TASKS = ("forget_rephrasings", "forget_adversarial", "forget_adversarial_combined")
_AVG_TASKS = ("forget_rephrasings", "forget_rephrasings_gibberish",
              "forget_adversarial", "forget_adversarial_combined",
              "retain", "retain_train_rephrasing")

def _build_rgq(d):
    counts = d.get("counts") or {}
    return {"winrate": d.get("winrate"),
            **{k: counts.get(k) for k in ("wins", "losses", "ties")}}


# Utility metrics (model-level — identical across a run's topics). For each metric:
#   (db_key, builder, [(folder, file_prefix), ...] in precedence order)
# The (folder, prefix) list mirrors collect_results.py exactly. rgq_bi has two sources:
# `rgqOutputs/RGQbi_*` (preferred) and `wrOutputs/WRnew3bi_*`. The prefix is required
# to pick the right variant — e.g. WRnew3bi excludes the unidirectional
# `WR_`/`WRnew3_` and the `WRnew3bi300_` files in the same folder.
_UTIL_SOURCES = [
    ("mmlu",   lambda d: {"acc": d.get("acc")},         [("mmluOutputs", "MMLU")]),
    ("rep",    lambda d: {"entropy": d.get("entropy")}, [("repOutputs", "Rep")]),
    ("rgq_bi", _build_rgq,                               [("rgqOutputs", "RGQbi"),
                                                          ("wrOutputs", "WRnew3bi")]),
]


def parse_multi_topic(multitopic: str):
    """('comb_A+B+C') -> ('comb', ['A','B','C']). Returns (None, [topic]) if not multi."""
    if multitopic.startswith("seq_"):
        return "seq", multitopic[len("seq_"):].split("+")
    if multitopic.startswith("comb_"):
        return "comb", multitopic[len("comb_"):].split("+")
    return None, [multitopic]


def _list_jsonl(dir_path: Path):
    """List *.jsonl files in dir_path, Windows-long-path safe. [] if dir is missing."""
    try:
        with os.scandir(_win_long(str(dir_path))) as it:
            return [dir_path / e.name for e in it
                    if e.name.endswith(".jsonl") and e.is_file()]
    except (FileNotFoundError, NotADirectoryError, OSError):
        return []


def _read_first_jsonl(dir_path: Path):
    """Load the single JSON object from the first *.jsonl in dir_path (or None)."""
    files = _list_jsonl(dir_path)
    if not files:
        return None
    f = files[0]
    if not is_real_file(f):
        return None
    return load_json(f)


def _read_metric_jsonl(dir_path: Path, model: str, prefix: str):
    """Load the metric file in dir_path whose name matches `{prefix}_{model}[_exp].jsonl`.
    The prefix match is what excludes sibling variants in the same folder (e.g. picking
    WRnew3bi while skipping WR_/WRnew3_/WRnew3bi300_). Returns None if no match."""
    for f in _list_jsonl(dir_path):
        if parse_metric_filename(f.name, prefix, model) is not None and is_real_file(f):
            return load_json(f)
    return None


def discover_exp(method_dir: Path, first_topic: str):
    """Auto-discover the exp name: the single non-'relearn' subdir under eval_<first_topic>/."""
    topic_dir = method_dir / f"eval_{first_topic}"
    try:
        with os.scandir(_win_long(str(topic_dir))) as it:
            subs = [e.name for e in it if e.is_dir() and e.name != "relearn"]
    except OSError:
        subs = []
    if len(subs) == 1:
        return subs[0]
    if not subs:
        raise SystemExit(f"[ERROR] no exp dir found under {topic_dir}; pass --exp explicitly")
    raise SystemExit(f"[ERROR] multiple exp dirs under {topic_dir}: {subs}; pass --exp explicitly")


def collect_topic_metrics(worst_method_dir: Path, eval_topic: str, exp: str):
    """Read the per-task metric dicts for one eval topic. Returns {task: metrics}."""
    out = {}
    base = worst_method_dir / f"eval_{eval_topic}" / exp
    for task in _AVG_TASKS:
        data = _read_first_jsonl(base / task)
        if data is None:
            continue
        if task in _FORGET_EXTRACT_TASKS:
            out[task] = extract_forget_metrics(data)
        else:
            out[task] = extract_retain_metrics(data)
    return out


def collect_utility(eval_dir: Path, multitopic: str, model: str, method: str, topics: list):
    """Best-effort read of model-level utility metrics. These are computed once per run
    and are identical across its topics (the unlearn scripts write them under the *first*
    eval topic — which is NOT necessarily the alphabetically-first one in the comb_/seq_
    name — then symlink to the rest). So scan ALL topics and use the first file found per
    metric. Returns (util_dict, source_topic_per_metric)."""
    util, sources = {}, {}
    for db_key, builder, folder_prefixes in _UTIL_SOURCES:
        found = False
        for folder, prefix in folder_prefixes:          # precedence order (preferred source first)
            for topic in topics:
                d = _read_metric_jsonl(
                    eval_dir / folder / multitopic / model / method / f"eval_{topic}",
                    model, prefix)
                if d is not None:
                    util[db_key] = builder(d)
                    sources[db_key] = f"{folder}/eval_{topic}"
                    found = True
                    break
            if found:
                break
    return util, sources


def _pct(x):
    return f"{x * 100:5.1f}" if isinstance(x, (int, float)) else "  --"


def _fmt_per_topic(per_topic, task, key):
    parts = []
    for t, tasks in per_topic.items():
        v = (tasks.get(task) or {}).get(key)
        parts.append(f"{_short(t)}:{_pct(v).strip()}")
    return "  ".join(parts)


def _short(topic: str, n: int = 10):
    return topic if len(topic) <= n else topic[:n - 1] + "."


def print_banner(multitopic, multi_type, model, method, exp, eval_topics, per_topic, avg, util, util_src):
    line = "=" * 72
    print(f"\n{line}")
    print(f" AVERAGE ACROSS TOPICS - {multitopic}")
    print(line)
    print(f" model={model}  method={method}")
    print(f" exp={exp}")
    print(f" topics ({len(eval_topics)}): {', '.join(eval_topics)}")
    print("-" * 72)
    rows = [
        ("forget WC   (J_W_Total)", "forget_rephrasings", "J_W_Total"),
        ("forget DI   (J_W_DI)   ", "forget_rephrasings", "J_W_DI"),
        ("forget adv  (J_W_Total)", "forget_adversarial", "J_W_Total"),
        ("retain      (J_avg)    ", "retain", "J_avg"),
    ]
    for label, task, key in rows:
        if task not in avg:
            continue
        mean = avg[task].get(key)
        print(f" {label}: mean={_pct(mean)}%   [{_fmt_per_topic(per_topic, task, key)}]")
    # Utility (model-level). Always print the line; show "n/a" + source when found.
    print("-" * 72)
    mmlu = (util.get("mmlu") or {}).get("acc")
    rep = (util.get("rep") or {}).get("entropy")
    rgq = (util.get("rgq_bi") or {}).get("winrate")
    print(" utility (model-level, identical across topics; read from the topic shown):")
    print(f"   mmlu.acc       = {_pct(mmlu)}%   {_src_note(util_src, 'mmlu')}")
    print(f"   rep.entropy    = {_num(rep)}   {_src_note(util_src, 'rep')}")
    print(f"   rgq_bi.winrate = {_pct(rgq)}%   {_src_note(util_src, 'rgq_bi')}")
    print(line + "\n")


def _num(x):
    return f"{x:6.3f}" if isinstance(x, (int, float)) else "   n/a"


def _src_note(util_src, key):
    t = util_src.get(key)
    return f"(from {t})" if t else "(not found in mmlu/rep/rgq/wrOutputs)"


def main():
    ap = argparse.ArgumentParser(description="Average eval metrics across a multi-topic run's topics.")
    ap.add_argument("--eval-dir", default=str(EVAL_DIR_DEFAULT),
                    help="evaluations/ root (default: repo-root evaluations/)")
    ap.add_argument("--multi-topic", required=True,
                    help="multi-topic dir name, e.g. comb_A+B+C or seq_A+B")
    ap.add_argument("--model", required=True, help="model config name, e.g. Llama-3.2-3B-Instruct")
    ap.add_argument("--method", required=True, help="method folder, e.g. jensen")
    ap.add_argument("--exp", default=None,
                    help="exp suffix; auto-discovered from the first topic if omitted")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    multitopic = args.multi_topic
    multi_type, eval_topics = parse_multi_topic(multitopic)

    if len(eval_topics) < 2:
        print(f"[INFO] '{multitopic}' has < 2 topics - nothing to average. Skipping.")
        return 0

    worst_method_dir = eval_dir / "worstCase" / multitopic / args.model / args.method
    exp = args.exp or discover_exp(worst_method_dir, eval_topics[0])

    # Per-topic metric dicts
    per_topic = {}
    for t in eval_topics:
        m = collect_topic_metrics(worst_method_dir, t, exp)
        if m:
            per_topic[t] = m
        else:
            print(f"[WARN] no metrics found for topic '{t}' under {worst_method_dir / ('eval_' + t) / exp}")

    if not per_topic:
        print(f"[ERROR] no per-topic metrics found for {multitopic}/{args.model}/{args.method}/{exp} - nothing saved.")
        return 1

    # Average each task across the topics that have it
    avg = {}
    for task in _AVG_TASKS:
        dicts = [tasks[task] for tasks in per_topic.values() if task in tasks]
        if dicts:
            avg[task] = _avg_task_dicts(dicts)

    # Utility is identical across topics — scan ALL of them, not just the first.
    util, util_src = collect_utility(eval_dir, multitopic, args.model, args.method, eval_topics)

    print_banner(multitopic, multi_type, args.model, args.method, exp,
                 list(per_topic.keys()), per_topic, avg, util, util_src)

    # Save to a folder collect_results.py never crawls (worstCase/mmluOutputs/repOutputs/rgqOutputs only).
    out_dir = eval_dir / "topicAverages" / multitopic / args.model / args.method / exp
    out_path = out_dir / "topic_average.json"
    payload = {
        "note": "Auto-generated by scripts/results/average_topics.py. NOT a run; "
                "not read by collect_results.py. Averages metrics across the topics of one multi-topic run.",
        "generated_by": "scripts/results/average_topics.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "multi_type": multi_type,
        "multitopic": multitopic,
        "model": args.model,
        "method": args.method,
        "exp": exp,
        "eval_topics": list(per_topic.keys()),
        "n_topics": len(per_topic),
        "average": avg,
        "utility": util,
        "utility_source_topic": util_src,
        "per_topic": per_topic,
    }
    try:
        os.makedirs(_win_long(str(out_dir)), exist_ok=True)
        with open(_win_long(str(out_path)), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[SAVED] {out_path}")
    except OSError as e:
        print(f"[WARN] could not save topic average to {out_path}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
