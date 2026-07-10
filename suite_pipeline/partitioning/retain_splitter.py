"""
Retain set partitioning for suite_pipeline: splits the semantic, general-knowledge,
lexical, and syntax retain sets into train/eval partitions.
"""
from __future__ import annotations
import re
import random
from pathlib import Path

from suite_pipeline.io_utils.loaders import load_json
from suite_pipeline.io_utils.exporters import save_json
from suite_pipeline.config import PipelineConfig


def _parse_semantic_level(key: str) -> int:
    """
    Extract the numeric level from a semantic-topic key.

    Keys are expected to start with a number followed by a dash or space,
    e.g. "1-Columbia Disaster", "11-Deepwater Horizon", "0-The Challenger".
    If no leading number is found, returns -1 (treated as unknown / level 0).
    """
    m = re.match(r'^(\d+)', key.strip())
    return int(m.group(1)) if m else -1


def _split_semantic(semantic_json: dict, cfg: PipelineConfig):
    """
    Split the semantic retain set by topic level.

    Level 0 behavior is controlled by cfg.semantic_level_0_policy:
      "none"  -> skip entirely
      "both"  -> same questions in both train and eval
      "split" -> different questions in train vs eval (like levels 1-10)

    Levels 1-10: randomly split cfg.semantic_levels_1_10_train / cfg.semantic_levels_1_10_eval.
    Levels 11+:  eval only.
    """
    train_rows, eval_rows = [], []
    for topic, qas in semantic_json.items():
        level = _parse_semantic_level(topic)
        rows_with_label = [
            {"question": qa["question"], "answer": qa["answer"], "label": f"Semantic-{topic}"}
            for qa in qas
        ]
        if level == 0:
            policy = cfg.semantic_level_0_policy
            if policy == "none":
                pass
            elif policy == "both":
                train_rows.extend(rows_with_label[:cfg.semantic_level_0_train])
                eval_rows.extend(rows_with_label[:cfg.semantic_level_0_train])
            elif policy == "split":
                random.shuffle(rows_with_label)
                n_train = cfg.semantic_level_0_train
                train_rows.extend(rows_with_label[:n_train])
                eval_rows.extend(rows_with_label[n_train:n_train + cfg.semantic_levels_1_10_eval])
            else:
                raise ValueError(
                    f"Unknown semantic_level_0_policy: '{policy}'. "
                    f"Must be 'none', 'both', or 'split'."
                )
        elif 1 <= level <= 10:
            random.shuffle(rows_with_label)
            n_train = cfg.semantic_levels_1_10_train
            train_rows.extend(rows_with_label[:n_train])
            eval_rows.extend(rows_with_label[n_train:n_train + cfg.semantic_levels_1_10_eval])
        else:
            eval_rows.extend(rows_with_label)
    print(f"  Semantic: {len(train_rows)} train, {len(eval_rows)} eval")
    return train_rows, eval_rows


def _split_lexical(lexical_json: list, cfg):
    """
    Split lexical (word-level) retain into train and eval.

    Each lexical group's questions are shuffled before splitting so that
    train and eval receive a uniform random sample rather than always
    taking the first/second half of the JSON order.

    Emits ``Lexical-<target>`` labels. Downstream consumers also accept the
    ``Words-<target>`` form.
    """
    train_rows, eval_rows = [], []
    for row_group in lexical_json:
        target = row_group.get("target_string", "unknown")
        pairs  = list(row_group.get("generated_pairs", []))
        random.shuffle(pairs)
        mid = len(pairs) // 2
        for qa in pairs[:mid]:
            train_rows.append({
                "question": qa["question"], "answer": qa["answer"],
                "label": f"Lexical-{target}",
            })
        for qa in pairs[mid:]:
            eval_rows.append({
                "question": qa["question"], "answer": qa["answer"],
                "label": f"Lexical-{target}",
            })
    print(f"  Lexical: {len(train_rows)} train, {len(eval_rows)} eval")
    return train_rows, eval_rows


def _split_general_knowledge(gk_json: dict, cfg):
    """
    Split GK by topic, keeping train and eval topics strictly separated.

    Topics are interleaved by dict-insertion order:
      even-indexed topics (0, 2, 4, ...) → train
      odd-indexed  topics (1, 3, 5, ...) → eval

    cfg.gk_questions_per_train_topic questions are sampled per train topic.
    cfg.gk_questions_per_eval_topic  questions are sampled per eval topic.

    Also produces eval_short_rows: the first 2 questions per eval topic (by JSON
    insertion order), assumed to have shorter answers than the rest.  A warning
    is printed for any topic where this assumption does not hold; if it holds
    everywhere a single OK line is printed instead.

    With 100 topics, 1 question/train-topic, 3 questions/eval-topic this yields
    50 train / 150 eval counts, with zero topic overlap between the two sets.

    Topic assignment is determined by position in the dict (no shuffle needed),
    so it is fully deterministic and stable across different unlearning datasets
    as long as the GK JSON file stays the same.
    """
    train_rows, eval_rows, eval_short_rows = [], [], []
    all_short_ok = True
    for i, (topic, qas) in enumerate(gk_json.items()):
        rows = [
            {"question": qa["question"], "answer": qa["answer"], "label": f"GK-{topic}"}
            for qa in qas
        ]
        if i % 2 == 0:  # train topic
            n = cfg.gk_questions_per_train_topic
            sampled = random.sample(rows, min(n, len(rows)))
            train_rows.extend(sampled)
        else:  # eval topic
            n = cfg.gk_questions_per_eval_topic
            sampled = random.sample(rows, min(n, len(rows)))
            eval_rows.extend(sampled)

            # eval_short: first 2 by JSON order
            short = rows[:2]
            eval_short_rows.extend(short)

            # Validate: max answer length among first 2 should be < min among the rest
            if len(rows) > 2:
                max_short = max(len(r["answer"]) for r in short)
                min_rest  = min(len(r["answer"]) for r in rows[2:])
                if max_short >= min_rest:
                    all_short_ok = False
                    print(
                        f"  [GK short WARNING] topic '{topic}': first-2 answers are not "
                        f"all shorter than the rest "
                        f"(max_len_first_2={max_short}, min_len_rest={min_rest})"
                    )

    if all_short_ok:
        print("  GK short: all eval topics OK — first 2 questions have shorter answers than the rest")
    print(
        f"  GK: {len(train_rows)} train ({len(gk_json) // 2 + len(gk_json) % 2} topics × "
        f"{cfg.gk_questions_per_train_topic}), "
        f"{len(eval_rows)} eval ({len(gk_json) // 2} topics × "
        f"{cfg.gk_questions_per_eval_topic}), "
        f"{len(eval_short_rows)} eval_short ({len(gk_json) // 2} topics × 2)"
    )
    return train_rows, eval_rows, eval_short_rows


def _split_syntax(syntax_json: list, cfg: PipelineConfig):
    """
    Split syntactic rephrasings into train / eval.

    Flat per-question format (one row per original question):
      {label, question, answer,
       q_llama1, q_llama1_answer, q_llama2, q_llama2_answer, ...,
       q_phi1, q_phi1_answer, ...,
       blank_llama1, blank_llama1_answer, ...}

    Each q_*/blank_* key has a sibling <key>_answer with its own independent answer.
    Falls back to the row's main answer if the sibling key is absent.

    The ({fact_id, generated_pairs}) input format is also supported.

    Split: first cfg.syntax_eval_n rephrases → eval; the rest → train.
    """
    train_rows, eval_rows = [], []

    for row in syntax_json:
        if "generated_pairs" in row:
            # ---- {fact_id, generated_pairs} format -----------------------
            fact_id = row.get("fact_id", "unknown")
            pairs   = row["generated_pairs"]
            for i, qa in enumerate(pairs):
                entry = {
                    "question": qa["question"],
                    "answer":   qa["answer"],
                    "label":    f"Syntax-{fact_id}",
                }
                if i < cfg.syntax_eval_n:
                    eval_rows.append(entry)
                elif i < cfg.syntax_eval_n + cfg.syntax_train_n:
                    train_rows.append(entry)
        else:
            # ---- Flat per-question format --------------------------------
            label       = row.get("label", "unknown")
            main_answer = row.get("answer", "")

            # Collect all rephrase (key, question_text) pairs, excluding *_answer siblings
            rephrase_pairs = [
                (k, row[k])
                for k in row
                if re.match(r'^(q_|blank_)', k)
                and not k.endswith('_answer')
                and isinstance(row[k], str)
                and row[k].strip()
            ]
            # Include the original question as one of the rephrase options,
            # then shuffle everything together for a uniform train/eval split.
            rephrase_pairs.append(("original", row.get("question", "")))
            random.shuffle(rephrase_pairs)

            for i, (rkey, q) in enumerate(rephrase_pairs):
                entry_answer = row.get(f"{rkey}_answer", main_answer)
                entry = {
                    "question": q,
                    "answer":   entry_answer,
                    "label":    f"Syntax-{label}@{rkey}",
                }
                if i < cfg.syntax_eval_n:
                    eval_rows.append(entry)
                else:
                    train_rows.append(entry)

    print(f"  Syntax: {len(train_rows)} train, {len(eval_rows)} eval")
    return train_rows, eval_rows


def partition_retain_sets(cfg: PipelineConfig) -> dict[str, list[dict]]:
    """
    Step 1: Partition all retain sets (semantic + GK + lexical + syntax).

    Writes each partition to step1_partitions/ and returns the full dict.
    Semantic is loaded from disk if already written (allows re-running step 1
    without re-computing the semantic split).
    """
    cfg.apply_seed()
    print("=" * 60)
    print("STEP 1: Partition retain sets")
    print("=" * 60)

    out_dir = Path(cfg.output_dir) / "step1_partitions"
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[dict]] = {}

    # ---- Semantic ----------------------------------------------------
    sem_train_path = out_dir / "semantic_train.json"
    sem_eval_path  = out_dir / "semantic_eval.json"
    if sem_train_path.exists() and sem_eval_path.exists():
        result["semantic_train"] = load_json(str(sem_train_path))
        result["semantic_eval"]  = load_json(str(sem_eval_path))
        print(
            f"  Semantic: loaded from disk "
            f"({len(result['semantic_train'])} train, {len(result['semantic_eval'])} eval)"
        )
    elif cfg.retain_semantic_json_path:
        semantic_json = load_json(cfg.retain_semantic_json_path)
        if isinstance(semantic_json, list) and len(semantic_json) == 1 and isinstance(semantic_json[0], dict):
            semantic_json = semantic_json[0]
        elif isinstance(semantic_json, list):
            semantic_json = {str(i): (v if isinstance(v, list) else [v]) for i, v in enumerate(semantic_json)}

        if cfg.retain_semantic_level0_json_path:
            level0 = load_json(cfg.retain_semantic_level0_json_path)
            if isinstance(level0, list):
                # Label = cfg override if set, else filename stem
                name = (cfg.retain_semantic_level0_label
                        or Path(cfg.retain_semantic_level0_json_path).stem)
                semantic_json = {f"0-{name}": level0, **semantic_json}
                print(f"  Loaded {len(level0)} level-0 questions, labelled '0-{name}'")

        sem_train, sem_eval = _split_semantic(semantic_json, cfg)
        result["semantic_train"] = sem_train
        result["semantic_eval"]  = sem_eval

    # ---- GK ----------------------------------------------------------
    if cfg.retain_gk_json_path:
        gk_data = load_json(cfg.retain_gk_json_path)
        if isinstance(gk_data, list) and len(gk_data) == 1 and isinstance(gk_data[0], dict):
            gk_data = gk_data[0]
        elif isinstance(gk_data, list):
            gk_data = {str(i): (v if isinstance(v, list) else [v]) for i, v in enumerate(gk_data)}
        gk_train, gk_eval, gk_eval_short = _split_general_knowledge(gk_data, cfg)
        result["gk_train"]      = gk_train
        result["gk_eval"]       = gk_eval
        result["gk_eval_short"] = gk_eval_short

    # ---- Lexical -----------------------------------------------------
    if cfg.retain_lexical_json_path:
        lexical_data = load_json(cfg.retain_lexical_json_path)
        lexical_train, lexical_eval = _split_lexical(lexical_data, cfg)
        result["lexical_train"] = lexical_train
        result["lexical_eval"]  = lexical_eval

    # ---- Syntax (per-rephrase answers) -------------------
    if cfg.retain_syntax_json_path:
        syntax_data = load_json(cfg.retain_syntax_json_path)
        syntax_train, syntax_eval = _split_syntax(syntax_data, cfg)
        result["syntax_train"] = syntax_train
        result["syntax_eval"]  = syntax_eval

    # Save all partitions
    for key, rows in result.items():
        save_json(rows, out_dir / f"{key}.json")

    total_train = sum(len(v) for k, v in result.items() if k.endswith("_train"))
    total_eval  = sum(len(v) for k, v in result.items() if k.endswith("_eval"))
    print(f"\nStep 1 complete: {total_train} total retain train, {total_eval} total retain eval")
    return result