"""
Step 2: Build syntax pairs and organise GK / Lexical retain rows.

Syntax:  deterministic forget↔retain pairing via the rephrase key encoded in
         each syntax retain label (e.g. "Syntax-M1-direct@q_llama2" → use
         M1-direct's q_llama2 rephrase).  Up to syntax_train_n retain entries
         per base forget label.

GK / Lexical:  retain rows are returned in their natural partition order.
               Forget assignment and reverse-question injection are handled
               entirely in the final assembly step (dataset_builder.py) via the
               uniform 15-block scheme, so no forget cycling or reverse-question
               injection happens here.

Output: step2_sampled/non_semantic_mappings.json
  {
    "syntax_pairs": [ {"forget_row": {...}, "retain_row": {...}}, ... ],
    "gk_rows":      [ {"question": ..., "answer": ..., "label": ...}, ... ],
    "lexical_rows": [ {"question": ..., "answer": ..., "label": ...}, ... ],
  }
"""
from __future__ import annotations
import re
import random
from collections import defaultdict
from pathlib import Path

from suite_pipeline.io_utils.exporters import save_json
from suite_pipeline.config import PipelineConfig


def _row(question: str, answer: str, label: str) -> dict:
    return {"question": question, "answer": answer, "label": label}


def build_non_semantic_mappings(
    cfg: PipelineConfig,
    forget_train_rephrases: list[dict],
    syntax_train: list[dict],
    gk_train: list[dict],
    lexical_train: list[dict],
) -> dict:
    """
    Build syntax forget↔retain pairs and organise GK / Lexical retain rows.

    forget_train_rephrases: raw list of {label, question, answer, q_*/blank_* ...}
    Returns a dict with keys: syntax_pairs, gk_rows, lexical_rows.

    Forget assignment for GK and Lexical rows (including reverse-question
    injection) is deferred to final assembly, which uses the uniform 15-block
    scheme.
    """
    forget_by_label = {e["label"]: e for e in forget_train_rephrases}

    # ---- Syntax pairs ------------------------------------------------
    # Group syntax retain entries by their base forget label.
    # Label format: "Syntax-M1-direct@q_llama2"  →  base="M1-direct", key="q_llama2"
    by_base: dict[str, list[dict]] = defaultdict(list)
    for item in syntax_train:
        stripped = re.sub(r'^Syntax-', '', item.get("label", ""))
        base = stripped.split("@")[0]
        by_base[base].append(item)

    n_valid = cfg.syntax_train_n
    syntax_pairs: list[dict] = []
    # Track which forget keys were used per label in syntax rows, so that
    # block direct cycling in assembly can avoid repeating those keys first.
    syntax_keys_by_label: dict[str, list[str]] = {}

    for base_label, syntax_items in by_base.items():
        forget_entry = forget_by_label.get(base_label)
        if not forget_entry:
            continue

        # Pre-collect forget keys by type for sequential type-matched assignment
        q_keys_f     = sorted(k for k in forget_entry
                              if k.startswith("q_") and not k.endswith("_answer")
                              and isinstance(forget_entry.get(k), str) and forget_entry[k].strip())
        blank_keys_f = sorted(k for k in forget_entry
                              if k.startswith("blank_") and not k.endswith("_answer")
                              and isinstance(forget_entry.get(k), str) and forget_entry[k].strip())
        type_counters: dict[str, int] = {"q": 0, "blank": 0}

        random.shuffle(syntax_items)
        selected = syntax_items[:n_valid]
        while len(selected) < n_valid and selected:
            selected.append(selected[0])

        for ret_item in selected:
            ret_label    = ret_item.get("label", "")
            rephrase_key = ret_label.split("@", 1)[1] if "@" in ret_label else "original"

            # Map the retain key's type to an actual forget rephrase of the same type.
            # This ensures syntax forget rows use real forget rephrases (not fallback
            # to original) and consume positions 0..n_syntax_per-1 of the key cycle,
            # leaving positions n_syntax_per..14 for block direct rows.
            if rephrase_key == "original":
                actual_key = "original"
            elif rephrase_key.startswith("q_"):
                if rephrase_key in forget_entry and forget_entry[rephrase_key].strip():
                    actual_key = rephrase_key          # exact key exists in forget entry
                elif q_keys_f:
                    actual_key = q_keys_f[type_counters["q"] % len(q_keys_f)]
                    type_counters["q"] += 1
                else:
                    actual_key = "original"
            elif rephrase_key.startswith("blank_"):
                if rephrase_key in forget_entry and forget_entry[rephrase_key].strip():
                    actual_key = rephrase_key
                elif blank_keys_f:
                    actual_key = blank_keys_f[type_counters["blank"] % len(blank_keys_f)]
                    type_counters["blank"] += 1
                else:
                    actual_key = "original"
            else:
                actual_key = "original"

            f_question = forget_entry["question"] if actual_key == "original" \
                         else forget_entry.get(actual_key, forget_entry["question"])
            syntax_pairs.append({
                "forget_row": _row(f_question, forget_entry["answer"], f"{base_label}@{actual_key}"),
                "retain_row": _row(ret_item["question"], ret_item["answer"], ret_label),
            })
            syntax_keys_by_label.setdefault(base_label, []).append(actual_key)

    # ---- GK / Lexical rows (retain side only; forget assigned in assembly) --
    gk_rows      = [_row(i["question"], i["answer"], i.get("label", "")) for i in gk_train]
    lexical_rows = [_row(i["question"], i["answer"], i.get("label", "")) for i in lexical_train]

    mappings = {
        "syntax_pairs":         syntax_pairs,
        "gk_rows":              gk_rows,
        "lexical_rows":         lexical_rows,
        "syntax_keys_by_label": syntax_keys_by_label,
    }

    out_dir = Path(cfg.output_dir) / "step2_sampled"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(mappings, out_dir / "non_semantic_mappings.json")
    print(
        f"  Non-semantic mappings: {len(syntax_pairs)} syntax pairs, "
        f"{len(gk_rows)} GK rows, {len(lexical_rows)} lexical rows"
    )
    return mappings
