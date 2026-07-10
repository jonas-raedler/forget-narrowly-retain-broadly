"""
Final dataset assembly for suite_pipeline.

forget_train and retain_train are built in lock-step across 15 uniform blocks:

  Block layout (N_BLOCKS = 15):
    Blocks  0 … n_levels-1   : one block per semantic level (sorted numerically)
    Blocks  n_levels … +1    : GK first half / second half
    Blocks  n_levels+2 … +3  : Lexical first half / second half

  Each block contains all N_LABELS (25) forget questions exactly once.

  Opposite assignment (N_OPP_PER_LABEL = 3 per label → 5 per block):
    Label i is opposite in blocks  { (i*3 + j) % 15  |  j ∈ {0,1,2} }
    This gives a uniform distribution: every block has exactly 5 opposite
    and 20 direct entries, and every label appears 3× as opposite and 12×
    as direct across the 15 blocks.

  Rephrase key cycling:
    For direct appearances  (12 per label): keys cycle through
      ["original"] + sorted(q_*) + sorted(blank_*) of the direct entry.
    For opposite appearances (3 per label): keys cycle through
      the same ordered key list of the opposite entry.
    Key for the k-th appearance of type T = sorted_keys[k % len(sorted_keys)].

  Semantic retain type-matching:
    For semantic blocks the retain question's rephrase type is chosen to match
    the forget rephrase type:
      forget key == "original"   → use original retain question
      forget key starts "q_"     → use first available q_* rephrase
      forget key starts "blank_" → use first available blank_* rephrase
    If retain_semantic_rephrases_path is not set (or no matching rephrase
    exists), the original retain question is used as fallback.

Row i of forget_train always corresponds to row i of retain_train.
"""
from __future__ import annotations
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # `datasets` only needed at the HF-serialization boundary (build_final_dataset)
    from datasets import DatasetDict

from suite_pipeline.assembly.schema import make_row, rows_to_dataset
from suite_pipeline.io_utils.exporters import save_json, push_to_hub, save_dataset_locally
from suite_pipeline.config import PipelineConfig

# ---------------------------------------------------------------------------
# Block-structure constants
# ---------------------------------------------------------------------------
_N_LABELS        = 25   # canonical forget questions per block; the actual count is derived
                        # from the data at runtime (n_labels = len(sorted_labels)), so topics
                        # with fewer than 25 forget questions are supported. Kept as a doc/default.
_N_BLOCKS        = 15   # semantic (11) + GK (2) + lexical (2); default for _opposite_block_set
_N_OPP_PER_LABEL = 3    # each label appears as the reverse-question variant this many times
# Derived: each block has n_labels * _N_OPP_PER_LABEL // n_blocks reverse rows (e.g. 25*3//15 = 5)


# ---------------------------------------------------------------------------
# Canonical label normalization (applied at every emission point)
# ---------------------------------------------------------------------------
# Input files (forget_*_rephrases.json) and step1_partitions/*.json may use any
# of these label forms, which are normalized on emission:
#   -opposite (+ compound -even-opposite, -odd-opposite, -indirect-opposite,
#              -direct-opposite)  →  -reverse
#   -even  →  -direct
#   -odd   →  -indirect
#   Words- → Lexical-       (retain lexical category prefix)
# The pipeline PARSES every input form (input data is unchanged) but EMITS the
# canonical form. The function is idempotent on already-canonical labels (no-op).
_OPPOSITE_COMPOUNDS = (
    "-even-opposite", "-odd-opposite",
    "-indirect-opposite", "-direct-opposite", "-opposite",
)


def _normalize_label(label: str) -> str:
    if not isinstance(label, str):
        return label
    base, sep, key = label.partition("@")
    # 1. Collapse every *-opposite variant to -reverse.
    for suffix in _OPPOSITE_COMPOUNDS:
        if base.endswith(suffix):
            base = base[: -len(suffix)] + "-reverse"
            break
    else:
        # 2. -even → -direct, -odd → -indirect (only when -opposite didn't match).
        if base.endswith("-even"):
            base = base[: -len("-even")] + "-direct"
        elif base.endswith("-odd"):
            base = base[: -len("-odd")] + "-indirect"
    # 3. Words- → Lexical- prefix rename (retain lexical rows).
    if base.startswith("Words-"):
        base = "Lexical-" + base[len("Words-"):]
    return base + sep + key


def _emit(question: str, answer: str, label: str) -> dict:
    """Helper used at every emission point: builds the canonical 3-col row
    with the normalized label."""
    return make_row(question, answer, _normalize_label(label))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sorted_rephrase_keys(entry: dict) -> list[str]:
    """
    Stable key ordering for rephrase cycling:
      ["original"] + sorted(q_* keys) + sorted(blank_* keys)
    """
    q_keys     = sorted(k for k in entry
                        if re.match(r'^q_', k)
                        and not k.endswith('_answer')
                        and isinstance(entry.get(k), str)
                        and entry[k].strip())
    blank_keys = sorted(k for k in entry
                        if re.match(r'^blank_', k)
                        and not k.endswith('_answer')
                        and isinstance(entry.get(k), str)
                        and entry[k].strip())
    return ["original"] + q_keys + blank_keys


def _opposite_block_set(label_idx: int, n_blocks: int = _N_BLOCKS) -> frozenset:
    """Blocks where label_idx uses its opposite-question variant."""
    return frozenset(
        (label_idx * _N_OPP_PER_LABEL + j) % n_blocks
        for j in range(_N_OPP_PER_LABEL)
    )


def _rephrase_type(key: str) -> str:
    """Return 'original', 'q', or 'blank' based on rephrase key name."""
    if key == "original":
        return "original"
    if key.startswith("q_"):
        return "q"
    if key.startswith("blank_"):
        return "blank"
    return "original"


def _get_question(entry: dict, key: str) -> str:
    if key == "original":
        return entry["question"]
    return entry.get(key, entry["question"])


def _pick_semantic_retain(
    retain_item: dict,
    rtype: str,
    rephrases_by_question: dict,
) -> tuple[str, str]:
    """
    Select the retain question matching the given rephrase type.
    Returns (question, answer).  Falls back to original if no rephrase found.
    """
    answer = retain_item["answer"]
    if rtype == "original":
        return retain_item["question"], answer

    question       = retain_item.get("question", "")
    rephrase_entry = rephrases_by_question.get(question)
    if rephrase_entry is None:
        return retain_item["question"], answer   # no rephrase file → fallback

    prefix = "q_" if rtype == "q" else "blank_"
    keys   = sorted(
        k for k in rephrase_entry
        if k.startswith(prefix)
        and not k.endswith('_answer')
        and isinstance(rephrase_entry.get(k), str)
        and rephrase_entry[k].strip()
    )
    if keys:
        return rephrase_entry[keys[0]], answer
    return retain_item["question"], answer       # no matching rephrase → fallback


def _group_semantic_by_level(semantic_train: list[dict]) -> dict[str, list[dict]]:
    """Group semantic_train items by their level string ('0', '1', …)."""
    by_level: dict[str, list[dict]] = defaultdict(list)
    for item in semantic_train:
        parts = item.get("label", "").split("-")
        level = parts[1] if len(parts) >= 2 else "unknown"
        by_level[level].append(item)
    return dict(by_level)


# ---------------------------------------------------------------------------
# Block row builder
# ---------------------------------------------------------------------------

def _build_block_rows(
    sorted_labels: list[str],
    direct_by_label: dict,
    opp_by_label: dict,
    semantic_by_level: dict[str, list[dict]],
    gk_rows: list[dict],
    lexical_rows: list[dict],
    semantic_rephrases_by_question: dict,
    syntax_keys_by_label: dict | None = None,
) -> tuple[list, list]:
    """
    Build all n_blocks × n_labels forget/retain row pairs (e.g. 15 × 25 = 375).

    The per-block forget count (n_labels) is derived from len(sorted_labels), so a
    topic with fewer than 25 forget questions (e.g. 20 → 15 × 20 = 300) works without
    any config change; existing 25-question topics are unaffected.

    sorted_labels          : direct-question labels, alphabetically sorted (n_labels of them)
    direct_by_label        : label → raw forget entry (direct question)
    opp_by_label           : label → raw forget entry (opposite question), may be empty
    semantic_by_level      : level_str → list of retain items (≥ n_labels entries each)
    gk_rows / lexical_rows : flat retain row lists (≥ 2 × n_labels entries for 2 blocks)
    semantic_rephrases_by_question : question → rephrase entry with q_*/blank_* keys (may be {})
    syntax_keys_by_label   : label → list of forget keys already used in syntax rows.
                             Block direct cycling places unused keys first so all 15
                             appearances together cover as many distinct keys as possible.
    """
    forget_rows: list = []
    retain_rows: list = []

    sorted_levels = sorted(semantic_by_level.keys(),
                           key=lambda x: int(x) if x.isdigit() else 999)
    n_levels = len(sorted_levels)   # typically 11 (or 10 when level-0 is absent)
    n_blocks = n_levels + 4         # semantic + 2 GK + 2 Lexical
    # Forget questions per block — derived from the data, not the _N_LABELS constant,
    # so a topic with fewer (or more) than 25 forget questions just works.
    n_labels = len(sorted_labels)

    # Precompute per-label: sorted opposite blocks and sorted direct blocks
    opp_sets          = [_opposite_block_set(i, n_blocks) for i in range(n_labels)]
    opp_block_lists   = [sorted(opp_sets[i]) for i in range(n_labels)]
    direct_block_lists = [
        sorted(b for b in range(n_blocks) if b not in opp_sets[i])
        for i in range(n_labels)
    ]

    for block_idx in range(n_blocks):
        for pos in range(n_labels):
            label     = sorted_labels[pos]
            use_opp   = (block_idx in opp_sets[pos]) and (label in opp_by_label)

            if use_opp:
                entry   = opp_by_label[label]
                app_idx = opp_block_lists[pos].index(block_idx)   # 0, 1, or 2

                # Opposite entry has one original + a few q_* + a few blank_*.
                # Each of the 3 appearances uses a different type (original / q_ / blank_).
                # The type order rotates by pos so each label starts from a different type,
                # giving variety across the 25 positions within a single block.
                _opp_q_keys     = sorted(k for k in entry
                                         if k.startswith("q_") and not k.endswith("_answer")
                                         and isinstance(entry.get(k), str) and entry[k].strip())
                _opp_blank_keys = sorted(k for k in entry
                                         if k.startswith("blank_") and not k.endswith("_answer")
                                         and isinstance(entry.get(k), str) and entry[k].strip())
                _type_order = ["original", "q", "blank"]
                _rtype = _type_order[(app_idx + pos) % 3]
                if _rtype == "q" and _opp_q_keys:
                    key = _opp_q_keys[pos % len(_opp_q_keys)]
                elif _rtype == "blank" and _opp_blank_keys:
                    key = _opp_blank_keys[pos % len(_opp_blank_keys)]
                else:
                    key = "original"

            else:
                entry   = direct_by_label[label]
                # If this block was supposed to be opposite but no entry exists,
                # fall back to direct and use the opposite-block position for cycling.
                if block_idx in opp_sets[pos]:
                    app_idx = opp_block_lists[pos].index(block_idx)
                else:
                    app_idx = direct_block_lists[pos].index(block_idx)  # 0..11

                full_keys = _sorted_rephrase_keys(entry)
                # Separate keys not yet used in syntax (new_keys) from those already used.
                # Block appearances cycle within new_keys only — never into used_keys —
                # so the pos offset cannot wrap into the used zone and cause repeats.
                # Fallback to full_keys only if all keys were consumed by syntax.
                used_in_syntax = set(
                    (syntax_keys_by_label or {}).get(label, [])
                )
                new_keys = [k for k in full_keys if k not in used_in_syntax]
                cycle    = new_keys if new_keys else full_keys
                key      = cycle[(app_idx + pos) % len(cycle)]
            f_q      = _get_question(entry, key)
            f_label  = f"{entry['label']}@{key}"
            rtype    = _rephrase_type(key)

            # ---- Retain side ----
            if block_idx < n_levels:
                level       = sorted_levels[block_idx]
                level_items = semantic_by_level[level]
                ret_item    = level_items[pos % len(level_items)]
                ret_q, ret_a = _pick_semantic_retain(ret_item, rtype, semantic_rephrases_by_question)
                ret_label   = f"{ret_item.get('label', '')}@{rtype}"

            elif block_idx < n_levels + 2:
                half    = block_idx - n_levels          # 0 or 1
                row_idx = half * n_labels + pos
                ret     = gk_rows[row_idx % len(gk_rows)]
                ret_q, ret_a, ret_label = ret["question"], ret["answer"], ret.get("label", "")

            else:
                half    = block_idx - n_levels - 2      # 0 or 1
                row_idx = half * n_labels + pos
                ret     = lexical_rows[row_idx % len(lexical_rows)]
                ret_q, ret_a, ret_label = ret["question"], ret["answer"], ret.get("label", "")

            forget_rows.append(_emit(f_q,   entry["answer"], f_label))
            retain_rows.append(_emit(ret_q, ret_a,           ret_label))

    n_rev = sum(
        1 for r in forget_rows
        if re.search(r'-reverse@', r.get("label", ""))
    )
    print(
        f"  Block rows: {len(forget_rows)} entries "
        f"({n_blocks} blocks × {n_labels} positions, {n_rev} reverse)"
    )
    return forget_rows, retain_rows


# ---------------------------------------------------------------------------
# Statistics printer
# ---------------------------------------------------------------------------

def _label_rtype(label: str) -> str:
    """Return 'original', 'q_*', or 'blank_*' based on the @key suffix of a label."""
    key = label.split("@")[-1] if "@" in label else "original"
    if key == "original":
        return "original"
    if key.startswith("q_"):
        return "q_*"
    if key.startswith("blank_"):
        return "blank_*"
    return key   # e.g. "@q" / "@blank" for semantic retain labels


def _print_dataset_stats(
    forget_train: list,
    retain_train: list,
    forget_eval: list,
    retain_eval: list,
    forget_eval_rephrasings: list,
    n_syntax: int,
    direct_by_label: dict,
    opp_by_label: dict,
) -> None:
    """Print a comprehensive quality report for the assembled dataset."""
    from collections import Counter
    W = 70

    # Forget questions per block — derived from the data (count of direct labels),
    # so stats stay correct for topics with fewer than 25 forget questions.
    n_labels = len(direct_by_label)

    # Derived expected values — use actual block count inferred from data
    block_f = forget_train[n_syntax:]
    _n_actual_blocks    = len(block_f) // n_labels if n_labels else _N_BLOCKS
    _EXP_OPP_TOTAL      = n_labels * _N_OPP_PER_LABEL                           # n_labels × 3
    _EXP_OPP_PER_BLOCK  = _EXP_OPP_TOTAL / _n_actual_blocks                     # may be non-integer
    _EXP_DIRECT_TOTAL   = _n_actual_blocks * n_labels - _EXP_OPP_TOTAL
    _EXP_BLOCK_ROWS     = _n_actual_blocks * n_labels

    print("\n" + "=" * W)
    print("DATASET STATISTICS")
    print("=" * W)

    # ── Split sizes ──────────────────────────────────────────────────────
    print("\n[Split sizes]")
    for name, rows in [
        ("forget_train",            forget_train),
        ("retain_train",            retain_train),
        ("forget_eval",             forget_eval),
        ("retain_eval",             retain_eval),
        ("forget_eval_rephrasings", forget_eval_rephrasings),
    ]:
        print(f"  {name:<30} {len(rows):>5} rows")

    assert len(forget_train) == len(retain_train), (
        f"MISMATCH: forget_train={len(forget_train)}, retain_train={len(retain_train)}"
    )
    print(f"  forget_train / retain_train aligned: OK")

    # ── forget_train breakdown ───────────────────────────────────────────
    print("\n[forget_train breakdown]")
    block_r = retain_train[n_syntax:]
    print(f"  Syntax rows : {n_syntax}")
    print(f"  Block rows  : {len(block_f)}  (expected {_EXP_BLOCK_ROWS})")

    # Rephrase type distribution (forget side)
    rtypes: Counter = Counter()
    for r in block_f:
        rtypes[_label_rtype(r.get("label", ""))] += 1
    total = len(block_f) or 1
    print(f"  Forget rephrase type distribution (block rows):")
    for t, n in sorted(rtypes.items()):
        print(f"    {t:<12} {n:>4}  ({100*n/total:.1f}%)")

    # Reverse vs direct (-reverse@... vs -direct@...)
    def _is_reverse_emitted(label: str) -> bool:
        return "-reverse@" in label or "-opposite@" in label  # accept both for safety

    n_rev   = sum(1 for r in block_f if _is_reverse_emitted(r.get("label", "")))
    n_dir   = len(block_f) - n_rev
    rev_ok  = n_rev == _EXP_OPP_TOTAL
    print(f"  Direct entries : {n_dir:>4}  (expected {_EXP_DIRECT_TOTAL})  {'OK' if n_dir == _EXP_DIRECT_TOTAL else 'MISMATCH'}")
    print(f"  Reverse entries: {n_rev:>4}  (expected {_EXP_OPP_TOTAL})  {'OK' if rev_ok else 'MISMATCH'}")

    # Per-block reverse count
    per_block_rev = []
    for b in range(_n_actual_blocks):
        chunk = block_f[b * n_labels: (b + 1) * n_labels]
        per_block_rev.append(sum(1 for r in chunk if _is_reverse_emitted(r.get("label", ""))))
    counts_str = "  ".join(f"B{b}:{n}" for b, n in enumerate(per_block_rev))
    exp_str = f"~{_EXP_OPP_PER_BLOCK:.1f}" if _EXP_OPP_PER_BLOCK != int(_EXP_OPP_PER_BLOCK) else str(int(_EXP_OPP_PER_BLOCK))
    print(f"  Reverse per block (expected {exp_str} each):")
    print(f"    {counts_str}")

    # Forget label appearance counts across ALL forget_train
    # Base label = strip @key suffix AND -opposite suffix → groups K1-direct + K1-direct-opposite
    # Expected: 3 syntax + 12 block-direct + 3 block-opposite = 18 per label
    _EXP_TOTAL_PER_LABEL  = n_syntax // n_labels + _n_actual_blocks
    _EXP_DIRECT_PER_LABEL = n_syntax // n_labels + (_n_actual_blocks - _N_OPP_PER_LABEL)
    _EXP_OPP_PER_LABEL_   = _N_OPP_PER_LABEL

    direct_counts:  Counter = Counter()
    reverse_counts: Counter = Counter()
    for r in forget_train:
        label = r.get("label", "")
        base  = label.split("@")[0] if "@" in label else label   # strip rephrase key
        # Accept both the -reverse and -opposite suffixes for safety.
        if base.endswith("-reverse"):
            base = base[: -len("-reverse")]
            reverse_counts[base] += 1
        elif base.endswith("-opposite"):
            base = base[: -len("-opposite")]
            reverse_counts[base] += 1
        else:
            if base.endswith("-direct"):
                base = base[: -len("-direct")]
            direct_counts[base] += 1

    all_bases = sorted(set(direct_counts) | set(reverse_counts))
    total_counts = {b: direct_counts[b] + reverse_counts[b] for b in all_bases}
    totals = list(total_counts.values())
    mismatched = [b for b in all_bases if total_counts[b] != _EXP_TOTAL_PER_LABEL]

    _n_syntax_per_label = n_syntax // n_labels
    print(f"  Appearances per forget question across all forget_train")
    print(f"    expected: {_EXP_TOTAL_PER_LABEL} total = {_EXP_DIRECT_PER_LABEL} direct ({_n_syntax_per_label} syntax + {_n_actual_blocks - _N_OPP_PER_LABEL} block) + {_EXP_OPP_PER_LABEL_} reverse per label")
    if totals:
        print(f"    min={min(totals)}  max={max(totals)}  unique_labels={len(all_bases)}")
    if not mismatched:
        print(f"    OK — all {len(all_bases)} labels are perfectly uniform")
    else:
        print(f"    MISMATCH — {len(mismatched)} labels differ from expected:")
    print(f"\n  {'Label':<30} {'Direct':>8} {'Reverse':>9} {'Total':>7}  Status")
    print(f"  {'-'*30} {'-'*8} {'-'*9} {'-'*7}  ------")
    for b in all_bases:
        d = direct_counts[b]
        o = reverse_counts[b]
        t = d + o
        ok = "OK" if t == _EXP_TOTAL_PER_LABEL else "MISMATCH"
        print(f"  {b:<30} {d:>8} {o:>9} {t:>7}  {ok}")

    # ── forget/retain type alignment (all rows except GK and Lexical retain) ─
    print("\n[forget / retain rephrase type alignment]")
    print("  (GK and Lexical retain rows excluded — they use original questions only)")

    def _key_rtype(label: str) -> str:
        """Broad type: 'original', 'q', or 'blank'."""
        key = label.split("@")[-1] if "@" in label else "original"
        if key == "original":                          return "original"
        if key.startswith("q_") or key == "q":        return "q"
        if key.startswith("blank_") or key == "blank": return "blank"
        return "other"

    total_checked = ok_count = bad_count = 0
    bad_examples: list = []
    cat_totals: Counter = Counter()
    cat_bad:    Counter = Counter()

    for i, (f, r) in enumerate(zip(forget_train, retain_train)):
        r_label = r.get("label", "")
        # Skip GK and Lexical retain — they have no rephrase types
        # (`Words-` prefix accepted alongside `Lexical-`).
        if (r_label.startswith("GK-") or r_label.startswith("Lexical-")
                or r_label.startswith("Words-")):
            continue
        total_checked += 1
        # Determine retain category for reporting
        if r_label.startswith("Syntax-"):
            cat = "Syntax"
        elif r_label.startswith("Semantic-"):
            cat = "Semantic"
        else:
            cat = "other"
        cat_totals[cat] += 1

        ft = _key_rtype(f.get("label", ""))
        rt = _key_rtype(r_label)
        if ft == rt:
            ok_count += 1
        else:
            bad_count += 1
            cat_bad[cat] += 1
            if len(bad_examples) < 10:
                bad_examples.append((i, cat, ft, rt, f.get("label", ""), r_label))

    for cat, n in sorted(cat_totals.items()):
        n_bad = cat_bad.get(cat, 0)
        status = "OK" if n_bad == 0 else f"{n_bad} MISMATCHES"
        print(f"  {cat:<10} {n:>4} rows checked — {status}")

    if bad_count == 0:
        print(f"  All {total_checked} checked rows have matching forget/retain types: OK")
    else:
        print(f"  {bad_count} mismatches out of {total_checked} checked rows:")
        print(f"  {'Row':>5}  {'Cat':<10} {'Forget type':<12} {'Retain type':<12}  Forget label  →  Retain label")
        for i, cat, ft, rt, fl, rl in bad_examples:
            print(f"  {i:>5}  {cat:<10} {ft:<12} {rt:<12}  {fl}  →  {rl}")

    # ── retain_train breakdown ───────────────────────────────────────────
    print("\n[retain_train breakdown]")
    cat_counts: Counter = Counter()
    retain_rtypes: Counter = Counter()
    for r in retain_train:
        label = r.get("label", "")
        if label.startswith("Syntax-"):
            cat_counts["Syntax"] += 1
        elif label.startswith("Semantic-"):
            parts = label.split("-")
            lvl = parts[1] if len(parts) >= 2 else "?"
            cat_counts[f"Semantic-{lvl}"] += 1
            retain_rtypes[_key_rtype(label)] += 1
        elif label.startswith("GK-"):
            cat_counts["GK"] += 1
        elif label.startswith("Lexical-") or label.startswith("Words-"):
            cat_counts["Lexical"] += 1
        else:
            cat_counts["other"] += 1
    for cat, n in sorted(cat_counts.items()):
        print(f"  {cat:<22} {n:>4} rows")
    if retain_rtypes:
        total_sem = sum(retain_rtypes.values()) or 1
        print(f"  Semantic retain rephrase types:")
        for t, n in sorted(retain_rtypes.items()):
            print(f"    {t:<12} {n:>4}  ({100*n/total_sem:.1f}%)")

    # ── forget_eval breakdown ────────────────────────────────────────────
    # Emission uses exactly three suffixes: -direct, -indirect, -reverse.
    # The -even, -odd, -opposite, *-opposite suffixes are also counted in case
    # the printer is called on data that uses them.
    print("\n[forget_eval breakdown]")
    eval_types: Counter = Counter()
    for r in forget_eval:
        label = r.get("label", "")
        if label.endswith("-reverse"):     eval_types["reverse"]   += 1
        elif label.endswith("-direct"):    eval_types["direct"]    += 1
        elif label.endswith("-indirect"):  eval_types["indirect"]  += 1
        # fallback suffixes
        elif label.endswith("-opposite"):  eval_types["reverse (-opposite)"] += 1
        elif label.endswith("-even"):      eval_types["direct (-even)"]      += 1
        elif label.endswith("-odd"):       eval_types["indirect (-odd)"]     += 1
        else:                              eval_types["other"]     += 1
    for t, n in sorted(eval_types.items()):
        print(f"  {t:<30} {n:>4} rows")

    # ── retain_eval breakdown ────────────────────────────────────────────
    print("\n[retain_eval breakdown]")
    eval_cat: Counter = Counter()
    for r in retain_eval:
        label = r.get("label", "")
        if label.startswith("Syntax-"):                                     eval_cat["Syntax"]   += 1
        elif label.startswith("Semantic-"):                                 eval_cat["Semantic"] += 1
        elif label.startswith("GK-"):                                       eval_cat["GK"]       += 1
        elif label.startswith("Lexical-") or label.startswith("Words-"):    eval_cat["Lexical"]  += 1
        else:                                                               eval_cat["other"]    += 1
    for cat, n in sorted(eval_cat.items()):
        print(f"  {cat:<12} {n:>4} rows")

    # ── forget_eval_rephrasings ──────────────────────────────────────────
    if forget_eval_rephrasings:
        reph_keys = sorted({
            k for r in forget_eval_rephrasings
            for k in r if k.startswith("q_") or k.startswith("blank_")
        })
        print(f"\n[forget_eval_rephrasings]")
        print(f"  {len(forget_eval_rephrasings)} rows  |  {len(reph_keys)} rephrase keys: {reph_keys}")

    # ── Unique rephrases per forget question (forget_train) ──────────────
    print("\n[Unique rephrases per forget question in forget_train]")
    # Group all forget_train rows by base label (strip @key AND -reverse/-opposite suffix)
    # Separate direct vs reverse appearances. Accept -opposite for safety.
    direct_rephrases:  dict[str, set] = {}
    reverse_rephrases: dict[str, set] = {}
    for r in forget_train:
        label = r.get("label", "")
        base, key = label.rsplit("@", 1) if "@" in label else (label, "original")
        if base.endswith("-reverse"):
            base = base[: -len("-reverse")]
            reverse_rephrases.setdefault(base, set()).add(key)
        elif base.endswith("-opposite"):
            base = base[: -len("-opposite")]
            reverse_rephrases.setdefault(base, set()).add(key)
        else:
            if base.endswith("-direct"):
                base = base[: -len("-direct")]
            direct_rephrases.setdefault(base, set()).add(key)

    n_syntax_per   = n_syntax // n_labels
    n_block_direct = _n_actual_blocks - _N_OPP_PER_LABEL
    n_direct_total = n_syntax_per + n_block_direct

    def _rephrase_variety_summary(
        by_label: dict[str, set],
        kind: str,
        n_appearances: int,
        source_entries: dict,   # label → raw entry with q_*/blank_* keys
    ) -> None:
        if not by_label:
            return
        # How many rephrase keys are available in the source vs how many we actually used
        print(f"  {kind} ({len(by_label)} labels, {n_appearances} appearances each):")
        print(f"  {'Label':<30} {'Available':>10} {'Used':>6}  {'Note'}")
        print(f"  {'-'*30} {'-'*10} {'-'*6}  {'-'*30}")
        for base in sorted(by_label.keys()):
            used = len(by_label[base])
            entry = (source_entries.get(base)
                     or source_entries.get(base + "-direct")
                     or source_entries.get(base + "-opposite")
                     or source_entries.get(base + "-direct-opposite"))
            if entry:
                avail_keys = _sorted_rephrase_keys(entry)
                avail = len(avail_keys)
            else:
                avail = "?"
            note = ""
            if isinstance(avail, int) and avail < n_appearances:
                note = f"only {avail} keys → cycling repeats {n_appearances - avail} times"
            elif used < n_appearances:
                note = f"used {used} of {avail}"
            print(f"  {base:<30} {str(avail):>10} {used:>6}  {note}")

    # Build source lookup from the raw rephrases lists passed into build_final_dataset.
    # We piggyback on direct_by_label and opp_by_label already built above.
    _source_direct   = direct_by_label   # label → raw entry
    _source_opposite = {b: opp_by_label[b] for b in opp_by_label}  # base → opposite entry

    _rephrase_variety_summary(direct_rephrases,  f"Direct  ({n_syntax_per} syntax + {n_block_direct} block)", n_direct_total, _source_direct)
    _rephrase_variety_summary(reverse_rephrases, f"Reverse ({_N_OPP_PER_LABEL} block)", _N_OPP_PER_LABEL, _source_opposite)

    print("\n" + "=" * W)


# ---------------------------------------------------------------------------
# Pairs → rows helper (for syntax)
# ---------------------------------------------------------------------------

def _pairs_to_rows(pairs: list[dict]) -> tuple[list, list]:
    """Unzip a list of {forget_row, retain_row} dicts into two parallel lists."""
    forget_rows, retain_rows = [], []
    for p in pairs:
        fr = p["forget_row"]
        rr = p["retain_row"]
        forget_rows.append(_emit(fr["question"], fr["answer"], fr["label"]))
        retain_rows.append(_emit(rr["question"], rr["answer"], rr["label"]))
    return forget_rows, retain_rows


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_final_dataset(
    cfg: PipelineConfig,
    retain_partitions: dict[str, list[dict]],
    non_semantic_mappings: dict,
    forget_train_rephrases: list[dict],
    forget_eval_rephrases: list[dict],
    semantic_rephrases: list[dict] | None = None,
) -> DatasetDict:
    """
    Assemble the final DatasetDict with splits:
      forget_train, retain_train, forget_eval, retain_eval, forget_eval_rephrasings.

    Parameters
    ----------
    retain_partitions       : output of partition_retain_sets() (step 1)
    non_semantic_mappings   : output of build_non_semantic_mappings() (step 2)
                              keys: syntax_pairs, gk_rows, lexical_rows
    forget_train_rephrases  : loaded from forget_train_rephrases_path
    forget_eval_rephrases   : loaded from forget_eval_rephrases_path
    semantic_rephrases      : optional list of {label, question, answer, q_*, blank_*}
                              for semantic retain type-matching; None = use originals
    """
    cfg.apply_seed()
    print("=" * 60)
    print("FINAL ASSEMBLY: Building dataset splits")
    print("=" * 60)

    out_dir = Path(cfg.output_dir) / "final"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Prepare forget lookups --------------------------------------
    # Sort direct labels alphabetically (these are the N_LABELS positions)
    direct_entries = [
        e for e in forget_train_rephrases
        if re.search(r'-direct$', e.get("label", ""))
    ]
    opp_entries = [
        e for e in forget_train_rephrases
        if re.search(r'-(opposite|reverse)$', e.get("label", ""))
    ]
    sorted_labels  = sorted(e["label"] for e in direct_entries)
    direct_by_label = {e["label"]: e for e in direct_entries}
    # Map "K1-direct" → reverse-question entry.
    # Supports these label forms:
    #   "K1-reverse"         → strip "-reverse" + append "-direct" → "K1-direct"  (canonical)
    #   "K1-direct-opposite" → strip "-opposite"                   → "K1-direct"  (compound)
    #   "K1-opposite"        → strip "-opposite" + append "-direct" → "K1-direct"  (bare)
    opp_by_label: dict[str, dict] = {}
    for e in opp_entries:
        lbl = e["label"]
        if lbl.endswith("-reverse"):
            base = lbl[: -len("-reverse")] + "-direct"   # "K1" → "K1-direct"
        elif lbl.endswith("-direct-opposite"):
            base = lbl[: -len("-opposite")]              # "K1-direct"
        else:
            base = lbl[: -len("-opposite")] + "-direct"  # "K1" → "K1-direct"
        opp_by_label[base] = e

    # ---- Prepare semantic data ---------------------------------------
    semantic_train = retain_partitions.get("semantic_train", [])
    semantic_by_level = _group_semantic_by_level(semantic_train)

    # Build semantic rephrase lookup (question → entry)
    semantic_rephrases_by_question: dict = {}
    if semantic_rephrases:
        semantic_rephrases_by_question = {e.get("question", ""): e for e in semantic_rephrases}

    # ---- Build forget_train / retain_train ---------------------------
    syntax_forget, syntax_retain = _pairs_to_rows(non_semantic_mappings["syntax_pairs"])

    block_forget, block_retain = _build_block_rows(
        sorted_labels               = sorted_labels,
        direct_by_label             = direct_by_label,
        opp_by_label                = opp_by_label,
        semantic_by_level           = semantic_by_level,
        gk_rows                     = non_semantic_mappings["gk_rows"],
        lexical_rows                = non_semantic_mappings.get("lexical_rows", non_semantic_mappings.get("words_rows", [])),
        semantic_rephrases_by_question      = semantic_rephrases_by_question,
        syntax_keys_by_label        = non_semantic_mappings.get("syntax_keys_by_label"),
    )

    forget_train = syntax_forget + block_forget
    retain_train = syntax_retain + block_retain

    assert len(forget_train) == len(retain_train), (
        f"Length mismatch: forget_train={len(forget_train)}, retain_train={len(retain_train)}"
    )
    print(f"  forget_train / retain_train: {len(forget_train)} rows each")

    # ---- forget_eval -------------------------------------------------
    # Accept both the -opposite and -reverse input labels as the gate for
    # cfg.has_reverse_questions. Emission normalizes to the canonical form.
    def _is_reverse_input(label: str) -> bool:
        return any(label.endswith(s) for s in _OPPOSITE_COMPOUNDS) or label.endswith("-reverse")

    forget_eval_rows: list = []
    for entry in forget_eval_rephrases:
        label = entry.get("label", "")
        if _is_reverse_input(label) and not cfg.has_reverse_questions:
            continue
        forget_eval_rows.append(_emit(entry["question"], entry["answer"], label))

    # ---- retain_eval -------------------------------------------------
    # retain_eval combines the semantic, syntax, GK (short: 2 questions per eval
    # topic), and lexical eval partitions. `lexical_eval` is the canonical key;
    # `words_eval` is the step1_partitions filename and is also read.
    retain_eval: list = []
    for key in ["semantic_eval", "syntax_eval", "gk_eval_short", "lexical_eval", "words_eval"]:
        for item in retain_partitions.get(key, []):
            retain_eval.append(_emit(
                item["question"], item["answer"], item.get("label", "")
            ))

    # ---- forget_eval_rephrasings -------------------------------------
    forget_eval_rephrasings: list = []
    for entry in forget_eval_rephrases:
        label = entry.get("label", "")
        if _is_reverse_input(label) and not cfg.has_reverse_questions:
            continue
        rephrase_keys = [
            k for k, v in entry.items()
            if re.match(r'^(q_|blank_)', k)
            and not k.endswith('_answer')
            and isinstance(v, str)
        ]
        if not rephrase_keys:
            continue
        row: dict = {"question": entry["question"], "answer": entry["answer"],
                     "label": _normalize_label(label)}
        for k in rephrase_keys:
            row[k] = entry[k]
        forget_eval_rephrasings.append(row)

    # Pad missing rephrase columns for a consistent HF schema
    if forget_eval_rephrasings:
        all_keys = sorted({
            k for row in forget_eval_rephrasings
            for k in row if k.startswith("q_") or k.startswith("blank_")
        })
        for row in forget_eval_rephrasings:
            for k in all_keys:
                row.setdefault(k, "")

    # ---- Statistics --------------------------------------------------
    _print_dataset_stats(
        forget_train            = forget_train,
        retain_train            = retain_train,
        forget_eval             = forget_eval_rows,
        retain_eval             = retain_eval,
        forget_eval_rephrasings = forget_eval_rephrasings,
        n_syntax                = len(syntax_forget),
        direct_by_label         = direct_by_label,
        opp_by_label            = opp_by_label,
    )

    # ---- Package and save --------------------------------------------
    # There is a single `retain_eval` split (its GK portion uses the short,
    # 2-question-per-topic variant). A wider retain_eval is not produced here.
    from datasets import DatasetDict  # local import: only this assembly step needs `datasets`
    dataset = DatasetDict({
        "forget_train":            rows_to_dataset(forget_train),
        "retain_train":            rows_to_dataset(retain_train),
        "forget_eval":             rows_to_dataset(forget_eval_rows),
        "retain_eval":             rows_to_dataset(retain_eval),
        "forget_eval_rephrasings": rows_to_dataset(forget_eval_rephrasings),
    })

    print("\n--- Dataset Summary ---")
    for split_name, ds in dataset.items():
        print(f"  {split_name}: {len(ds)} rows")

    save_dataset_locally(dataset, out_dir)
    push_to_hub(
        dataset,
        cfg.hf_dataset_name,
        cfg.hf_private,
        rephrasings_repo_name=getattr(cfg, "hf_rephrasings_dataset_name", ""),
    )
    return dataset