"""
suite_pipeline — GPU-free pipeline with type-matched semantic alignment.

Steps:
  1      Partition all retain sets (CPU)
           → step1_partitions/

  2      Build non-semantic mappings (syntax pairs + GK/Lexical retain rows)
           → step2_sampled/non_semantic_mappings.json

  final  Assemble the dataset and (optionally) push to HuggingFace Hub.
         Reads:
           step1_partitions/               (from step 1)
           step2_sampled/                  (from step 2)
           forget_train_rephrases_path     (config)
           forget_eval_rephrases_path      (config)
           retain_semantic_rephrases_path  (config, optional)

Block structure (forget_train):
  2 syntax blocks  (syntax_train_n pairs per forget question)
  +
  15 non-syntax blocks × 25 forget questions each:
    • 11 semantic blocks (one per level: level 0 + levels 1-10)
    • 2  GK blocks (first / second half of gk_train)
    • 2  Lexical blocks (first / second half of lexical_train; emitted as "Lexical-<…>")

  Reverse assignment: label i uses the reverse question in blocks {(i*3+j)%15 | j∈{0,1,2}},
  giving exactly 5 reverse and 20 direct entries per block. Emitted labels carry
  the suffix "-reverse" (the "-opposite" form is also parsed).

  Rephrase cycling: direct/reverse appearances cycle through the entry's
  sorted rephrase keys ["original", q_*..., blank_*...] independently.

  Semantic type-matching: the retain question uses the rephrase type that
  matches the paired forget question (original/q_/blank_).  Requires
  retain_semantic_rephrases_path to be set; otherwise uses original question.

Usage:
    python -m suite_pipeline.run_pipeline config.yaml
    python -m suite_pipeline.run_pipeline config.yaml --step 1
    python -m suite_pipeline.run_pipeline config.yaml --step 2
    python -m suite_pipeline.run_pipeline config.yaml --step final
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

from suite_pipeline.config import PipelineConfig


def main():
    parser = argparse.ArgumentParser(description="SUITE Dataset Pipeline Final (GPU-free)")
    parser.add_argument("config", nargs="?", default=None)
    parser.add_argument(
        "--step", type=str, default=None,
        choices=["1", "2", "final", "all", "eval_only"],
        help="Run only a specific step (default: all). "
             "Use 'eval_only' to rebuild and push forget_eval + forget_eval_rephrasings "
             "from the updated forget_eval_rephrases_path without touching other splits.",
    )
    parser.add_argument(
        "--no_upload", action="store_true",
        help="Skip HuggingFace upload in the final step",
    )
    args = parser.parse_args()

    cfg      = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
    if args.no_upload:
        cfg.hf_dataset_name = ""
        cfg.hf_rephrasings_dataset_name = ""
    cfg.apply_seed()
    run_step = args.step or "all"

    # ------------------------------------------------------------------
    # Step 1: Partition all retain sets
    # ------------------------------------------------------------------
    retain_partitions = None

    if run_step in ("1", "all"):
        from suite_pipeline.partitioning.retain_splitter import partition_retain_sets
        retain_partitions = partition_retain_sets(cfg)
    else:
        out_dir = Path(cfg.output_dir) / "step1_partitions"
        if out_dir.exists():
            from suite_pipeline.io_utils.loaders import load_json
            retain_partitions = {f.stem: load_json(str(f)) for f in out_dir.glob("*.json")}
            print(f"  Loaded retain partitions from disk: {list(retain_partitions.keys())}")
        else:
            print(f"WARNING: step1_partitions not found at {out_dir}. Run --step 1 first.")

    # ------------------------------------------------------------------
    # eval_only: rebuild forget_eval + forget_eval_rephrasings and push
    # ------------------------------------------------------------------
    if run_step == "eval_only":
        from suite_pipeline.io_utils.loaders import load_json
        from suite_pipeline.assembly.schema import make_row, rows_to_dataset
        from suite_pipeline.assembly.dataset_builder import _normalize_label, _OPPOSITE_COMPOUNDS
        from suite_pipeline.io_utils.exporters import push_splits

        def _is_reverse_input(label: str) -> bool:
            return any(label.endswith(s) for s in _OPPOSITE_COMPOUNDS) or label.endswith("-reverse")

        if not cfg.forget_eval_rephrases_path:
            raise ValueError("forget_eval_rephrases_path must be set in config")
        forget_eval_rephrases = load_json(cfg.forget_eval_rephrases_path)
        print(f"  Loaded {len(forget_eval_rephrases)} forget eval entries")

        forget_eval_rows = []
        for entry in forget_eval_rephrases:
            label = entry.get("label", "")
            if _is_reverse_input(label) and not cfg.has_reverse_questions:
                continue
            forget_eval_rows.append(make_row(entry["question"], entry["answer"],
                                             _normalize_label(label)))

        forget_eval_rephrasings = []
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
            row = {"question": entry["question"], "answer": entry["answer"],
                   "label": _normalize_label(label)}
            for k in rephrase_keys:
                row[k] = entry[k]
            forget_eval_rephrasings.append(row)

        # Pad missing rephrase columns for consistent HF schema
        if forget_eval_rephrasings:
            all_keys = sorted({
                k for row in forget_eval_rephrasings
                for k in row if k.startswith("q_") or k.startswith("blank_")
            })
            for row in forget_eval_rephrasings:
                for k in all_keys:
                    row.setdefault(k, "")

        print(f"  forget_eval:             {len(forget_eval_rows)} rows")
        print(f"  forget_eval_rephrasings: {len(forget_eval_rephrasings)} rows")

        # Save locally
        out_dir = Path(cfg.output_dir) / "final"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Write via Dataset.to_json so the files are JSON Lines, matching --step final's output format
        rows_to_dataset(forget_eval_rows).to_json(str(out_dir / "forget_eval.json"))
        rows_to_dataset(forget_eval_rephrasings).to_json(str(out_dir / "forget_eval_rephrasings.json"))

        splits = {
            "forget_eval":             rows_to_dataset(forget_eval_rows),
            "forget_eval_rephrasings": rows_to_dataset(forget_eval_rephrasings),
        }

        if not args.no_upload:
            push_splits(
                splits,
                repo_name=cfg.hf_dataset_name,
                rephrasings_repo_name=getattr(cfg, "hf_rephrasings_dataset_name", ""),
                private=cfg.hf_private,
            )
        else:
            print("  [skip] --no_upload set; skipping HuggingFace push.")

        print("\neval_only complete — only forget_eval and forget_eval_rephrasings were updated.")
        return

    if run_step == "1":
        print("\nStep 1 complete. Run --step 2 next.")
        return

    # ------------------------------------------------------------------
    # Step 2: Build non-semantic mappings
    # ------------------------------------------------------------------
    non_semantic_mappings = None

    if run_step in ("2", "all"):
        from suite_pipeline.io_utils.loaders import load_json
        from suite_pipeline.alignment.non_semantic_mapper import build_non_semantic_mappings

        if not cfg.forget_train_rephrases_path:
            raise ValueError("forget_train_rephrases_path must be set in config")
        forget_train_rephrases = load_json(cfg.forget_train_rephrases_path)

        print("\n--- Step 2: Build non-semantic mappings ---")
        syntax_train  = (retain_partitions or {}).get("syntax_train", [])
        gk_train      = (retain_partitions or {}).get("gk_train", [])
        # The `words_train` key is also accepted for step1_partitions/ on disk.
        lexical_train = (retain_partitions or {}).get("lexical_train",
                          (retain_partitions or {}).get("words_train", []))
        non_semantic_mappings = build_non_semantic_mappings(
            cfg, forget_train_rephrases, syntax_train, gk_train, lexical_train
        )

        print(
            "\nStep 2 complete."
            "\n\nNext: run --step final to assemble the dataset."
            "\n  Optionally set retain_semantic_rephrases_path in the config"
            "\n  to provide blank/q_ rephrases for semantic retain questions."
            "\n  Format: [{label, question, answer, q_claude1, blank_claude1, ...}]"
            "\n  (one entry per semantic train question, label must match exactly)"
        )
    else:
        out_dir = Path(cfg.output_dir) / "step2_sampled"
        if out_dir.exists():
            from suite_pipeline.io_utils.loaders import load_json
            raw = load_json(str(out_dir / "non_semantic_mappings.json"))
            non_semantic_mappings = raw[0] if isinstance(raw, list) else raw
            print(f"  Loaded non_semantic_mappings from disk")
        else:
            print(f"WARNING: step2_sampled not found at {out_dir}. Run --step 2 first.")

    if run_step == "2":
        return

    # ------------------------------------------------------------------
    # Final: Assemble dataset
    # ------------------------------------------------------------------
    if run_step in ("final", "all"):
        from suite_pipeline.io_utils.loaders import load_json
        from suite_pipeline.assembly.dataset_builder import build_final_dataset

        # Load any inputs not already in memory
        if retain_partitions is None:
            out_dir = Path(cfg.output_dir) / "step1_partitions"
            retain_partitions = {f.stem: load_json(str(f)) for f in out_dir.glob("*.json")}

        # Back-fill gk_eval_short if the loaded step1_partitions don't contain it.
        # This is pure deterministic (rows[:2] per eval/odd-indexed topic) — no RNG.
        if "gk_eval_short" not in retain_partitions and cfg.retain_gk_json_path:
            print("  gk_eval_short not found in step1_partitions — computing on-the-fly from GK JSON")
            gk_raw = load_json(cfg.retain_gk_json_path)
            if isinstance(gk_raw, list) and len(gk_raw) == 1 and isinstance(gk_raw[0], dict):
                gk_raw = gk_raw[0]
            elif isinstance(gk_raw, list):
                gk_raw = {str(i): (v if isinstance(v, list) else [v]) for i, v in enumerate(gk_raw)}
            eval_short = []
            for i, (topic, qas) in enumerate(gk_raw.items()):
                if i % 2 == 1:  # eval topic
                    for qa in list(qas)[:2]:
                        eval_short.append({
                            "question": qa["question"],
                            "answer":   qa["answer"],
                            "label":    f"GK-{topic}",
                        })
            retain_partitions["gk_eval_short"] = eval_short
            from suite_pipeline.io_utils.exporters import save_json
            out_dir = Path(cfg.output_dir) / "step1_partitions"
            save_json(eval_short, out_dir / "gk_eval_short.json")
            print(f"  Saved gk_eval_short ({len(eval_short)} rows) to step1_partitions/")

        if non_semantic_mappings is None:
            out_dir = Path(cfg.output_dir) / "step2_sampled"
            raw = load_json(str(out_dir / "non_semantic_mappings.json"))
            non_semantic_mappings = raw[0] if isinstance(raw, list) else raw

        if not cfg.forget_train_rephrases_path:
            raise ValueError("forget_train_rephrases_path must be set in config")
        forget_train_rephrases = load_json(cfg.forget_train_rephrases_path)

        if not cfg.forget_eval_rephrases_path:
            raise ValueError("forget_eval_rephrases_path must be set in config")
        forget_eval_rephrases = load_json(cfg.forget_eval_rephrases_path)

        # Optional: semantic retain rephrases for type-matching
        semantic_rephrases = None
        if cfg.retain_semantic_rephrases_path:
            semantic_rephrases = load_json(cfg.retain_semantic_rephrases_path)
            print(f"  Loaded {len(semantic_rephrases)} semantic rephrase entries")
        else:
            print("  retain_semantic_rephrases_path not set — semantic rows use original questions")

        dataset = build_final_dataset(
            cfg=cfg,
            retain_partitions=retain_partitions,
            non_semantic_mappings=non_semantic_mappings,
            forget_train_rephrases=forget_train_rephrases,
            forget_eval_rephrases=forget_eval_rephrases,
            semantic_rephrases=semantic_rephrases,
        )

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Output directory: {cfg.output_dir}")
        for name, ds in dataset.items():
            print(f"  {name}: {len(ds)} rows")


if __name__ == "__main__":
    main()
