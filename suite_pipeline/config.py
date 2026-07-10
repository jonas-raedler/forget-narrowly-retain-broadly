"""
Pipeline configuration for suite_pipeline.
GPU-free variant: rephrases are pre-generated externally;
semantic retain questions are matched to forget rephrase type (original/blank/q_).
No LLM alignment step, no judge-based filtering step.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class PipelineConfig:
    seed: int = 42

    # ---- Input paths -------------------------------------------------
    # Pre-generated forget rephrases (both train and eval)
    forget_train_rephrases_path: str = ""
    forget_eval_rephrases_path: str = ""

    # Raw retain source files
    retain_semantic_json_path: str = ""
    retain_semantic_level0_json_path: str = ""
    retain_lexical_json_path: str = ""       # also accepts the YAML key `retain_words_json_path`
    retain_gk_json_path: str = ""
    retain_syntax_json_path: str = ""

    # Optional: blank/q_ rephrases for semantic retain questions.
    # Format: [{label, question, answer, q_claude1, blank_claude1, ...}, ...]
    # If not set, semantic retain rows always use the original question.
    retain_semantic_rephrases_path: str = ""

    # Optional: override the Semantic-0 label suffix. When set, level-0 rows get
    # labels of the form "Semantic-0-<retain_semantic_level0_label>". When empty,
    # the level-0 source filename stem is used. Use this to keep HF labels stable
    # across filename renames.
    retain_semantic_level0_label: str = ""

    # ---- Output ------------------------------------------------------
    output_dir: str = "./dataset"

    # ---- Step 1: Retain partitioning ---------------------------------
    semantic_level_0_policy: str = "none"        # "none" | "both" | "split"
    semantic_level_0_train: int = 25
    semantic_levels_1_10_train: int = 25
    semantic_levels_1_10_eval: int = 25
    semantic_levels_11_15_policy: str = "eval_only"
    gk_questions_per_train_topic: int = 1        # questions sampled per even-indexed GK topic
    gk_questions_per_eval_topic: int = 3         # questions sampled per odd-indexed GK topic
    syntax_train_n: int = 6   # max rephrases per question going to train
    syntax_eval_n: int = 2    # rephrases per question going to eval

    # ---- Final assembly ----------------------------------------------
    has_reverse_questions: bool = False    # also accepts the YAML key `has_opposite_questions`

    # ---- HuggingFace upload ------------------------------------------
    hf_dataset_name: str = ""
    hf_rephrasings_dataset_name: str = ""
    hf_private: bool = False

    def apply_seed(self) -> None:
        random.seed(self.seed)
        try:
            import numpy as np
            np.random.seed(self.seed)
        except ImportError:
            pass

    # YAML key aliases: either spelling loads; the canonical key wins if both present.
    _LEGACY_KEY_ALIASES = {
        "retain_words_json_path": "retain_lexical_json_path",
        "has_opposite_questions": "has_reverse_questions",
    }

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for legacy, new in cls._LEGACY_KEY_ALIASES.items():
            if legacy in data:
                if new not in data:
                    data[new] = data[legacy]
                data.pop(legacy)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str | Path) -> None:
        from dataclasses import asdict
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)