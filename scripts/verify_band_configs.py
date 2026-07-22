#!/usr/bin/env python3
"""Fail-fast check of the proximity-band configs against the local dataset copy.

Applies each band YAML's split + exclude_label_prefixes to
dataset/{topic}/final/{split}.json (the local mirror of apeleg/SUITE) and
asserts the surviving rows are exactly the intended band. Run locally before
any GPU time:

    python scripts/verify_band_configs.py [--topic challenger_disaster]

No torch/transformers needed — pure stdlib + PyYAML-free (naive YAML read via
the structure gen_band_configs.py emits is avoided: we use omegaconf if
available, else a tiny fallback parser for exactly these files).
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# band -> (split, predicate description, keep-checker)
EXPECT = {
    "R0": ("retain_train", {"Semantic-0": 25}),
    "R1": ("retain_train", {"Semantic-1": 25, "Semantic-2": 25}),
    "R2": ("retain_train", {"Semantic-3": 25, "Semantic-4": 25}),
    "R3": ("retain_train", {"Semantic-5": 25, "Semantic-6": 25}),
    "R4": ("retain_train", {"Semantic-7": 25, "Semantic-8": 25}),
    "R5": ("retain_train", {"Semantic-9": 25, "Semantic-10": 25}),
    "R6": ("retain_eval", {"Semantic-11": 25, "Semantic-12": 25}),
    "R7": ("retain_eval", {"Semantic-13": 25, "Semantic-14": 25}),
    "C_gk": ("retain_train", {"GK": 50}),
    "C_lex": ("retain_train", {"Lexical": 50}),
    "A_forget_partial": ("forget_train",
                         {f"K{i}": 18 for i in range(1, 6)}),
    "A_forget_full": ("forget_train",
                      {**{f"K{i}": 18 for i in range(1, 21)},
                       **{f"M{i}": 18 for i in range(1, 6)}}),
}


def group_of(label: str) -> str:
    m = re.match(r"(Semantic-\d+)-", label)
    if m:
        return m.group(1)
    for g in ("GK", "Lexical", "Syntax"):
        if label.startswith(g):
            return g
    m = re.match(r"([KM]\d+)-", label)
    if m:
        return m.group(1)
    return f"UNKNOWN({label})"


def load_yaml(path: Path) -> dict:
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except ImportError:
        import yaml  # PyYAML fallback
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    (key, body), = cfg.items()
    return body["args"]


def load_rows(topic: str, split: str) -> list:
    path = ROOT / "dataset" / topic / "final" / f"{split}.json"
    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="challenger_disaster")
    args = ap.parse_args()
    topic = args.topic

    band_dir = ROOT / "configs" / "data" / "datasets" / topic / "bands"
    failures = 0
    for band, (want_split, want_counts) in EXPECT.items():
        cfg = load_yaml(band_dir / f"{band}.yaml")
        split = cfg["hf_args"]["split"]
        excludes = cfg.get("exclude_label_prefixes") or []
        ok = True
        msgs = []
        if split != want_split:
            ok = False
            msgs.append(f"split={split!r}, expected {want_split!r}")
        # The local per-topic JSONs have no `topic` column (the unified HF
        # dataset does); filter on it only when present.
        rows = [r for r in load_rows(topic, split)
                if r.get("topic", topic) == topic
                and not any(r["label"].startswith(p) for p in excludes)]
        got = Counter(group_of(r["label"]) for r in rows)
        if dict(got) != want_counts:
            ok = False
            msgs.append(f"groups={dict(got)}, expected {want_counts}")
        n_want = sum(want_counts.values())
        status = "OK  " if ok else "FAIL"
        print(f"[{status}] {band:<18} {split:<13} rows={len(rows):<4}"
              f" (expected {n_want})" + ("; " + "; ".join(msgs) if msgs else ""))
        failures += (not ok)

    print(f"\n{len(EXPECT) - failures}/{len(EXPECT)} band configs verified.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
