"""
Unified data loading from JSON files and HuggingFace datasets.
"""
from __future__ import annotations
import json
from pathlib import Path


def load_json(path: str | Path) -> list[dict]:
    """Load a JSON file; returns a list of dicts."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"Expected list or dict in {path}, got {type(data)}")
    return data

def load_hf_dataset(name: str, split: str) -> list[dict]:
    """Load a split from a HuggingFace dataset as a list of dicts."""
    from datasets import load_dataset
    ds = load_dataset(name, split=split)
    return [dict(row) for row in ds]

def load_source_data(
    path: str | Path | list[str] | None = None,
    hf_name: str | None = None,
    hf_split: str | None = None,
) -> list[dict]:
    """
    Flexible loader: local JSON path(s) take priority; falls back to HuggingFace.
    Parameters
    ----------
    path : str, Path, or list of paths (optional)
        Local JSON file(s). When provided, hf_name/hf_split are ignored.
    hf_name : str (optional)
        HuggingFace dataset identifier (e.g. "apeleg/SUITE").
    hf_split : str (optional)
        Split name (e.g. "train", "forget_eval").
    """
    if path is not None:
        paths = [path] if isinstance(path, (str, Path)) else list(path)
        rows: list[dict] = []
        for p in paths:
            print(f"  Loading local JSON: {p}")
            rows.extend(load_json(p))
        print(f"  {len(rows)} entries loaded from local file(s)")
        return rows
    if hf_name and hf_split:
        print(f"  Loading HF dataset {hf_name} split={hf_split}")
        rows = load_hf_dataset(hf_name, hf_split)
        print(f"  {len(rows)} entries loaded from HuggingFace")
        return rows
    raise ValueError("Must provide either `path` or both `hf_name` and `hf_split`.")
