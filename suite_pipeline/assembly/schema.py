"""
Canonical dataset row schema and conversion helpers.
Every row in the final dataset has these columns:
  question, answer, label

Some HF-published datasets also carry `count` and `rep` columns (always set
to 1). They are never read semantically by any consumer and are dropped from
outputs. Code reading those datasets still works because the extra columns
are simply ignored.
"""
from __future__ import annotations

COLUMNS = ["question", "answer", "label"]

def make_row(question: str, answer: str, label: str) -> dict:
    return {"question": question, "answer": answer, "label": label}

def rows_to_dataset(rows: list[dict]):
    """Convert a list of row dicts to a HuggingFace Dataset, ensuring all canonical columns are present."""
    from datasets import Dataset  # local import: keeps this module importable without `datasets`
    for row in rows:
        for col in COLUMNS:
            row.setdefault(col, "")
    return Dataset.from_list(rows)