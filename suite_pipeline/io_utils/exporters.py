"""
Saving results to JSON and uploading to HuggingFace Hub.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # `datasets` only needed for type hints here; keep module import-light
    from datasets import DatasetDict

def save_json(data: list[dict], path: str | Path) -> None:
    """Write a list of dicts to a JSON file, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(data)} entries to {path}")

def push_to_hub(
    dataset_dict: DatasetDict,
    repo_name: str,
    private: bool = False,
    rephrasings_repo_name: str = "",
) -> None:
    """Push each split to HuggingFace Hub.

    forget_eval_rephrasings has extra q_*/blank_* columns that differ from the
    other splits, causing HF schema-validation errors when pushed to the same
    repo, so it gets its own dedicated repo. If that repo name is empty, falls
    back to repo_name.
    """
    if not repo_name:
        print("  [skip] No hf_dataset_name configured; not uploading.")
        return

    # Map split name → target repo
    forget_rephrasings_target = rephrasings_repo_name or repo_name

    split_targets = {
        "forget_eval_rephrasings": forget_rephrasings_target,
    }

    try:
        from huggingface_hub import HfApi, CommitOperationDelete
        from huggingface_hub.utils import RepositoryNotFoundError
        api = HfApi()

        # Wipe + recreate every distinct target repo
        repos_to_wipe = {repo_name, forget_rephrasings_target}
        for r in repos_to_wipe:
            _wipe_repo(api, r, private, CommitOperationDelete, RepositoryNotFoundError)
    except ImportError:
        print("  Warning: huggingface_hub not installed; skipping repo wipe.")

    # Push splits to their respective repos
    for split_name, ds in dataset_dict.items():
        target = split_targets.get(split_name, repo_name)
        try:
            ds.push_to_hub(target, split=split_name, private=private)
            print(f"  Pushed split '{split_name}' ({len(ds)} rows) → {target}")
        except Exception as e:
            print(f"  Failed to push split '{split_name}': {e}")


def _wipe_repo(api, repo_name, private, CommitOperationDelete, RepositoryNotFoundError):
    """Delete all files in a HF dataset repo and recreate it fresh."""
    try:
        api.create_repo(repo_id=repo_name, repo_type="dataset",
                        private=private, exist_ok=True)
        all_files = list(api.list_repo_files(repo_id=repo_name, repo_type="dataset"))
        wipe = [CommitOperationDelete(path_in_repo=f)
                for f in all_files if not f.startswith(".git")]
        if wipe:
            api.create_commit(
                repo_id=repo_name, repo_type="dataset",
                operations=wipe,
                commit_message="wipe repo before schema-changing push",
            )
            print(f"  Wiped {len(wipe)} file(s) from '{repo_name}'")
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_name, repo_type="dataset",
                        private=private, exist_ok=True)
    except Exception as e:
        print(f"  Warning: could not wipe '{repo_name}': {e}")

def push_splits(
    splits: dict,
    repo_name: str,
    rephrasings_repo_name: str = "",
    private: bool = False,
) -> None:
    """Push a subset of splits to HuggingFace Hub WITHOUT wiping the repo first.

    Useful for updating only forget_eval / forget_eval_rephrasings after
    changing the source eval rephrase file, without disturbing other splits.

    splits: {split_name: Dataset}
    forget_eval_rephrasings is routed to rephrasings_repo_name (or repo_name).
    """
    if not repo_name:
        print("  [skip] No hf_dataset_name configured; not uploading.")
        return

    rephrasings_target = rephrasings_repo_name or repo_name
    split_targets = {"forget_eval_rephrasings": rephrasings_target}

    for split_name, ds in splits.items():
        target = split_targets.get(split_name, repo_name)
        try:
            ds.push_to_hub(target, split=split_name, private=private)
            print(f"  Pushed split '{split_name}' ({len(ds)} rows) → {target}")
        except Exception as e:
            print(f"  Failed to push split '{split_name}': {e}")


def save_dataset_locally(dataset_dict: DatasetDict, output_dir: str | Path) -> None:
    """Save each split as a separate JSON file under output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, ds in dataset_dict.items():
        path = output_dir / f"{split_name}.json"
        ds.to_json(str(path))
        print(f"  Saved split '{split_name}' ({len(ds)} rows) -> {path}")
