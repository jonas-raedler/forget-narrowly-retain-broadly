"""
Full evaluation pipeline with ALL generation first, then ALL judging.

WORKFLOW:
1. PHASE 1: Generate all responses for all tasks in PARALLEL (one task per GPU)
   - Covers: forget/retain/forget_rephrasings (Evaluator-based) AND mmlu/repet (utility)
   - If more tasks than GPUs, they run in batches
2. PHASE 2: Judge forget/retain results across N parallel judge subprocesses
   (each on an NVLink-preferring GPU pair)
3. PHASE 2b: rgq_bi shares the same judge queue — each subprocess claims rgq tasks
   after finishing its judge work (rgq_bi depends on repet outputs from Phase 1)
4. PHASE 3: Compute final evaluation metrics

Shell script usage:
    bash scripts/suite_evaluation_optimized.sh
"""

import json
import logging
import multiprocessing as mp
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from queue import Queue as _Queue

import torch
import warnings
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer

from model import load_causal_model
from evals.eval_repetitiveness import eval_repet, load_rep_data
from evals.judge_eval_qwen import EvalJUDGE
from evals.judge_quality import EvalRGQ
from evals.mmlu_utils import eval_mmlu, load_mmlu
from evals.utils import Evaluator, load_eval_dataset
from evals.worst_eval import AvgEval, WorstEval

warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:None for open-end generation.")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task categories
# ---------------------------------------------------------------------------
FORGET_TASKS  = ('forget_rephrasings', 'forget_train_rephrasing', 'forget_adversarial')
FORGET_NO_ICR = ('forget_adversarial',)
RETAIN_TASKS  = ('retain', 'retain_train', 'retain_train_rephrasing')
UTILITY_TASKS = ('mmlu', 'repet')   # GPU generation, no judge

# Tasks that skip Phase-1 generation and reuse an existing task's generation output,
# then apply the gibberish judge prompt to detect gibberish/nonsense responses.
# Key = overlay task name, Value = base task whose generation file is reused.
GIBBERISH_OVERLAY_TASKS: Dict[str, str] = {
    'forget_rephrasings_gibberish': 'forget_rephrasings',
    'retain_gibberish':             'retain',
}

def is_utility_task(task: str) -> bool:
    """True for 'mmlu' or 'repet'."""
    return task in UTILITY_TASKS

RGQ_TASKS = ('rgq_bi',)  # judge-only, depends on repet

ALL_PARALLEL_TASKS = FORGET_TASKS + RETAIN_TASKS + UTILITY_TASKS  # run in Phase 1

# ---------------------------------------------------------------------------
# Trainer → short method folder name.
# Single source of truth: scripts/trainer_method_map.txt (also read by
# trainer_to_method in scripts/_suite_common.sh, which decides where the
# checkpoint is SAVED — this map decides where eval READS / groups results).
# A trainer not listed there falls back to lowercase(name) on BOTH sides.
# ---------------------------------------------------------------------------
def _load_method_map() -> Dict[str, str]:
    map_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "trainer_method_map.txt")
    if not os.path.isfile(map_path):
        # Fail loud: an empty map would silently break non-lowercase mappings
        # (e.g. JensUnPP → jensen) and desync save vs eval paths.
        raise FileNotFoundError(
            f"Trainer→folder map not found at {map_path}. This file is the single "
            f"source of truth shared with scripts/_suite_common.sh; it must exist.")
    mapping: Dict[str, str] = {}
    with open(map_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _sep, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if key and val:
                mapping[key] = val
    return mapping


METHOD_MAP: Dict[str, str] = _load_method_map()


def parse_path_parts(topic: str, model_path: str, hf_model_name: str) -> Tuple[str, str]:
    """Return (subpath, exp_name) for the hierarchical output layout.

    subpath  = <topic>/<model>/<method>[/relearn]
    exp_name = leaf run identifier (empty string for pretrained models)

    New hierarchical format (primary):
        saves/unlearn/{topic}/{model}/{method}/{exp}
        saves/unlearn/{topic}/{model}/{method}/{src_exp}/relearn/{exp}

    Examples
    --------
    pretrained:
        model_path='meta-llama/Llama-3.2-3B-Instruct'
        → ('challenger_disaster/Llama-3.2-3B-Instruct/pretrained', '')

    unlearn:
        model_path='./saves/unlearn/challenger_disaster/Llama-3.2-3B-Instruct/jensen/epochs_20_lrs_4e-6_…'
        → ('challenger_disaster/Llama-3.2-3B-Instruct/jensen', 'epochs_20_lrs_4e-6_…')

    relearn:
        model_path='./saves/unlearn/challenger_disaster/Llama-3.2-3B-Instruct/jensen/epochs_20_lrs_4e-6_gamma0.5_alpha1_scale_10000/relearn/GradLearn_epochs_10_lrs_4e-6'
        → ('challenger_disaster/Llama-3.2-3B-Instruct/jensen/epochs_20_lrs_4e-6_gamma0.5_alpha1_scale_10000/relearn', 'GradLearn_epochs_10_lrs_4e-6')
    """
    # Normalize missing ./ prefix (e.g. paths written as 'saves/unlearn/...' without './')
    if model_path.startswith('saves/unlearn/'):
        model_path = './' + model_path
    # Normalize absolute path (e.g. '/saves/unlearn/...' → './saves/unlearn/...')
    elif model_path.startswith('/saves/unlearn/'):
        model_path = '.' + model_path

    model_short = hf_model_name.split("/")[-1]

    # Pretrained: model_path is a HF hub ID (no local save directory)
    if not (model_path.startswith('./') or model_path.startswith('/')):
        return f"{topic}/{model_short}/pretrained", ""

    # New hierarchical format: ./saves/unlearn/{topic}/{model}/{method}[/relearn]/{exp}
    _unlearn_prefix = './saves/unlearn/'
    if model_path.startswith(_unlearn_prefix):
        rel = model_path[len(_unlearn_prefix):].rstrip('/')
        parts = rel.split('/')
        if len(parts) >= 4:
            path_topic, path_model, path_method = parts[0], parts[1], parts[2]
            # Detect relearn: look for a 'relearn' component anywhere after the method
            try:
                relearn_idx = parts.index('relearn', 3)
            except ValueError:
                relearn_idx = -1
            if relearn_idx >= 4:
                # Relearn path with source exp: {topic}/{model}/{method}/{src_exp}[/...]/relearn/{exp}
                src_exp  = '/'.join(parts[3:relearn_idx])
                exp_name = '/'.join(parts[relearn_idx + 1:]) if len(parts) > relearn_idx + 1 else ''
                return f"{path_topic}/{path_model}/{path_method}/{src_exp}/relearn", exp_name
            elif relearn_idx == 3:
                # Relearn path without source exp: {topic}/{model}/{method}/relearn/{exp}
                exp_name = '/'.join(parts[4:]) if len(parts) > 4 else ''
                return f"{path_topic}/{path_model}/{path_method}/relearn", exp_name
            else:
                exp_name = '/'.join(parts[3:])
                return f"{path_topic}/{path_model}/{path_method}", exp_name

    # Flat path fallback: saves/unlearn/Topic_new_fixed_{Model}_{Trainer}_…
    folder = os.path.basename(model_path.rstrip('/'))
    marker = f"_new_fixed_{model_short}_"
    idx = folder.find(marker)
    if idx == -1:
        return f"{topic}/{model_short}/unknown", folder

    after = folder[idx + len(marker):]

    if after.endswith('_relearn'):
        after_stripped = after[:-len('_relearn')]
        from_match = re.search(r'_from_([A-Za-z][A-Za-z0-9]*)_?', after_stripped)
        if from_match:
            src_token = from_match.group(1)
            method    = METHOD_MAP.get(src_token, src_token.lower())
            exp_name  = re.sub(rf'_from_{re.escape(src_token)}_', '_from_', after_stripped)
        else:
            method   = 'unknown'
            exp_name = after_stripped
        return f"{topic}/{model_short}/{method}/relearn", exp_name

    flat_parts   = after.split('_', 1)
    method_token = flat_parts[0]
    method       = METHOD_MAP.get(method_token, method_token.lower())
    exp_name     = flat_parts[1] if len(flat_parts) > 1 else after
    return f"{topic}/{model_short}/{method}", exp_name


# Judge batch sizes per task — tasks with longer prompts need a smaller batch to avoid OOM.
# Default (unspecified) tasks use judge_batch_size_default from eval.yaml (currently 4).
JUDGE_BATCH_SIZE = {
    'forget_adversarial': 4,
}

# Generation batch size (max prompts per model.generate call) for Evaluator tasks.
# For rephrasings tasks each row already has ~15-20 variant columns, so 1 row ≈ batch already.
# For plain tasks (forget/retain) each row has 1 prompt, so N rows are batched together.
# Both tiers pinned at 32 — safe on A100-40GB for every supported model size.
GEN_BATCH_SIZE_SMALL = 32   # Llama-3B and other ≤4B models
GEN_BATCH_SIZE_LARGE = 32   # Llama-8B, Qwen-9B, and other 7B/8B/9B models

# Treat any model with >= this many billion parameters as "large" for batch sizing.
LARGE_MODEL_MIN_B = 7.0


def model_param_billions(name: str) -> float:
    """Best-effort parse of a model's parameter count (in billions) from its name.

    Matches the number immediately preceding a 'B'/'b' token, e.g.
    'Qwen3.5-9B' -> 9.0, 'Llama-3.2-3B' -> 3.0,
    'Ministral-3-3B-Instruct-2512-BF16' -> 3.0 (the 'BF16' tag has no digit before
    its 'B', so it is ignored). Returns 0.0 when no size token is found.
    """
    sizes = [float(m) for m in re.findall(r'(\d+(?:\.\d+)?)[bB]', name)]
    return max(sizes) if sizes else 0.0


def is_large_model(name: str) -> bool:
    """True when the model has >= LARGE_MODEL_MIN_B billion parameters (7B+)."""
    return model_param_billions(name) >= LARGE_MODEL_MIN_B


def icr_variants_for(eval_task: str) -> List[bool]:
    """Return the list of ICR flag values required for *eval_task*."""
    if eval_task in FORGET_TASKS and eval_task not in FORGET_NO_ICR:
        return [False, True]
    return [False]


def needs_judge(eval_task: str) -> bool:
    """True if the task result needs to pass through EvalJUDGE (Phase 2)."""
    return (eval_task in FORGET_TASKS or eval_task in RETAIN_TASKS
            or eval_task in GIBBERISH_OVERLAY_TASKS)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GenerationTask:
    """One Evaluator-based generation run: (model, eval_task, icr)."""
    model_name: str
    model_path: str
    eval_task:  str
    icr:        bool
    cfg:        DictConfig

    @property
    def result_key(self) -> str:
        return f"{self.model_name}|{self.eval_task}|{self.icr}|{self.cfg.output.task_name}"

    def __hash__(self):
        return hash(self.result_key)

    def __eq__(self, other):
        return self.result_key == other.result_key


@dataclass
class UtilityTask:
    """A utility generation task (mmlu or repet) that runs in parallel in Phase 1."""
    model_name: str
    model_path: str
    eval_task:  str   # 'mmlu' or 'repet'
    task_name:  str   # experiment label (used for output filename)
    cfg:        DictConfig

    @property
    def result_key(self) -> str:
        return f"utility|{self.eval_task}|{self.task_name}"

    @property
    def out_path(self) -> str:
        """Expected output file path for this task's generation results."""
        if self.eval_task == 'mmlu':
            prefix   = 'MMLU'
            base_dir = self.cfg.output.gendir
        else:
            prefix   = 'Rep'
            base_dir = self.cfg.output.repdir
        subpath    = getattr(self.cfg.output, 'subpath',  '') or ''
        exp_name   = getattr(self.cfg.output, 'exp_name', '') or ''
        model_short = self.model_name.split("/")[-1]
        if subpath:
            subdir   = os.path.join(base_dir, subpath)
            suffix   = f"_{exp_name}" if exp_name else ""
            filename = f"{prefix}_{model_short}{suffix}.jsonl"
        else:
            subdir   = base_dir
            filename = f"{prefix}_{model_short}_{self.task_name}.jsonl"
        return os.path.join(subdir, filename)


@dataclass
class RGQTask:
    """A Relative Generation Quality (RGQ) task – runs after all generation, uses EvalRGQ (its own judge)."""
    model_name:   str
    model_path:   str   # path of the *unlearned* model (not used for inference, just config)
    task_name:    str   # experiment label for the unlearned model
    pretrained_task_name: str  # task_name used for the pretrained repet file
    cfg:          DictConfig

    @property
    def result_key(self) -> str:
        return f"rgq_bi|{self.task_name}"


# ---------------------------------------------------------------------------
# Generation registry
# ---------------------------------------------------------------------------
class GenerationRegistry:
    """Persist generated-file paths so we don't regenerate on re-runs."""

    REGISTRY_FILE = "./evaluations/generation_registry.json"

    def __init__(self):
        self._data: Dict[str, str] = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.REGISTRY_FILE):
            with open(self.REGISTRY_FILE, encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save(self):
        os.makedirs(os.path.dirname(self.REGISTRY_FILE), exist_ok=True)
        with open(self.REGISTRY_FILE, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def _key(self, topic: str, model_name: str, eval_task: str, icr: bool, task_name: str) -> str:
        return f"{topic}|{model_name}|{eval_task}|{icr}|{task_name}"

    def register(self, topic: str, model_name: str, eval_task: str, icr: bool,
                 task_name: str, out_path: str):
        self._data[self._key(topic, model_name, eval_task, icr, task_name)] = out_path
        self._save()

    def get(self, topic: str, model_name: str, eval_task: str, icr: bool,
            task_name: str) -> Optional[str]:
        path = self._data.get(self._key(topic, model_name, eval_task, icr, task_name))
        if path and not os.path.exists(path):
            # Stale entry — file was deleted or never written; treat as not cached
            logger.debug(f"Registry entry exists but file missing, invalidating: {path}")
            del self._data[self._key(topic, model_name, eval_task, icr, task_name)]
            self._save()
            return None
        return path

    def all_generated(self, tasks: List[GenerationTask]) -> bool:
        return all(
            self.get(t.cfg.output.topic, t.model_name, t.eval_task, t.icr, t.cfg.output.task_name)
            for t in tasks
        )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def load_model_tokenizer(cfg):
    # Fail fast with a clear message when the local path doesn't exist — avoids
    # the confusing "Repo id must be in the form..." HF Hub error.
    if (cfg.model.path.startswith('./') or cfg.model.path.startswith('/')) \
            and not os.path.isdir(cfg.model.path):
        raise FileNotFoundError(
            f"Model path does not exist: {cfg.model.path!r}\n"
            "Check that the model was saved to this path (task_name mismatch?)"
        )

    # Load the model first so we know its actual vocab/embedding size.
    # Local DeepSpeed / ZeRO saves sometimes lack a `model_type` in their
    # config.json.  Passing the HF config from cfg.model.name as a fallback
    # tells AutoModel which architecture to use without re-downloading weights.
    model = load_causal_model(
        cfg.model.path,
        is_local=os.path.isdir(cfg.model.path),
        device_map="auto",
    )

    # Always try to load the tokenizer from the *same path as the model* first.
    # This is critical when the model's vocab was resized during training
    # (added special tokens): using the base HF tokenizer would produce token
    # IDs that exceed the embedding table → CUDA index-OOB crash.
    tokenizer_source = cfg.model.path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, padding_side='left')
    except Exception:
        # Fall back to the HF model name/ID if the local path has no tokenizer files.
        logger.warning(
            f"Tokenizer not found at model.path={cfg.model.path!r}; "
            f"falling back to model.name={cfg.model.name!r}"
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, padding_side='left')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure model.config agrees on pad_token_id so generate() never inserts
    # an out-of-vocabulary ID into the embedding lookup.
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    return model, tokenizer


def make_task_cfg(base_cfg: DictConfig, model_path: str, task_name: str,
                  eval_task: str, split: str, dataset_name: str,
                  max_tokens: int, model_name: str = None) -> DictConfig:
    """Create a per-task OmegaConf config from the base config."""
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg.model.path            = model_path
    cfg.output.task_name      = task_name
    cfg.output.eval_task      = eval_task
    cfg.dataset.split         = split
    cfg.generation.max_new_tokens = max_tokens
    if dataset_name:
        cfg.dataset.name = dataset_name
    if model_name:
        cfg.model.name = model_name  # override the default Llama name in eval.yaml
    # Compute hierarchical path parts when topic is set
    topic = cfg.output.get('topic', '') or ''
    if topic:
        subpath, exp_name = parse_path_parts(topic, model_path, cfg.model.name)
        # Cross-topic eval: if the eval topic differs from the training topic embedded in
        # the model path, insert an eval_{topic} subdirectory (mirrors the relearn pattern).
        subpath_parts = subpath.split('/')
        training_topic = subpath_parts[0] if subpath_parts else ''
        method_part    = subpath_parts[2] if len(subpath_parts) > 2 else ''
        # Compound training topic (e.g. "challenger_disaster+salem_witch_trials"):
        # always add eval_{topic} so each topic's results land in a separate subdir.
        # Single-topic model: only add the subdir for cross-topic evaluations.
        is_compound = '+' in training_topic
        if training_topic and method_part != 'pretrained':
            if is_compound or topic != training_topic:
                subpath = f"{subpath}/eval_{topic}"
        cfg.output.subpath  = subpath
        cfg.output.exp_name = exp_name
    return cfg


# ---------------------------------------------------------------------------
# Generation metadata header
# ---------------------------------------------------------------------------
def _generation_metadata(cfg, icr=None) -> dict:
    """First entry written to every generation JSONL – for easy identification."""
    return {
        "__metadata__": True,
        "model_name":   cfg.model.name,
        "model_path":   cfg.model.path,
        "eval_task":    cfg.output.eval_task,
        "task_name":    cfg.output.task_name,
        "dataset":      cfg.dataset.name,
        "split":        cfg.dataset.split,
        "icr":          icr,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Evaluator-based generation
# ---------------------------------------------------------------------------
def generate_responses(cfg, ev):
    dataset  = load_eval_dataset(cfg.dataset.name, cfg.dataset.split,
                                 filter_topic=cfg.output.get("topic", "") or "")
    map_fn   = lambda ex: {k: ev.get_template(v) for k, v in ex.items()}
    model, tokenizer = load_model_tokenizer(cfg)

    try:
        is_large  = is_large_model(cfg.model.name)
        gen_batch = GEN_BATCH_SIZE_LARGE if is_large else GEN_BATCH_SIZE_SMALL

        pending   = []   # list of (batch_dict, question, gt_answer, label, raw_questions)
        pending_n = [0]  # mutable counter (list so nested function can mutate it)

        def _flush():
            if pending:
                ev.evaluate_rows(model, tokenizer, pending)
                pending.clear()
                pending_n[0] = 0

        for i in tqdm(range(len(dataset)), desc="Generating"):
            item  = dataset[i]
            batch = {k: v for k, v in item.items()
                     if k not in ("answer", "label", "refusal", "count", "rep", "topic")
                     and v is not None and str(v).strip() != ""}   # skip empty/null columns
            # Preserve raw rephrase question texts (q_qwen1, q_llama1, blank_qwen1, …)
            # so the judge can use the correct question text for each rephrased answer.
            raw_questions = {k: v for k, v in batch.items() if k != "question"}
            mapped = map_fn(batch)
            n = len(mapped)
            # Flush before adding if this row would push us over the batch limit
            if pending_n[0] + n > gen_batch and pending:
                _flush()
            pending.append((mapped, item["question"], item["answer"],
                            item.get("label"), raw_questions))
            pending_n[0] += n

        _flush()   # process any remaining rows

        assert len(ev.logs) == len(dataset), \
            f"Output length mismatch: {len(ev.logs)} vs {len(dataset)}"

        meta = _generation_metadata(cfg, icr=ev.icr_data)
        ev.logs = [meta] + ev.logs
        ev.save_logs()
    finally:
        # Free GPU memory so the next task on this slot starts with a clean slate
        del model, tokenizer
        torch.cuda.empty_cache()


def _evaluator_worker(task: GenerationTask, gpu_id: str
                      ) -> Tuple[GenerationTask, Optional[str], Optional[str]]:
    """Subprocess worker for Evaluator-based tasks.
    gpu_id is the PHYSICAL GPU id string (e.g. '4'), not a slot index.
    """
    try:
        # Configure expandable segments before first CUDA alloc to prevent fragmentation
        # when multiple tasks run sequentially on the same GPU slot.
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        ev = Evaluator(task.cfg, icr=task.icr, task=task.eval_task)
        done, _ = ev.load_logs_from_file()
        if done:
            logger.info(f"[GPU {gpu_id}] Already done: {task.result_key}")
            return task, ev.out_path, None
        # Remove any leftover .tmp file from a previous crashed run
        tmp_path = ev.out_path + ".tmp"
        if os.path.exists(tmp_path):
            logger.warning(f"[GPU {gpu_id}] Removing stale .tmp file: {tmp_path}")
            os.remove(tmp_path)
        logger.info(f"[GPU {gpu_id}] Generating: {task.result_key}")
        generate_responses(task.cfg, ev)
        out_path = ev.out_path
        logger.info(f"[GPU {gpu_id}] ✓ Done: {task.result_key}")
        del ev
        torch.cuda.empty_cache()
        return task, out_path, None
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] ✗ {task.result_key}: {e}")
        torch.cuda.empty_cache()
        return task, None, str(e)


# ---------------------------------------------------------------------------
# Utility generation (mmlu / repet)
# ---------------------------------------------------------------------------
def _utility_worker(task: UtilityTask, gpu_id: str
                    ) -> Tuple[UtilityTask, Optional[str], Optional[str]]:
    """Subprocess worker for mmlu / repet tasks.
    gpu_id is the PHYSICAL GPU id string (e.g. '4'), not a slot index.
    """
    try:
        # Configure expandable segments before first CUDA alloc to prevent fragmentation
        # when multiple tasks run sequentially on the same GPU slot.
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        out_path = task.out_path

        if os.path.exists(out_path):
            logger.info(f"[GPU {gpu_id}] Already done: {task.result_key}")
            return task, out_path, None

        # Remove any leftover .tmp file from a previous crashed run
        tmp_path = out_path + ".tmp"
        if os.path.exists(tmp_path):
            logger.warning(f"[GPU {gpu_id}] Removing stale .tmp file: {tmp_path}")
            os.remove(tmp_path)

        logger.info(f"[GPU {gpu_id}] Running {task.eval_task}: {task.result_key}")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        model, tokenizer = load_model_tokenizer(task.cfg)

        # Detect how many GPUs device_map spread the model across.
        # Qwen3.5-9B in float32 (~36 GiB) overflows one A100-40GB → 2 GPUs → 80 GiB total,
        # so we can afford a larger repet batch than for single-GPU models.
        n_model_gpus = len({v for v in getattr(model, 'hf_device_map', {}).values()
                            if str(v).startswith('cuda')}) or 1
        # Batch sizes tuned from nvidia-smi on A100-40GB:
        #   Llama-8B   mmlu : model ~22 GiB; per-sample ~3.2 GiB → batch=4 ≈35 GiB (batch=5 → ~38 GiB, borderline)
        #   Qwen3.5-9B repet: float32 ~36 GiB → 2 GPUs; ~44 GiB free → batch=16 safe
        #   Llama-3B   mmlu : batch=8 safe (~10 GiB total) after fixing GPU cleanup between tasks
        #   Llama-3B   repet: batch=8 safe at 1000-token sequences
        params_b = model_param_billions(task.model_name)
        repet_max_tokens = task.cfg.generation.max_new_tokens
        mmlu_batch  = 4 if params_b >= LARGE_MODEL_MIN_B else 8
        repet_batch = 64
        # A ~12B model fills ~30 GiB of one 40 GiB GPU in bf16, leaving too little for
        # the default utility batches → OOM. Trim ONLY this tier; <12B models keep the
        # values above.
        # Batch size does not change MMLU argmax / deterministic repet outputs, so this
        # is metric-safe.
        if params_b >= 12:
            mmlu_batch  = 2     # mmlu OOMs at 4 for a 12B model on a 40 GiB card
            repet_batch = 16    # 64 OOMs at 1000-token gen; 16 fits the ~10 GiB headroom

        try:
            if task.eval_task == 'mmlu':
                eval_mmlu(model, tokenizer, load_mmlu(), batch_size=mmlu_batch,
                          output_result_dir=out_path, use_prompt=False, num_samples=2000)
            elif task.eval_task == 'repet':
                eval_repet(model, tokenizer, load_rep_data(), batch_size=repet_batch,
                           output_result_dir=out_path, use_prompt=False, num_samples=500,
                           max_new_tokens=repet_max_tokens)
            else:
                raise NotImplementedError(f"Unknown utility task: {task.eval_task}")
        finally:
            # Explicitly free GPU memory before this slot is reused
            del model, tokenizer
            torch.cuda.empty_cache()

        logger.info(f"[GPU {gpu_id}] ✓ Done: {task.result_key}")
        return task, out_path, None
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] ✗ {task.result_key}: {e}")
        return task, None, str(e)


def _visible_gpu_ids() -> List[str]:
    """
    Return the list of physical GPU IDs that are visible to this process.

    If CUDA_VISIBLE_DEVICES is set (e.g. "4,5,6,7"), return ["4","5","6","7"].
    If it is unset or empty, fall back to ["0".."N-1"] using torch.cuda.device_count().
    Workers must set CUDA_VISIBLE_DEVICES to ONE of these IDs so they stay
    within the allowed set and never land on training GPUs.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd.strip():
        return [g.strip() for g in cvd.split(",") if g.strip()]
    n = torch.cuda.device_count() or 1
    return [str(i) for i in range(n)]


# ---------------------------------------------------------------------------
# PHASE 1 – parallel generation (Evaluator tasks + utility tasks)
# ---------------------------------------------------------------------------
def phase1_generate_all(
        gen_tasks:     List[GenerationTask],
        utility_tasks: List[UtilityTask],
        num_gpus:      Optional[int] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Run ALL generation in parallel (one slot per GPU).

    Each subprocess is assigned one physical GPU from the parent's
    CUDA_VISIBLE_DEVICES list, so generation never touches GPUs that are
    reserved for training.

    Returns:
        gen_results     – {result_key -> out_path}  for Evaluator tasks
        utility_results – {result_key -> out_path}  for utility (mmlu/repet) tasks
    """
    gpu_ids = _visible_gpu_ids()          # e.g. ["4","5"] or ["4","5","6","7"]
    if num_gpus is None:
        num_gpus = len(gpu_ids)

    logger.info(f"Available GPU slots: {gpu_ids}  (num_gpus={num_gpus})")

    registry = GenerationRegistry()
    gen_results:     Dict[str, str] = {}
    utility_results: Dict[str, str] = {}

    # ---- Evaluator tasks: check cache ----
    to_generate: List[GenerationTask] = []
    for task in gen_tasks:
        cached = registry.get(task.cfg.output.topic, task.model_name, task.eval_task, task.icr,
                              task.cfg.output.task_name)
        if cached and os.path.exists(cached):
            logger.info(f"✓ Cached (gen):     {task.result_key}")
            gen_results[task.result_key] = cached
        else:
            to_generate.append(task)

    # ---- Utility tasks: check output file ----
    to_run_util: List[UtilityTask] = []
    for task in utility_tasks:
        if os.path.exists(task.out_path):
            logger.info(f"✓ Cached (utility): {task.result_key}")
            utility_results[task.result_key] = task.out_path
        else:
            to_run_util.append(task)

    total_pending = len(to_generate) + len(to_run_util)
    if total_pending == 0:
        logger.info("All tasks already generated – skipping Phase 1.")
        return gen_results, utility_results

    logger.info("=" * 80)
    logger.info(f"PHASE 1: PARALLEL GENERATION")
    logger.info(f"  Evaluator tasks : {len(to_generate)}  (forget/retain/…)")
    logger.info(f"  Utility tasks   : {len(to_run_util)}  (mmlu/repet)")
    logger.info(f"  Total pending   : {total_pending}  on {min(total_pending, num_gpus)} GPU slot(s)")
    logger.info("=" * 80)

    # Interleave both task types in one pool so GPU slots are fully used
    # Tag each item as ('gen', task) or ('util', task)
    tagged: List[Tuple[str, Union[GenerationTask, UtilityTask]]] = (
        [('gen',  t) for t in to_generate] +
        [('util', t) for t in to_run_util]
    )

    n_ok = n_fail = 0

    # Dynamic GPU slot queue: each GPU runs at most ONE task at a time.
    # When a task finishes it returns its GPU id to the free-slot queue so
    # the next waiting task can be dispatched immediately.
    free_slots: _Queue = _Queue()
    for gid in gpu_ids:
        free_slots.put(gid)

    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as pool:
        future_map: Dict = {}   # future -> (tag, task, gpu_id)
        queue = list(tagged)    # tasks still waiting to be submitted

        # Seed: submit one task per available GPU slot
        while queue and not free_slots.empty():
            gpu_id = free_slots.get()
            tag, task = queue.pop(0)
            if tag == 'gen':
                f = pool.submit(_evaluator_worker, task, gpu_id)
            else:
                f = pool.submit(_utility_worker, task, gpu_id)
            future_map[f] = (tag, task, gpu_id)
            logger.info(f"[GPU {gpu_id}] Queued: {task.result_key}")

        # Process completions and dispatch remaining tasks
        while future_map:
            for future in as_completed(future_map):
                tag, task, gpu_id = future_map.pop(future)
                try:
                    _, path, err = future.result()
                    if err or path is None:
                        logger.error(f"✗ {task.result_key}: {err}")
                        n_fail += 1
                    else:
                        if tag == 'gen':
                            registry.register(task.cfg.output.topic, task.model_name, task.eval_task, task.icr,
                                               task.cfg.output.task_name, path)
                            gen_results[task.result_key] = path
                        else:
                            utility_results[task.result_key] = path
                        n_ok += 1
                except Exception as e:
                    logger.error(f"✗ Process error for {task.result_key}: {e}")
                    n_fail += 1

                # GPU slot is free — dispatch next waiting task if any
                if queue:
                    tag2, task2 = queue.pop(0)
                    if tag2 == 'gen':
                        f2 = pool.submit(_evaluator_worker, task2, gpu_id)
                    else:
                        f2 = pool.submit(_utility_worker, task2, gpu_id)
                    future_map[f2] = (tag2, task2, gpu_id)
                    logger.info(f"[GPU {gpu_id}] Queued (next): {task2.result_key}")
                else:
                    free_slots.put(gpu_id)
                break  # restart as_completed with updated future_map

    logger.info(f"Phase 1 complete: {n_ok} succeeded, {n_fail} failed")
    if n_fail:
        raise RuntimeError(f"{n_fail} generation task(s) failed")
    return gen_results, utility_results


# ---------------------------------------------------------------------------
# Judge path helper (module-level so both sequential and parallel paths use it)
# ---------------------------------------------------------------------------
def _jg_path_for(eval_task_name: str, icr_bool: bool, judge_cfg: DictConfig,
                 judge_tag: Optional[str] = None) -> str:
    """Return the expected JG_EVAL output path for a given task/config/tag combination.

    Mirrors EvalJUDGE.set_out_dirs so both the cache-check and Phase 3 agree on the path.
    When judge_tag is set the prefix becomes JG_EVAL_{tag}_, else JG_EVAL_.
    """
    model_short = judge_cfg.model.name.split("/")[-1]
    task_name   = judge_cfg.output.task_name
    subpath     = getattr(judge_cfg.output, 'subpath',  '') or ''
    exp_name    = getattr(judge_cfg.output, 'exp_name', '') or ''
    # Mirror EvalJUDGE.set_out_dirs: hierarchical dir_key when topic/subpath is set
    if subpath:
        dir_key = os.path.join(subpath, exp_name) if exp_name else subpath
    else:
        dir_key = task_name
    suffix      = f"{model_short}_{eval_task_name}_icr_{icr_bool}.jsonl"
    jg_subdir   = os.path.join(judge_cfg.output.evaldir, dir_key)
    tag_prefix  = f"JG_EVAL_{judge_tag}_" if judge_tag else "JG_EVAL_"
    return os.path.join(jg_subdir, tag_prefix + suffix)


# ---------------------------------------------------------------------------
# Refusal overlay helpers
# ---------------------------------------------------------------------------
def _base_gen_file_path(task_cfg: DictConfig, base_task: str) -> str:
    """Return the EVAL_* generation file path for *base_task* given an overlay task config.

    Mirrors Evaluator.set_out_dirs with icr=False so the path is deterministic
    without instantiating a full Evaluator.
    """
    model_short = task_cfg.model.name.split("/")[-1]
    subpath     = getattr(task_cfg.output, 'subpath',  '') or ''
    exp_name    = getattr(task_cfg.output, 'exp_name', '') or ''
    if subpath:
        dir_key = os.path.join(subpath, exp_name) if exp_name else subpath
    else:
        dir_key = task_cfg.output.task_name
    out_dir = os.path.join(task_cfg.output.dir, dir_key)
    return os.path.join(out_dir, f"EVAL_{model_short}_{base_task}_icr_False.jsonl")


def _inject_overlay_gen_results(
        overlay_tasks: List[Tuple[str, 'DictConfig']],
        gen_results:   Dict[str, str],
) -> None:
    """Resolve generation file paths for gibberish overlay tasks and inject into gen_results.

    For each overlay task, tries (in order):
      1. Base task's generation file already exists on disk.
      2. Base task was just generated and is already in gen_results.
      3. Re-runs Phase 1 for the missing base task (single sequential pass).
    """
    pending: List[Tuple[str, 'DictConfig', str, str]] = []

    for overlay_eval_task, task_cfg in overlay_tasks:
        base_task = GIBBERISH_OVERLAY_TASKS[overlay_eval_task]
        gen_file  = _base_gen_file_path(task_cfg, base_task)
        base_key  = f"{task_cfg.model.name}|{base_task}|False|{task_cfg.output.task_name}"
        result_key = f"{task_cfg.model.name}|{overlay_eval_task}|False|{task_cfg.output.task_name}"

        if os.path.exists(gen_file):
            gen_results[result_key] = gen_file
            logger.info(f"✓ Overlay [{overlay_eval_task}] → {os.path.basename(gen_file)} (existing file)")
        elif base_key in gen_results:
            gen_results[result_key] = gen_results[base_key]
            logger.info(f"✓ Overlay [{overlay_eval_task}] → {os.path.basename(gen_results[base_key])} (just generated)")
        else:
            pending.append((overlay_eval_task, task_cfg, base_task, gen_file))

    if not pending:
        return

    logger.info(f"  {len(pending)} gibberish overlay task(s) need base-task generation first …")
    extra_gen_tasks: List[GenerationTask] = []
    seen_keys: set = set()
    for overlay_eval_task, task_cfg, base_task, _ in pending:
        base_cfg = OmegaConf.create(OmegaConf.to_container(task_cfg, resolve=True))
        base_cfg.output.eval_task = base_task
        gen_task = GenerationTask(
            model_name=task_cfg.model.name,
            model_path=task_cfg.model.path,
            eval_task=base_task,
            icr=False,
            cfg=base_cfg,
        )
        if gen_task.result_key not in seen_keys:
            extra_gen_tasks.append(gen_task)
            seen_keys.add(gen_task.result_key)

    extra_results, _ = phase1_generate_all(extra_gen_tasks, [])

    for overlay_eval_task, task_cfg, base_task, gen_file in pending:
        base_key   = f"{task_cfg.model.name}|{base_task}|False|{task_cfg.output.task_name}"
        result_key = f"{task_cfg.model.name}|{overlay_eval_task}|False|{task_cfg.output.task_name}"
        if not os.path.exists(gen_file) and base_key in extra_results:
            gen_file = extra_results[base_key]
        if os.path.exists(gen_file):
            gen_results[result_key] = gen_file
            logger.info(f"✓ Overlay [{overlay_eval_task}] → {os.path.basename(gen_file)}")
        else:
            logger.warning(f"⚠ Overlay [{overlay_eval_task}]: base gen file not found – {gen_file}")


# ---------------------------------------------------------------------------
# GPU pair/group detection (NVLink-aware, SLURM-safe)
# ---------------------------------------------------------------------------
def _nvlink_pairs(gpu_ids: List[str]) -> List[Tuple[str, str]]:
    """Return NVLink-preferring pairs from the visible GPU ID list.

    gpu_ids are the strings from CUDA_VISIBLE_DEVICES (e.g. ["4","5","6","7"]).
    In SLURM these ARE the physical GPU IDs that nvidia-smi uses, so the topo
    table is directly readable without additional re-mapping.
    Falls back to consecutive pairs on any error or when no NVLink is found.
    """
    import subprocess as _sp
    import re as _re
    from collections import defaultdict as _dd

    def _consecutive(ids):
        return [(ids[i], ids[i + 1]) for i in range(0, len(ids) - 1, 2)]

    if len(gpu_ids) < 2:
        return []

    try:
        topo_result = _sp.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True, text=True, timeout=15
        )
        if topo_result.returncode != 0:
            return _consecutive(gpu_ids)

        lines = [l for l in topo_result.stdout.splitlines() if l.strip()]

        # Find the column-header line (e.g. "        GPU0  GPU1  GPU2 ...")
        header_line = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("GPU") and "GPU" in stripped[3:]:
                header_line = stripped
                break
        if header_line is None:
            return _consecutive(gpu_ids)

        col_gpus = [tok[3:] for tok in header_line.split() if tok.startswith("GPU")]
        gpu_id_set = set(gpu_ids)

        nvlink_adj: dict = _dd(set)
        for line in lines:
            parts = line.split()
            if not parts or not parts[0].startswith("GPU"):
                continue
            row_gpu = parts[0][3:]
            if row_gpu not in gpu_id_set:
                continue
            for col_idx, token in enumerate(parts[1:]):
                if col_idx >= len(col_gpus):
                    break
                col_gpu = col_gpus[col_idx]
                if col_gpu == row_gpu or col_gpu not in gpu_id_set:
                    continue
                if _re.match(r'^NV\d+$', token):
                    nvlink_adj[row_gpu].add(col_gpu)

        # Greedily build pairs, preferring NVLink partners
        used: set = set()
        pairs: List[Tuple[str, str]] = []
        remaining = list(gpu_ids)
        while len(remaining) >= 2:
            a = remaining.pop(0)
            if a in used:
                continue
            # Prefer NVLink partner; fall back to first available
            partner = next(
                (b for b in remaining if b not in used and b in nvlink_adj.get(a, set())),
                next((b for b in remaining if b not in used), None)
            )
            if partner is None:
                break
            pairs.append((a, partner))
            used.update({a, partner})
            remaining = [g for g in remaining if g not in used]

        return pairs if pairs else _consecutive(gpu_ids)

    except Exception:
        return _consecutive(gpu_ids)


# ---------------------------------------------------------------------------
# Judge subprocess worker (module-level for multiprocessing spawn compatibility)
# ---------------------------------------------------------------------------
def _judge_subprocess_worker(
        task_queue,                      # multiprocessing.Queue of tagged items (see below)
        task_configs_plain: dict,        # OmegaConf serialised as plain dicts
        gpu_pair: Tuple[str, str],
        judge_hf_model_id: str,
        judge_tag: Optional[str],
        judge_batch_size_default: int,
) -> None:
    """Run in a separate process: own 2 GPUs, pull tasks from the shared queue until empty.

    Queue items are tagged tuples:
      ('judge', key, out_path, eval_task_name, icr_bool_str, task_name_key)
      ('rgq',   rgq_dict)   — rgq_dict is a serialised RGQTask plain dict

    Both EvalJUDGE and EvalRGQ share the same loaded model weights, so the first task
    (of either type) that loads the model makes it available to all subsequent tasks.
    RGQ tasks are put into the same queue so they are claimed dynamically by
    whichever subprocess goes idle first — exactly like judge tasks.
    """
    import sys as _sys
    import logging as _logging
    import queue as _queue
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{gpu_pair[0]},{gpu_pair[1]}"

    _src_dir = os.path.join(os.path.dirname(__file__), '..')
    if _src_dir not in _sys.path:
        _sys.path.insert(0, _src_dir)

    from evals.judge_eval_qwen import EvalJUDGE as _EvalJUDGE
    from evals.judge_quality import EvalRGQ as _EvalRGQ
    from omegaconf import OmegaConf as _OmegaConf

    _logging.basicConfig(level=_logging.INFO,
                         format='%(asctime)s [judge-%(process)d] %(levelname)s %(message)s')
    _log = _logging.getLogger(__name__)
    _log.info(f"Judge subprocess started on GPUs {gpu_pair}, pulling from shared task queue")

    task_configs = {k: _OmegaConf.create(v) for k, v in task_configs_plain.items()}

    ej             = None   # EvalJUDGE instance (loaded lazily)
    shared_model     = None
    shared_tokenizer = None

    while True:
        try:
            item = task_queue.get(timeout=5)
        except _queue.Empty:
            break

        task_type = item[0]

        if task_type == 'judge':
            _, key, out_path, eval_task_name, icr_bool_str, task_name_key = item
            icr_bool      = icr_bool_str == "True"
            config_lookup = f"{eval_task_name}|{task_name_key}"
            judge_cfg     = task_configs.get(config_lookup)
            if judge_cfg is None:
                _log.error(f"  No config for '{config_lookup}' – skipping.")
                continue

            judge_cfg = _OmegaConf.create(_OmegaConf.to_container(judge_cfg, resolve=True))
            judge_cfg.output.eval_task = eval_task_name

            bs_override  = JUDGE_BATCH_SIZE.get(eval_task_name)
            effective_bs = bs_override if bs_override is not None else judge_batch_size_default

            if ej is None:
                kwargs = dict(task=eval_task_name, gen_file=out_path, icr=icr_bool,
                              batch_size=effective_bs)
                ej = _EvalJUDGE(judge_cfg, hf_model_id=judge_hf_model_id,
                                 judge_tag=judge_tag, **kwargs)
                shared_model     = ej.model
                shared_tokenizer = ej.tokenizer
                _log.info("✓ EvalJUDGE model loaded")
            else:
                ej.eval_cfg = judge_cfg
                ej.name     = judge_cfg.model.name
                ej.task     = eval_task_name
                ej.gen_file = out_path
                ej.set_icr(icr_bool)
                if ej.batch_size != effective_bs:
                    ej.batch_size = effective_bs

            if os.path.exists(ej.jg_file_path):
                _log.info(f"✓ Already judged – {os.path.basename(ej.jg_file_path)}")
            else:
                ej.generate()
                _log.info(f"✓ Judged → {os.path.basename(ej.jg_file_path)}")

        elif task_type == 'rgq':
            _, rgq_dict = item
            rgq_cfg = _OmegaConf.create(rgq_dict['cfg'])
            try:
                ew = _EvalRGQ(rgq_cfg, model=shared_model, tokenizer=shared_tokenizer,
                              pretrained_task_name=rgq_dict['pretrained_task_name'])
                if shared_model is None and ew.model is not None:
                    shared_model     = ew.model
                    shared_tokenizer = ew.tokenizer
                ew.rgq_bi()
                _log.info(f"✓ RGQ complete inside judge subprocess")
            except Exception as e:
                _log.warning(f"⚠ RGQ failed in judge subprocess: {e}")

        else:
            _log.error(f"Unknown task type in queue: {task_type!r} – skipping.")


# ---------------------------------------------------------------------------
# PHASE 2 – EvalJUDGE (forget / retain)
# ---------------------------------------------------------------------------
def phase2_judge_all(gen_results: Dict[str, str],
                     task_configs: Dict[str, DictConfig],
                     judge_hf_model_id: str = "Qwen/Qwen3.5-35B-A3B",
                     judge_tag: Optional[str] = None,
                     judge_batch_size_default: int = 16,
                     ) -> Optional[EvalJUDGE]:
    """
    Load EvalJUDGE ONCE, then judge every forget/retain result sequentially.
    Returns the EvalJUDGE instance so its model can be reused by EvalRGQ (rgq_bi).

    gen_results  – {result_key -> out_path}    (only forget/retain keys)
    task_configs – {"eval_task|task_name" -> task_cfg}
    """
    if not gen_results:
        logger.info("No generation results to judge.")
        return None

    logger.info("=" * 80)
    logger.info(f"PHASE 2: UNIFIED JUDGING  (single EvalJUDGE instance)")
    logger.info(f"  {len(gen_results)} task(s) to judge")
    logger.info("=" * 80)

    all_jg_paths = []
    for key in gen_results:
        parts          = key.split("|")
        eval_task_name = parts[1]
        icr_bool       = parts[2] == "True"
        task_name_key  = parts[3] if len(parts) >= 4 else None
        judge_cfg      = task_configs.get(f"{eval_task_name}|{task_name_key}")
        if judge_cfg is not None:
            all_jg_paths.append(_jg_path_for(eval_task_name, icr_bool, judge_cfg, judge_tag))

    already_done = [p for p in all_jg_paths if os.path.exists(p)]
    if len(already_done) == len(all_jg_paths) and all_jg_paths:
        logger.info(f"✓ All {len(all_jg_paths)} judge output file(s) already exist – skipping Phase 2 entirely.")
        return None

    logger.info(f"  {len(already_done)}/{len(all_jg_paths)} already judged; "
                f"{len(all_jg_paths) - len(already_done)} to run.")

    already_done_set = set(already_done)

    ej: Optional[EvalJUDGE] = None

    for idx, (key, out_path) in enumerate(gen_results.items(), 1):
        parts          = key.split("|")
        model_name     = parts[0]
        eval_task_name = parts[1]
        icr_bool       = parts[2] == "True"
        task_name_key  = parts[3] if len(parts) >= 4 else None

        config_lookup = f"{eval_task_name}|{task_name_key}"
        judge_cfg = task_configs.get(config_lookup)
        if judge_cfg is None:
            logger.error(f"  No config found for '{config_lookup}' – skipping.")
            continue

        if judge_cfg.output.eval_task != eval_task_name:
            judge_cfg = OmegaConf.create(OmegaConf.to_container(judge_cfg, resolve=True))
            judge_cfg.output.eval_task = eval_task_name

        jg_path = _jg_path_for(eval_task_name, icr_bool, judge_cfg, judge_tag)
        if jg_path in already_done_set:
            logger.info(f"  [{idx}/{len(gen_results)}] ✓ Already judged: {os.path.basename(jg_path)} – skipping.")
            continue

        logger.info(f"\n[{idx}/{len(gen_results)}] Judging: {model_name} / "
                    f"{eval_task_name} / icr={icr_bool} / task={judge_cfg.output.task_name}")
        logger.info(f"  gen_file  : {os.path.basename(out_path)}")

        try:
            # Per-task override → config default → EvalJUDGE hardcoded default
            effective_bs = JUDGE_BATCH_SIZE.get(eval_task_name, judge_batch_size_default)
            if ej is None:
                kwargs = dict(task=eval_task_name, gen_file=out_path, icr=icr_bool,
                              batch_size=effective_bs)
                ej = EvalJUDGE(judge_cfg, hf_model_id=judge_hf_model_id,
                               judge_tag=judge_tag, **kwargs)
                logger.info("✓ EvalJUDGE model loaded")
            else:
                ej.eval_cfg = judge_cfg
                ej.name     = judge_cfg.model.name
                ej.task     = eval_task_name
                ej.gen_file = out_path
                ej.set_icr(icr_bool)
                if ej.batch_size != effective_bs:
                    logger.info(f"  batch_size: {ej.batch_size} → {effective_bs} for {eval_task_name}")
                    ej.batch_size = effective_bs

            logger.info(f"  judge_out : {os.path.basename(ej.jg_file_path)}")
            if os.path.exists(ej.jg_file_path):
                logger.info(f"✓ Already judged (skipping) → {os.path.basename(ej.jg_file_path)}")
            else:
                ej.generate()
                logger.info(f"✓ Judged → {os.path.basename(ej.jg_file_path)}")
        except Exception as e:
            logger.error(f"✗ Judgment failed for {key}: {e}")
            import traceback; traceback.print_exc()
            raise

    logger.info("\n✓ All EvalJUDGE judgments complete")
    return ej  # caller can reuse model for EvalRGQ (rgq_bi)


def _spawn_rgq_only(rgq_plain, pairs, task_configs_plain,
                    judge_hf_model_id, judge_tag, judge_batch_size_default) -> None:
    """Run only the RGQ tasks (no pending judge work) across NVLink GPU pairs."""
    if not (rgq_plain and pairs):
        return
    logger.info(f"Running {len(rgq_plain)} EvalRGQ task(s) across {min(len(rgq_plain), len(pairs))} GPU pair(s).")
    rgq_queue = mp.Queue()
    for rt in rgq_plain:
        rgq_queue.put(('rgq', rt))
    n_procs = min(len(rgq_plain), len(pairs))
    procs = [
        mp.Process(target=_judge_subprocess_worker,
                   args=(rgq_queue, task_configs_plain, pairs[j],
                         judge_hf_model_id, judge_tag, judge_batch_size_default),
                   daemon=False)
        for j in range(n_procs)
    ]
    for p in procs: p.start()
    for p in procs: p.join()


def phase2_judge_parallel(
        gen_results: Dict[str, str],
        task_configs: Dict[str, DictConfig],
        judge_hf_model_id: str,
        judge_tag: Optional[str],
        judge_batch_size_default: int,
        gpu_ids: List[str],
        num_parallel_judges: int,
        rgq_tasks: list,
) -> None:
    """Run N judge subprocesses in parallel, each on a NVLink-preferring GPU pair.

    Both judge and rgq tasks go into a single shared queue so each subprocess
    claims whatever work is available — whichever GPU pair finishes its judge tasks
    first picks up the next rgq task automatically.
    Returns None (model is not accessible in the main process).
    """

    # Serialise rgq tasks for pickling (shared across both branches below)
    rgq_plain = [{'cfg': OmegaConf.to_container(t.cfg, resolve=True),
                  'pretrained_task_name': t.pretrained_task_name}
                 for t in rgq_tasks]
    task_configs_plain = {k: OmegaConf.to_container(v, resolve=True)
                          for k, v in task_configs.items()}
    pairs = _nvlink_pairs(gpu_ids)

    if not gen_results:
        logger.info("No generation results to judge.")
        _spawn_rgq_only(rgq_plain, pairs, task_configs_plain,
                        judge_hf_model_id, judge_tag, judge_batch_size_default)
        return

    # Build list of pending (not already judged) tasks
    pending = []
    for key, out_path in gen_results.items():
        parts          = key.split("|")
        eval_task_name = parts[1]
        icr_bool_str   = parts[2]
        task_name_key  = parts[3] if len(parts) >= 4 else None
        judge_cfg      = task_configs.get(f"{eval_task_name}|{task_name_key}")
        if judge_cfg is None:
            continue
        icr_bool = icr_bool_str == "True"
        jg_path  = _jg_path_for(eval_task_name, icr_bool, judge_cfg, judge_tag)
        if os.path.exists(jg_path):
            logger.info(f"✓ Already judged – {os.path.basename(jg_path)}")
            continue
        pending.append(('judge', key, out_path, eval_task_name, icr_bool_str, task_name_key))

    if not pending:
        logger.info("✓ All judge outputs already exist – skipping parallel Phase 2.")
        _spawn_rgq_only(rgq_plain, pairs, task_configs_plain,
                        judge_hf_model_id, judge_tag, judge_batch_size_default)
        return

    logger.info("=" * 80)
    logger.info(f"PHASE 2: PARALLEL JUDGING  ({num_parallel_judges} judge subprocess(es))")
    logger.info(f"  {len(pending)} judge task(s), {len(rgq_plain)} rgq task(s)")
    logger.info("=" * 80)

    # Detect NVLink-preferring GPU pairs
    if len(pairs) < num_parallel_judges:
        logger.warning(
            f"Only {len(pairs)} GPU pair(s) available; reducing to {len(pairs)} judge(s)."
        )
        num_parallel_judges = max(1, len(pairs))

    if num_parallel_judges == 1:
        logger.info("Falling back to sequential judging (only 1 GPU pair available).")
        judge = phase2_judge_all(gen_results, task_configs,
                                 judge_hf_model_id, judge_tag, judge_batch_size_default)
        phase2b_rgq(rgq_tasks, judge=judge)
        return

    # Single shared queue: judge tasks first, then rgq tasks.
    # Whichever subprocess drains its judge work first picks up rgq tasks.
    task_queue = mp.Queue()
    for task in pending:
        task_queue.put(task)
    for rt in rgq_plain:
        task_queue.put(('rgq', rt))

    procs = []
    for j in range(num_parallel_judges):
        p = mp.Process(
            target=_judge_subprocess_worker,
            args=(task_queue, task_configs_plain, pairs[j],
                  judge_hf_model_id, judge_tag, judge_batch_size_default),
            daemon=False,
        )
        procs.append(p)
        logger.info(f"  Starting judge-{j} on GPUs {pairs[j]}")

    for p in procs:
        p.start()
    for j, p in enumerate(procs):
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Judge subprocess {j} exited with code {p.exitcode}")

    logger.info("\n✓ All parallel EvalJUDGE judgments complete")


# ---------------------------------------------------------------------------
# PHASE 2b – EvalRGQ (rgq_bi) — runs after all generation + after Phase 2
# ---------------------------------------------------------------------------
def phase2b_rgq(rgq_tasks: List[RGQTask],
                judge: Optional[EvalJUDGE] = None):
    """
    Run RGQ (Relative Generation Quality) evaluation sequentially using the SAME
    judge model as Phase 2. EvalRGQ uses a different prompt but the identical
    model weights — no reload needed.

    rgq_bi depends on repet files being present (generated in Phase 1).
    judge – the EvalJUDGE instance returned by phase2_judge_all; its model and
            tokenizer are passed into EvalRGQ to avoid loading the judge model again.
    """
    if not rgq_tasks:
        return

    logger.info("=" * 80)
    logger.info(f"PHASE 2b: RGQ EVALUATION  ({len(rgq_tasks)} task(s))")
    if judge is not None:
        logger.info("  Reusing EvalJUDGE model weights for EvalRGQ (no second load)")
    logger.info("=" * 80)

    # Extract model/tokenizer from EvalJUDGE if available.
    # If judge is None (no forget/retain tasks were run), we load the model on the first
    # rgq task and then reuse it for all subsequent ones — loading it once per pipeline run.
    shared_model     = judge.model     if judge is not None else None
    shared_tokenizer = judge.tokenizer if judge is not None else None

    for task in rgq_tasks:
        logger.info(f"\nRGQ: {task.task_name}  (pretrained ref: {task.pretrained_task_name})")
        try:
            ew = EvalRGQ(task.cfg, model=shared_model, tokenizer=shared_tokenizer,
                         pretrained_task_name=task.pretrained_task_name)
            # After the first task the model is loaded — reuse it for all remaining tasks
            if shared_model is None and ew.model is not None:
                shared_model     = ew.model
                shared_tokenizer = ew.tokenizer
                logger.info("  EvalRGQ model loaded — will be reused for remaining rgq tasks")
            ew.rgq_bi()
            logger.info(f"✓ RGQ complete: {task.task_name}")
        except AssertionError as e:
            msg = f"⚠ RGQ skipped for {task.task_name}: {e}"
            logger.warning(msg)
            print(msg, flush=True)   # also to stdout so it shows in .out log
        except Exception as e:
            msg = f"✗ RGQ failed for {task.task_name}: {e}"
            logger.error(msg)
            print(msg, flush=True)   # also to stdout
            import traceback; traceback.print_exc()
            # Do NOT re-raise: an rgq failure for one model should not abort the entire pipeline


# ---------------------------------------------------------------------------
# PHASE 3b – combined adversarial worst-case (requires both forget_rephrasings
#             and forget_adversarial judge files to be complete)
# ---------------------------------------------------------------------------
def phase3b_combined_adversarial(task_configs: Dict[str, DictConfig],
                                 base_cfg: DictConfig,
                                 judge_tag: Optional[str] = None):
    """For each model/topic with a forget_adversarial judge file, compute the combined
    worst-case with forget_rephrasings if those judge files exist on disk (regardless of
    whether forget_rephrasings was part of the current run)."""
    from evals.worst_eval import evaluate_combined_adversarial

    adv_configs = {cfg.output.task_name: cfg
                   for key, cfg in task_configs.items()
                   if key.startswith('forget_adversarial|')}

    for task_name, adv_cfg in adv_configs.items():
        ev_adv = Evaluator(adv_cfg, icr=False, task='forget_adversarial', judge_tag=judge_tag)
        jg_adv = ev_adv.jg_file_path

        if not os.path.exists(jg_adv):
            logger.warning(f'⚠ Combined adversarial skipped for {task_name}: adversarial judge file missing')
            continue

        # Derive rephrasings paths by filename substitution — no extra HF download needed
        jg_reph_noicr = jg_adv.replace('_forget_adversarial_icr_False', '_forget_rephrasings_icr_False')
        jg_reph_icr   = jg_adv.replace('_forget_adversarial_icr_False', '_forget_rephrasings_icr_True')

        if not (os.path.exists(jg_reph_noicr) and os.path.exists(jg_reph_icr)):
            logger.warning(f'⚠ Combined adversarial skipped for {task_name}: forget_rephrasings judge files not found')
            logger.warning(f'   expected noicr: {jg_reph_noicr}')
            logger.warning(f'   expected icr:   {jg_reph_icr}')
            continue

        subpath  = getattr(adv_cfg.output, 'subpath',  '') or ''
        exp_name = getattr(adv_cfg.output, 'exp_name', '') or ''
        run_name = exp_name or 'pretrained'
        if subpath and exp_name:
            dir_key = os.path.join(subpath, exp_name)
        elif subpath:
            dir_key = subpath
        else:
            dir_key = task_name
        out_dir = os.path.join(base_cfg.output.worstdir, dir_key, 'forget_adversarial_combined')

        try:
            out_path = evaluate_combined_adversarial(
                jg_reph_noicr, jg_reph_icr, jg_adv,
                name=adv_cfg.model.name, run_name=run_name, out_dir=out_dir,
            )
            logger.info(f'✓ Combined adversarial worst-case: {out_path}')
        except Exception as e:
            logger.warning(f'⚠ Combined adversarial failed for {task_name}: {e}')


# ---------------------------------------------------------------------------
# PHASE 3 – final evaluation metrics
# ---------------------------------------------------------------------------
def phase3_evaluate_all(task_configs: Dict[str, DictConfig],
                        base_cfg: DictConfig,
                        judge_tag: Optional[str] = None):
    """
    Compute WorstEval / AvgEval metrics for every (eval_task, task_name) combination.
    Results are saved under  <worstdir>/<task_name>/<eval_task>/
    judge_tag must match the tag used in Phase 2 so Evaluator finds the correct JG_EVAL files.
    """
    logger.info("=" * 80)
    logger.info("PHASE 3: FINAL EVALUATION METRICS")
    logger.info("=" * 80)

    for config_key, task_cfg in task_configs.items():
        eval_task_name = task_cfg.output.eval_task
        task_name      = task_cfg.output.task_name
        subpath        = getattr(task_cfg.output, 'subpath',  '') or ''
        exp_name       = getattr(task_cfg.output, 'exp_name', '') or ''
        run_name       = exp_name or 'pretrained'

        if subpath:
            dir_key      = os.path.join(subpath, exp_name) if exp_name else subpath
            task_out_dir = os.path.join(base_cfg.output.worstdir, dir_key, eval_task_name)
        else:
            task_out_dir = os.path.join(base_cfg.output.worstdir, task_name, eval_task_name)

        logger.info(f"\nFinal evaluation: {eval_task_name} / {task_name}  →  {task_out_dir}")

        if eval_task_name in FORGET_TASKS:
            icr_variants = icr_variants_for(eval_task_name)
            evaluators   = {icr: Evaluator(task_cfg, icr=icr, task=eval_task_name,
                                           judge_tag=judge_tag)
                            for icr in icr_variants}

            if all(ev.everything_evaluated() for ev in evaluators.values()):
                jg_noicr = evaluators[False].jg_file_path
                jg_icr   = evaluators[False].jg_file_path if eval_task_name in FORGET_NO_ICR \
                           else evaluators[True].jg_file_path

                we = WorstEval(task_cfg.model.name, run_name, task_out_dir,
                               [jg_noicr, jg_icr], task=eval_task_name)
                we.evaluate()
                logger.info(f"✓ WorstEval complete")

                # After adversarial eval: show combined (adv+rephrasings) if it exists
                if eval_task_name == 'forget_adversarial':
                    model_short = task_cfg.model.name.split('/')[-1]
                    if subpath:
                        comb_dir = os.path.join(base_cfg.output.worstdir, dir_key,
                                                'forget_adversarial_combined')
                    else:
                        comb_dir = os.path.join(base_cfg.output.worstdir, task_name,
                                                'forget_adversarial_combined')
                    comb_path = os.path.join(
                        comb_dir,
                        f'worst_case_eval_{model_short}_{run_name}_forget_adversarial_combined.jsonl')
                    if os.path.exists(comb_path):
                        with open(comb_path, encoding='utf-8') as _cf:
                            _comb = json.load(_cf)
                        _j = _comb.get('J_W_Total')
                        print(f"\n--- Combined (forget_adversarial + forget_rephrasings) [cached] ---")
                        print(f"  J_W_Total: {_j:.2%}" if _j is not None else "  J_W_Total: N/A")
                    else:
                        print(f"\n--- Combined (forget_adversarial + forget_rephrasings): not yet computed ---")
                        print(f"    Evaluate forget_rephrasings first, then re-run.")
            else:
                missing = [ev.jg_file_path for ev in evaluators.values()
                           if not os.path.exists(ev.jg_file_path)]
                logger.warning(f"⚠ Not all judge files present for {config_key} – skipping.")
                for p in missing:
                    logger.warning(f"   missing: {p}")

        elif eval_task_name in RETAIN_TASKS:
            ev = Evaluator(task_cfg, icr=False, task=eval_task_name, judge_tag=judge_tag)
            if ev.everything_evaluated():
                ae = AvgEval(task_cfg.model.name, run_name, task_out_dir,
                             [ev.jg_file_path], task=eval_task_name)
                ae.evaluate()
                logger.info(f"✓ AvgEval complete")
            else:
                logger.warning(f"⚠ Judge file missing for {config_key} – skipping.")
                logger.warning(f"   missing: {ev.jg_file_path}")

        elif eval_task_name in GIBBERISH_OVERLAY_TASKS:
            ev = Evaluator(task_cfg, icr=False, task=eval_task_name, judge_tag=judge_tag)
            if ev.everything_evaluated():
                ae = AvgEval(task_cfg.model.name, run_name, task_out_dir,
                             [ev.jg_file_path], task=eval_task_name)
                ae.evaluate()
                logger.info(f"✓ AvgEval complete (gibberish overlay: {eval_task_name})")
            else:
                logger.warning(f"⚠ Judge file missing for gibberish overlay {config_key} – skipping.")
                logger.warning(f"   missing: {ev.jg_file_path}")

        # mmlu / repet / rgq_bi have their own metrics inside eval_mmlu / eval_repet / EvalRGQ
        # nothing extra needed here


# ---------------------------------------------------------------------------
# Task-spec parser (used by multi-task mode)
# ---------------------------------------------------------------------------
def parse_task_spec(spec: str, base_cfg: DictConfig
                    ) -> Optional[Tuple[str, str, str, str, str, str, int, str]]:
    """
    Parse a colon-separated task spec string.
    Format: model_key:model_path:eval_task:task_name:split:dataset_arg:max_tokens[:hf_model_name]
    Returns (model_key, model_path, eval_task, task_name, split, dataset_name, max_tokens, hf_model_name)
    or None if malformed.
    hf_model_name defaults to model_path if not provided.
    """
    # Split from the RIGHT: in the 7-field form a model_path containing colons never shifts the
    # trailing fixed fields (the last 5 fields are colon-free); in the 6-field form the path must
    # be colon-free.
    right = spec.rsplit(':', 6)
    if len(right) < 6:
        logger.warning(f"Skipping malformed task spec (need ≥6 fields): {spec!r}")
        return None

    if len(right) == 7:
        # 7-field form: model_key:model_path:eval_task:task_name:split:dataset_arg:max_tokens:hf_name
        # rsplit(':',6) gives 7 parts: [prefix, eval_task, task_name, split, dataset_arg, max_tokens, hf_name]
        prefix, eval_task, task_name, split, dataset_arg, max_tokens_str, hf_model_name = right
    else:
        # 6-field form: no hf_name field
        prefix, eval_task, task_name, split, dataset_arg, max_tokens_str = right
        hf_model_name = None

    key_path = prefix.split(':', 1)
    if len(key_path) < 2:
        logger.warning(f"Skipping malformed task spec (no model_key:model_path): {spec!r}")
        return None
    model_key, model_path = key_path[0], key_path[1]
    if not hf_model_name:
        hf_model_name = model_path  # fallback: use path as name (pretrained HF models)
    dataset_name = dataset_arg.replace("dataset.name=", "") if dataset_arg else ""
    try:
        max_tokens = int(max_tokens_str)
    except ValueError:
        logger.warning(f"Skipping malformed task spec (max_tokens not int: {max_tokens_str!r}): {spec!r}")
        return None
    return model_key, model_path, eval_task, task_name, split, dataset_name, max_tokens, hf_model_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    base_cfg = OmegaConf.load("configs/eval.yaml")
    cli_cfg  = OmegaConf.from_dotlist(sys.argv[1:])
    cfg      = OmegaConf.merge(base_cfg, cli_cfg)

    print("\n" + "=" * 80)
    print("FULL EVALUATION PIPELINE: ALL GENERATION → ALL JUDGING → METRICS")
    print("=" * 80)
    logger.info(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")

    multi_task_mode = cfg.get('multi_task_mode', False)

    # ---------------------------------------------------------------
    # Judge configuration  (read once; used in both branches below)
    # ---------------------------------------------------------------
    judge_cfg_section  = cfg.get('judge', {}) or {}
    judge_hf_model     = judge_cfg_section.get('hf_model_id', 'Qwen/Qwen3.5-35B-A3B')
    judge_tag          = judge_cfg_section.get('judge_tag', None) or None   # "" → None
    judge_batch_size   = int(judge_cfg_section.get('batch_size', 16))
    gpus_per_judge     = int(judge_cfg_section.get('gpus_per_judge', 6))
    n_parallel_cfg     = int(judge_cfg_section.get('num_parallel_judges', 1))

    gpu_ids = _visible_gpu_ids()
    if n_parallel_cfg == -1:
        n_parallel = max(1, len(gpu_ids) // gpus_per_judge)
    else:
        n_parallel = n_parallel_cfg

    parallel_mode = n_parallel > 1

    logger.info(f"Judge config: model={judge_hf_model}  tag={judge_tag!r}  "
                f"batch={judge_batch_size}  gpus_per_judge={gpus_per_judge}  "
                f"parallel={n_parallel}  gpu_ids={gpu_ids}")

    # ---------------------------------------------------------------
    # MULTI-TASK MODE  (used by suite_evaluation_optimized.sh)
    # ---------------------------------------------------------------
    if multi_task_mode and 'multi_task_specs' in cfg:
        logger.info("Multi-task mode: aggregating all model×task combinations")

        task_specs = cfg.multi_task_specs
        if isinstance(task_specs, str):
            task_specs = [task_specs]

        all_gen_tasks:     List[GenerationTask]         = []
        all_utility_tasks: List[UtilityTask]            = []
        all_rgq_tasks:     List[RGQTask]                = []
        all_overlay_tasks: List[Tuple[str, DictConfig]] = []   # (overlay_eval_task, task_cfg)
        task_configs: Dict[str, DictConfig]             = {}   # "eval_task|task_name" -> cfg

        # Track pretrained repet task_name per HF model name (one per model family).
        # e.g. "meta-llama/Llama-3.2-3B-Instruct" -> "pretrained"
        #      "Qwen/Qwen3.5-9B"                   -> "qwen_pretrained"
        pretrained_task_name: Optional[str] = None
        pretrained_task_name_by_model: Dict[str, str] = {}

        for spec in task_specs:
            parsed = parse_task_spec(spec, cfg)
            if parsed is None:
                continue
            model_key, model_path, eval_task, task_name, split, dataset_name, max_tokens, hf_model_name = parsed

            task_cfg = make_task_cfg(cfg, model_path, task_name, eval_task,
                                     split, dataset_name, max_tokens,
                                     model_name=hf_model_name)
            logger.info(f"  Spec: {eval_task}|{task_name}  model={model_path}  hf={hf_model_name}")

            if eval_task in FORGET_TASKS or eval_task in RETAIN_TASKS:
                config_key = f"{eval_task}|{task_name}"
                task_configs[config_key] = task_cfg
                for icr in icr_variants_for(eval_task):
                    all_gen_tasks.append(GenerationTask(
                        model_name=hf_model_name,
                        model_path=model_path,
                        eval_task=eval_task,
                        icr=icr,
                        cfg=task_cfg,
                    ))

            elif eval_task in GIBBERISH_OVERLAY_TASKS:
                # No generation — reuses the base task's gen file in Phase 1 injection step.
                config_key = f"{eval_task}|{task_name}"
                task_configs[config_key] = task_cfg
                all_overlay_tasks.append((eval_task, task_cfg))
                logger.info(f"  ↳ gibberish overlay: will reuse '{GIBBERISH_OVERLAY_TASKS[eval_task]}' gen file")

            elif is_utility_task(eval_task):
                all_utility_tasks.append(UtilityTask(
                    model_name=hf_model_name,
                    model_path=model_path,
                    eval_task=eval_task,
                    task_name=task_name,
                    cfg=task_cfg,
                ))
                # Track which task_name is the pretrained baseline for each model family.
                # A spec is considered "pretrained" when its model_path == hf_model_name
                # (i.e. it points directly to the HF hub, not a local save).
                if model_path == hf_model_name and eval_task == 'repet':
                    pretrained_task_name = task_name
                    pretrained_task_name_by_model[hf_model_name] = task_name

            elif eval_task in RGQ_TASKS:
                # Look up the pretrained baseline for this specific model family first,
                # then fall back to a model-family-specific task name convention.
                def _default_pretrained_task(name: str) -> str:
                    n = name.lower()
                    if 'llama' in n:   return 'llama_pretrained'
                    if 'mistral' in n: return 'mistral_pretrained'
                    if 'qwen' in n:    return 'qwen_pretrained'
                    if 'phi' in n:     return 'phi_pretrained'
                    return 'pretrained'
                ref_task = pretrained_task_name_by_model.get(
                    hf_model_name, pretrained_task_name or _default_pretrained_task(hf_model_name)
                )
                all_rgq_tasks.append(RGQTask(
                    model_name=hf_model_name,
                    model_path=model_path,
                    task_name=task_name,
                    pretrained_task_name=ref_task,
                    cfg=task_cfg,
                ))

        # ---- PHASE 1: parallel generation (Evaluator + utility) ----
        gen_results, _utility_results = phase1_generate_all(
            all_gen_tasks, all_utility_tasks
        )

        # ---- Inject gibberish overlay tasks (reuse base-task gen files) ----
        if all_overlay_tasks:
            _inject_overlay_gen_results(all_overlay_tasks, gen_results)

        # ---- PHASE 2 (+2b): EvalJUDGE + optional EvalRGQ ----
        if parallel_mode:
            # Parallel judges: EvalRGQ handled inside subprocess 0
            if gen_results or all_rgq_tasks:
                phase2_judge_parallel(
                    gen_results, task_configs,
                    judge_hf_model, judge_tag, judge_batch_size,
                    gpu_ids, n_parallel, all_rgq_tasks,
                )
        else:
            judge = None
            if gen_results:
                judge = phase2_judge_all(
                    gen_results, task_configs,
                    judge_hf_model_id=judge_hf_model,
                    judge_tag=judge_tag,
                    judge_batch_size_default=judge_batch_size,
                )
            if all_rgq_tasks:
                phase2b_rgq(all_rgq_tasks, judge=judge)

        # ---- PHASE 3: metrics ----
        phase3_evaluate_all(task_configs, cfg, judge_tag=judge_tag)
        phase3b_combined_adversarial(task_configs, cfg, judge_tag=judge_tag)

    # ---------------------------------------------------------------
    # SINGLE-TASK MODE  (direct python call / suite_evaluation.sh)
    # ---------------------------------------------------------------
    else:
        eval_task_raw = cfg.output.eval_task
        eval_tasks    = [t.strip() for t in eval_task_raw.split(',')]
        logger.info(f"Single-task mode: {eval_tasks}")

        all_gen_tasks:     List[GenerationTask]         = []
        all_utility_tasks: List[UtilityTask]            = []
        all_rgq_tasks:     List[RGQTask]                = []
        all_overlay_tasks: List[Tuple[str, DictConfig]] = []
        task_configs:      Dict[str, DictConfig]        = {}

        for et in eval_tasks:
            if is_utility_task(et):
                all_utility_tasks.append(UtilityTask(
                    model_name=cfg.model.name,
                    model_path=cfg.model.path,
                    eval_task=et,
                    task_name=cfg.output.task_name,
                    cfg=cfg,
                ))
            elif et in RGQ_TASKS:
                all_rgq_tasks.append(RGQTask(
                    model_name=cfg.model.name,
                    model_path=cfg.model.path,
                    task_name=cfg.output.task_name,
                    pretrained_task_name='pretrained',
                    cfg=cfg,
                ))
            elif et in FORGET_TASKS or et in RETAIN_TASKS:
                task_cfg = make_task_cfg(
                    cfg, cfg.model.path, cfg.output.task_name, et,
                    cfg.dataset.split, cfg.dataset.name, cfg.generation.max_new_tokens,
                    model_name=cfg.model.name,
                )
                config_key = f"{et}|{cfg.output.task_name}"
                task_configs[config_key] = task_cfg
                for icr in icr_variants_for(et):
                    all_gen_tasks.append(GenerationTask(
                        model_name=cfg.model.name,
                        model_path=cfg.model.path,
                        eval_task=et,
                        icr=icr,
                        cfg=task_cfg,
                    ))
            elif et in GIBBERISH_OVERLAY_TASKS:
                task_cfg = make_task_cfg(
                    cfg, cfg.model.path, cfg.output.task_name, et,
                    cfg.dataset.split, cfg.dataset.name, cfg.generation.max_new_tokens,
                    model_name=cfg.model.name,
                )
                config_key = f"{et}|{cfg.output.task_name}"
                task_configs[config_key] = task_cfg
                all_overlay_tasks.append((et, task_cfg))
            else:
                raise NotImplementedError(f"Unknown eval task: '{et}'")

        gen_results, _utility_results = phase1_generate_all(
            all_gen_tasks, all_utility_tasks
        )

        # ---- Inject gibberish overlay tasks (reuse base-task gen files) ----
        if all_overlay_tasks:
            _inject_overlay_gen_results(all_overlay_tasks, gen_results)
        if parallel_mode:
            if gen_results or all_rgq_tasks:
                phase2_judge_parallel(
                    gen_results, task_configs,
                    judge_hf_model, judge_tag, judge_batch_size,
                    gpu_ids, n_parallel, all_rgq_tasks,
                )
        else:
            judge = None
            if gen_results:
                judge = phase2_judge_all(
                    gen_results, task_configs,
                    judge_hf_model_id=judge_hf_model,
                    judge_tag=judge_tag,
                    judge_batch_size_default=judge_batch_size,
                )
            if all_rgq_tasks:
                phase2b_rgq(all_rgq_tasks, judge=judge)
        phase3_evaluate_all(task_configs, cfg, judge_tag=judge_tag)
        phase3b_combined_adversarial(task_configs, cfg, judge_tag=judge_tag)

    logger.info(f"\nPipeline complete: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logging.shutdown()


if __name__ == "__main__":
    main()
