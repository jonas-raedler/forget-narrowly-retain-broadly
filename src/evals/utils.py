import json
import random
import re

import torch
import numpy as np
import logging
import time
import os
from datasets import load_dataset
from omegaconf import DictConfig

# Qwen3+ tends to continue generating after the answer (prompt echoes, self-corrections, notes).
# Cut at the earliest occurrence of any of these sequences.
# To remove: delete this constant + Evaluator._qwen_trim + the 2-line guard in _run_generate.
_QWEN_STOP_SEQS = ("\nuser\n", "\nassistant\n", "\n\n", "\n", "   ")

def setup_logger(log_file_name: str = "application.log", log_dir: str = "./",
                 console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Sets up a logger that prints to the console and saves to a file.
    Safe to call multiple times: the file handler is always replaced so that
    each task writes to its own log file without duplication.
    """
    logger = logging.getLogger("SUITEEVAL")
    logger.setLevel(logging.DEBUG)
    # Prevent messages from also reaching the root logger (avoids duplicate console lines)
    logger.propagate = False

    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_file_name)

    # Remove any existing FileHandlers (they point to the previous task's log file)
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            h.close()
            logger.removeHandler(h)

    # Add a console handler only once (it is shared across all tasks)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(console_handler)

    # Always add a fresh file handler for this task's log file
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)

    return logger


def load_eval_dataset(name: str, split: str, filter_topic: str = ""):
    """Load a dataset from HuggingFace or a local JSON file.

    For local .json paths (e.g. ./assets/refusals_eval.json), the HuggingFace
    datasets library requires the 'json' format key + data_files dict.
    For HF hub names, the standard load_dataset call is used.

    If `filter_topic` is set and the loaded dataset has a `topic` column,
    filter to rows where topic == filter_topic. This is how the unified
    apeleg/SUITE dataset is sliced down to one topic at eval time. If the
    dataset has no `topic` column (per-topic repos, RWKU, local JSONs)
    the filter is a no-op so callers can pass it unconditionally.
    """
    if name.endswith('.json') and os.path.exists(name):
        ds = load_dataset('json', data_files={split: name}, split=split)
    else:
        ds = load_dataset(name, split=split)
    if filter_topic and "topic" in ds.column_names:
        ds = ds.filter(lambda x: x["topic"] == filter_topic)
    return ds


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Evaluator:
    """Handles the evaluation process for a language model across forget/retain tasks.

    This class manages loading evaluation data, constructing prompts (with or without
    in-context examples), running model inference, logging results, and managing
    output directories for evaluation outcomes. It supports evaluation for "forget"
    and other tasks, and can incorporate in-context retention (ICR) data.

    Attributes:
        name (str): The name of the model being evaluated.
        icr_data (bool): Flag indicating whether in-context retention (ICR) data is used.
        eval_cfg (DictConfig): Configuration object containing evaluation settings
            (e.g., model name, output paths, generation parameters).
        task (str): The specific evaluation task (e.g., 'forget', 'retain').
        icr_dataset (Dataset): Forget set containing ICR examples, loaded on init.
        logs (list): A list to store dictionaries of generated responses for logging.
        out_path (str): Full path to the main output JSONL file for generations.
        jg_file_path (str): Full path to the JudgeEvals evaluation JSONL file.
        log_path (str): Full path to the evaluation log file.
        logger (Logger): Logger instance for logging evaluation progress and information.
    """
    def __init__(self, eval_cfg:DictConfig, icr:bool, task:str, judge_tag:str=None):

        self.name = eval_cfg.model.name
        self.icr_data = icr
        self.eval_cfg = eval_cfg
        self.task = task
        self.judge_tag = judge_tag or None  # normalize "" to None
        self.icr_dataset = self.load_icr() if self.icr_data else None
        self.logs = []
        self.set_out_dirs()
        self.set_template()
        self.logger = setup_logger(self.log_path, self.log_dir)
        self.logger.info(f"Output filepath {self.log_path}")
        self.logger.info(f"Evaluating {self.task} with ICR? {self.icr_data}")

        seed_everything()

    def set_template(self):
        # The "refusal" template applies only when a custom --task named "refusal" is passed
        if self.task == "refusal":
            self.TEMP_P = f"You are a helpful assistant."
        else:
            self.TEMP_P = f"You are a good model. I want you to answer this question with a short response. \
                The output should just be the answer without any preceding/succeeding text. Respond with a short, relevant answer only. \
                A few examples: \
                1. Question: Where did Olympics 2012 happen? Answer: London \
                2. Question: What is the capital city of Australia? Answer: The capital is Canberra \
                3. Question: Which year did World-war 2 end? Answer: 1945 \
                Now it's your turn.\\ "


    # Maps topic name → (hf_repo, split) for ICR retain_train examples.
    # Topics covered by the unified apeleg/SUITE repo use it + are filtered by
    # the topic column inside load_icr. Topics with non-standard splits keep a
    # bespoke entry.
    _ICR_DATASET_BY_TOPIC = {
        "challenger_disaster":            ("apeleg/SUITE", "retain_train"),
        "salem_witch_trials":             ("apeleg/SUITE", "retain_train"),
        "steve_jobs_medical":             ("apeleg/SUITE", "retain_train"),
        "britney_spears_conservatorship": ("apeleg/SUITE", "retain_train"),
        "challenger_baseline":            ("apeleg/LKF-baseline-challenger-train", "retain_train_semantic"),
    }

    def load_icr(self):
        """Loads the in-context retention (ICR) dataset.

        The dataset is selected based on output.topic so that the ICR examples
        come from the same topic the model was trained on.  Falls back to the
        Challenger Disaster dataset for unknown topics.

        Returns:
            Dataset: The loaded Hugging Face Dataset for ICR.
        """
        topic = getattr(self.eval_cfg.output, "topic", "") or ""
        if topic not in self._ICR_DATASET_BY_TOPIC:
            known = list(self._ICR_DATASET_BY_TOPIC.keys())
            raise ValueError(
                f"No ICR dataset configured for topic '{topic}'. "
                f"Known topics: {known}. "
                f"Add an entry to Evaluator._ICR_DATASET_BY_TOPIC in src/evals/utils.py."
            )
        hf_name, split = self._ICR_DATASET_BY_TOPIC[topic]
        ds = load_dataset(hf_name, split=split)
        # Filter unified repo down to the requested topic. No-op on
        # per-topic repos (no `topic` column).
        if "topic" in ds.column_names:
            ds = ds.filter(lambda x: x["topic"] == topic)
        return ds

    def get_ret_icr(self, num_ic_examples:int=3) -> list[list[str, str]]:
        """Retrieves a specified number of in-context retention examples.

        Randomly samples `num_ic_examples` examples from the loaded `icr_dataset` and
        formats them as a list of `[question, answer]` pairs.

        Args:
            num_ic_examples (int, optional): The number of in-context examples to retrieve. Defaults to 3.

        Returns:
            list[list[str, str]]: A list of lists, where each inner list contains
                                   a question and its corresponding answer from the ICR dataset.
        """
        sampled_dataset = self.icr_dataset.shuffle().select(range(num_ic_examples))
        icr_examples = [[queries['question'], queries['answer']] for queries in sampled_dataset]
        
        return icr_examples

    def get_template(self, ex:str) -> str:
        """Constructs a prompt template for the LLM.

        The template includes a base prefix (`TEMP_P`). If `self.icr_data` is True,
        it prepends a set of in-context examples retrieved by `get_ret_icr`.

        Args:
            ex (str): The specific question or example to be included in the prompt.

        Returns:
            str: The fully constructed prompt string ready for model inference.
        """
        if not self.icr_data:
            return f"{self.TEMP_P}QUESTION:{ex}, \n ANSWER:"

        else:
            num_ic_examples=3
            in_context_examples = self.get_ret_icr(num_ic_examples)
            ic_string_parts = [
                f"{idx + 4} Question: {q} Answer: {a}"
                for idx, (q, a) in enumerate(in_context_examples)
            ]
            ic_examples_str = "\n".join(ic_string_parts)

            return f"{self.TEMP_P}{ic_examples_str}\n\nNow it's your turn." +  f"QUESTION:{ex}, ANSWER: "
            

    def set_out_dirs(self, prefix:str='EVAL_'):
        """Sets up output directories and file paths for evaluation results and logs.

        Files are grouped under a hierarchical subdirectory when topic is set:
            evalOutputs/{topic}/{model}/{method}[/relearn]/{exp}/EVAL_{model}_{task}_icr_{bool}.jsonl
            evalJudge/{topic}/{model}/{method}[/relearn]/{exp}/JG_EVAL_{model}_{task}_icr_{bool}.jsonl
        Falls back to flat {task_name}/ layout when topic is not configured.
        """
        task_name   = self.eval_cfg.output.task_name
        model_short = self.name.split("/")[-1]
        subpath     = getattr(self.eval_cfg.output, 'subpath',  '') or ''
        exp_name    = getattr(self.eval_cfg.output, 'exp_name', '') or ''

        if subpath:
            dir_key = os.path.join(subpath, exp_name) if exp_name else subpath
        else:
            dir_key = task_name

        out_subdir = os.path.join(self.eval_cfg.output.dir,     dir_key)
        jg_subdir  = os.path.join(self.eval_cfg.output.evaldir, dir_key)
        os.makedirs(out_subdir, exist_ok=True)
        os.makedirs(jg_subdir,  exist_ok=True)

        suffix      = f"{model_short}_{self.eval_cfg.output.eval_task}_icr_{self.icr_data}.jsonl"
        filename    = prefix     + suffix
        jg_prefix   = f"JG_EVAL_{self.judge_tag}_" if self.judge_tag else "JG_EVAL_"
        jg_filename = jg_prefix + suffix

        log_subdir = os.path.join(self.eval_cfg.output.logdir, dir_key)
        os.makedirs(log_subdir, exist_ok=True)

        self.out_path     = os.path.join(out_subdir, filename)
        self.jg_file_path = os.path.join(jg_subdir,  jg_filename)
        self.log_path     = prefix + f"{model_short}_{self.eval_cfg.output.eval_task}_icr_{self.icr_data}.log"
        self.log_dir      = log_subdir

    def everything_evaluated(self) -> bool:
        """Checks if all necessary evaluation results already exist on disk.

        For forget tasks that require both ICR variants, checks for both the
        current `jg_file_path` and its counterpart with `_icr_False`.
        For other tasks, it only checks the current `jg_file_path`.
        """
        primary_file_exists = os.path.exists(self.jg_file_path)

        # All forget tasks that need both ICR=True and ICR=False judge files
        forget_tasks_needing_icr = ('forget_rephrasings', 'forget_train_rephrasing')
        if self.task in forget_tasks_needing_icr:
            other_file_path = self.jg_file_path.replace("icr_True", "icr_False")
            other_file_exists = os.path.exists(other_file_path)
            return primary_file_exists and other_file_exists
        else:
            # Tasks like forget_adversarial: only icr=False needed
            return primary_file_exists



    def load_logs_from_file(self) -> tuple[bool, bool]:
        """Returns the cache of existing results"""
        gen_exists = os.path.exists(self.out_path)
        jg_eval_exists = os.path.exists(self.jg_file_path)
        
        if gen_exists:
            self.logger.info(f"Existing evaluations are at {self.log_path}")
            if jg_eval_exists:
                self.logger.info(f"JudgeEvals also exist")
            return (gen_exists, jg_eval_exists)
        else:
            return (gen_exists, jg_eval_exists)


    def save_logs(self):
        """Save logs sorted by label so all variants of the same question
        (direct / indirect / reverse) appear together for easy visual inspection."""
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

        logs = self.logs

        # Separate metadata header if present
        if logs and isinstance(logs[0], dict) and logs[0].get("__metadata__"):
            header, entries = [logs[0]], logs[1:]
        else:
            header, entries = [], list(logs)

        def _label_sort_key(entry):
            raw = entry.get("label", "")
            return raw.split("@")[0] if "@" in raw else raw

        sorted_entries = sorted(entries, key=_label_sort_key)

        tmp_path = self.out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(header + sorted_entries, f, indent=4, ensure_ascii=False)
        os.replace(tmp_path, self.out_path)


    @staticmethod
    def _qwen_trim(text: str) -> str:
        """Cut Qwen verbosity: truncate at the earliest stop sequence, then at sentence boundary."""
        positions = [text.find(s) for s in _QWEN_STOP_SEQS if text.find(s) != -1]
        cut = min(positions, default=len(text))
        text = text[:cut].strip()
        # Cut at the first sentence boundary (". Capital") to remove multi-sentence rambling.
        # Skips abbreviations like "Jr.", "Dr.", "No." since those aren't followed by a capital+space.
        m = re.search(r'\.\s+(?=[A-Z])', text)
        if m:
            text = text[:m.start() + 1].strip()  # keep the period
        # Strip role tokens (user/assistant) that bled in without a preceding newline.
        text = re.sub(r'[^\w\s]?\s*(user|assistant)\s*$', '', text, flags=re.IGNORECASE).strip()
        return text

    @staticmethod
    def _extract_answers(decoded: list) -> list:
        """Extract clean answer strings from a list of decoded model outputs."""
        answer_marker   = "ANSWER:"
        question_pattern = r"(?:QUESTION:|[ \t]*\w+\.\s*Question)"
        results = []
        for x in decoded:
            after = x.split(answer_marker, 1)[1]
            after = re.sub(r'^\s*assistant\s*\n\s*', '', after, flags=re.IGNORECASE)
            answer = re.split(question_pattern, after, maxsplit=1,
                              flags=re.MULTILINE | re.IGNORECASE)[0]
            results.append(answer.strip())
        return results

    def _run_generate(self, prompts: list, model, tokenizer) -> list:
        """Tokenize prompts, run model.generate, return extracted answer strings."""
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_device = model.get_input_embeddings().weight.device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        # Some models (e.g. Llama-3.1) use <|eot_id|> as the chat turn terminator
        # while tokenizer.eos_token_id points to a different token (<|end_of_text|>).
        # Without including <|eot_id|> in the stop list, generation never ends after a
        # chat response and the model falls into repetition loops.
        _eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        _unk_id = tokenizer.unk_token_id
        eos_ids = [tokenizer.eos_token_id]
        if _eot_id not in (_unk_id, tokenizer.eos_token_id, 0, None):
            eos_ids.append(_eot_id)
        # Suppress the thinking-mode token for models that support it (Qwen3+ uses
        # "<think>"). bad_words_ids prevents the token from ever being emitted. No-op
        # for Llama/Ministral — convert_tokens_to_ids returns unk for them.
        _think_ids = {tokenizer.convert_tokens_to_ids(t) for t in ("<think>",)}
        _think_ids -= {_unk_id, 0, None}
        _bad_words = [[tid] for tid in _think_ids] if _think_ids else None
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.eval_cfg.generation.max_new_tokens,
                do_sample=self.eval_cfg.generation.do_sample,
                top_p=0,
                temperature=self.eval_cfg.generation.temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids,
                bad_words_ids=_bad_words,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        extracted = self._extract_answers(decoded)
        if "Qwen" in self.name:
            extracted = [self._qwen_trim(a) for a in extracted]
        return extracted

    def _build_result(self, batch: dict, extracted: list, question: str,
                      gt_answer: str, label, raw_questions: dict) -> dict:
        """Build the per-row result dict from extracted answers."""
        result_dict = {f'ans_{qid}': resp
                       for qid, resp in zip(batch.keys(), extracted)
                       if str(batch.get(qid, "")).strip() != ""}
        if raw_questions:
            for k, v in raw_questions.items():
                if v is not None and str(v).strip():
                    result_dict[k] = v
        result_dict["question"]  = question
        result_dict["gt_answer"] = gt_answer
        if label is not None:
            result_dict["label"] = label
        return result_dict

    def evaluate(self, model, batch: dict, tokenizer, question: str,
                 gt_answer: str, label=None, raw_questions: dict = None):
        """Generate and log responses for one dataset row (all its prompt variants)."""
        extracted = self._run_generate(list(batch.values()), model, tokenizer)
        self.logs.append(self._build_result(batch, extracted, question, gt_answer,
                                            label, raw_questions))

    def evaluate_rows(self, model, tokenizer,
                      rows: list):
        """Generate and log responses for multiple dataset rows in one forward pass.

        rows: list of (batch_dict, question, gt_answer, label, raw_questions)
        All prompts from every row are concatenated into a single model.generate call,
        then outputs are split back per row and logged individually.
        """
        # Flatten prompts and track per-row slice boundaries
        all_prompts   = []
        row_boundaries = []   # (start, end) indices into all_prompts for each row
        for batch_dict, *_ in rows:
            start = len(all_prompts)
            all_prompts.extend(batch_dict.values())
            row_boundaries.append((start, len(all_prompts)))

        all_extracted = self._run_generate(all_prompts, model, tokenizer)

        for (batch_dict, question, gt_answer, label, raw_questions), (start, end) in zip(rows, row_boundaries):
            extracted = all_extracted[start:end]
            self.logs.append(self._build_result(batch_dict, extracted, question,
                                                gt_answer, label, raw_questions))
