import json
import os
import argparse
import logging
import warnings
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Standard setup
from evals.judge_utils import *
from evals.utils import setup_logger, load_eval_dataset

# Judge system-prompt files live alongside this module under prompts/.
# Anchored to __file__ so the resolution doesn't depend on CWD.
_PROMPT_DIR = Path(__file__).parent / "prompts"

warnings.filterwarnings("ignore")


class EvalJUDGE:
    def __init__(self, eval_cfg, task='forget', gen_file=None, icr=False,
                 hf_model_id="Qwen/Qwen3.5-35B-A3B", batch_size=4,
                 judge_tag=None):
        """
        judge_tag: optional string included in output filenames to avoid mixing
                   experiments across different judge models/prompts.
                   e.g. judge_tag="qwen35b" → JG_EVAL_qwen35b_...
                   Default None preserves the original filename format (backward-compatible).
        """
        self.eval_cfg = eval_cfg
        self.name = eval_cfg.model.name
        self.task = task

        self.batch_size = batch_size
        self.hf_model_id = hf_model_id
        self.judge_tag = judge_tag  # None = no tag, backward-compatible

        self.set_icr(icr)
        self.set_gen_file(gen_file)

        # Load Model
        self.logger.info("Loading Model...")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        n_gpus = torch.cuda.device_count()
        # Leave headroom for inference activations (batch_size=8 needs ~2 GiB per GPU).
        # 34 GiB cap on A100-40GB leaves ~6 GiB free per GPU; excess model layers
        # spill to the next GPU (or CPU) which is acceptable.
        # 80B/72B models need an even tighter cap on GPU 0 due to embedding layers.
        _gib = lambda i: max(int(torch.cuda.get_device_properties(i).total_memory // (1024**3)) - 6, 1)
        max_memory = {i: f"{_gib(i)}GiB" for i in range(n_gpus)}
        if "80B" in hf_model_id or "72B" in hf_model_id:
            max_memory[0] = f"{max(_gib(0) - 6, 1)}GiB"
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            device_map="auto",
            dtype=torch.float16,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()

    def set_icr(self, icr: bool):
        self.icr_data = icr
        self.set_out_dirs()
        self.logger = setup_logger(self.log_path, self._log_dir)
        self.logger.info(f"Evaluating {self.task} | Batch Size: {self.batch_size}")
        print(
            f"Evaluating {self.task} | log path: {self.log_path} | output path: {self.jg_file_path} | ICR: {self.icr_data}")

    def save_logs(self, final_output):
        # Sort by base label so all rephrasings of the same question appear
        # together (strip @rephrase_key suffix; stable sort keeps insertion order
        # within each group, so the original question stays before its rephrasings)
        def _label_key(entry):
            raw = entry.get("label", "")
            return raw.split("@")[0] if "@" in raw else raw

        final_output = sorted(final_output, key=_label_key)

        with open(self.jg_file_path, "w", encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        self.logger.info(f"\n✅ Saved {len(final_output)} generations to: {self.jg_file_path}")

    def set_out_dirs(self, prefix='JG_EVAL_'):
        task_name   = self.eval_cfg.output.task_name
        model_short = self.name.split("/")[-1]
        subpath     = getattr(self.eval_cfg.output, 'subpath',  '') or ''
        exp_name    = getattr(self.eval_cfg.output, 'exp_name', '') or ''

        # Mirror Evaluator.set_out_dirs: use hierarchical path when topic is set
        if subpath:
            dir_key = os.path.join(subpath, exp_name) if exp_name else subpath
        else:
            dir_key = task_name

        jg_subdir = os.path.join(self.eval_cfg.output.evaldir, dir_key)
        os.makedirs(jg_subdir, exist_ok=True)

        # When judge_tag is set (e.g. "qwen35b"), include it in the filename so
        # outputs from different judge models never collide.  Default None keeps
        # the original format for full backward-compatibility.
        tag_prefix = f"{prefix}{self.judge_tag}_" if self.judge_tag else prefix

        suffix = f"{model_short}_{self.eval_cfg.output.eval_task}_icr_{self.icr_data}.jsonl"
        self.jg_file_path = os.path.join(jg_subdir, tag_prefix + suffix)

        # setup_logger(log_file_name, log_dir) joins them — store only the
        # filename here and pass jg_subdir as the directory.
        # Use exp_name (last path component) so slashes in task_name don't
        # create phantom subdirectories inside an already-hierarchical jg_subdir.
        log_label     = exp_name if exp_name else task_name.replace('/', '_')
        log_filename  = tag_prefix + f"{log_label}_icr_{self.icr_data}.log"
        self.log_path = log_filename
        self._log_dir = jg_subdir

    def _apply_chat_template(self, messages: list) -> str:
        """Apply chat template with enable_thinking=False when supported (Qwen3.5+)."""
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def create_prompt(self, gt, question, answer):
        # system prompt is loaded from the task-specific prompt file
        sys_prompt = self.get_system_prompt()
        return [
            {"role": "system",
             "content": sys_prompt},
            {"role": "user", "content": f"""
Evaluate the **Test Response** against the **Ground Truth**.

**Question:** {question}
**Ground Truth:** {gt}
**Test Response:** {answer}

Output (ONLY "YES" or "NO"):
"""}
        ]

    def get_system_prompt(self) -> str:
        # The "refusal" prompt is selected only when a custom --task name containing
        # "refusal" is passed; no pipeline-configured task uses it.
        # Allow explicit override via prompt_paths dict set at init time
        if hasattr(self, '_prompt_paths'):
            if "refusal" in self.task:
                path = _PROMPT_DIR / "judge_prompt_qwen_refusal.txt"
            elif "gibberish" in self.task:
                path = _PROMPT_DIR / "judge_prompt_qwen_gibberish.txt"
            elif "retain" in self.task and "retain" in self._prompt_paths:
                path = self._prompt_paths["retain"]
            elif "forget" in self.task and "forget" in self._prompt_paths:
                path = self._prompt_paths["forget"]
            else:
                path = self._prompt_paths.get("forget", _PROMPT_DIR / "judge_prompt_qwen.txt")
        elif "refusal" in self.task:
            path = _PROMPT_DIR / "judge_prompt_qwen_refusal.txt"
        elif "gibberish" in self.task:
            path = _PROMPT_DIR / "judge_prompt_qwen_gibberish.txt"
        elif "retain" in self.task:
            path = _PROMPT_DIR / "judge_retain_prompt_qwen.txt"
        else:
            path = _PROMPT_DIR / "judge_prompt_qwen.txt"
        with open(path, "r") as file:
            sys_prompt = file.read().replace("\n", " ")
        return sys_prompt

    def set_prompt_paths(self, forget_path: str = None, retain_path: str = None):
        """Override default prompt file paths. Call before generate()."""
        self._prompt_paths = {}
        if forget_path:
            self._prompt_paths["forget"] = forget_path
        if retain_path:
            self._prompt_paths["retain"] = retain_path

    def generate(self):
        if not os.path.exists(self.gen_file):
            raise FileNotFoundError(f"File not found: {self.gen_file}")

        # 1. Load responses (skip __metadata__ header if present)
        with open(self.gen_file, 'r', encoding='utf-8') as f:
            responses = json.load(f)
        if responses and isinstance(responses[0], dict) and responses[0].get("__metadata__"):
            self.logger.info(
                f"  metadata: model={responses[0].get('model_name')} "
                f"task={responses[0].get('eval_task')}"
            )
            responses = responses[1:]

        # 2. Detect format and normalise to grouped structure.
        #
        #  Flat (per-rephrase) – one entry per rephrase, label contains '@':
        #    {"label":"K1-direct@q_qwen1","question":"<rephrase>","ans_question":"...","gt_answer":"..."}
        #
        #  Grouped (per-question) – one entry per base question, label has no '@':
        #    {"label":"K1-direct","question":"<original>","q_qwen1":"<rephrase>",
        #     "ans_question":"...","ans_q_qwen1":"...","GT":"..."}
        #
        #  Flat entries are merged on-the-fly into the grouped structure so both
        #  formats produce identical judge output.

        def _is_flat_format(entries):
            return any('@' in str(e.get('label', '')) for e in entries[:5])

        if _is_flat_format(responses):
            self.logger.info("Detected flat (per-rephrase) format – merging entries by base label...")
            from collections import OrderedDict
            grouped: OrderedDict = OrderedDict()
            for ent in responses:
                raw_label = ent.get('label', '')
                base_label, rephrase_key = (raw_label.split('@', 1)
                                            if '@' in raw_label
                                            else (raw_label, None))
                gt = ent.get('gt_answer') or ent.get('GT', '')
                if base_label not in grouped:
                    grouped[base_label] = {
                        'GT':           gt,
                        'label':        base_label,
                        'question':     ent.get('question', ''),
                        'ans_question': str(ent.get('ans_question', '')).lower().replace('assistant', '--'),
                    }
                elif rephrase_key:
                    g = grouped[base_label]
                    g[rephrase_key] = ent.get('question', '')
                    g[f'ans_{rephrase_key}'] = str(ent.get('ans_question', '')).lower().replace('assistant', '--')
            merged_data = list(grouped.values())
            self.logger.info(f"  Merged {len(responses)} flat → {len(merged_data)} grouped entries")
        else:
            self.logger.info("Detected grouped (per-question) format – reading directly...")
            merged_data = []
            for ent in responses:
                entry = {'GT': ent.get('GT') or ent.get('gt_answer', ''),
                         'label': ent.get('label', '')}
                if 'question' in ent:
                    entry['question'] = ent['question']
                for k, v in ent.items():
                    if k in ('GT', 'gt_answer', 'label', 'question', 'topic'):
                        continue
                    if k == 'ans_question' or k.startswith('ans_'):
                        entry[k] = str(v).lower().replace('assistant', '--')
                    elif k.startswith('q_') or k.startswith('blank_'):
                        entry[k] = v
                merged_data.append(entry)

        # 3. Build flat judge tasks: one per (row_idx, ans_* key)
        flat_tasks = []
        self.logger.info("Preparing judge prompts...")
        for row_idx, entry in enumerate(merged_data):
            gt = entry.get('GT', '')
            for key, val in entry.items():
                if not key.startswith('ans_'):
                    continue
                base_key = key[len('ans_'):]  # "question" | "q_qwen1" | "blank_llama1"
                question_text = entry.get('question', '') if base_key == 'question' else entry.get(base_key, '')
                # Fall back to the original question text when the rephrase question text is
                # missing from the eval-output file (older outputs that did not store q_*/blank_* fields).
                if not question_text or not str(question_text).strip():
                    question_text = entry.get('question', '')
                if not question_text or not str(question_text).strip():
                    continue  # skip if even the original question is missing
                messages = self.create_prompt(gt, question_text, val)
                prompt = self._apply_chat_template(messages)
                flat_tasks.append({'row_idx': row_idx, 'key': key, 'prompt': prompt})

        self.logger.info(f"Total pairs to judge: {len(flat_tasks)}")
        if not flat_tasks:
            self.logger.warning("0 judge pairs – check that ans_* keys exist in the EVAL file.")
            return

        # 4. Batch inference
        evaluation_map = {}
        for i in tqdm(range(0, len(flat_tasks), self.batch_size), desc="Judging"):
            batch   = flat_tasks[i: i + self.batch_size]
            prompts = [item['prompt'] for item in batch]
            inputs  = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
            first_device = next(self.model.parameters()).device
            inputs = {k: v.to(first_device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=10, do_sample=False, temperature=0.0, top_p=1.0, use_cache=True)
            texts = self.tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for item, resp in zip(batch, texts):
                if "</think>" in resp:
                    resp = resp.split("</think>")[-1]
                decision = 'Yes' if 'yes' in resp.strip().lower() else 'No'
                evaluation_map.setdefault(item['row_idx'], {})[item['key']] = decision

        # 5. Reconstruct: add judge_* keys alongside the ans_* keys
        final_output = []
        for row_idx, entry in enumerate(merged_data):
            new_entry = entry.copy()
            for key, decision in evaluation_map.get(row_idx, {}).items():
                new_entry[key.replace('ans_', 'judge_')] = decision
            final_output.append(new_entry)

        self.save_logs(final_output)

    def set_gen_file(self, gen_file):
        self.gen_file = gen_file


class MockConfig:
    def __init__(self, model_name, task, data, savefolder):
        self.model = type('obj', (), {'name': model_name})
        self.output = type('obj', (), {
            'evaldir': savefolder, 'logdir': os.path.join(savefolder, 'logs'),
            'task_name': data, 'eval_task': task
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run EvalJUDGE on an existing eval-output file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Run 35B judge on pretrained forget rephrasings (outputs tagged 'qwen35b'):
CUDA_VISIBLE_DEVICES=1,2 python src/evals/judge_eval_qwen.py \\
    --responses_file evaluations/evalOutputs/llama_pretrained/EVAL_Llama-3.2-3B-Instruct_forget_rephrasings_icr_False.jsonl \\
    --model_name meta-llama/Llama-3.2-3B-Instruct \\
    --task forget_rephrasings --task_name llama_pretrained \\
    --evaldir evaluations/evalJudge \\
    --hf_model Qwen/Qwen3.5-35B-A3B \\
    --batch_size 4 --judge_tag qwen35b \\
    --forget_prompt src/evals/prompts/judge_prompt_qwen.txt

# Run 35B judge on pretrained retain:
CUDA_VISIBLE_DEVICES=1,2 python src/evals/judge_eval_qwen.py \\
    --responses_file evaluations/evalOutputs/llama_pretrained/EVAL_Llama-3.2-3B-Instruct_retain_icr_False.jsonl \\
    --model_name meta-llama/Llama-3.2-3B-Instruct \\
    --task retain --task_name llama_pretrained \\
    --evaldir evaluations/evalJudge \\
    --hf_model Qwen/Qwen3.5-35B-A3B \\
    --batch_size 4 --judge_tag qwen35b \\
    --retain_prompt src/evals/prompts/judge_retain_prompt_qwen.txt
""")
    parser.add_argument("--responses_file", type=str, required=True,
                        help="Path to an EVAL_*.jsonl file from evalOutputs/")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="HF model name of the evaluated model (used in output filename)")
    parser.add_argument("--task", type=str, default="forget",
                        help="eval_task: forget / retain / forget_rephrasings / ...")
    parser.add_argument("--task_name", type=str, default="llama_pretrained",
                        help="Experiment name — determines the output subdirectory")
    parser.add_argument("--evaldir", type=str, default="evaluations/evalJudge",
                        help="Root eval-judge output directory")
    parser.add_argument("--hf_model", default="Qwen/Qwen3.5-35B-A3B", type=str,
                        help="HF model ID of the JUDGE model")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--judge_tag", type=str, default="qwen35b",
                        help="Tag added to the output filename to separate judge models")
    parser.add_argument("--icr", action="store_true", default=False)
    parser.add_argument("--forget_prompt", type=str, default=None,
                        help="Override forget system prompt path (default: judge_prompt_qwen.txt)")
    parser.add_argument("--retain_prompt", type=str, default=None,
                        help="Override retain system prompt path (default: judge_retain_prompt_qwen.txt)")
    args = parser.parse_args()

    cfg = MockConfig(args.model_name, args.task, args.task_name, args.evaldir)

    evaluator = EvalJUDGE(
        eval_cfg=cfg,
        task=args.task,
        gen_file=args.responses_file,
        hf_model_id=args.hf_model,
        batch_size=args.batch_size,
        judge_tag=args.judge_tag,
        icr=args.icr,
    )
    if args.forget_prompt or args.retain_prompt:
        evaluator.set_prompt_paths(
            forget_path=args.forget_prompt,
            retain_path=args.retain_prompt,
        )
    evaluator.generate()
