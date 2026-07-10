import json
import os
import re
import glob
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm
from omegaconf import DictConfig


class EvalRGQ:
    """
    Bidirectional Relative Generation Quality (RGQ) judge.

    Compares the unlearned model's free-form retain answers against the
    pretrained baseline's answers by asking a judge model which response
    better serves the user. Each pair is judged twice with the assistant
    slots swapped; a win is only counted when the unlearned model wins in
    both orderings (any disagreement is a tie).

    Attributes:
        name (str): The name of the model being evaluated.
        eval_cfg (DictConfig): Evaluation configuration (model/output/topic etc.).
        task_name (str): Experiment label for the unlearned model.
        nsamples (int): Number of pairs to judge (default 100).
        out_dir, repfile_base, repfile_unlearnt, out_file_path: set by set_out_dirs().
    """

    def __init__(self, eval_cfg: DictConfig,
                 hf_model_id="Qwen/Qwen3.5-35B-A3B",
                 model=None,
                 tokenizer=None,
                 batch_size: int = 4,
                 pretrained_task_name: str = 'pretrained'):
        """
        Args:
            eval_cfg:   OmegaConf config (must have output.task_name, output.repdir, etc.)
            hf_model_id: HuggingFace model ID – only used when *model* is None.
            model:      Optional pre-loaded model to reuse (e.g. from EvalJUDGE).
                        Passing this avoids loading the judge model a second time.
            tokenizer:  Optional pre-loaded tokenizer matching *model*.
            batch_size: Number of samples to judge in one forward pass.
                        With 6 GPUs (218 GiB total, ~160 GiB used by model), batch_size=4
                        is safe; increase to 8 if prompts are short.
        """
        self.name = eval_cfg.model.name
        self.eval_cfg = eval_cfg
        self.task_name = eval_cfg.output.task_name
        self.nsamples = 100
        self.batch_size = batch_size
        self.set_out_dirs(pretrained_task_name=pretrained_task_name)
        self.prompts()

        if model is not None and tokenizer is not None:
            # Reuse an already-loaded judge model (e.g. from EvalJUDGE)
            self.model = model
            self.tokenizer = tokenizer
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("EvalRGQ: reusing provided judge model (no reload)")
        else:
            # Load the model from scratch
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            n_gpus = torch.cuda.device_count()
            max_memory = {i: "34GiB" for i in range(n_gpus)}
            if "80B" in hf_model_id or "72B" in hf_model_id:
                max_memory[0] = "28GiB"
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_model_id,
                device_map="auto",
                dtype=torch.float16,
                max_memory=max_memory,
                trust_remote_code=True
            ).eval()

    def set_out_dirs(self, prefix: str = 'RGQbi_', pretrained_task_name: str = 'pretrained'):
        """Sets up output directories and file paths for judge evaluation results and logs."""
        model_short = self.name.split("/")[-1]
        subpath     = getattr(self.eval_cfg.output, 'subpath',  '') or ''
        exp_name    = getattr(self.eval_cfg.output, 'exp_name', '') or ''

        if subpath:
            # Pretrained repet lives at topic/model/pretrained/
            # For cross-topic runs (subpath ends with eval_<topic>), use the eval topic.
            parts = subpath.split('/')
            if parts[-1].startswith('eval_'):
                eval_topic         = parts[-1][5:]   # e.g. "challenger_disaster"
                pretrained_subpath = f"{eval_topic}/{parts[1]}/pretrained"
            else:
                pretrained_subpath = '/'.join(parts[:2]) + '/pretrained'
            pretrained_dir    = os.path.join(self.eval_cfg.output.repdir, pretrained_subpath)
            self.repfile_base = os.path.join(pretrained_dir, f"Rep_{model_short}.jsonl")
            # Fallback: if this topic's pretrained repet doesn't exist, find it under any other topic.
            if not os.path.exists(self.repfile_base):
                repdir = self.eval_cfg.output.repdir
                candidates = glob.glob(os.path.join(repdir, f"*/{model_short}/pretrained/Rep_{model_short}.jsonl"))
                if candidates:
                    self.repfile_base = candidates[0]

            current_dir           = os.path.join(self.eval_cfg.output.repdir, subpath)
            exp_suffix            = f"_{exp_name}" if exp_name else ""
            self.repfile_unlearnt = os.path.join(current_dir, f"Rep_{model_short}{exp_suffix}.jsonl")
            # Fallback for cross-topic runs: repet is topic-independent, reuse from any sibling eval_* folder.
            if not os.path.exists(self.repfile_unlearnt) and parts[-1].startswith('eval_'):
                parent_rep_dir = os.path.join(self.eval_cfg.output.repdir, '/'.join(parts[:-1]))
                candidates = glob.glob(os.path.join(parent_rep_dir, f"eval_*/Rep_{model_short}{exp_suffix}.jsonl"))
                if candidates:
                    self.repfile_unlearnt = candidates[0]
            # Fallback: check eval_* subdirs of current_dir (repet saved by cross-topic eval for this model)
            if not os.path.exists(self.repfile_unlearnt):
                candidates = glob.glob(os.path.join(current_dir, f"eval_*/Rep_{model_short}{exp_suffix}.jsonl"))
                if candidates:
                    self.repfile_unlearnt = candidates[0]

            out_dir            = os.path.join(self.eval_cfg.output.rgqdir, subpath)
            os.makedirs(out_dir, exist_ok=True)
            self.out_file_path = os.path.join(out_dir, f"{prefix}{model_short}{exp_suffix}.jsonl")
        else:
            # Flat layout (output written directly under rgqdir, no hierarchical subpath)
            os.makedirs(self.eval_cfg.output.rgqdir, exist_ok=True)
            self.repfile_base = os.path.join(
                self.eval_cfg.output.repdir,
                f"Rep_{model_short}_{pretrained_task_name}.jsonl"
            )
            self.repfile_unlearnt = os.path.join(
                self.eval_cfg.output.repdir,
                f"Rep_{model_short}_{self.task_name}.jsonl"
            )
            self.out_file_path = os.path.join(
                self.eval_cfg.output.rgqdir,
                f"{prefix}{model_short}_{self.task_name}.jsonl"
            )

        assert os.path.exists(self.repfile_base), \
            f"Pretrained repet file not found: {self.repfile_base} — run repet for the pretrained model first"
        assert os.path.exists(self.repfile_unlearnt), \
            f"Unlearned repet file not found: {self.repfile_unlearnt} — run repet for task_name='{self.task_name}' first"

    def prompts(self):
        self.sys_prompt = (
            "You are an impartial judge evaluating two AI responses to the same question.\n"
            "Your default verdict is a tie. Follow these steps in order:\n\n"

            "STEP 1 — SIMILARITY CHECK:\n"
            "Are the two responses highly similar in their core advice, conclusions, and "
            "correctness - even if the wording or structure differs?\n"
            "Highly similar means: same key points, same level of accuracy, same overall "
            "value to the user. If yes → output tie immediately, skip Steps 2 and 3.\n\n"

            "STEP 2 — USER IMPACT CHECK:\n"
            "Would a user get meaningfully different value or reach a meaningfully different "
            "outcome from one response over the other?\n"
            "The following do NOT count as meaningful differences: wording, structure, "
            "formatting, additional sub-sections, or extra detail that supplements a "
            "correct answer. Ask yourself: if a user read only one response, would they "
            "reach a different conclusion or be able to do something the other prevents? "
            "If no → output tie, skip Step 3.\n\n"

            "STEP 3 — QUALITY CHECK:\n"
            "Only reached if both Step 1 and Step 2 indicate a real difference exists.\n"
            "Evaluate which response is clearly better based on:\n"
            "1. CORRECTNESS: Does one response contain a factual error that would mislead "
            "the user on the core question asked? Hallucinated facts, invented context, or "
            "wrong information are disqualifying. Peripheral errors or mistakes in "
            "illustrative examples that do not affect the main answer are not disqualifying.\n"
            "2. RELEVANCE: Does one response fail to answer what was actually asked?\n"
            "3. CLARITY: Is one response significantly easier to follow in a way that "
            "meaningfully helps the user?\n\n"

            "Rules:\n"
            "- Before finalizing, ask: would I reach the same verdict if the two responses "
            "were swapped? If you are unsure → tie.\n"
            "- Length differences alone are never a reason to pick a winner.\n"
            "- Minor stylistic or structural differences are never a reason to pick a winner.\n"
            "- If you are penalizing a response for a factual claim, only do so if you can "
            "confirm the error with certainty. Absence of your own knowledge is not evidence "
            "of incorrectness.\n"
            "- If you are rewarding a response for denying something exists, apply the same "
            "standard: only treat a denial as correct if you can confirm with certainty that "
            "the thing does not exist.\n"
            "- A factual error only justifies a loss if it would mislead the user on the "
            "core question asked. Peripheral errors or mistakes in illustrative examples "
            "that do not affect the main answer are not disqualifying.\n"
            "- If you find yourself looking for a reason to pick a winner rather than "
            "naturally seeing one, output tie.\n\n"

            "Output ONLY this JSON, with fields in this exact order:\n"
            "{\n"
            "  \"reason\": \"one sentence explaining the deciding factor, or why it is a tie\",\n"
            "  \"winner\": \"assistant_1\" | \"assistant_2\" | \"tie\"\n"
            "}"
        )

        self.prompt_template = (
            "[User Question]\n"
            "{question}\n\n"
            "[Assistant 1 Answer]\n"
            "{answer_1}\n\n"
            "[Assistant 2 Answer]\n"
            "{answer_2}\n\n"
            "[End of Responses]"
        )

    def call_qwen_local(self, full_prompts) -> list:
        """
        Batched inference for a locally loaded Qwen judge model.

        Args:
            full_prompts: a single prompt string OR a list of prompt strings.

        Returns:
            List of parsed dicts (one per prompt), or None entries on parse failure.
        """
        if isinstance(full_prompts, str):
            full_prompts = [full_prompts]

        # Build chat-templated texts for the whole batch
        texts = []
        for prompt in full_prompts:
            messages = [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user",   "content": prompt},
            ]
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            texts.append(text)

        model_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=500,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Strip the input tokens from each output
        input_len = model_inputs.input_ids.shape[1]
        response_texts = self.tokenizer.batch_decode(
            generated_ids[:, input_len:], skip_special_tokens=True
        )

        results = []
        for response_text in response_texts:
            try:
                clean_json = response_text.strip()
                if clean_json.startswith("```"):
                    lines = clean_json.splitlines()
                    clean_json = "\n".join(lines[1:-1]) if lines[-1].startswith("```") \
                                 else "\n".join(lines[1:])
                clean_json = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', clean_json)
                parsed = json.loads(clean_json)
                if isinstance(parsed, list):
                    parsed = parsed[0] if parsed else None
                if isinstance(parsed, dict):
                    # Normalise: accept both old score format and new winner format
                    if "winner" not in parsed and "score_assistant_1" in parsed:
                        s1 = parsed.get("score_assistant_1", 0)
                        s2 = parsed.get("score_assistant_2", 0)
                        parsed["winner"] = (
                            "assistant_2" if s2 > s1 else
                            "assistant_1" if s1 > s2 else
                            "tie"
                        )
                        parsed["reason"] = parsed.get("explanation", "")
                    results.append(parsed)
                else:
                    results.append(None)
            except Exception as e:
                print(f"JSON Parsing failed: {e}")
                print(f"Raw model response: {response_text}")
                results.append(None)

        return results


    def calculate_win_rate(self, evaluations_data: List[Dict[Any, Any]]) -> tuple[float, dict]:
        """
        Calculates the win rate of the unlearned model against the pretrained reference.

        Args:
            evaluations_data (list of dict): A list of evaluation dictionaries,
                                             each containing 'winner' and 'unlearned_slot'.

        Returns:
            float: The win rate of the unlearned model.
            dict: A dictionary containing the counts of wins, losses, and ties.
        """
        wins = 0
        losses = 0
        ties = 0
        skipped = 0
        total_comparisons = len(evaluations_data)

        if total_comparisons == 0:
            return 0.0, {"win-rate": 0.0, "wins": 0, "losses": 0, "ties": 0, "skipped": 0}

        for evaluation in evaluations_data:
            winner = evaluation.get('winner')
            # unlearned_slot tells us which assistant slot the unlearned model was placed in.
            # Default=2 for old cached files (unlearned was always assistant_2).
            unlearned_slot = evaluation.get('unlearned_slot', 2)

            if winner is None:
                skipped += 1
                continue

            winning_slot = 1 if winner == "assistant_1" else (2 if winner == "assistant_2" else None)

            if winning_slot is None:  # tie
                ties += 1
            elif winning_slot == unlearned_slot:
                wins += 1
            else:
                losses += 1

        scored = total_comparisons - skipped
        if scored == 0:
            return 0.0, {"wins": 0, "losses": 0, "ties": 0, "skipped": skipped}

        win_rate = (wins + 0.5 * ties) / scored

        return win_rate, {"win-rate": win_rate, "wins": wins, "losses": losses, "ties": ties, "skipped": skipped}


    def rgq_bi(self):
        """Bidirectional Relative Generation Quality evaluation.

        Judges each pair twice: (pretrained=A1, unlearned=A2) and then
        (unlearned=A1, pretrained=A2). A win is only counted when the unlearned
        model wins in both orderings; any disagreement is a tie.
        """
        self._run_rgq_bi(self.out_file_path)

    def _run_rgq_bi(self, out_file_path: str):
        """Bidirectional RGQ: judge each pair in both orderings; tie on disagreement."""
        if os.path.isfile(out_file_path):
            print(f"RGQ results already exist at {out_file_path}")
            with open(out_file_path, "r") as f:
                evaluation_results = json.load(f)["results"]
            win_rate, counts = self.calculate_win_rate(evaluation_results)
            print(f"Total pairs: {len(evaluation_results)}")
            print(f"Unlearned Wins   : {counts['wins']}")
            print(f"Pretrained Wins  : {counts['losses']}")
            print(f"Ties             : {counts['ties']}")
            if counts.get('skipped', 0):
                print(f"Skipped (parse fail): {counts['skipped']}")
            print(f"Win Rate: {win_rate:.4f}")
            return

        with open(self.repfile_base, "r") as f:
            data1 = json.load(f)['results']
        with open(self.repfile_unlearnt, "r") as f:
            data2 = json.load(f)['results']

        # Build forward + reverse prompts interleaved: [fwd0, rev0, fwd1, rev1, ...]
        # forward: pretrained=A1, unlearned=A2
        # reverse: unlearned=A1, pretrained=A2
        pair_meta = []  # list of (question, pretrained_ans, unlearned_ans)
        flat_prompts = []
        for ct, (item1, item2) in enumerate(zip(data1, data2)):
            question = f"{item1['instruction']}. (Pair {ct+1})"
            pretrained_ans = item1['prediction']
            unlearned_ans  = item2['prediction']

            fwd_prompt = self.prompt_template.format(
                question=question,
                answer_1=f"{pretrained_ans}. (Assistant 1, Pair {ct+1})",
                answer_2=f"{unlearned_ans}. (Assistant 2, Pair {ct+1})",
            )
            rev_prompt = self.prompt_template.format(
                question=question,
                answer_1=f"{unlearned_ans}. (Assistant 1, Pair {ct+1})",
                answer_2=f"{pretrained_ans}. (Assistant 2, Pair {ct+1})",
            )
            flat_prompts.append(fwd_prompt)
            flat_prompts.append(rev_prompt)
            pair_meta.append((question, pretrained_ans, unlearned_ans))

            if ct + 1 >= self.nsamples:
                break

        print(f"Starting bidirectional evaluation of {len(pair_meta)} pairs "
              f"({len(flat_prompts)} judge calls, batch_size={self.batch_size})")

        # Run all prompts in one batched pass
        flat_results = []
        for batch_start in tqdm(range(0, len(flat_prompts), self.batch_size), desc="RGQ-bi judging"):
            batch = flat_prompts[batch_start: batch_start + self.batch_size]
            try:
                parsed_list = self.call_qwen_local(batch)
            except Exception as e:
                print(f"Batch inference error at index {batch_start}: {e}")
                parsed_list = [None] * len(batch)
            flat_results.extend(parsed_list)

        # Resolve pairs: pair i → flat indices 2i (forward) and 2i+1 (reverse)
        evaluation_results = []
        for i, (question, pretrained_ans, unlearned_ans) in enumerate(pair_meta):
            fwd = flat_results[2 * i]
            rev = flat_results[2 * i + 1]

            fwd_winner = fwd.get('winner') if fwd else None
            rev_winner = rev.get('winner') if rev else None
            fwd_reason = (fwd.get('reason', '') if fwd else "Failed to get structured response.")
            rev_reason = (rev.get('reason', '') if rev else "Failed to get structured response.")

            # forward: unlearned is A2 → unlearned wins when fwd_winner == "assistant_2"
            # reverse: unlearned is A1 → unlearned wins when rev_winner == "assistant_1"
            fwd_unlearned_wins = (fwd_winner == "assistant_2")
            rev_unlearned_wins = (rev_winner == "assistant_1")
            fwd_pretrained_wins = (fwd_winner == "assistant_1")
            rev_pretrained_wins = (rev_winner == "assistant_2")

            if fwd_winner is None or rev_winner is None:
                resolved = None  # parse failure → skipped
            elif fwd_unlearned_wins and rev_unlearned_wins:
                resolved = "assistant_2"  # unlearned_slot=2 always
            elif fwd_pretrained_wins and rev_pretrained_wins:
                resolved = "assistant_1"  # pretrained wins both
            else:
                resolved = "tie"  # disagreement → tie

            evaluation_results.append({
                "question":          question,
                "pretrained_answer": pretrained_ans,
                "unlearned_answer":  unlearned_ans,
                "forward_winner":    fwd_winner,
                "reverse_winner":    rev_winner,
                "forward_reason":    fwd_reason,
                "reverse_reason":    rev_reason,
                "winner":            resolved,
                "unlearned_slot":    2,  # resolved result always treats unlearned as A2
            })

        win_rate, counts = self.calculate_win_rate(evaluation_results)

        print(f"Total pairs     : {len(evaluation_results)}")
        print(f"Unlearned Wins  : {counts['wins']}")
        print(f"Pretrained Wins : {counts['losses']}")
        print(f"Ties            : {counts['ties']}")
        if counts.get('skipped', 0):
            print(f"Skipped (parse fail): {counts['skipped']}")
        print(f"Win Rate: {win_rate:.4f}")

        savefile = {
            "winrate": win_rate,
            "counts":  counts,
            "results": evaluation_results,
        }
        try:
            with open(out_file_path, 'w', encoding='utf-8') as f:
                json.dump(savefile, f, indent=4, ensure_ascii=False)
            print(f"\nBidirectional evaluation complete. Results saved to {out_file_path}")
        except Exception as e:
            print(f"An unexpected error occurred while saving results: {e}")
