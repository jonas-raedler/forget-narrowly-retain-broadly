"""
get_refusal_strings.py — Elicit natural refusal strings from a model.

Feeds a set of deliberately unanswerable questions
(dataset/refusal_detection/refusal_detection_questions.json — e.g. "What will be the
exact closing price of Apple stock on November 15, 2031?") to a chat model and prints
how it naturally refuses. Use it to choose a model's refusal string: the per-model
`model_tokens.refusal_string` in configs/model/*.yaml and the REFUSAL_KEY prefixes in
scripts/_suite_common.sh (e.g. "I am unable...") that the JensUn++ trainer trains against.

The model is loaded with its own built-in chat template. Pass --system-prompt to mimic a
particular eval/training setup; by default no system prompt is used.

Usage (from repo root):
    python scripts/get_refusal_strings.py --model meta-llama/Llama-3.2-3B-Instruct
    python scripts/get_refusal_strings.py --model Qwen/Qwen3.5-9B --n 20 --device cuda:0
    python scripts/get_refusal_strings.py --system-prompt "You are a helpful assistant."

Prints each question with the model's response, then a summary of the most common
opening phrases (the natural refusal prefixes) across all responses.
"""

import argparse
import collections
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = _REPO_ROOT / "dataset" / "refusal_detection" / "refusal_detection_questions.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct",
                   help="HF hub id or local path of the chat model to probe.")
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS,
                   help="JSON list of {question, answer} probe items.")
    p.add_argument("--n", type=int, default=None,
                   help="Only use the first N questions (default: all).")
    p.add_argument("--system-prompt", default=None,
                   help="Optional system prompt (default: none).")
    p.add_argument("--max-new-tokens", type=int, default=64,
                   help="Generation length per question.")
    p.add_argument("--device", default=None,
                   help="Device to place the model on (e.g. cuda:0). Default: device_map=auto.")
    p.add_argument("--summary-words", type=int, default=4,
                   help="How many leading words of each response to group in the prefix summary.")
    return p.parse_args()


def load_model(model_id: str, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
    if device is None:
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, tokenizer


def refusal_for(question, model, tokenizer, system_prompt, max_new_tokens):
    """Return the model's response to one probe question, decoded from the newly
    generated tokens only (the prompt is stripped)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    completion = out[0][prompt_len:]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    items = json.loads(args.questions.read_text(encoding="utf-8"))
    if args.n is not None:
        items = items[: args.n]

    print(f"Probing {args.model} with {len(items)} unanswerable question(s)\n" + "=" * 80)
    model, tokenizer = load_model(args.model, args.device)

    openers = collections.Counter()
    for i, item in enumerate(items, 1):
        question = item["question"]
        response = refusal_for(question, model, tokenizer, args.system_prompt, args.max_new_tokens)
        print(f"[{i}/{len(items)}] Q: {question}\n        A: {response}\n")
        opener = " ".join(response.split()[: args.summary_words])
        if opener:
            openers[opener] += 1

    print("=" * 80)
    print(f"Most common opening phrases (first {args.summary_words} words) — candidate refusal prefixes:")
    for opener, count in openers.most_common(15):
        print(f"  {count:3d}x  {opener}")


if __name__ == "__main__":
    main()
