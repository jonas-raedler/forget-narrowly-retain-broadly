"""Answer-NLL evaluation for the proximity-stratified relearning experiment.

Computes the negative log-likelihood of each ground-truth answer given its
question (chat-formatted exactly as training data via `preprocess_chat_instance`,
but with NO few-shot context and NO refusal context), for a given checkpoint
over one or more SUITE splits. This is the continuous forget/retain metric
(paper Sec. B.8 style) — the released upstream eval pipeline is judge-based
only, so this file adds the NLL readout.

The number that matters downstream is the DELTA vs. the unlearned checkpoint's
NLL on the same split; run this script on the unlearned checkpoint once to get
the baseline. Formatting choices (no context) differ from the training-time
rendering on purpose: the metric must be identical across attack arms, and the
few-shot pool would otherwise leak band-dependent context into the readout.

Usage (from the repo root, after training):
    python src/evals/nll_eval.py \
        --model_path ./saves/unlearn/challenger_disaster/Llama-3.2-3B-Instruct/jensen/exp/relearn/band_R2_... \
        --topic challenger_disaster \
        --model_config configs/model/Llama-3.2-3B-Instruct.yaml \
        --splits forget_eval retain_eval

Writes evaluations/nllOutputs/{task_name}/NLL_summary.json with per-example
NLLs and per-label-group aggregates.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from omegaconf import OmegaConf  # noqa: E402

from data.qa import QADataset  # noqa: E402
from data.collators import DataCollatorForSupervisedDataset  # noqa: E402
from data.utils import IGNORE_INDEX  # noqa: E402


def label_group(label: str) -> str:
    """Coarse group key for aggregation: Semantic-N, GK, Lexical, Syntax, K*, M*."""
    m = re.match(r"(Semantic-\d+)-", label)
    if m:
        return m.group(1)
    for g in ("GK", "Lexical", "Syntax"):
        if label.startswith(g):
            return g
    m = re.match(r"([KM]\d+)-", label)
    if m:
        return m.group(1)
    return "other"


@torch.no_grad()
def batch_answer_nll(model, batch, device):
    """Per-example (sum NLL over answer tokens, answer token count)."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # next-token shift
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        shift_logits.transpose(1, 2), shift_labels,
        ignore_index=IGNORE_INDEX, reduction="none",
    )  # [B, T-1]
    mask = (shift_labels != IGNORE_INDEX)
    return (loss * mask).sum(dim=1), mask.sum(dim=1)


def evaluate_split(model, tokenizer, template_args, device, args, split):
    hf_args = {"path": args.hf_path, "name": "default", "split": split}
    ds = QADataset(
        hf_args=OmegaConf.create(hf_args),
        template_args=template_args,
        tokenizer=tokenizer,
        question_key="question",
        answer_key="answer",
        max_length=args.max_length,
        add_context=False,
        add_refusal_context=False,
        filter_topic=args.topic,
    )
    labels_by_index = {int(r["index"]): r["label"] for r in ds.data}
    collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, padding_side="right", index="index")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    per_example = []
    for batch in loader:
        nll_sum, n_tok = batch_answer_nll(model, batch, device)
        for i, idx in enumerate(batch["index"].tolist()):
            per_example.append({
                "index": int(idx),
                "label": labels_by_index[int(idx)],
                "nll_sum": round(nll_sum[i].item(), 6),
                "n_answer_tokens": int(n_tok[i].item()),
            })

    groups = {}
    for ex in per_example:
        groups.setdefault(label_group(ex["label"]), []).append(ex)

    def agg(exs):
        tot_tok = sum(e["n_answer_tokens"] for e in exs)
        tot_nll = sum(e["nll_sum"] for e in exs)
        return {
            "n_examples": len(exs),
            # mean over examples of per-example mean-token NLL (each Q weighs equally)
            "mean_nll_per_token": sum(
                e["nll_sum"] / max(e["n_answer_tokens"], 1) for e in exs) / len(exs),
            # corpus-level per-token NLL (long answers weigh more)
            "corpus_nll_per_token": tot_nll / max(tot_tok, 1),
            "mean_nll_sum": tot_nll / len(exs),
        }

    return {
        "split": split,
        "overall": agg(per_example),
        "groups": {g: agg(exs) for g, exs in sorted(groups.items())},
        "per_example": per_example,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="checkpoint dir (./saves/unlearn/...) or HF model id")
    ap.add_argument("--task_name", default=None,
                    help="output subdir under evaluations/nllOutputs/; default: "
                         "model_path relative to saves/unlearn/")
    ap.add_argument("--topic", default="challenger_disaster")
    ap.add_argument("--model_config",
                    default="configs/model/Llama-3.2-3B-Instruct.yaml",
                    help="model YAML providing template_args + tokenizer_args")
    ap.add_argument("--splits", nargs="+",
                    default=["forget_eval", "retain_eval"])
    ap.add_argument("--hf_path", default="apeleg/SUITE")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--attn", default="sdpa",
                    help="attn_implementation for the eval forward pass "
                         "(sdpa is safe everywhere; loss is implementation-independent)")
    ap.add_argument("--out_root", default="evaluations/nllOutputs")
    args = ap.parse_args()

    model_cfg = OmegaConf.load(REPO_ROOT / args.model_config)
    template_args = model_cfg.template_args
    # QADataset probes template_args for model_tokens in refusal mode only,
    # but pass it through anyway so behavior matches the training pipeline.
    template_args = OmegaConf.merge(
        template_args, {"model_tokens": model_cfg.get("model_tokens", {})})

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok_path = model_cfg.tokenizer_args.pretrained_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn).to(device).eval()

    task_name = args.task_name
    if task_name is None:
        p = args.model_path.replace("\\", "/")
        marker = "saves/unlearn/"
        if marker in p:
            task_name = p.split(marker, 1)[1].strip("/")
        else:
            task_name = p.strip("./").replace("/", "_")

    result = {
        "model_path": args.model_path,
        "task_name": task_name,
        "topic": args.topic,
        "model_config": args.model_config,
        "hf_path": args.hf_path,
        "max_length": args.max_length,
        "formatting": {"add_context": False, "add_refusal_context": False,
                       "system_prompt": bool(template_args.get("system_prompt"))},
        "splits": {},
    }
    for split in args.splits:
        print(f"[nll_eval] scoring split={split} ...")
        result["splits"][split] = evaluate_split(
            model, tokenizer, template_args, device, args, split)
        o = result["splits"][split]["overall"]
        print(f"[nll_eval] {split}: n={o['n_examples']} "
              f"mean_nll_per_token={o['mean_nll_per_token']:.4f} "
              f"mean_nll_sum={o['mean_nll_sum']:.4f}")

    out_dir = REPO_ROOT / args.out_root / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "NLL_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[nll_eval] wrote {out_path}")


if __name__ == "__main__":
    main()
