import torch
import tqdm
import json
import numpy as np
import scipy
import os
from typing import List, Dict, Any
import nltk
from datasets import load_dataset


def load_rep_data():

    rep_data = load_dataset("jinzhuoran/RWKU", 'utility_fluency') # repetitiveness
    rep_data = rep_data.shuffle(seed=0)
    return rep_data



def n_gram_entropy(gen_texts:list, agg:str="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()

def compute_n_gram_entropy(sentence:str, ns:List=None, weights:List=None, agg:str="arith") -> float:
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()
        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))
    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


@torch.no_grad()
def eval_repet(model, tokenizer, dataset, batch_size:int=1, output_result_dir:str=None, use_prompt:bool=False, num_samples:int=1000, max_new_tokens:int=500):
    nltk.download('punkt')
    tokenizer.padding_side = 'left'
    prompts_text = []
    questions = []

    for ix, sample in enumerate(dataset['test']):
        instruction = sample['instruction']
        prompt = 'Instruction: {}\n'.format(instruction)
        if use_prompt:
            prompt = "You are an AI Assistant who is supposed to unlearn about {} " \
                     "and provide answers without its knowledge as if you never knew about it. " \
                     "Don't tell anyone that you unlearned anything.\n".format(sample['subject']) + prompt
        messages = [{"role": "user", "content": prompt}]
        # enable_thinking=False tells Qwen3+ it's done thinking (inserts <think>\n\n</think>).
        # Non-Qwen tokenizers don't support the flag → fall back gracefully.
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prompts_text.append(formatted)
        questions.append(sample)
        if ix >= num_samples:
            break

    # Build EOS id list: always include eos_token_id; also add <|eot_id|> (Llama) and
    # <|im_end|> (Qwen) when they exist and differ from eos_token_id/unk. Without the
    # model's own turn terminator, generation runs to max_new_tokens and the
    # over-generated text skews the n-gram entropy. No-op for families that lack a
    # given token (convert_tokens_to_ids returns unk).
    _unk_id = tokenizer.unk_token_id
    eos_ids = [tokenizer.eos_token_id]
    for special_tok in ("<|eot_id|>", "<|im_end|>"):
        tok_id = tokenizer.convert_tokens_to_ids(special_tok)
        if tok_id not in (_unk_id, tokenizer.eos_token_id, 0, None):
            eos_ids.append(tok_id)

    input_device = model.get_input_embeddings().weight.device

    outputs_text = []
    for i in tqdm.tqdm(range(0, len(prompts_text), batch_size), desc="Generating repet completions"):
        batch = prompts_text[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=False)
        enc = {k: v.to(input_device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )
        # Decode only the newly generated tokens — avoids the string-length mismatch
        # that occurs when special tokens in the prompt change the decoded prompt length.
        new_tokens = out[:, prompt_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        outputs_text.extend(decoded)

    for answer, question in zip(outputs_text, questions):
        question['prediction'] = answer

    entropy = n_gram_entropy(outputs_text)
    print("Score {:.4f}".format(entropy*100))

    output_result = {
        'entropy': entropy*100,
        'results': questions,
    }
    tokenizer.padding_side = 'right'
    tmp_path = output_result_dir + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(output_result, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, output_result_dir)
    print(f"\n✅ Saved results to: {output_result_dir}")