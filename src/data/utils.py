import json

import torch
import random
import datasets
import numpy as np
from typing import List, Dict, Any, Union, Optional

IGNORE_INDEX = -100  # label value excluded from the loss (fixed PyTorch convention)


def _apply_chat_template_ids(tokenizer, chat, **kwargs) -> list:
    """apply_chat_template returns different types across transformers versions; always return a plain list."""
    try:
        ids = tokenizer.apply_chat_template(chat, tokenize=True, enable_thinking=False, **kwargs)
    except TypeError:
        ids = tokenizer.apply_chat_template(chat, tokenize=True, **kwargs)
    if isinstance(ids, list):
        return ids
    if hasattr(ids, 'input_ids'):
        ids = ids.input_ids
    return ids.tolist() if hasattr(ids, 'tolist') else list(ids)


def _turn_close_ids(tokenizer) -> list:
    """Tokens the chat template appends after the assistant's content to close a turn.

    Llama ``[<|eot_id|>]``, Ministral ``[</s>]``, Qwen ``[<|im_end|>, '\\n']``. The
    first token is the turn terminator (the token the model emits to stop); any tokens
    after it are template padding, e.g. a trailing newline.
    """
    sentinel = "z"
    sent_ids = tokenizer.encode(sentinel, add_special_tokens=False)
    full_ids = _apply_chat_template_ids(
        tokenizer,
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": sentinel}],
        add_generation_prompt=False,
    )
    # Return whatever follows the sentinel content tokens (the turn closer).
    for i in range(len(full_ids) - len(sent_ids), -1, -1):
        if full_ids[i : i + len(sent_ids)] == sent_ids:
            return full_ids[i + len(sent_ids):]
    return []


def _end_on_terminator(ids: list, tokenizer) -> list:
    """Return ``ids`` ending on exactly one turn terminator (the model's stop token).

    The answer turn should close on the single token the model actually emits to stop
    (Llama ``<|eot_id|>``, Ministral ``</s>``, Qwen ``<|im_end|>``), with no trailing
    newline or duplicate EOS after it.
    """
    closer = _turn_close_ids(tokenizer)
    terminator = closer[0] if closer else tokenizer.eos_token_id
    trailing = closer[1:]
    if trailing and ids[-len(trailing):] == trailing:
        ids = ids[: -len(trailing)]
    if not ids or ids[-1] != terminator:
        ids = ids + [terminator]
    return ids


def load_hf_dataset(path, _is_main=True, **kwargs):
    if _is_main:
        print('HF PATH', path)
    dataset = datasets.load_dataset(path, **kwargs)
    return dataset


def preprocess_chat_instance(
    tokenizer,
    template_config: Dict[str, Any],
    prompt_msgs: Union[List[str], str],
    response_msgs: Union[List[str], str],
    max_length: int,
    predict_with_generate: bool = False,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, torch.Tensor]:
    """Preprocesses a chat instance for training or generation.
    When in training, both the returned `input_ids` and `labels` cover the entire conversation.
    `input_ids` has no padding, and `labels` assign `IGNORE_INDEX` to tokens where loss is not computed (i.e. all tokens except the final response message).
    When in generation, `input_ids` are returned only up to the last user prompt, excluding the assistant's response. The `labels` returned are the same as during training.
    `attention_mask` is always 1 over the full `input_ids` token sequence.

    `prompt_msgs` and `response_msgs` are lists where, except for the last pair, all
    corresponding pairs are in-context examples. When they are a string and not
    a list, there are no in-context examples.

    Args:
        tokenizer: Tokenizer to apply on text
        template_config (Dict[str, Any]): Configuration for the chat template (comes from model-specific config).
        prompt_msgs (Union[List[str], str]): List of prompt messages or a single prompt message string.
        response_msgs (Union[List[str], str]): List of response messages or a single response message string.
        max_length (int): Maximum sequence length after tokenization.
        predict_with_generate (bool, optional): Whether to prepare inputs for generation.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing 'input_ids', 'labels', and 'attention_mask' tensors for model input.
    """
    assert len(prompt_msgs) == len(response_msgs)
    if isinstance(prompt_msgs, str):
        assert isinstance(response_msgs, str)
        prompt_msgs, response_msgs = [prompt_msgs], [response_msgs]

    if template_config["apply_chat_template"]:
        chat = []
        system_prompt = template_config.get("system_prompt", None)
        if system_prompt:
            chat += [{"role": "system", "content": system_prompt}]
        for prompt, response in zip(prompt_msgs, response_msgs):
            chat += [{"role": "user", "content": prompt}]
            chat += [{"role": "assistant", "content": response}]

        chat_ids = _apply_chat_template_ids(tokenizer, chat, add_generation_prompt=False)
        # all except last response are in-context examples
        try:
            wrapped_prompt = tokenizer.apply_chat_template(
                chat[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            wrapped_prompt = tokenizer.apply_chat_template(
                chat[:-1], tokenize=False, add_generation_prompt=True
            )
        prompt_ids = _apply_chat_template_ids(tokenizer, chat[:-1], add_generation_prompt=True)
    else:
        wrapped_prompt = ""
        system_prompt_with_special_tokens = template_config.get(
            "system_prompt_with_special_tokens", None
        )
        if system_prompt_with_special_tokens:
            wrapped_prompt += system_prompt_with_special_tokens
        # add in-context examples
        n_few_shot = len(prompt_msgs) - 1
        for i in range(n_few_shot):
            fs_prompt, fs_response = prompt_msgs[i], response_msgs[i]
            wrapped_prompt += (
                template_config["user_start_tag"]
                + fs_prompt
                + template_config["user_end_tag"]
                + template_config["asst_start_tag"]
                + fs_response
                + template_config["asst_end_tag"]
            )

        # add actual example
        final_prompt, final_response = prompt_msgs[-1], response_msgs[-1]
        wrapped_prompt += (
            template_config["user_start_tag"]
            + final_prompt
            + template_config["user_end_tag"]
            + template_config["asst_start_tag"]
        )
        chat_ids = tokenizer(
            wrapped_prompt + final_response,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
        )["input_ids"]

        prompt_ids = tokenizer(
            wrapped_prompt,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
        )["input_ids"]

    if template_config["apply_chat_template"]:
        # End the answer turn on the model's terminator (its stop token), dropping any
        # trailing newline or duplicate EOS the template leaves after it — so the
        # supervised answer stops exactly where generation stops.
        chat_ids = _end_on_terminator(chat_ids, tokenizer)
    elif chat_ids[-1] != tokenizer.eos_token_id:
        chat_ids += [tokenizer.eos_token_id]

    # Mask boundary = number of leading prompt tokens. When the generation prompt is an
    # exact prefix of the full conversation, this is len(prompt_ids). Some chat templates
    # break that: the generation prompt opens a block that a rendered assistant turn omits,
    # so prompt_ids is NOT a prefix of chat_ids — using len(prompt_ids) would mask the answer
    # away (or supervise misaligned tokens). When the prefix assumption fails, fall back to
    # the true divergence point (common-prefix length). Prefix-compatible templates leave
    # len_matched at len(prompt_ids), so their masking is unaffected.
    len_matched = len(prompt_ids)
    if chat_ids[:len_matched] != prompt_ids:
        len_matched = next(
            (i for i, (a, b) in enumerate(zip(prompt_ids, chat_ids)) if a != b),
            min(len(prompt_ids), len(chat_ids)),
        )

    item = {}
    if predict_with_generate:
        item["input_ids"] = prompt_ids
        labels = chat_ids  # contains the entire conversation
    elif len_matched != len(prompt_ids):
        # The generation prompt is not a prefix of the rendered conversation: it opens a
        # block that the history render omits, so len_matched fell back below
        # len(prompt_ids). Rebuild the final turn from the generation prompt + the history
        # tail (answer + turn terminator) so the masked prefix INCLUDES that block —
        # matching what the model sees at inference and the framing the forget path uses.
        # The supervised tokens (chat_ids[len_matched:]) are unchanged.
        item["input_ids"] = prompt_ids + chat_ids[len_matched:]
        labels = [ignore_index] * len(prompt_ids) + chat_ids[len_matched:]
    else:
        item["input_ids"] = chat_ids
        labels = [ignore_index] * len_matched + chat_ids[len_matched:]
    item["labels"] = labels
    item["attention_mask"] = [1] * len(item["input_ids"])
    for attr in item:
        item[attr] = torch.tensor(item[attr])
    return item

def preprocess_chat_instance_with_refusal(
    tokenizer,
    template_config: Dict[str, Any],
    prompt_msgs: Union[List[str], str],
    response_msgs: Union[List[str], str],
    refusal_msgs: Union[List[str], str],
    max_length: int,
    p: float,  # Probability of immediate refusal
    predict_with_generate: bool = False,
    ignore_index: int = IGNORE_INDEX,
    push_prefix_to_refusal_start: bool = False,
) -> Dict[str, torch.Tensor]:
    """Tokenize a chat instance whose final answer is supervised toward a refusal (JensUn forget target).

    Builds the prompt (system + few-shot context + final question), then the target sequence
    `[prompt] + [optional sampled answer prefix] + [refusal] + [turn terminator]`. With probability `p` the target
    is an immediate refusal (no prefix); otherwise a short prefix of the real answer (`response_msgs[-1]`)
    is sampled and prepended before the refusal — stochastic prefix mixing. When
    `push_prefix_to_refusal_start` is set, the answer-prefix label positions are supervised toward the
    refusal-start token (so the model learns to bail into the refusal at every step) instead of being
    ignored. Loss is computed over the refusal and the trailing turn terminator; the prompt is always masked.
    Only the chat-template path is supported.

    Args:
        tokenizer: tokenizer to apply.
        template_config (Dict[str, Any]): chat-template config from the model YAML.
        prompt_msgs / response_msgs / refusal_msgs: equal-length lists; all but the last pair are
            in-context examples, the last triple is (question, real answer, refusal string).
        max_length (int): maximum sequence length (truncation).
        p (float): probability of an immediate refusal (no sampled answer prefix).
        predict_with_generate (bool, optional): accepted for signature parity with
            `preprocess_chat_instance`; not used here (this path is training-only).
        ignore_index (int): label id for masked (no-loss) positions.
        push_prefix_to_refusal_start (bool, optional): supervise the answer-prefix positions toward the
            refusal start instead of ignoring them.

    Returns:
        Dict[str, torch.Tensor]: 'input_ids', 'labels', and 'attention_mask' tensors.
    """
    assert len(prompt_msgs) == len(response_msgs) == len(refusal_msgs)
    # Standardize inputs to lists
    if isinstance(prompt_msgs, str):
        prompt_msgs = [prompt_msgs]
        response_msgs = [response_msgs]
        refusal_msgs = [refusal_msgs]

    # Extract context (few-shot) and final pair
    # We only apply sampling to the LAST response/refusal in the list
    context_prompts = prompt_msgs[:-1]
    context_responses = response_msgs[:-1]
    final_prompt = prompt_msgs[-1]
    final_response = response_msgs[-1]
    final_refusal = refusal_msgs[-1]

    # Tokenize the "Prompt" part (System + Few Shot + Final User Prompt)
    if template_config["apply_chat_template"]:
        chat_context = []
        system_prompt = template_config.get("system_prompt", None)
        if system_prompt:
            chat_context.append({"role": "system", "content": system_prompt})

        for p_msg, r_msg in zip(context_prompts, context_responses):
            chat_context.append({"role": "user", "content": p_msg})
            chat_context.append({"role": "assistant", "content": r_msg})

        chat_context.append({"role": "user", "content": final_prompt})

        # Get IDs up to the assistant start header
        prompt_ids = _apply_chat_template_ids(tokenizer, chat_context, add_generation_prompt=True)

    else:
        raise RuntimeError("else is not implemented for preprocess_chat_instance_with_refusal")

    # Tokenize the final Response and Refusal separately (no special tokens/headers)
    # We use add_special_tokens=False to get raw content tokens
    response_ids = tokenizer.encode(final_response, add_special_tokens=False)

    # Token-space Sampling Logic
    sampled_prefix_ids = []

    added_prefix = False
    # Only sample if we aren't doing a direct full refusal
    if random.random() > p and len(response_ids) > 0:
        n = len(response_ids)
        # Weights biased toward smaller numbers (shorter prefixes)
        weights = [1 / (i + 1) for i in range(n)]
        k = random.choices(range(1, n + 1), weights=weights, k=1)[0]
        sampled_prefix_ids = response_ids[:k]
        added_prefix = True

    # Encode refusal both ways for push_prefix mode:
    #   - no-space version: "I am unable..." → first token is "I" (sentence-start, no leading space)
    #   - space version:   " I am unable..." → first token is " I" (mid-sentence word with space)
    # The space version is used for the actual input_ids sequence (so the decoded text reads
    # "...Disaster I am unable..." naturally), and as the target for mid-prefix label positions.
    # The no-space version is used only for the very first prefix label position (right after the
    # prompt/question ends), mimicking a sentence that starts fresh with "I".
    refusal_ids_no_space = tokenizer.encode(final_refusal, add_special_tokens=False)
    if added_prefix:
        if not final_refusal.startswith(" "):
            final_refusal = " " + final_refusal
    refusal_ids = tokenizer.encode(final_refusal, add_special_tokens=False)

    # Construct final sequence: [Prompt] + [Optional Response Prefix] + [Refusal] + [terminator]
    # Close the turn on the model's terminator token, so the refusal ends identically to the retain target (see `_end_on_terminator`).
    input_ids = prompt_ids + sampled_prefix_ids + refusal_ids
    input_ids = _end_on_terminator(input_ids, tokenizer)

    # Construct Labels
    # Everything before the refusal_ids starts is ignored
    len_ignored = len(prompt_ids) + len(sampled_prefix_ids)

    # When push_prefix_to_refusal_start is on and a prefix was sampled, replace the
    # IGNORE labels on prefix positions so the model learns to predict the refusal start
    # at every step through the answer prefix.
    # - First position (right after the question): "I" without space (fresh sentence start).
    # - Remaining positions (mid-prefix): " I" with space (natural mid-sentence word).
    if push_prefix_to_refusal_start and len(sampled_prefix_ids) > 0:
        _prefix_labels = (
            [refusal_ids_no_space[0]]
            + [refusal_ids[0]] * (len(sampled_prefix_ids) - 1)
        )
    else:
        _prefix_labels = [ignore_index] * len(sampled_prefix_ids)

    # Loss on the full refusal, including the trailing turn terminator.
    labels = [ignore_index] * len(prompt_ids) + _prefix_labels + input_ids[len_ignored:]

    # Truncation and Tensor conversion
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]

    item = {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
    }

    return item

def preprocess_pretraining_instance(
    tokenizer,
    prefix: str,
    text_content: str,
    max_length: int,
    predict_with_generate: bool = False,
    insert_space: bool = False,
) -> Dict[str, torch.Tensor]:
    """Preprocesses a pretraining instance for training or generation.
    When in training, both the returned `input_ids` and `labels` are over the entire token sequence. `input_ids` has no padding, `labels` assigns `IGNORE_INDEX` to ignore all tokens that we don't compute loss over (i.e. the the 0th index token, all prefix tokens)
    When in generation, `input_ids` are returned only until the prefix portion. The `labels` returned are the same as during training.
    `attention_mask` is always 1 over the full input token sequence.
    Args:
        tokenizer: Tokenizer to apply on text
        prefix (str): The prefix string to prepend to the content.
        text_content (str): The main text content (following the prefix) to be tokenized.
        max_length (int): Maximum text content length after tokenization.
        predict_with_generate (bool, optional): Whether to prepare inputs for generation.
        insert_space (bool, optional): Whether to insert a space between prefix and content.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing 'input_ids', 'labels', and 'attention_mask' tensors for model input.
    """
    full_seq_ids = tokenizer(
        prefix + (" " if insert_space else "") + text_content, add_special_tokens=True
    )["input_ids"]
    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    prefix_len = len(prefix_ids)
    full_seq_ids = full_seq_ids[: prefix_len + max_length]  # manual truncation

    len_matched = prefix_len
    if len_matched == 0:  # never give loss on index 0, when prefix is empty
        len_matched = 1
    labels = [IGNORE_INDEX] * len_matched + full_seq_ids[len_matched:]
    item = {}
    if predict_with_generate:
        item["input_ids"] = prefix_ids
    else:
        item["input_ids"] = full_seq_ids
    item["labels"] = labels
    item["attention_mask"] = [1] * len(item["input_ids"])
    for attr in item:
        item[attr] = torch.tensor(item[attr])
    return item


def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column("index", indexing)
    return dataset
 