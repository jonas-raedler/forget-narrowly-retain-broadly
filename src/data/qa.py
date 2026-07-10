"""Question-answer datasets for unlearning training and evaluation.

Each dataset loads a HuggingFace QA split and tokenizes every (question, answer) pair into a chat
example (`input_ids` / `labels` / `attention_mask` / `index`). `QADataset` is the general loader
(retain data and baseline forget data); `QAwithRefusalStringDataset` adds a model-specific refusal
target used by the JensUn++ forget loss; `QAwithIdkDataset` pairs each question with a random
"I don't know" answer.
"""

import json

import torch
from torch.utils.data import Dataset

from data.utils import load_hf_dataset, preprocess_chat_instance, add_dataset_index, preprocess_chat_instance_with_refusal, IGNORE_INDEX

import random



def _get_model_tokens(template_args: dict, field: str):
    """Retrieve a required per-model token config value from template_args.

    template_args is expected to contain a 'model_tokens' sub-dict (populated
    from the model YAML's model_tokens section).  Raises a clear error if the
    field is missing so unsupported models fail fast.
    """
    model_tokens = template_args.get("model_tokens")
    if model_tokens is None:
        raise NotImplementedError(
            "template_args does not contain 'model_tokens'. "
            "Add a 'model_tokens' section to your model YAML config "
            "(see configs/model/Llama-3.2-3B-Instruct.yaml for an example)."
        )
    value = model_tokens.get(field) if hasattr(model_tokens, "get") else getattr(model_tokens, field, None)
    if value is None:
        raise NotImplementedError(
            f"model_tokens.{field} is not defined in your model YAML config. "
            f"Add it under the 'model_tokens' section."
        )
    return value



class QADataset(Dataset):
    """Load a HuggingFace QA dataset and tokenize each (question, answer) into a chat example.

    Supports filtering to a single `topic`, excluding rows by label prefix, row slicing
    (`skip_first_n_rows` / `max_rows` / `skip_range_*`), and optional few-shot context examples
    prepended to each prompt. Used for retain data and for baseline (non-JensUn++) forget data.
    """

    # Placeholder used in context-with-refusal JSON files.
    # At runtime it is replaced with the model-specific refusal string.
    _CONTEXT_REFUSAL_PLACEHOLDER = "I am unable to verify this information."

    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key="question",
        answer_key="answer",
        few_shot_dataset_hf_args=None,
        max_length=512,
        predict_with_generate=False,
        add_context=False,
        add_refusal_context=False,
        context_path=None,
        forget_direct_path=None,
        exclude_label_prefixes=None,
        skip_first_n_rows=0,
        max_rows=0,
        skip_range_start=-1,
        skip_range_end=-1,
        push_prefix_to_refusal_start=False,  # only used by QAwithRefusalStringDataset; accepted here for config compatibility
        filter_topic=None,
    ):
        super(QADataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        import os
        from omegaconf import OmegaConf
        _is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0
        hf_args_parsed = OmegaConf.to_container(hf_args, resolve=True)
        if _is_main:
            print("HF ARGS", hf_args_parsed)
        self.data = load_hf_dataset(**hf_args, _is_main=_is_main)
        if filter_topic:
            # Filter unified multi-topic datasets (apeleg/SUITE) down to a single topic.
            # No-op for per-topic repos that don't carry the `topic` column.
            if "topic" not in self.data.column_names:
                raise ValueError(
                    f"filter_topic={filter_topic!r} was set but dataset {hf_args_parsed.get('path')!r} "
                    f"has no 'topic' column. Either point at the unified repo or remove filter_topic."
                )
            self.data = self.data.filter(lambda x: x["topic"] == filter_topic)
            if _is_main:
                print(f"filter_topic={filter_topic!r}: dataset filtered to {len(self.data)} rows")
        if exclude_label_prefixes:
            prefixes = list(exclude_label_prefixes)
            self.data = self.data.filter(
                lambda x: not any(x["label"].startswith(p) for p in prefixes)
            )
            if _is_main:
                print(f"exclude_label_prefixes={prefixes}: dataset filtered to {len(self.data)} rows")
        if skip_first_n_rows:
            self.data = self.data.select(range(skip_first_n_rows, len(self.data)))
            if _is_main:
                print(f"skip_first_n_rows={skip_first_n_rows}: dataset trimmed to {len(self.data)} rows")
        if max_rows > 0:
            self.data = self.data.select(range(min(max_rows, len(self.data))))
            if _is_main:
                print(f"max_rows={max_rows}: dataset truncated to {len(self.data)} rows")
        if skip_range_start >= 0 and skip_range_end > skip_range_start:
            indices = list(range(skip_range_start)) + list(range(skip_range_end, len(self.data)))
            self.data = self.data.select(indices)
            if _is_main:
                print(f"skip_range=[{skip_range_start},{skip_range_end}): dataset trimmed to {len(self.data)} rows")
        if _is_main:
            print("DATA", self.data)
            print(self.data[0])
            print(self.data[-1])
        self.data = add_dataset_index(self.data)
        self.fs_data = None
        if few_shot_dataset_hf_args is not None:
            raw_data = load_hf_dataset(**few_shot_dataset_hf_args)
            self.fs_data = {}
            self.fs_data[question_key] = raw_data[question_key]
            self.fs_data[answer_key] = raw_data[answer_key]
        self.template_args = template_args
        self.question_key = question_key
        self.answer_key = answer_key
        self.predict_with_generate = predict_with_generate

        # IGNORE_INDEX is a fixed PyTorch convention (-100), not model-specific
        self.ignore_index = IGNORE_INDEX

        self.CONTEXT = add_context
        self.REFUSAL_CONTEXT = add_refusal_context

        _default_context_path = "./dataset/few_shot_context/short_context.json"

        # When refusal context is active, load the model-specific refusal string first
        # so we can use it when building the context pool below.
        self._context_refusal_string = None
        if self.REFUSAL_CONTEXT:
            self._context_refusal_string = _get_model_tokens(template_args, "refusal_string")

        if self.CONTEXT and self.REFUSAL_CONTEXT:
            assert forget_direct_path is not None, (
                "add_context=True and add_refusal_context=True require forget_direct_path "
                "to be set in the dataset YAML config (path to the topic's forget_direct.json)."
            )
            with open(context_path or _default_context_path, "r") as f:
                standard_context = json.load(f)
            with open(forget_direct_path, "r") as f:
                forget_direct = json.load(f)
            refusal_entries = [
                {"prompt": entry["question"], "response": self._context_refusal_string}
                for entry in forget_direct
            ]
            self.static_context_pool = standard_context + refusal_entries
        elif self.CONTEXT:
            with open(context_path or _default_context_path, "r") as f:
                self.static_context_pool = json.load(f)
        else:
            self.static_context_pool = None

    def __len__(self):
        return len(self.data)

    def _process_sample(self, question, answer, index=-1):
        # 1. Initialize lists based on CONTEXT
        selected_examples = []
        num_max_context = 2
        if self.CONTEXT:
            # random.sample handles the pool size vs requested count automatically with min()
            count = random.randint(0, min(num_max_context, len(self.static_context_pool)))
            selected_examples = random.sample(self.static_context_pool, count)

        # 2. Extract static components, substituting the refusal placeholder when active.
        s_questions = [ex["prompt"] for ex in selected_examples]
        s_answers = [
            self._context_refusal_string if (
                self.REFUSAL_CONTEXT and self._context_refusal_string
                and ex["response"] == self._CONTEXT_REFUSAL_PLACEHOLDER
            ) else ex["response"]
            for ex in selected_examples
        ]

        # 3. Handle Few-Shot data (defaulting to empty lists if None)
        fs = self.fs_data or {
            self.question_key: [],
            self.answer_key: [],
        }

        # 4. Final Assembly (One clean path)
        prompt_msgs = s_questions + fs[self.question_key] + [question]
        response_msgs = s_answers + fs[self.answer_key] + [answer]

        tokenized_data = preprocess_chat_instance(
            self.tokenizer,
            self.template_args,
            prompt_msgs,
            response_msgs,
            self.max_length,
            self.predict_with_generate,
            ignore_index=self.ignore_index,
        )

        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
            "index": index,
        }
        return item_dct

    def __getitem__(self, idx):
        question = self.data[idx][self.question_key]
        answer = self.data[idx][self.answer_key]
        index = self.data[idx]["index"]
        if isinstance(answer, str):
            item = self._process_sample(question=question, answer=answer, index=index)
        elif isinstance(answer, list):
            item = {}
            for i, ans in enumerate(answer):
                sample_item = self._process_sample(
                    question=question, answer=ans, index=index
                )
                item[i] = sample_item
        else:
            print(idx)
            print(question)
            print(answer)
            print(type(answer))
            raise NotImplementedError("answer format not found")
        return item


class QAwithRefusalStringDataset(QADataset):
    """QA dataset that also carries a model-specific refusal answer, for the JensUn++ forget loss.

    Each example is tokenized against the model's `refusal_string` (model YAML
    `model_tokens.refusal_string`) via `preprocess_chat_instance_with_refusal`: with probability
    `self.p` the target is an immediate refusal, otherwise a short sampled prefix of the real answer
    precedes the refusal (stochastic prefix mixing). When `push_prefix_to_refusal_start` is set, the
    model is trained to start the refusal at every prefix position.
    """

    def __init__(self, hf_args, template_args, tokenizer, question_key="question", answer_key="answer",
                 few_shot_dataset_hf_args=None, max_length=512, predict_with_generate=False, add_context=False,
                 context_path=None, exclude_label_prefixes=None, skip_first_n_rows=0,
                 max_rows=0, skip_range_start=-1, skip_range_end=-1,
                 push_prefix_to_refusal_start=False, filter_topic=None):
        super().__init__(hf_args, template_args, tokenizer, question_key, answer_key, few_shot_dataset_hf_args,
                         max_length, predict_with_generate, add_context,
                         context_path=context_path,
                         exclude_label_prefixes=exclude_label_prefixes, skip_first_n_rows=skip_first_n_rows,
                         max_rows=max_rows, skip_range_start=skip_range_start, skip_range_end=skip_range_end,
                         filter_topic=filter_topic)
        self.p = 0.5  # probability of an immediate refusal; otherwise a sampled answer prefix precedes it
        self.refusal_key = "refusal"
        # Refusal string is model-specific: defined in model YAML under model_tokens.refusal_string
        self.refusal_string = _get_model_tokens(template_args, "refusal_string")
        self.push_prefix_to_refusal_start = push_prefix_to_refusal_start

    def _process_sample(self, question, answer, index=-1):
        # 1. Initialize lists based on CONTEXT
        selected_examples = []
        num_max_context = 2
        if self.CONTEXT:
            # random.sample handles the pool size vs requested count automatically with min()
            count = random.randint(0, min(num_max_context, len(self.static_context_pool)))
            selected_examples = random.sample(self.static_context_pool, count)

        # 2. Extract static components, swapping the hardcoded placeholder with
        #    the model-specific refusal string so each model sees its own phrasing.
        s_questions = [ex["prompt"] for ex in selected_examples]
        s_answers = [
            self.refusal_string if ex["response"] == self._CONTEXT_REFUSAL_PLACEHOLDER
            else ex["response"]
            for ex in selected_examples
        ]

        # 3. Handle Few-Shot data (defaulting to empty lists if None)
        fs = self.fs_data or {
            self.question_key: [],
            self.answer_key: [],
            self.refusal_key: []
        }

        # Refusal string is the single model-specific value from template_args.model_tokens.refusal_string
        refusal = self.refusal_string

        # 4. Final Assembly (One clean path)
        prompt_msgs = s_questions + fs[self.question_key] + [question]
        response_msgs = s_answers + fs[self.answer_key] + [answer]
        refusal_msgs = s_answers + fs[self.refusal_key] + [refusal]

        tokenized_data = preprocess_chat_instance_with_refusal(
            self.tokenizer,
            self.template_args,
            prompt_msgs,
            response_msgs,
            refusal_msgs,
            self.max_length,
            self.p,
            self.predict_with_generate,
            ignore_index=self.ignore_index,
            push_prefix_to_refusal_start=self.push_prefix_to_refusal_start,
        )

        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
            "index": index,
        }
        return item_dct


class QAwithIdkDataset(QADataset):
    """QA dataset that returns each example together with a variant answered by a random "I don't know" line.

    Candidate refusals are read from `idk_path` (one per line). `__getitem__` returns
    `{"original": <real>, "alternate": <idk>}`, or just the alternate when `return_original=False`.
    """

    def __init__(self, idk_path, return_original=True, *args, **kwargs):
        self.idk_path = idk_path
        self.return_original = return_original
        self.idk_responses = open(self.idk_path, "r").readlines()
        super().__init__(*args, **kwargs)

    def item_with_idk(self, question):
        rand_pos = torch.randint(0, len(self.idk_responses), (1,)).item()
        idk_response = self.idk_responses[rand_pos].strip()
        idk_item = self._process_sample(question=question, answer=idk_response)
        return idk_item

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        question = self.data[idx][self.question_key]
        if isinstance(item, dict):
            return_item = {"original": item}
            idk_item = self.item_with_idk(question)
            return_item["alternate"] = idk_item
            # return_item = [item, idk_item]
        elif isinstance(item, list) or isinstance(item, tuple):
            return_item = []
            for sample_item in item:
                return_item = {"original": sample_item}
                idk_item = self.item_with_idk(question)
                return_item["alternate"] = idk_item
                # return_item.append([sample_item, idk_item])
        return return_item if self.return_original else return_item["alternate"]
