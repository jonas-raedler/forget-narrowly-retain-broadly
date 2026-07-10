from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from omegaconf import DictConfig, open_dict
import os
import torch
import logging

hf_home = os.getenv("HF_HOME", default=None)


logger = logging.getLogger(__name__)


def load_causal_model(model_name_or_path: str, is_local: bool = False, **kwargs):
    """Load a causal LM, handling architectures not registered with AutoModelForCausalLM.

    For Ministral / Mistral3: the HF hub checkpoint is stored as FP8 and must be
    dequantized to BF16 at load time using FineGrainedFP8Config(dequantize=True).
    A locally saved fine-tuned checkpoint is already BF16 — no quant config needed.
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        # Clear quantization_config so Trainer.validate_quantization_for_training doesn't
        # reject BF16 hub checkpoints that carry a residual quantization_config field.
        # model.is_quantized checks model.hf_quantizer (not config), so clear both.
        if not is_local:
            if getattr(getattr(model, 'config', None), 'quantization_config', None) is not None:
                model.config.quantization_config = None
            if getattr(model, 'hf_quantizer', None) is not None:
                model.hf_quantizer = None
        return model
    except Exception as e:
        hf_config = AutoConfig.from_pretrained(model_name_or_path)
        model_type = getattr(hf_config, 'model_type', '')
        if model_type == 'mistral3':
            from transformers import Mistral3ForConditionalGeneration, FineGrainedFP8Config
            # Mistral3ForConditionalGeneration does not accept use_cache as an __init__ kwarg.
            mistral3_kwargs = {k: v for k, v in kwargs.items() if k != 'use_cache'}
            # Local checkpoints (unlearned/finetuned) are already BF16 — no quant config.
            # Hub checkpoints may be FP8 (original) or BF16 (the -BF16 variant).
            # Try without quant config first; fall back to FP8 dequantization only if needed.
            if is_local:
                return Mistral3ForConditionalGeneration.from_pretrained(
                    model_name_or_path, **mistral3_kwargs
                )
            try:
                model = Mistral3ForConditionalGeneration.from_pretrained(
                    model_name_or_path, **mistral3_kwargs
                )
                # The BF16 hub variant stores weights in BF16 but may have a
                # quantization_config in config.json.  Recent transformers versions
                # call validate_quantization_for_training() in Trainer.__init__ and
                # refuse to fine-tune any model that carries a quantization_config.
                # model.is_quantized checks model.hf_quantizer (not config), so clear both.
                if getattr(getattr(model, 'config', None), 'quantization_config', None) is not None:
                    model.config.quantization_config = None
                if getattr(model, 'hf_quantizer', None) is not None:
                    model.hf_quantizer = None
                return model
            except Exception:
                return Mistral3ForConditionalGeneration.from_pretrained(
                    model_name_or_path, quantization_config=FineGrainedFP8Config(dequantize=True), **mistral3_kwargs
                )
        raise


def get_dtype(model_args):
    with open_dict(model_args):
        torch_dtype = model_args.pop("torch_dtype", None)
    if model_args["attn_implementation"] == "flash_attention_2":
        # This check handles https://github.com/Dao-AILab/flash-attention/blob/7153673c1a3c7753c38e4c10ef2c98a02be5f778/flash_attn/flash_attn_triton.py#L820
        # If you want to run at other precisions consider running "training or inference using
        # Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):`
        # decorator" or using an attn_implementation compatible with the precision in the model
        # config.
        assert torch_dtype in ["float16", "bfloat16"], ValueError(
            f"Invalid torch_dtype '{torch_dtype}' for the requested attention "
            f"implementation: 'flash_attention_2'. Supported types are 'float16' "
            f"and 'bfloat16'."
        )
    if torch_dtype == "float16":
        return torch.float16
    elif torch_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def get_model(model_cfg: DictConfig):
    assert model_cfg is not None and model_cfg.model_args is not None, ValueError(
        "Model config not found or model_args absent in configs/model."
    )
    model_args = model_cfg.model_args
    tokenizer_args = model_cfg.tokenizer_args
    torch_dtype = get_dtype(model_args)
    try:
        model_name = model_args.pretrained_model_name_or_path
        model = load_causal_model(
            model_name,
            is_local=os.path.isdir(model_name),
            torch_dtype=torch_dtype,
            **{k: v for k, v in model_args.items() if k != 'pretrained_model_name_or_path'},
            cache_dir=hf_home,
        )
    except Exception as e:
        logger.warning(
            f"Model {model_args.pretrained_model_name_or_path} requested with {model_cfg.model_args}"
        )
        raise ValueError(
            f"Error {e} while fetching model using load_causal_model()."
        )
    tokenizer = get_tokenizer(tokenizer_args)
    return model, tokenizer


def _add_or_replace_eos_token(tokenizer, eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.info("New tokens have been added, make sure `resize_vocab` is True.")


def get_tokenizer(tokenizer_cfg: DictConfig):
    try:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_cfg, cache_dir=hf_home)
    except Exception as e:
        error_message = (
            f"{'--' * 40}\n"
            f"Error {e} fetching tokenizer using AutoTokenizer.\n"
            f"Tokenizer requested from path: {tokenizer_cfg.get('pretrained_model_name_or_path', None)}\n"
            f"Full tokenizer config: {tokenizer_cfg}\n"
            f"{'--' * 40}"
        )
        raise RuntimeError(error_message)

    if tokenizer.eos_token_id is None:
        logger.info("replacing eos_token with <|endoftext|>")
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token as eos token: {}".format(tokenizer.pad_token))

    return tokenizer
