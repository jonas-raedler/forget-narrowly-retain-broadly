import hydra
from omegaconf import DictConfig, OmegaConf
from trainer.utils import seed_everything
import os
from data import get_collators, get_data
from model import get_model
from trainer import load_trainer


@hydra.main(version_base=None, config_path="../configs", config_name="unlearn.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    # ------------------------------------------------------------------ #
    # Rank-0 only printing                                                 #
    # With DeepSpeed / torchrun every process runs this script.           #
    # We silence all non-main processes so logs appear exactly once.      #
    # ------------------------------------------------------------------ #
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = (local_rank == 0)

    def _print(*args, **kwargs):
        """Print only on the main process."""
        if is_main:
            print(*args, **kwargs)

    # Silence HuggingFace Transformers / Datasets loggers on non-main ranks
    import logging
    if not is_main:
        for noisy_logger in ("transformers", "datasets", "accelerate", "deepspeed"):
            logging.getLogger(noisy_logger).setLevel(logging.ERROR)
    # httpx is used by HF hub on every rank for dataset/model resolution — silence globally
    logging.getLogger("httpx").setLevel(logging.WARNING)

    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    # Merge model_tokens into template_args so datasets can access per-model
    # constants (e.g. refusal_string) without needing a separate config key.
    if hasattr(model_cfg, "model_tokens") and model_cfg.model_tokens is not None:
        # Convert to a plain (non-struct) dict so merging a new key is allowed
        template_args = OmegaConf.create(OmegaConf.to_container(template_args, resolve=True))
        template_args["model_tokens"] = OmegaConf.to_container(model_cfg.model_tokens, resolve=True)
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)
    # Load Dataset
    data_cfg = cfg.data
    import yaml

    # QAwithRefusalStringDataset adds refusal-string supervision required by JensUn.
    # Other methods should receive plain QADataset for the forget split.
    _JENSUN_TRAINERS = {
        "JensUnPP",       # JensUn++ (canonical)
    }
    if cfg.trainer.get("handler") not in _JENSUN_TRAINERS:
        forget_cfg = data_cfg.get("forget")
        if forget_cfg is not None:
            for ds_name, ds_cfg in forget_cfg.items():
                if ds_cfg.get("handler") == "QAwithRefusalStringDataset":
                    OmegaConf.update(data_cfg, f"forget.{ds_name}.handler", "QADataset")
                    _print(f"[train] trainer={cfg.trainer.get('handler')}: overriding forget dataset handler → QADataset")
        # Non-JensUn methods don't use refusal context — force it off so retain
        # loads the plain context pool instead of the refusal-augmented one.
        retain_cfg = data_cfg.get("retain")
        if retain_cfg is not None:
            for ds_name, ds_cfg in retain_cfg.items():
                if ds_cfg.get("args", {}).get("add_refusal_context"):
                    OmegaConf.update(data_cfg, f"retain.{ds_name}.args.add_refusal_context", False)
                    _print(f"[train] trainer={cfg.trainer.get('handler')}: disabling refusal context in retain dataset")
    if is_main:
        with open('tmp-config-data.yaml', 'w') as f:
            yaml.dump(OmegaConf.to_container(data_cfg), f)
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")
    if is_main:
        with open('trainerargs-cfg.yaml', 'w') as f:
            yaml.dump(OmegaConf.to_container(trainer_cfg), f)

    evaluator = None

    _print(data.get("train", None))

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        data_collator=collator,
        evaluator=evaluator,
        template_args=template_args,
    )

    # save trainer args
    if is_main:
        with open('trainerargs.yaml', 'w') as f:
            _print(trainer_args)
            # trainer_args.to_dict() can contain non-serializable objects
            # (enums, partial funcs, etc.).  Round-trip through JSON to
            # coerce everything to plain Python types that yaml.dump accepts.
            import json
            safe_dict = json.loads(json.dumps(trainer_args.to_dict(), default=str))
            yaml.dump(safe_dict, f)

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)
        _print(f"FINAL_STRING:{trainer_args.output_dir}", flush=True)
        return trainer_args.output_dir

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")
        return trainer_args.output_dir

if __name__ == "__main__":
    main()
