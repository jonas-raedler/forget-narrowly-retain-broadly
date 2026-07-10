# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

from typing import Dict, List, Optional, Union, Any

import os
import time
import logging
from transformers import Trainer
from torch.utils.data import Dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

logger = logging.getLogger(__name__)


class FinetuneTrainer(Trainer):
    def __init__(self, evaluator=None, template_args=None, *args, **kwargs):
        self.evaluator = evaluator
        self.template_args = template_args
        self._train_start_time: Optional[float] = None
        super().__init__(*args, **kwargs)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Intercept every log call to inject ETA / elapsed / speed metrics."""
        if self.state.global_step > 0:
            # Seed the start time on the very first log call
            if self._train_start_time is None:
                self._train_start_time = time.time()

            elapsed = time.time() - self._train_start_time
            steps_done = self.state.global_step
            max_steps  = self.state.max_steps

            if max_steps and max_steps > 0 and steps_done > 0:
                secs_per_step = elapsed / steps_done
                steps_left    = max_steps - steps_done
                eta_secs      = secs_per_step * steps_left

                def _fmt(secs):
                    h = int(secs // 3600)
                    m = int((secs % 3600) // 60)
                    s = int(secs % 60)
                    return f"{h:02d}:{m:02d}:{s:02d}"

                # Numeric values → TensorBoard / wandb
                logs["eta_secs"]     = round(eta_secs, 1)
                logs["elapsed_secs"] = round(elapsed, 1)
                logs["secs_per_step"]= round(secs_per_step, 2)
                logs["pct_done"]     = round(100.0 * steps_done / max_steps, 1)

                if self.accelerator.is_local_main_process:
                    msg = (
                        f"Step {steps_done}/{max_steps} ({logs['pct_done']}%) | "
                        f"Elapsed {_fmt(elapsed)} | "
                        f"ETA {_fmt(eta_secs)} | "
                        f"{secs_per_step:.2f}s/step"
                    )
                    logger.info(msg)
                    print(msg, flush=True)

        super().log(logs, start_time=start_time)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        # Run a custom evaluator and save results
        if self.evaluator:
            if self.accelerator.is_local_main_process:
                eval_metrics = {}
                if self.accelerator.num_processes == 1:
                    run_dir = self._get_output_dir(trial=trial)
                    checkpoint_folder = (
                        f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                    )
                    output_dir = os.path.join(run_dir, checkpoint_folder, "evals")
                    os.makedirs(output_dir, exist_ok=True)
                    eval_args = {
                        "output_dir": output_dir,
                        "template_args": self.template_args,
                        "model": self.model,
                        "tokenizer": self.tokenizer,
                    }
                    eval_metrics = self.evaluator.evaluate(**eval_args)
                    eval_metrics = self.evaluator.summarize(eval_metrics)
                    self.log(eval_metrics)
                else:
                    logger.warning(
                        "Custom evaluator can be run with this Trainer only on a single GPU"
                    )
                return eval_metrics

        if eval_dataset is None:
            return {}
        # Run the default HF Trainer evaluate method when eval dataset is provided
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)