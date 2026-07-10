"""JensUn unlearning trainers.

Every trainer here minimizes ``gamma * forget_loss + alpha * retain_loss``. The retain loss is the
same for all of them — a Jensen-Shannon divergence against a frozen reference model
(``jensun_retain_loss``). They differ in the **forget loss** and the **forget dataset**:

- ``JensUnPP`` (the method): the forget target is the per-example **refusal
  string read from the dataset labels** (a ``QAwithRefusalStringDataset``), applied with
  ``jensun_label_forget_loss`` — first-token weighting and optional gradient-norm balancing
  (``use_grad_norm_scaling``). Trains the model to answer the forget questions with a natural refusal.
- ``JensUnBaseline`` / ``JensUnBaselineHash`` (baselines): the forget target is a **fixed token string** (``'No Idea'``
  / ``'#'``) cycled over the answer positions of a plain ``QADataset``, applied with
  ``jensun_fixedtok_forget_loss``. Trains the model to emit that fixed token instead of the real answer.

Which forget dataset a trainer receives (refusal-context vs plain QA) is enforced in
``src/train.py``: for any trainer outside the JensUn++ family it downgrades a
``QAwithRefusalStringDataset`` forget config to a plain ``QADataset``, so the pairing above
always holds.

Class hierarchy::

    JensUnBase              base: forget + retain, optional grad-norm scaling (label-based forget)
    ├── _JensUnFixedTarget  shared base for the fixed-token baselines
    │   ├── JensUnBaseline      fixed target 'No Idea'
    │   └── JensUnBaselineHash  fixed target '#'
    └── JensUnPP            the method (refusal-string target)
"""

import copy
import torch
from transformers import AutoModelForCausalLM
from typing import Dict

from trainer.unlearn.base import UnlearnTrainer
from trainer.utils import IGNORE_INDEX, jensun_retain_loss, jensun_fixedtok_forget_loss, jensun_label_forget_loss, get_forget_target_tokens


class JensUnBase(UnlearnTrainer):
    def __init__(self, gamma: float = 1.0, alpha: float = 1.0,
                 use_grad_norm_scaling: bool = False,
                 *args, **kwargs):
        """Initializes the JensUnBase trainer for unlearning with JensUn.

        This trainer implements an unlearning strategy that combines a "forget" loss
        (using a multi-token Jensen-Shannon-like divergence) with a "retain" loss
        (a Jensen-Shannon divergence against a frozen reference model that keeps the
        retained knowledge intact).

        Args:
            gamma (float, optional): Weighting factor for the `forget_loss`. Defaults to 1.0.
            alpha (float, optional): Weighting factor for the `retain_loss`. Defaults to 1.0.
            use_grad_norm_scaling (bool, optional): If True, scales `forget_loss` by the
                ratio of logit-level gradient norms (retain / forget) so neither loss
                dominates the combined gradient. Uses raw per-step gnorms (no smoothing).
                When False, `forget_loss` flows through unchanged. Defaults to False.
            *args: Variable length argument list to pass to the parent `UnlearnTrainer` class.
            **kwargs: Arbitrary keyword arguments to pass to the parent `UnlearnTrainer` class.
        """
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.use_grad_norm_scaling = use_grad_norm_scaling
        self.ignore_index = IGNORE_INDEX  # -100, PyTorch CrossEntropyLoss convention
        self.ref_model = self._prepare_ref_model(self.model)

        if self.accelerator.is_local_main_process:
            self.combined = 0
            self.forget_scaled_down = 0
            self.forget_scaled_up = 0

    def _prepare_ref_model(self, model):
        # Deep-copy on CPU to avoid a memory spike.
        # With DeepSpeed ZeRO-3 the .to(device) call would materialise the
        # entire 7B model on a single GPU *before* sharding — instant OOM.
        # Instead, let DeepSpeed handle placement via _prepare_deepspeed.
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        # Freeze every parameter so DeepSpeed does NOT create an optimizer
        # (avoids DeepSpeedCPUAdam crash and partition_numel errors).
        for param in ref_model.parameters():
            param.requires_grad = False
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    def compute_retain_loss(self, model: AutoModelForCausalLM, retain_inputs: Dict[str, Dict[str, torch.Tensor]],
                            return_logits: bool = False):
        """Jensen-Shannon divergence between the trained model and the frozen reference model
        on the retain set, keeping the retained knowledge intact. When `return_logits` is set,
        also returns the model's retain logits (needed for grad-norm scaling)."""
        js_loss, retain_outputs = jensun_retain_loss(model, self.ref_model, retain_inputs, self.ignore_index,
                                                     weight_first_token=0.5)
        if return_logits:
            return js_loss, retain_outputs.logits
        return js_loss

    def compute_loss(self, model: AutoModelForCausalLM, inputs: Dict[str, Dict[str, torch.Tensor]],
                     return_outputs: bool = False, **kwargs):
        """Computes the total loss for the JensUn unlearning strategy.

        Args:
            model (`torch.nn.Module`): The language model being trained/unlearned.
            inputs (`Dict[str, Dict[str, torch.Tensor]]`): A dictionary containing
                input tensors, expected to have two main keys:
                - "forget": A dictionary with "input_ids", "attention_mask", and "labels"
                            for the data to be forgotten.
                - "retain": A dictionary with "input_ids", "attention_mask", and "labels"
                            for the data to be retained.
            return_outputs (bool, optional): Whether to return the model's outputs
                from the `forget_loss` calculation along with the total loss. Defaults to False.

        Returns:
            The total computed `loss` (a scalar `torch.Tensor`). If `return_outputs` is True,
            returns a tuple containing the `loss` and the `forget_outputs` from the
            `jensun_label_forget_loss` calculation.
        """
        forget_inputs = inputs["forget"]
        forget_loss, forget_outputs = jensun_label_forget_loss(
            model, forget_inputs, self.ignore_index,
            weights_first_token=0.5,
        )

        # Keep logits alive for the grad-norm branch; free outputs otherwise to
        # prevent two full sets of model activations from being on GPU at once.
        forget_logits_ref = forget_outputs.logits if self.use_grad_norm_scaling else None
        if not return_outputs:
            del forget_outputs
            forget_outputs = None

        retain_inputs = inputs["retain"]
        if self.use_grad_norm_scaling:
            retain_loss, retain_logits_ref = self.compute_retain_loss(
                model=model, retain_inputs=retain_inputs, return_logits=True
            )
        else:
            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
            retain_logits_ref = None

        # Reduce across GPUs for logging and the saturation check.
        mean_forget_loss = self.accelerator.reduce(forget_loss.detach(), reduction="mean")
        mean_retain_loss = self.accelerator.reduce(retain_loss.detach(), reduction="mean")

        _EPS = 1e-5                # precision guard against div-by-zero
        _SCALE_UP_CAP = 1_000_000  # safety cap on the scale-up factor

        scaled_forget_loss = forget_loss
        applied_scale = 1.0
        forget_gnorm = retain_gnorm = None  # only populated in the gnorm branch

        if self.use_grad_norm_scaling:
            # Gradient-norm branch: differentiate each loss w.r.t. its own logits.
            # retain_graph=True keeps the graph alive for the trainer's subsequent backward().
            if mean_forget_loss.item() > _EPS and mean_retain_loss.item() > _EPS:
                g_forget = torch.autograd.grad(forget_loss, forget_logits_ref, retain_graph=True)[0]
                g_retain = torch.autograd.grad(retain_loss, retain_logits_ref, retain_graph=True)[0]
                _dev = g_forget.device
                forget_gnorm = self.accelerator.reduce(g_forget.norm().to(_dev), reduction="mean").item()
                retain_gnorm = self.accelerator.reduce(g_retain.norm().to(_dev), reduction="mean").item()
                del g_forget, g_retain

                scale, tag = self._grad_norm_scale(forget_gnorm, retain_gnorm, _EPS, _SCALE_UP_CAP)
                if tag != "combined":
                    scaled_forget_loss = forget_loss * scale
                    applied_scale = scale
                self.update_loss_count(tag, scale if tag != "combined" else None)
            else:
                # Saturated — skip the autograd cost; forget_loss flows through.
                self.update_loss_count("combined")
            del forget_logits_ref, retain_logits_ref
        else:
            # No scaling — forget_loss passes through unchanged.
            self.update_loss_count("combined")

        loss = self.gamma * scaled_forget_loss + self.alpha * retain_loss

        if self.accelerator.is_local_main_process:
            log_dict = {
                "retain_loss": mean_retain_loss.item(),
                "forget_loss": mean_forget_loss.item(),
                "forget_scale": applied_scale,
            }
            if forget_gnorm is not None:
                log_dict["forget_gnorm"] = forget_gnorm
                log_dict["retain_gnorm"] = retain_gnorm
            self.log(log_dict)

        return (loss, forget_outputs) if return_outputs else loss

    @staticmethod
    def _grad_norm_scale(forget_gnorm, retain_gnorm, eps: float = 1e-5, scale_up_cap: float = 1_000_000):
        """Return ``(scale, tag)`` for the forget loss given the two logit-gradient norms.

        Balances the forget and retain gradients so neither dominates the update:
          * forget dominates (``forget_gnorm > retain_gnorm``) → scale DOWN, clamped to ``[0, 1]``.
          * retain dominates (``forget_gnorm < retain_gnorm``) → scale UP, capped at ``scale_up_cap``.
          * equal gnorms → no scaling (``1.0``).
        ``eps`` guards the division against a zero forget gnorm.
        """
        if forget_gnorm > retain_gnorm:
            return max(min(retain_gnorm / (forget_gnorm + eps), 1.0), 0.0), "scaled_down"
        if forget_gnorm < retain_gnorm:
            return min(retain_gnorm / (forget_gnorm + eps), scale_up_cap), "scaled_up"
        return 1.0, "combined"

    def update_loss_count(self, loss_type, scale=None):
        """Tally how the grad-norm scaling treated the forget loss this step.

        Keeps running counts (on the main process) of how often the forget loss was scaled down,
        scaled up, or left unchanged ("combined"), and prints each update. The per-step scale is
        also logged in `compute_loss` as `forget_scale` (1.0 = combined, <1 = down, >1 = up).
        """
        if self.accelerator.is_local_main_process:
            if loss_type == "scaled_down":
                self.forget_scaled_down += 1
                print(f"Forget scaled down loss used. Count:{self.forget_scaled_down}, Scale: {scale}")
            elif loss_type == "scaled_up":
                self.forget_scaled_up += 1
                print(f"Forget scaled up loss used. Count:{self.forget_scaled_up}, Scale: {scale}")
            elif loss_type == "combined":
                self.combined += 1
                print("Combined loss used. Count:", self.combined)
            else:
                raise RuntimeError("Unknown loss type.")


class _JensUnFixedTarget(JensUnBase):
    """Shared logic for the baseline JensUn variants whose forget target is a fixed token string.

    The target string is looked up per subclass name in `_MODEL_FORGET_TARGET_STR` (via
    `get_forget_target_tokens`) and cycled over the answer positions by `jensun_fixedtok_forget_loss`.
    Subclasses differ only in that
    target token; they read plain QA datasets (no refusal context).
    """

    def __init__(self, gamma=1.0, alpha=1.0, *args, **kwargs):
        super().__init__(gamma=gamma, alpha=alpha, *args, **kwargs)
        self.tok_id = get_forget_target_tokens(self.data_collator.tokenizer, type(self).__name__)

    def compute_retain_loss(self, model, retain_inputs, **kwargs):
        # Same JensUn retain loss as the base class, but without first-token up-weighting.
        js_loss, _ = jensun_retain_loss(model, self.ref_model, retain_inputs,
                                        self.ignore_index, weight_first_token=None)
        return js_loss

    def compute_loss(self, model: AutoModelForCausalLM, inputs: Dict[str, Dict[str, torch.Tensor]],
                     return_outputs: bool = False, **kwargs):
        forget_inputs = inputs["forget"]
        forget_loss, forget_outputs = jensun_fixedtok_forget_loss(model, forget_inputs, self.tok_id, self.ignore_index)

        retain_inputs = inputs["retain"]
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        if self.accelerator.is_local_main_process:
            self.log({
                "retain_loss": retain_loss.item(),
                "forget_loss": forget_loss.item(),
            })

        return (loss, forget_outputs) if return_outputs else loss


class JensUnBaseline(_JensUnFixedTarget):
    """Baseline JensUn: fixed target tokens 'No Idea' cycled over the answer positions — the
    fixed-token baseline forget loss. Uses plain QA datasets (no refusal context).
    """


class JensUnBaselineHash(_JensUnFixedTarget):
    """Baseline JensUn variant using '#' as the fixed forget target token."""


class JensUnPP(JensUnBase):
    """JensUn++ — the canonical method (paper: "Forget Narrowly, Retain Broadly").
    Reads the refusal answer from the dataset labels (baked into the refusal-context
    dataset) — no hardcoded target tokens. Uses refusal-context datasets.    """
