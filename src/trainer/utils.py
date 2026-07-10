import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# IGNORE_INDEX is a PyTorch convention (-100) used by CrossEntropyLoss to
# skip masked label positions.  It is the same for every model family.
# ---------------------------------------------------------------------------
IGNORE_INDEX = -100

# ---------------------------------------------------------------------------
# Per-model token tables
#
# The unlearning target tokens differ per model family and per trainer. They are
# stored here as strings and encoded to IDs at runtime via
# get_forget_target_tokens(tokenizer, class_name), so the IDs always match the
# loaded tokenizer. Only the keys of _MODEL_END_TOKEN_STR are read — _model_family()
# matches them against name_or_path to detect the model family; the token values
# themselves are not used.
# ---------------------------------------------------------------------------

# End-of-turn token string per model family (substring matched against name_or_path)
_MODEL_END_TOKEN_STR: Dict[str, str] = {
    "Llama": "<|eot_id|>",
    "Phi": "<|end|>",
    "Ministral": "</s>",
    "Qwen": "<|im_end|>",
}

# Unlearning target-token *strings* per (model-family, trainer-class).
# Values are the human-readable strings; get_forget_target_tokens() encodes them at runtime.
_MODEL_FORGET_TARGET_STR: Dict[str, Dict[str, str]] = {
    "Llama": {
        "JensUnBaseline": "No Idea",
        "JensUnBaselineHash": "#",
    },
    "Phi": {
        "JensUnBaseline": "No Idea",
        "JensUnBaselineHash": "#",
    },
    "Ministral": {
        "JensUnBaseline": "No Idea",
        "JensUnBaselineHash": "#",
    },
    "Qwen": {
        "JensUnBaseline": "No Idea",
        "JensUnBaselineHash": "#",
    },
}


def _model_family(tokenizer) -> str:
    """Return the model-family key (e.g. 'Llama', 'Qwen', 'Phi') from name_or_path."""
    name = getattr(tokenizer, "name_or_path", "") or ""
    for family in _MODEL_END_TOKEN_STR:
        if family in name:
            return family
    raise NotImplementedError(
        f"Model '{name}' is not in the supported list "
        f"({list(_MODEL_END_TOKEN_STR.keys())}). "
        f"Add an entry to _MODEL_END_TOKEN_STR and _MODEL_FORGET_TARGET_STR in trainer/utils.py."
    )


def get_forget_target_tokens(tokenizer, class_name: str) -> List[int]:
    """Return the unlearning target token IDs for (model family, trainer class).

    The token string is looked up from _MODEL_FORGET_TARGET_STR and then encoded by
    the tokenizer so the IDs are always correct for the loaded model.

    Args:
        tokenizer: The model's tokenizer (used for name_or_path lookup + encoding).
        class_name: The trainer subclass name (e.g. 'JensUnBaseline', 'JensUnBaselineHash', …).

    Returns:
        List of token IDs (may be more than one for multi-token targets).

    Raises:
        NotImplementedError: model family not registered.
        KeyError: class_name not registered for this model family.
    """
    family = _model_family(tokenizer)
    family_map = _MODEL_FORGET_TARGET_STR.get(family)
    if family_map is None:
        raise NotImplementedError(
            f"No forget-target entries for model family '{family}'. "
            f"Add them to _MODEL_FORGET_TARGET_STR in trainer/utils.py."
        )
    if class_name not in family_map:
        raise KeyError(
            f"Trainer class '{class_name}' has no forget-target entry for model family '{family}'. "
            f"Add it to _MODEL_FORGET_TARGET_STR['{family}'] in trainer/utils.py."
        )
    tok_str = family_map[class_name]
    tok_ids = tokenizer.encode(tok_str, add_special_tokens=False)
    if not tok_ids:
        raise ValueError(
            f"forget-target string {tok_str!r} for '{family}/{class_name}' "
            f"encoded to an empty list — check the tokenizer."
        )
    return tok_ids


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_kl_divergence(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(**inputs)

    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = ref_probs.view(-1, ref_outputs.logits.shape[-1])

    outputs = model(**inputs)
    current_probs = F.log_softmax(outputs.logits, dim=-1)
    current_probs = current_probs.view(-1, outputs.logits.shape[-1])

    # minimum KL divergence
    return nn.functional.kl_div(
        current_probs, ref_probs, reduction="batchmean", log_target=True
    ), outputs


def jensun_retain_loss(model, target_model, inputs, ignore_index, weight_first_token):
    """JensUn retain loss: Jensen-Shannon divergence between the model and a frozen reference model.

    Evaluated only at label (answer) positions on the retain set. Keeps the model's next-token
    distribution close to the reference where knowledge must be preserved, balancing the forget
    loss. `weight_first_token` gives the first answer token of each sample a fixed share of the
    loss; the remaining weight is spread over the rest.
    """
    outputs, shift_labels, shift_logits = get_shift_labels_and_logits(inputs, model)
    model_probs = F.softmax(shift_logits, dim=-1)
    del shift_logits  # free (B, T, V) tensor

    mask = (shift_labels != ignore_index).float()

    with torch.no_grad():
        ref_outputs = target_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
    ref_logits = ref_outputs.logits
    del ref_outputs  # free the full outputs object
    shift_ref_logits = ref_logits[..., :-1, :].contiguous()  # to align with the labels
    del ref_logits  # free the unshifted logits
    ref_probs = F.softmax(shift_ref_logits, dim=-1)
    del shift_ref_logits  # free another (B, T, V) tensor

    js_div = compute_jensen_loss(ref_probs, model_probs, mask, weight_first_token=weight_first_token)
    del ref_probs, model_probs  # free before returning

    return js_div, outputs


def compute_batch_nll(model, inputs) -> tuple[float, torch.Tensor]:
    # get the sum loss for each sequence in a batch
    # NOTE: not same as model(**inputs).loss but has sum loss for each seq in a batch
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss, outputs


def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")

    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
        win_log_ratio = -(win_loss - win_ref_loss)

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss)

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)


def compute_undial_loss(model, ref_model, inputs, beta):
    # Forward pass on the student (trainable) model
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    # Forward pass on the teacher model (no grad)
    with torch.no_grad():
        teacher_logits = ref_model(**inputs).logits
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

    # Build the mask that identifies the tokens need to be unlearned
    mask = torch.zeros_like(shift_teacher_logits)
    batch_idx = torch.arange(mask.shape[0]).view(-1, 1, 1)
    seq_idx = torch.arange(mask.shape[1]).view(1, -1, 1)
    mask[batch_idx, seq_idx, shift_labels.unsqueeze(-1)] = 1.0

    # Adjust teacher logits: subtract di_strength on the correct token
    pre_softmax = shift_teacher_logits - mask * beta
    soft_label = F.softmax(pre_softmax, dim=-1)

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        soft_label.view(-1, soft_label.size(-1)),
    )
    return loss.mean(), outputs

def compute_wga_loss(model, inputs, beta):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_ce = ((-lm_loss).exp().detach()) ** beta
    forget_loss = -(weight_ce * lm_loss)[shift_labels.view(-1) != -100].mean()
    return forget_loss, outputs

def compute_satimp_loss(model, inputs, beta1, beta2):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_sat = ((-lm_loss).exp().detach()) ** beta1
    weight_imp = (1 - (-lm_loss).exp().detach()) ** beta2
    forget_loss = -((weight_sat * weight_imp) * lm_loss)[
        shift_labels.view(-1) != -100
    ].mean()
    return forget_loss, outputs

def jensun_fixedtok_forget_loss(model, forget_inputs, tokids, ignore_index) -> Tuple[torch.Tensor, Any]:
    """Baseline JensUn forget loss using hardcoded target token IDs.

    At each label-masked position, cyclically repeats `tokids` to build a peaked
    distribution, then computes JSD between the model's output and that distribution.
    Used by the fixed-target baselines (JensUnBaseline / JensUnBaselineHash) — target tokens are
    fixed regardless of dataset content.
    """
    outputs, shift_labels, shift_logits = get_shift_labels_and_logits(forget_inputs, model)

    device = shift_logits.device
    batch_size, seq_len, vocab_size = shift_logits.size()

    mask = (shift_labels != ignore_index).float()

    if not isinstance(tokids, torch.Tensor):
        pattern = torch.tensor(tokids, device=device, dtype=torch.long)
    else:
        pattern = tokids.to(device)
    pattern_len = len(pattern)

    peaked_dist = torch.zeros_like(shift_logits)
    for i in range(batch_size):
        valid_indices = torch.nonzero(mask[i], as_tuple=True)[0]
        num_slots = len(valid_indices)
        if num_slots == 0:
            continue
        repeats = math.ceil(num_slots / pattern_len)
        cyclic_targets = pattern.repeat(repeats)[:num_slots]
        seq_target_probs = torch.zeros(num_slots, vocab_size, device=device, dtype=peaked_dist.dtype)
        seq_target_probs.scatter_(1, cyclic_targets.unsqueeze(1), 1.0)
        peaked_dist[i, valid_indices, :] = seq_target_probs

    model_probs = F.softmax(shift_logits, dim=-1)
    del shift_logits

    js_div = compute_jensen_loss(peaked_dist, model_probs, mask, weight_first_token=None)
    del peaked_dist, model_probs

    return js_div, outputs


def jensun_label_forget_loss(model, forget_inputs, ignore_index, weights_first_token) -> Tuple[torch.Tensor, Any]:
    """Canonical JensUn forget loss: train the model to emit the refusal answer instead of the real one.

    Builds a one-hot target distribution peaked on each sample's own label tokens — in JensUn++
    these are the refusal-string tokens — and returns the Jensen-Shannon divergence between the
    model's next-token distribution and that target at every label position. Minimizing it moves the
    model's output toward the refusal. Unlike `jensun_fixedtok_forget_loss`, the target tokens are read from
    each sample's labels rather than a fixed token list. `weights_first_token` sets the share of the
    loss assigned to the first answer token of each sample (one sample = one forget question with
    its refusal answer); the remaining weight is spread over that sample's other answer tokens.
    """
    outputs, shift_labels, shift_logits = get_shift_labels_and_logits(forget_inputs, model)
    probs = torch.softmax(shift_logits, dim=-1)
    del shift_logits  # free (B, T, V) tensor

    # get the indices of the refusal tokens (where the loss should be applied)
    mask = (shift_labels != ignore_index).float()

    peaked_dist = torch.zeros_like(probs)
    # Get indices where mask == 1
    idx_a, idx_b = torch.where(mask == 1)
    peaked_dist[idx_a, idx_b, shift_labels[idx_a, idx_b]] = 1

    js_div = compute_jensen_loss(peaked_dist, probs, mask, weight_first_token=weights_first_token)
    del peaked_dist, probs  # free (B, T, V) tensors

    return js_div, outputs


def get_shift_labels_and_logits(inputs, model) -> tuple[Any, Any, Any]:
    """Run the model and return (outputs, shifted labels, shifted logits) for next-token prediction.

    Shifts by one position so the logits at step t line up with the label at step t+1 (standard
    causal-LM teacher forcing). Shared by the JensUn forget and retain losses.
    """
    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    logits = outputs.logits

    labels = inputs["labels"]

    # SHIFT for Causal Prediction
    # Input @ t predicts Target @ t+1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return outputs, shift_labels, shift_logits


def compute_jensen_loss(
        target_probs: torch.Tensor,
        model_probs: torch.Tensor,
        mask: torch.Tensor,
        epsilon: float = 1e-10,
        weight_first_token=0.5
) -> torch.Tensor:
    """Compute a weighted Jensen-Shannon divergence between target_probs and model_probs.

    KL terms are computed **only on valid (unmasked) token positions**.
    This avoids the numerical jitter caused by clamping all-zero rows at masked
    positions to epsilon and then summing log(epsilon/epsilon) = 0 terms across
    the full vocabulary for every masked token.

    Args:
        target_probs: (B, T, V) – e.g. peaked distribution or reference probs.
        model_probs:  (B, T, V) – current model softmax probabilities.
        mask:         (B, T)   – 1.0 for valid tokens, 0.0 for padding/ignore.
        epsilon:      small constant for numerical stability.
        weight_first_token: weight assigned to the first valid token per sample;
                            remaining weight is spread uniformly over the rest.
                            Pass None to use a simple unweighted mean.
    """
    # ------------------------------------------------------------------ #
    # 1.  Gather only the valid positions → shape (N, V)                  #
    # ------------------------------------------------------------------ #
    valid_idx = (mask == 1)  # (B, T) bool
    n_valid = valid_idx.sum()

    if n_valid == 0:
        return torch.zeros((), device=model_probs.device, dtype=model_probs.dtype)

    # flat views over valid tokens only
    m_valid = model_probs[valid_idx]  # (N, V)
    t_valid = target_probs[valid_idx]  # (N, V)
    # Free the full (B, T, V) inputs — only the (N, V) slices are needed
    del model_probs, target_probs
    mid = 0.5 * (m_valid + t_valid)  # (N, V), can be 0 where both are 0

    # ------------------------------------------------------------------ #
    # 2.  JS divergence on the (N, V) slice                               #
    # ------------------------------------------------------------------ #
    log_mid = torch.log(torch.clamp(mid, min=epsilon))  # (N, V) safe log of midpoint
    del mid  # free (N, V)

    # Fuse log computation into the KL term directly to avoid extra (N,V) buffers
    kl_pm = torch.sum(m_valid * (torch.log(torch.clamp(m_valid, min=epsilon)) - log_mid), dim=-1)  # (N,)
    del m_valid  # free (N, V)
    kl_qm = torch.sum(t_valid * (torch.log(torch.clamp(t_valid, min=epsilon)) - log_mid), dim=-1)  # (N,)
    del t_valid, log_mid  # free remaining (N, V) tensors
    js = 0.5 * (kl_pm + kl_qm)  # (N,)
    js = torch.clamp(js, min=0.0)  # guard against epsilon-induced numerical negatives

    # ------------------------------------------------------------------ #
    # 3.  Reduce back to a scalar with the standard logic #
    # ------------------------------------------------------------------ #
    if weight_first_token is None:
        return js.mean()

    # ---- weighted average: first token gets weight_first_token, rest share the remainder ----
    B = mask.shape[0]
    valid_counts = mask.sum(dim=1).long()  # (B,)  number of valid tokens per sample

    # sample_ids[i] = which batch sample the i-th valid token belongs to
    sample_ids = torch.where(valid_idx)[0]  # (N,)

    # Vectorised per-token weight computation — no Python loop over B.
    #
    # For each valid token we need to know:
    #   (a) its within-sample rank (0 = first)
    #   (b) how many valid tokens its sample has
    #
    # rank[i] = arange(N)[i] - start_offset_of_its_sample
    # start offsets are obtained by cumsum-ing valid_counts, shifted by 1.
    counts_per_token = valid_counts[sample_ids]  # (N,)

    offsets = torch.zeros(B + 1, device=mask.device, dtype=torch.long)
    offsets[1:] = valid_counts.cumsum(0)  # offsets[b] = start of sample b in flat N
    starts = offsets[:-1]  # (B,)
    rank = torch.arange(n_valid, device=mask.device) - starts[sample_ids]  # (N,)

    is_first = (rank == 0)  # (N,) True for the first token of each sample
    is_single = (counts_per_token == 1)  # (N,) True when sample has only 1 token

    # "rest" weight; clamp denom to ≥1 to avoid div-by-zero for single-token samples
    rest_weight = (1.0 - weight_first_token) / (counts_per_token - 1).clamp(min=1).float()

    weights = torch.where(is_single,
                          torch.ones_like(rest_weight),  # cnt==1 → weight=1.0
                          torch.where(is_first,
                                      torch.full_like(rest_weight, weight_first_token),
                                      rest_weight))  # first → w, rest → (1-w)/(cnt-1)
    weights = weights.to(dtype=js.dtype)  # ensure same dtype as js for scatter_add_

    # per-sample weighted JS, then average over samples that have at least 1 valid token
    loss_flat = js * weights  # (N,)
    valid_b_mask = (valid_counts > 0)  # (B,)
    n_valid_samples = valid_b_mask.sum().clamp(min=1)

    # scatter-add flat per-token losses back to per-sample
    loss_per_sample = torch.zeros(B, device=js.device, dtype=loss_flat.dtype)
    loss_per_sample.scatter_add_(0, sample_ids, loss_flat)

    return loss_per_sample[valid_b_mask].sum() / n_valid_samples
