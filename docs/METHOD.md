# Method: JensUn++

JensUn++ is implemented as the **`JensUnPP`** trainer
([`src/trainer/unlearn/jensun.py`](../src/trainer/unlearn/jensun.py)). Its three contributions map
directly to training knobs:

| JensUn++ component (paper) | In the code |
|---|---|
| Refusal-string loss (natural refusals, no gibberish) + stochastic prefix-mixing | `push_prefix_to_refusal_start=true` (on by default for JensUn++ runs) |
| Dynamic loss balancing (grad-norm rescaling of forget vs. retain) | `use_grad_norm_scaling=true` (the `gnorm` field) |
| Hard forget↔retain pairing by augmentation type | pairing (default), or `randpair` for the random-pairing ablation |

The grad-norm rescaling multiplies the forget loss by s = ‖∇𝓛ᵣ‖ / (‖∇𝓛_f‖ + ε), so the rescaled forget gradient matches the retain gradient in norm.

**JensUn++ vs. the JensUn baseline.** They differ in *both* the forget loss and the forget data.
JensUn++ (`JensUnPP`) trains the model to produce the dataset's **refusal sentence**: its forget target
is the per-example refusal string from a refusal-context dataset (`QAwithRefusalStringDataset`), applied
with first-token weighting and optional grad-norm balancing. The plain **JensUn** baseline trains the
model to emit **fixed tokens** (`'No Idea'`, or `'#'` for `JensUnBaselineHash`) over the answer positions
of a standard QA dataset (`QADataset`). The retain loss (a Jensen-Shannon divergence against a frozen
reference model) is identical for both.
