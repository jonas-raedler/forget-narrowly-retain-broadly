from trainer.unlearn.base import UnlearnTrainer


class GradLearn(UnlearnTrainer):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL", *args, **kwargs):
        """Relearning trainer: fine-tune only on the relearn set, with no forget term.

        Used to probe how easily an unlearned model recovers the forgotten knowledge — it keeps
        training the model with a plain next-token (NLL) loss on a relearn dataset and never
        applies a forget loss. `gamma`, `alpha`, and `retain_loss_type` are accepted so the
        trainer matches the interface of the other unlearning trainers, but this trainer does not
        use them.

        Args:
            *args, **kwargs: Passed through to the parent `UnlearnTrainer`.
        """
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type
        self.ref_model = None

    def compute_retain_loss(self, model, retain_inputs):
        """Next-token (NLL) loss on the relearn set, returned with the model outputs."""
        retain_outputs = model(**retain_inputs)
        retain_loss = retain_outputs.loss
        return retain_outputs, retain_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Training loss: the next-token loss on the relearn set only.

        There is no forget term — the model is simply trained further on the relearn data — so the
        forget loss is logged as 0.

        Args:
            model: the language model being trained.
            inputs: dict with a "retain" entry holding input_ids / attention_mask / labels.
            return_outputs: if True, return (loss, outputs) instead of just the loss.

        Returns:
            The scalar loss, or (loss, outputs) when return_outputs is True.
        """
        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_outputs, retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = retain_loss

        if self.accelerator.is_local_main_process:
            self.log({
                "retain_loss": retain_loss.item(),
                "forget_loss": 0.0,
            })

        return (loss, retain_outputs) if return_outputs else loss
