"""
CFG-based hallucination mitigation: contrast image-conditional logits
against text-only logits (language prior), then scale the difference.

Formula: l_steered = gamma * (l_img - l_text) + l_text
         = (1+gamma) * l_img - gamma * l_text

This is classifier-free guidance applied to VLMs: it amplifies what the
image contributes beyond the language prior. Unlike VCD (clean vs noisy
image), the negative here is the TEXT-ONLY model, so it directly suppresses
language priors (the documented failure root).

Key difference from VCD on 14B:
  VCD: noisy image logits ≈ clean image logits (small diff, can't flip)
  CFG: text-only logits = language prior (LARGE diff from image, CAN flip)
"""
import torch
from transformers import LogitsProcessor
from typing import Optional


class CFGLogitsProcessor(LogitsProcessor):
    """Classifier-free guidance at the logit level.

    At each generation step, computes:
      l_steered = (1 + gamma) * l_img - gamma * l_text

    where l_img = logits with image input, l_text = logits with text-only.
    """

    def __init__(self, model, base_inputs: dict, gamma: float = 1.5):
        self.model = model
        self.base_inputs = base_inputs  # inputs WITHOUT pixel_values
        self.gamma = gamma

    def __call__(self, input_ids, scores):
        # Forward with text-only inputs (same input_ids, no image)
        text_inputs = {
            "input_ids": input_ids,
            "attention_mask": self.base_inputs.get("attention_mask"),
            "use_cache": True,
            "return_dict": True,
        }
        with torch.no_grad():
            text_out = self.model(**text_inputs)
        logits_text = text_out.logits[:, -1, :]

        l_img = scores
        l_text = logits_text.to(l_img.device)
        l_fused = (1 + self.gamma) * l_img - self.gamma * l_text
        return l_fused


def apply_cfg(model, inputs, gamma=1.5):
    """Apply CFG to a generation call. Returns the processor for use."""
    # Extract text-only inputs (no pixel_values, no image-related keys)
    text_inputs = {}
    img_keys = {"pixel_values", "grid_sizes", "merge_sizes", "modals", "image_token_length"}
    for k, v in inputs.items():
        if k not in img_keys and k in ("input_ids", "attention_mask"):
            text_inputs[k] = v

    processor = CFGLogitsProcessor(model.model, text_inputs, gamma=gamma)
    return processor
