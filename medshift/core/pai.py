"""
PAI: Paying More Attention to Image — attention-level hallucination mitigation.

Operates on pre-softmax attention scores: amplifies image-token attention
for the last (generating) query token, then applies softmax. This forces
the model to attend more to visual evidence and less to language priors.

Key difference from MEDA (meda.py):
  - MEDA edits the residual-stream hidden states (post-attention).
  - PAI edits the attention scores directly (pre-softmax).
  - The two are complementary: PAI on attention + MEDA on hidden states.

Reference: "Paying More Attention to Image" ECCV 2024.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import types


def _pai_forward(self, *args, **kwargs):
    """Monkey-patched forward for a decoder self-attention module.

    Injects PAI's pre-softmax attention amplification for image tokens
    at the last query position (the token being generated).

    Mathematically (for the last query position q_last):
      attn(q_last, k_img) <- |attn(q_last, k_img)| * alpha + attn(q_last, k_img)
    then softmax is applied normally.
    """
    if not hasattr(self, "_pai_enabled") or not self._pai_enabled:
        return self._original_forward(*args, **kwargs)

    # Extract query, key, value from args or kwargs
    if "hidden_states" in kwargs:
        hidden_states = kwargs["hidden_states"]
        attention_mask = kwargs.get("attention_mask", None)
    else:
        hidden_states = args[0]
        attention_mask = args[1] if len(args) > 1 else None

    # Qwen3/Qwen2 attention style: compute Q, K, V
    bs, seq_len, _ = hidden_states.shape
    # Get head dimensions from config or infer from projections
    if hasattr(self, "num_heads"):
        n_heads = self.num_heads
        head_dim = self.head_dim if hasattr(self, "head_dim") else self.q_proj.out_features // n_heads
    elif hasattr(self, "config"):
        n_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // n_heads
    else:
        n_heads = 40  # Qwen3-14B default
        head_dim = 128
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)

    # Reshape for multi-head attention
    q = q.view(bs, seq_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(bs, seq_len, n_heads, head_dim).transpose(1, 2)
    v = v.view(bs, seq_len, n_heads, head_dim).transpose(1, 2)

    # Compute raw attention scores (pre-softmax)
    scale = head_dim ** -0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Apply causal mask
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # --- PAI: amplify image-token attention for last query token ---
    if hasattr(self, "_pai_img_start") and hasattr(self, "_pai_img_end"):
        img_s = self._pai_img_start
        img_e = min(self._pai_img_end, attn_weights.size(-1))
        if img_s < img_e and self._pai_alpha > 0:
            # Only modify the last query position (current generating token)
            attn_weights[:, :, -1, img_s:img_e] = (
                attn_weights[:, :, -1, img_s:img_e].abs() * self._pai_alpha
                + attn_weights[:, :, -1, img_s:img_e]
            )
    # -------------------------------------------------------------

    # Softmax
    attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

    # Weighted sum
    attn_output = torch.matmul(attn_probs, v)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bs, seq_len, -1)

    # Output projection
    attn_output = self.o_proj(attn_output)
    return attn_output


def find_image_token_indices(model_hf, processor, tokenizer, prompt_text, device="cuda"):
    """Find image token indices in input sequence for Hulu-Med/Qwen3.

    For Qwen3-based models, image tokens are inserted at <image>
    placeholder. Their IDs start with the image_token_index from config.
    """
    # Get image token ID from config
    img_token_id = None
    if hasattr(model_hf, "config") and hasattr(model_hf.config, "image_token_index"):
        img_token_id = model_hf.config.image_token_index
    if img_token_id is None and tokenizer is not None:
        if hasattr(tokenizer, "image_token_id"):
            img_token_id = tokenizer.image_token_id

    if img_token_id is None:
        print("[PAI] Could not find image_token_index")
        return None, None

    # Process a sample input to find image token positions
    messages = [{"role": "user", "content": f"<image>\n{prompt_text}"}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text_inputs = processor.process_text(text, {}, return_tensors="pt", truncation=True, max_length=512)
    input_ids = text_inputs["input_ids"][0]

    positions = (input_ids == img_token_id).nonzero(as_tuple=True)[0]
    if len(positions) > 0:
        img_start = int(positions[0])
        # All contiguous image tokens from img_start
        img_end = img_start
        while img_end < len(input_ids) and input_ids[img_end] == img_token_id:
            img_end += 1
        print(f"[PAI] Image tokens: {img_start} to {img_end} (img_token_id={img_token_id})")
        return img_start, img_end
    print(f"[PAI] No image token found (id={img_token_id})")
    return None, None


class PAIAttentionSteering:
    """PAI attention amplification: monkey-patches decoder self_attn layers.

    Usage:
        steer = PAIAttentionSteering(model, img_start_idx, img_end_idx, alpha=0.2)
        steer.enable_layers(start=2, end=32)
        # ... run generation ...
        steer.disable()
    """

    def __init__(self, model: nn.Module, img_start_idx: int, img_end_idx: int,
                 alpha: float = 0.2):
        self.model = model
        self.img_start = img_start_idx
        self.img_end = img_end_idx
        self.alpha = alpha
        self._patched = []

    def _patch_layer(self, layer: nn.Module, name: str):
        """Monkey-patch a single decoder layer's self_attn."""
        if hasattr(layer, "self_attn"):
            sa = layer.self_attn
        else:
            return

        if hasattr(sa, "_pai_enabled"):
            return  # already patched

        # Save original forward
        sa._original_forward = sa.forward
        sa._pai_enabled = True
        sa._pai_img_start = self.img_start
        sa._pai_img_end = self.img_end
        sa._pai_alpha = self.alpha
        sa.forward = types.MethodType(_pai_forward, sa)
        self._patched.append(sa)

    def enable_layers(self, start: int = 2, end: Optional[int] = None):
        """Patch decoder layers from `start` to `end` (exclusive).

        Default: layers 2 through last-1 (skipping first 2 and last 1),
        matching PAI's recommended layer prior.
        """
        decoder = self.model.model  # Qwen3 has model.layers directly
        if not hasattr(decoder, "layers"):
            # LLaMA/Mistral style: model.model.layers
            decoder = self.model.model.model

        if end is None:
            end = len(decoder.layers)
        for i in range(start, end):
            if i < len(decoder.layers):
                self._patch_layer(decoder.layers[i], f"layers.{i}")
        print(f"[PAI] Patched layers {start}-{end} "
              f"(alpha={self.alpha}, img=[{self.img_start},{self.img_end}))")

    def enable(self, start: int = 2, end: Optional[int] = None):
        """Alias for enable_layers."""
        self.enable_layers(start, end)

    def disable(self):
        """Restore original forward methods."""
        for sa in self._patched:
            sa._pai_enabled = False
        print(f"[PAI] Disabled on {len(self._patched)} layers")
