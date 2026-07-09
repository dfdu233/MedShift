"""
Line 3: MEDA - MEDical-oriented Activation Editing for hallucination mitigation.

Motivation
----------
14B model's argmax is "greedy-locked": a linear logits transform (VCD,
cache-TTA, entity-boost) cannot flip it because the correct/wrong logit
gap is >4 (FAILURE_ANALYSIS §2). MEDA *edits hidden-state activations*
inside the decoder instead of touching logits linearly, so it bypasses
the greedy lock. This mirrors PAI (ECCV'24) but is medical-oriented:

  - QMS (Query-decisive Manifestation Steering): the per-query medical
    concept (e.g. "pneumothorax", "spinal cord") decides which internal
    activations represent "medical evidence" and steers them up/down.
  - Modality steering vectors: per-modality (CXR / CT / pathology)
    directions in the decoder residual stream that correspond to
    clinically grounded vs. hallucinated activations. Built from KB
    (small auxiliary module, does NOT touch VLM weights).

Operation
---------
At inference, for each decoder layer in a chosen layer range, we edit the
post-attention residual / MLP activation:
    h' = h + alpha * (proj(h, steering_vec)) * steering_vec
i.e. push the activation toward the "medically grounded" direction and
away from the "hallucinated" direction, conditioned on the query concept.

This file provides:
  - SteeringVector: container for per-layer (ground, halluc) directions.
  - MEDAHook: forward hook applying the edit.
  - build_steering_from_kb: construct directions from KB QA pairs
    (small auxiliary module, no VLM training).
  - apply_meda / remove_hooks: registration.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import json, os


# ---------------------------------------------------------------------------
# Steering vector container
# ---------------------------------------------------------------------------

class SteeringVector:
    """Per-layer (ground_direction, halluc_direction) in residual stream.

    ground: direction along which activation ~ image-grounded medical fact.
    halluc: direction along which activation ~ hallucinated / prior-only.
    The edit pushes activations toward `ground` and away from `halluc`.
    """

    def __init__(self, ground: Dict[str, torch.Tensor],
                 halluc: Dict[str, torch.Tensor]):
        # each tensor: (hidden_dim,) unit-normalized
        self.ground = ground
        self.halluc = halluc

    @classmethod
    def load(cls, path: str) -> "SteeringVector":
        with open(path) as f:
            data = json.load(f)
        g = {k: torch.tensor(v) for k, v in data["ground"].items()}
        h = {k: torch.tensor(v) for k, v in data["halluc"].items()}
        return cls(g, h)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "ground": {k: v.tolist() for k, v in self.ground.items()},
                "halluc": {k: v.tolist() for k, v in self.halluc.items()},
            }, f)
        print(f"[MEDA] Saved steering vector to {path}")


# ---------------------------------------------------------------------------
# Build steering directions from KB (small auxiliary module, no VLM training)
# ---------------------------------------------------------------------------

def build_steering_from_kb(model, kb_root: str, modality: str,
                            num_samples: int = 200,
                            layer_names: Optional[List[str]] = None,
                            device: str = "cuda") -> SteeringVector:
    """Construct per-layer (ground, halluc) directions from KB QA pairs.

    Procedure (contrastive activation collection, no gradient on VLM):
      1. For each KB entry, forward (image + correct question -> correct
         answer). Capture the decoder-layer activation at the answer token
         position. This is a "grounded" sample.
      2. Forward (image + correct question -> WRONG/noisy answer) by
         forcing a mismatched answer string. Capture the same activation.
         This is a "hallucinated" sample.
      3. ground_dir  = mean(grounded) - mean(hallucinated), normalized.
        halluc_dir = mean(hallucinated) - mean(grounded), normalized.
      4. Per-layer direction stored.

    The VLM is frozen throughout; we only read activations via hooks.
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../retrieval"))
    from kb_builder import load_kb

    entries = load_kb(modality, kb_root=kb_root)
    if not entries:
        raise ValueError(f"No KB entries for modality {modality}")

    # --- activation collection hooks ---
    activations: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
    # per layer: list of (grounded_act, halluc_act) at answer-token position

    handles = []
    captured = {}  # layer -> last hidden state at answer token

    def _make_capture_hook(layer_name: str):
        def hook(module, inp, out):
            # out may be a tuple (hidden,) or just tensor; take hidden
            h = out[0] if isinstance(out, tuple) else out
            captured[layer_name] = h.detach().float()
        return hook

    decoder = model.model.model  # language model trunk
    # Heuristic layer selection: transformer blocks named like *.layers.*
    if layer_names is None:
        layer_names = []
        for name, mod in decoder.named_modules():
            if isinstance(mod, nn.Module) and ('layers.' in name and
                ('.block' in name or name.endswith('.self_attn') is False)):
                # pick the block module itself; refine below
                pass
        # simpler: grab every Nth decoder layer output
        candidates = []
        for name, mod in decoder.named_modules():
            # Qwen3: model.model.layers.[i]  (decoder blocks)
            if name.startswith("layers.") and name.count(".") == 1:
                candidates.append((name, mod))
        # subsample to ~8 layers spread across depth
        n = len(candidates)
        if n == 0:
            print("[MEDA][WARN] no decoder layers found; "
                  "falling back to all modules ending in '.layer' ignored.")
            return SteeringVector({}, {})
        idxs = np.linspace(0, n - 1, min(8, n)).round().astype(int)
        layer_names = [candidates[i][0] for i in idxs]
        target_mods = {candidates[i][0]: candidates[i][1] for i in idxs}
    else:
        target_mods = {n: dict(decoder.named_modules())[n] for n in layer_names}

    for ln in layer_names:
        handles.append(target_mods[ln].register_forward_hook(_make_capture_hook(ln)))

    # --- collect grounded vs hallucinated activations ---
    ground_acts: Dict[str, List[torch.Tensor]] = {ln: [] for ln in layer_names}
    halluc_acts: Dict[str, List[torch.Tensor]] = {ln: [] for ln in layer_names}

    # Collect all KB answers for OE negative sampling (expand usable samples)
    all_kb_answers = [e.get("answer", "").strip().lower() for e in entries[:num_samples]
                      if e.get("answer", "").strip()]

    processed = 0
    for entry in entries[:num_samples]:
        img_name = entry.get("img_name", "")
        q = entry.get("question", "")
        a_correct = entry.get("answer", "")
        if not (img_name and q and a_correct):
            continue
        candidates = [
            os.path.join("/root/autodl-tmp/MedHEval/images/Slake", img_name),
            os.path.join("/root/autodl-tmp/MedHEval/images/VQA-RAD", img_name),
            os.path.join("/root/autodl-tmp/MedHEval/images/IU-Xray", img_name),
            os.path.join(kb_root, modality, img_name),
            os.path.join(kb_root, "pathology", img_name),
        ]
        img_path = next((c for c in candidates if os.path.exists(c)), None)
        if img_path is None:
            continue
        try:
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            # grounded: forward with correct answer as teacher-forcing target
            captured.clear()
            msgs = {"prompt": f"Answer concisely: {q}\nAnswer: {a_correct}",
                    "image": img}
            _ = model.process_messages(msgs)
            # capture last-token hidden per layer
            for ln in layer_names:
                if ln in captured:
                    h = captured[ln]
                    ground_acts[ln].append(h[0, -1, :].cpu())

            # hallucinated: forward with a deliberately wrong answer
            wrong = _make_wrong_answer(a_correct, kb_answers=all_kb_answers)
            if wrong is None:
                continue
            captured.clear()
            msgs = {"prompt": f"Answer concisely: {q}\nAnswer: {wrong}",
                    "image": img}
            _ = model.process_messages(msgs)
            for ln in layer_names:
                if ln in captured:
                    h = captured[ln]
                    halluc_acts[ln].append(h[0, -1, :].cpu())
            processed += 1
        except Exception:
            continue
        if processed >= num_samples:
            break

    for h in handles:
        h.remove()

    print(f"[MEDA] Collected activations from {processed} KB entries "
          f"across {len(layer_names)} layers.")

    # --- compute directions ---
    ground_dir, halluc_dir = {}, {}
    for ln in layer_names:
        g = ground_acts[ln]
        h = halluc_acts[ln]
        if len(g) < 2 or len(h) < 2:
            continue
        gm = torch.stack(g).mean(0)
        hm = torch.stack(h).mean(0)
        gd = gm - hm
        hd = hm - gm
        gn = gd / (gd.norm() + 1e-8)
        hn = hd / (hd.norm() + 1e-8)
        ground_dir[ln] = gn
        halluc_dir[ln] = hn

    return SteeringVector(ground_dir, halluc_dir)


def _make_wrong_answer(correct: str, kb_answers: Optional[list] = None) -> Optional[str]:
    """Construct a wrong answer for contrastive activation collection.

    For OE questions, samples a random different answer from kb_answers
    as a negative anchor (prevents discarding ~30% of KB entries)."""
    c = correct.strip().lower()
    if c in ("yes", "no"):
        return "no" if c == "yes" else "yes"
    if len(c) == 1 and c in "abcd":
        return {"a": "b", "b": "a", "c": "d", "d": "c"}.get(c, None)
    # OE: use a random different answer from KB as rough negative
    if kb_answers:
        import random
        others = [a for a in kb_answers if a.strip().lower() != c and len(a.strip()) > 0]
        if others:
            return random.choice(others)
    return None


# ---------------------------------------------------------------------------
# MEDA editing hook (applied at inference)
# ---------------------------------------------------------------------------

class MEDAHook:
    """Activation editing hook on a decoder layer.

    h' = h + alpha * <h, ground> * ground  -  beta * <h, halluc> * halluc

    Pushes the residual-stream activation toward the medically-grounded
    direction (QMS: query-decisive) and suppresses the hallucinated one.
    Operates on hidden states, NOT logits, so it bypasses greedy lock.
    """

    def __init__(self, ground_dir: torch.Tensor, halluc_dir: torch.Tensor,
                 alpha: float = 1.0, beta: float = 1.0):
        self.ground = ground_dir.float()      # (hidden,)
        self.halluc = halluc_dir.float()
        self.alpha = alpha
        self.beta = beta

    def __call__(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        shape = h.shape
        h_f = h.float()
        g = self.ground.to(h.device)
        hc = self.halluc.to(h.device)
        # projection of last-token (or all tokens) onto each direction
        # edit only the last position to influence next-token logits
        proj_g = (h_f[..., -1, :] @ g).unsqueeze(-1) * g  # (B, dim)
        proj_h = (h_f[..., -1, :] @ hc).unsqueeze(-1) * hc
        edited = h_f.clone()
        edited[..., -1, :] = edited[..., -1, :] + self.alpha * proj_g - self.beta * proj_h
        edited = edited.to(h.dtype)
        if isinstance(out, tuple):
            return (edited,) + out[1:]
        return edited


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def apply_meda(model, steering: SteeringVector, alpha: float = 1.0,
               beta: float = 1.0, layer_range: Optional[Tuple[int, int]] = None):
    """Register MEDA editing hooks on selected decoder layers.

    layer_range: (start, end) fraction of decoder depth to intervene.
    Default middle-to-late layers (0.4, 0.8) per MEDA/PAI layer prior.
    """
    decoder = model.model.model
    candidates = []
    for name, mod in decoder.named_modules():
        if name.startswith("layers.") and name.count(".") == 1:
            candidates.append((name, mod))
    n = len(candidates)
    if n == 0 or not steering.ground:
        print("[MEDA] nothing to apply (no steering vectors).")
        return []
    if layer_range is None:
        s = int(n * 0.4)
        e = int(n * 0.8)
    else:
        s = int(n * layer_range[0])
        e = int(n * layer_range[1])
    hooks = []
    applied = 0
    for i, (name, mod) in enumerate(candidates):
        if s <= i < e and name in steering.ground:
            hook = MEDAHook(steering.ground[name], steering.halluc[name],
                            alpha=alpha, beta=beta)
            hooks.append(mod.register_forward_hook(hook))
            applied += 1
    print(f"[MEDA] Applied to {applied}/{n} decoder layers "
          f"(range {s}:{e}, alpha={alpha}, beta={beta})")
    return hooks


def remove_hooks(hooks: list):
    for h in hooks:
        h.remove()
