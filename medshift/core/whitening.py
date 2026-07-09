"""
Line 1: Feature Whitening for source-domain center alignment.

Replaces AdaIN's first-order (mean/std) alignment with second-order
(covariance / principal-subspace) whitening, motivated by:

  - Anisotropic Modality Align (2026): after a global centroid shift,
    the cross-modal residual is NOT isotropic noise but an anisotropic,
    low-effective-dimensional structure (Ar=28.6, deff/d=0.284).
    AdaIN (mean+std) only removes the centroid + isotropic scaling, leaving
    the dominant anisotropic residual untouched.
  - Feature-Whitening (Roy, CVPR 2019): layer-specific covariance matrices
    align marginal distributions without a dedicated loss.

Key design vs AdaIN (medshift/core/adain.py):
  - AdaIN: f' = (f - mu_test)/sigma_test * sigma_src + mu_src   (1st order only)
  - Whitening: f' = W_src @ (f - mu_src)  with W_src = Sigma_src^{-1/2}
               truncated to top-r principal subspace (anisotropic residual).
  - Shallow-layer only (1-9): deep layers carry semantics, must not be perturbed
    (FAILURE_ANALYSIS §1). Optional MMD gate skips layers with small shift.

Numerical robustness for 1152-dim covariance with ~500 samples:
  - feature grouping (g=64) per Roy, OR ridge regularization Sigma + lambda*I,
  - eigendecomposition truncated to top-r (r from spectral knee, ~16-64).
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import json, os
import math


# ---------------------------------------------------------------------------
# Source-domain second-order statistics collection (Welford online covariance)
# ---------------------------------------------------------------------------

class _CovAccumulator:
    """Online mean + covariance accumulator (Welford extension for cov)."""

    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0
        self.mean = None
        self.M2 = None
        self._device = None

    def _ensure_device(self, x: torch.Tensor):
        if self._device is None:
            self._device = x.device
            self.mean = torch.zeros(self.dim, dtype=torch.float64, device=self._device)
            self.M2 = torch.zeros(self.dim, self.dim, dtype=torch.float64, device=self._device)

    def update(self, x: torch.Tensor):
        # x: (N, dim) float
        x = x.detach().to(torch.float64).reshape(-1, self.dim)
        self._ensure_device(x)
        for row in x:
            self.n += 1
            n = self.n
            delta = row - self.mean
            self.mean += delta / n
            delta2 = row - self.mean
            self.M2 += torch.outer(delta, delta2)

    def stats(self):
        if self.mean is None or self.n < 2:
            return self.mean, None, None, None
        cov = self.M2 / (self.n - 1)  # unbiased
        eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending
        eigvals = eigvals.clamp_min(1e-8)
        return self.mean.float().cpu(), cov.float().cpu(), eigvals.float().cpu(), eigvecs.float().cpu()


def compute_source_cov_hook():
    """Returns (stats_dict, hook_factory). hook collects per-layer mean+cov."""
    stats: Dict[str, _CovAccumulator] = {}
    handles = []

    def _make_hook(layer_name: str, dim_hint: Optional[int] = None):
        def hook(module, inp, out):
            with torch.no_grad():
                feat = out.detach().float()
                if feat.ndim == 2:
                    feat = feat  # (N, C)
                elif feat.ndim == 3:
                    feat = feat.reshape(-1, feat.shape[-1])  # (B*N, C)
                elif feat.ndim == 4:
                    feat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
                else:
                    return
                dim = feat.shape[-1]
                if layer_name not in stats:
                    stats[layer_name] = _CovAccumulator(dim)
                stats[layer_name].update(feat)
        return hook

    return stats, _make_hook


def compute_source_cov_from_kb(model, kb_root: str, modality: str,
                                num_samples: int = 500, device: str = "cuda") -> Dict:
    """Compute source-domain mean+cov+eig per LayerNorm of the vision encoder."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../retrieval"))
    from kb_builder import load_kb

    entries = load_kb(modality, kb_root=kb_root)
    if not entries:
        raise ValueError(f"No KB entries for modality {modality}")

    print(f"[Whitening] Computing source cov for {modality} from "
          f"{min(len(entries), num_samples)} images...")

    ve = model.model.model.vision_encoder
    stats, hook_factory = compute_source_cov_hook()
    hooks = []
    for name, mod in ve.named_modules():
        if isinstance(mod, torch.nn.LayerNorm) or 'layer_norm' in name:
            hooks.append(mod.register_forward_hook(hook_factory(name)))

    processed = 0
    for entry in entries[:num_samples]:
        img_name = entry.get("img_name", "")
        if not img_name:
            continue
        candidates = [
            os.path.join("/root/autodl-tmp/MedHEval/images/Slake", img_name),
            os.path.join("/root/autodl-tmp/MedHEval/images/VQA-RAD", img_name),
            os.path.join("/root/autodl-tmp/MedHEval/images/IU-Xray", img_name),
            os.path.join(kb_root, modality, img_name),
            os.path.join(kb_root, "pathology", img_name),
        ]
        img_path = next((c for c in candidates if os.path.exists(c)), None)
        if img_path is None or not os.path.exists(img_path):
            continue
        try:
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            messages = {"prompt": "describe", "image": img}
            _ = model.process_messages(messages)
            model.generate_output(messages)
            processed += 1
        except Exception:
            continue
        if processed >= num_samples:
            break

    for h in hooks:
        h.remove()

    print(f"[Whitening] Processed {processed} images, "
          f"{len(stats)} layers with cov.")
    result = {}
    for layer_name, acc in stats.items():
        mean, cov, eigvals, eigvecs = acc.stats()
        if mean is None:
            continue
        result[layer_name] = {
            "mean": mean.tolist(),
            "eigvals": eigvals.tolist(),
            "eigvecs": eigvecs.tolist(),
        }
    return result


def save_source_cov(stats: Dict, kb_root: str, modality: str):
    path = os.path.join(kb_root, modality, "source_cov.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f)
    print(f"[Whitening] Saved source cov to {path}")


def load_source_cov(kb_root: str, modality: str) -> Optional["WhiteningStats"]:
    path = os.path.join(kb_root, modality, "source_cov.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    means, eigvals, eigvecs = {}, {}, {}
    for k, v in data.items():
        means[k] = torch.tensor(v["mean"])
        eigvals[k] = torch.tensor(v["eigvals"])
        eigvecs[k] = torch.tensor(v["eigvecs"])
    return WhiteningStats(means, eigvals, eigvecs)


# ---------------------------------------------------------------------------
# Whitening transform + hook
# ---------------------------------------------------------------------------

class WhiteningStats:
    """Per-layer mean, eigenvalues, eigenvectors of the source domain."""

    def __init__(self, means: Dict[str, torch.Tensor],
                 eigvals: Dict[str, torch.Tensor],
                 eigvecs: Dict[str, torch.Tensor]):
        self.means = means
        self.eigvals = eigvals  # ascending (from eigh)
        self.eigvecs = eigvecs  # columns are eigenvectors


class WhiteningHook:
    """
    Forward hook that whitens features to the source-domain principal subspace.

    f' = mu_src + V_top_r @ diag(1/sqrt(lambda_top_r)) @ V_top_r^T @ (f - mu_src)

    where (V, lambda) are the source-domain covariance eigendecomp, truncated
    to the top-r principal directions (largest eigenvalues = dominant residual
    subspace per Anisotropic Modality Align). Lambda regularized by +lambda_r.

    Optional test-time MMD gate: only whiten if the test batch's Mahalanobis
    distance to source exceeds `gate_threshold` (skips low-shift layers).
    """

    def __init__(self, source_mean: torch.Tensor, source_eigvals: torch.Tensor,
                 source_eigvecs: torch.Tensor, top_r: int = 32,
                 ridge: float = 1e-3, gate_threshold: float = 0.0,
                 eps: float = 1e-6):
        # eigh returns ascending; we want the largest r eigenvalues.
        d = source_eigvals.shape[0]
        idx = torch.arange(d - 1, d - 1 - top_r, -1)  # descending pick
        lam = source_eigvals[idx].clamp_min(eps) + ridge
        V = source_eigvecs[:, idx]  # (dim, r)
        # whitening matrix in the top-r subspace: W = V @ diag(1/sqrt(lam)) @ V^T
        W = V @ torch.diag(1.0 / torch.sqrt(lam)) @ V.t()
        self.source_mean = source_mean.float()
        self.W = W.float()          # (dim, dim) but rank-r
        self.top_r = top_r
        self.gate_threshold = gate_threshold

    @torch.no_grad()
    def _shift_score(self, feat: torch.Tensor) -> float:
        """Mahalanobis-style shift score of current batch vs source."""
        if self.gate_threshold <= 0:
            return float("inf")  # always on
        diff = feat.mean(dim=0) - self.source_mean.to(feat.device)
        # use whitening matrix as inverse-cov surrogate
        score = float((diff @ self.W.to(feat.device) @ diff).sqrt().item())
        return score

    def __call__(self, module, inp, output):
        shape = output.shape
        orig_dtype = output.dtype
        feat = output.float()
        if feat.ndim == 2:
            pass
        elif feat.ndim == 3:
            feat = feat.reshape(-1, feat.shape[-1])
        elif feat.ndim == 4:
            feat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        else:
            return output

        if self._shift_score(feat) < self.gate_threshold:
            return output  # low shift, skip

        sm = self.source_mean.to(feat.device)
        W = self.W.to(feat.device)
        out = sm + (feat - sm) @ W.t()
        out = out.reshape(shape).to(orig_dtype)
        return out


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------

def apply_whitening_to_shallow_layers(model, stats: WhiteningStats,
                                       num_shallow: int = 9, top_r: int = 32,
                                       ridge: float = 1e-3,
                                       gate_threshold: float = 0.0):
    """Register whitening hooks only on the first `num_shallow` LayerNorms.

    Deep layers (10+) carry semantic info and must not be perturbed
    (FAILURE_ANALYSIS §1: AdaIN on all 55 layers hurt deep semantics).
    """
    ve = model.model.model.vision_encoder
    ln_names = sorted([n for n, m in ve.named_modules()
                      if isinstance(m, torch.nn.LayerNorm) or 'layer_norm' in n
                      if n in stats.means])
    target = set(ln_names[:num_shallow])

    hooks = []
    for name in ln_names:
        if name in target:
            hook = WhiteningHook(
                stats.means[name], stats.eigvals[name], stats.eigvecs[name],
                top_r=top_r, ridge=ridge, gate_threshold=gate_threshold,
            )
            mod = dict(ve.named_modules())[name]
            hooks.append(mod.register_forward_hook(hook))
    print(f"[Whitening] Applied to {len(hooks)}/{len(ln_names)} shallow layers "
          f"(top_r={top_r}, ridge={ridge}, gate={gate_threshold})")
    return hooks


def remove_hooks(hooks: List):
    for h in hooks:
        h.remove()
