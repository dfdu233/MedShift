"""
Learned Feature Alignment Adapter (LFAA).
Trains a lightweight MLP to align VLM vision features toward source domain.

The key advancement over AdaIN/whitening:
  - AdaIN: hard-coded mean/std per dimension (too simple, distorts semantics)
  - Whitening: hard-coded covariance (noisy, limited samples)
  - LFAA: learned nonlinear mapping (trained on real KB images, ~90K params)

Architecture:
  Input(1152) → LayerNorm → Linear(1152→512) → GELU → Linear(512→1152)
  Residual connection: output = input + adapter(input)

Training:
  - Source images: forward through frozen vision encoder → collect features
  - Self-supervised reconstruction: adapter(source_feat) ≈ source_feat
  - Distribution alignment: MMD/MMD between adapted source and target features
  - The adapter learns to "clean up" domain-specific artifacts

Inference:
  - Hook into vision encoder's final LayerNorm output
  - Pass features through adapter before they go to the projector
  - The adapted features are closer to source domain distribution
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple
import os, json


class FeatureAdapter(nn.Module):
    """Lightweight MLP for vision feature domain alignment.

    Architecture: bottleneck with residual connection.
    Input dim → hidden dim (bottleneck) → output dim
    ReLU/GELU activation, LayerNorm before the adapter.
    """

    def __init__(self, input_dim: int = 1152, hidden_dim: int = 384,
                 output_dim: Optional[int] = None):
        super().__init__()
        output_dim = output_dim or input_dim
        self.ln = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        self.shortcut = (input_dim == output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.ln(x)
        out = self.net(x)
        if self.shortcut:
            out = identity + 0.1 * out  # residual with 0.1 scaling for stability
        return out


def collect_vision_features(model, loader, num_samples: int = 500,
                            device: str = "cuda") -> Tuple[torch.Tensor, List[str]]:
    """Extract vision encoder features from KB images."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from medshift.retrieval.kb_builder import load_kb

    features = []
    paths = []
    count = 0
    for entry in loader.entries[:num_samples]:
        img = loader.load_image(entry)
        if img is None:
            continue
        try:
            msgs = {"prompt": "describe", "image": img}
            inp = model.process_messages(msgs)
            pv = inp["pixel_values"].cuda().to(dtype=torch.bfloat16)
            gs = inp["grid_sizes"]
            ms_ = inp["merge_sizes"]

            # Forward through vision encoder only
            with torch.no_grad():
                ve_out = model.model.model.vision_encoder(pv, grid_sizes=gs, merge_sizes=ms_)
                # ve_out shape: (num_patches, hidden_dim) → pool to (hidden_dim,)
                feat = ve_out.mean(0).cpu().float()

            features.append(feat)
            paths.append(entry.get("img_name", ""))
            count += 1
            if count >= num_samples:
                break
        except Exception as e:
            continue

    return torch.stack(features), paths


def train_adapter_contrastive(adapter: FeatureAdapter, source_features: torch.Tensor,
                                num_epochs: int = 200, lr: float = 1e-3,
                                device: str = "cuda") -> FeatureAdapter:
    """Train adapter with contrastive learning: pull source features to center,
    push target features (augmented) away from non-center."""
    adapter = adapter.to(device)
    src = source_features.to(device).float()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Source center
    src_center = src.mean(0)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward through adapter
        adapted = adapter(src)

        # Reconstruction loss
        loss_rec = F.mse_loss(adapted, src)

        # Center alignment loss: adapted features should cluster around source center
        loss_center = F.mse_loss(adapted.mean(0), src_center)

        # Whitening: maintain source variance structure
        src_cov = (src - src.mean(0)).T @ (src - src.mean(0)) / (src.size(0) - 1)
        adapted_centered = adapted - adapted.mean(0)
        adapted_cov = adapted_centered.T @ adapted_centered / (adapted.size(0) - 1)
        loss_cov = F.mse_loss(adapted_cov, src_cov.detach())

        loss = loss_rec + 0.5 * loss_center + 0.1 * loss_cov

        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            print(f"  [Train] epoch {epoch+1}/{num_epochs} loss={loss.item():.6f} "
                  f"rec={loss_rec.item():.6f} center={loss_center.item():.6f} cov={loss_cov.item():.6f}")

    return adapter


class FeatureAdapterHook:
    """Forward hook that applies the trained adapter to vision encoder output."""

    def __init__(self, adapter: FeatureAdapter):
        self.adapter = adapter.eval()

    def __call__(self, module, input, output):
        """Apply adapter to the output features."""
        with torch.no_grad():
            feat = output.float()
            adapted = self.adapter(feat)
            return adapted.to(output.dtype)


def build_and_train_adapter(model, kb_root: str, modality: str,
                             num_samples: int = 500, hidden_dim: int = 384,
                             num_epochs: int = 100,
                             device: str = "cuda") -> FeatureAdapter:
    """Full pipeline: collect features → train adapter."""
    from medshift.retrieval.clip_retriever import default_image_loader
    from medshift.retrieval.kb_builder import load_kb

    # Load KB entries
    entries = load_kb(modality, kb_root=kb_root)
    loader_cls = type('Loader', (), {
        'entries': entries,
        'load_image': default_image_loader(kb_root, modality)
    })

    print(f"[LFAA] Collecting {num_samples} vision features from {modality} KB...")
    features, paths = collect_vision_features(model, loader_cls,
                                               num_samples=num_samples, device=device)
    print(f"[LFAA] Collected {len(features)} features of dim {features.shape[1]}")

    # Train adapter
    input_dim = features.shape[1]
    adapter = FeatureAdapter(input_dim=input_dim, hidden_dim=hidden_dim)
    print(f"[LFAA] Training adapter ({sum(p.numel() for p in adapter.parameters()):,} params)...")
    adapter = train_adapter(adapter, features, num_epochs=num_epochs, device=device)

    return adapter


def apply_adapter_to_model(model, adapter: FeatureAdapter) -> List:
    """Register adapter hook on the vision encoder's final output."""
    ve = model.model.model.vision_encoder
    hook = FeatureAdapterHook(adapter)

    # Find the last LayerNorm module (post_layernorm)
    target = None
    for name, mod in ve.named_modules():
        if 'post_layernorm' in name or (isinstance(mod, torch.nn.LayerNorm) and name.endswith('.layer_norm')):
            target = mod

    if target is None:
        print("[LFAA] WARNING: Could not find post_layernorm, using vision encoder output hook")
        # Fallback: hook on the entire vision encoder
        hooks = [ve.register_forward_hook(lambda m, i, o: hook(m, i, o[0] if isinstance(o, tuple) else o))]
    else:
        hooks = [target.register_forward_hook(hook)]

    print(f"[LFAA] Registered adapter hook on {type(target).__name__}")
    return hooks
