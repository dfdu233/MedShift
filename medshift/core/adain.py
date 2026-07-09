"""
AdaIN: Adaptive Instance Normalization for vision encoder feature alignment.
Computes source domain statistics and applies AdaIN at each encoder layer.
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
import json, os


class AdaINStats:
    """Source domain feature statistics per layer."""
    def __init__(self, means: Dict[str, torch.Tensor], stds: Dict[str, torch.Tensor]):
        self.means = means  # {"layer_0": (C,) tensor, ...}
        self.stds = stds


class AdaINHook:
    """Forward hook that applies AdaIN to align features to source domain."""
    
    def __init__(self, source_mean: torch.Tensor, source_std: torch.Tensor, eps: float = 1e-5):
        self.source_mean = source_mean  # (C,) or (1, 1, C)
        self.source_std = source_std
        self.eps = eps
    
    def __call__(self, module, input, output):
        shape = output.shape
        orig_dtype = output.dtype
        feat = output.float()
        if len(shape) == 2:  # (N, C)
            feat_mean = feat.mean(dim=0, keepdim=True)
            feat_std = feat.std(dim=0, keepdim=True) + self.eps
            s_mean = self.source_mean.float().view(1, -1).to(feat.device)
            s_std = self.source_std.float().view(1, -1).to(feat.device)
            result = (feat - feat_mean) / feat_std * s_std + s_mean
        elif len(shape) == 3:
            feat_mean = feat.mean(dim=(0, 1), keepdim=True)
            feat_std = feat.std(dim=(0, 1), keepdim=True) + self.eps
            s_mean = self.source_mean.float().view(1, 1, -1).to(feat.device)
            s_std = self.source_std.float().view(1, 1, -1).to(feat.device)
            result = (feat - feat_mean) / feat_std * s_std + s_mean
        elif len(shape) == 4:
            feat_mean = feat.mean(dim=(0, 2, 3), keepdim=True)
            feat_std = feat.std(dim=(0, 2, 3), keepdim=True) + self.eps
            s_mean = self.source_mean.float().view(1, -1, 1, 1).to(feat.device)
            s_std = self.source_std.float().view(1, -1, 1, 1).to(feat.device)
            result = (feat - feat_mean) / feat_std * s_std + s_mean
        else:
            return output
        return result.to(orig_dtype)


def compute_source_stats_hook():
    """Returns a forward hook that collects statistics."""
    stats = {"means": {}, "stds": {}, "count": 0}
    
    def _make_hook(layer_name):
        def hook(module, input, output):
            with torch.no_grad():
                feat = output.detach().float()
                if len(feat.shape) == 2:  # (N, C)
                    mean = feat.mean(dim=0)
                    std = feat.std(dim=0)
                elif len(feat.shape) == 3:  # (B, N, C)
                    mean = feat.mean(dim=(0, 1))
                    std = feat.std(dim=(0, 1))
                elif len(feat.shape) == 4:  # (B, C, H, W)
                    mean = feat.mean(dim=(0, 2, 3))
                    std = feat.std(dim=(0, 2, 3))
                else:
                    return
                if layer_name not in stats["means"]:
                    stats["means"][layer_name] = mean.cpu()
                    stats["stds"][layer_name] = std.cpu()
                else:
                    n = stats["count"]
                    stats["means"][layer_name] = (stats["means"][layer_name] * n + mean.cpu()) / (n + 1)
                    stats["stds"][layer_name] = (stats["stds"][layer_name] * n + std.cpu()) / (n + 1)
        return hook
    
    return stats, _make_hook


def compute_source_stats_from_kb(model, kb_root: str, modality: str, 
                                  num_samples: int = 200, device: str = "cuda") -> Dict:
    """Compute source domain feature statistics from KB images."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../retrieval"))
    from kb_builder import load_kb
    
    entries = load_kb(modality)
    if not entries:
        raise ValueError(f"No entries found for modality {modality}")
    
    print(f"[AdaIN] Computing source stats for {modality} from {min(len(entries), num_samples)} images...")
    
    # Register hooks on vision encoder layers
    ve = model.model.model.vision_encoder
    stats, hook_factory = compute_source_stats_hook()
    hooks = []
    for name, mod in ve.named_modules():
        if isinstance(mod, torch.nn.LayerNorm) or 'layer_norm' in name:
            hooks.append(mod.register_forward_hook(hook_factory(name)))
    
    # Process images from KB
    processed = 0
    for entry in entries[:num_samples]:
        img_name = entry.get("img_name", "")
        if not img_name:
            continue
        
        # Try to find the image
        candidates = [
            os.path.join("/root/autodl-tmp/MedHEval/images/Slake", img_name),
            os.path.join("/root/autodl-tmp/MedHEval/images/VQA-RAD", img_name),
            os.path.join("/root/autodl-tmp/MedHEval/images/IU-Xray", img_name),
            os.path.join("/root/autodl-tmp/MedShift/data/knowledge_bases", modality, img_name),
        ]
        img_path = None
        for c in candidates:
            if os.path.exists(c):
                img_path = c
                break
        
        if img_path is None and modality == "pathology":
            # PathVQA images stored in KB dir
            img_path = os.path.join("/root/autodl-tmp/MedShift/data/knowledge_bases/pathology", img_name)
        
        if img_path is None or not os.path.exists(img_path):
            continue
        
        try:
            img = Image.open(img_path).convert("RGB")
            messages = {"prompt": "describe", "image": img}
            # Forward through model to capture hook outputs
            _ = model.process_messages(messages)
            model.generate_output(messages)  # This triggers the forward pass
            stats["count"] = processed + 1
            processed += 1
            stats["count"] = processed
        except Exception as e:
            continue
        
        if processed >= num_samples:
            break
    
    # Clean up hooks
    for h in hooks:
        h.remove()
    
    print(f"[AdaIN] Processed {processed} images, computed stats for {len(stats['means'])} layers")
    
    # Convert to tensors
    result = {}
    for layer_name in stats["means"]:
        result[layer_name] = {
            "mean": stats["means"][layer_name].tolist(),
            "std": stats["stds"][layer_name].tolist(),
        }
    
    return result


def load_source_stats(kb_root: str, modality: str) -> Optional[AdaINStats]:
    """Load pre-computed source statistics from disk."""
    path = os.path.join(kb_root, modality, "source_stats.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    means = {k: torch.tensor(v["mean"]) for k, v in data.items()}
    stds = {k: torch.tensor(v["std"]) for k, v in data.items()}
    return AdaINStats(means, stds)


def save_source_stats(stats: Dict, kb_root: str, modality: str):
    """Save computed source statistics to disk."""
    path = os.path.join(kb_root, modality, "source_stats.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[AdaIN] Saved source stats to {path}")


def apply_adain_to_model(model, source_stats: AdaINStats):
    """Register AdaIN hooks on ALL encoder layers (use for ablation)."""
    return apply_adain_to_first_n_layers(model, source_stats, ratio=1.0)


def apply_adain_to_first_n_layers(model, source_stats: AdaINStats, ratio: float = 0.33):
    """
    Register AdaIN hooks only on the first `ratio` fraction of encoder layers.
    Later layers carry semantic info — applying AdaIN there hurts accuracy.
    """
    ve = model.model.model.vision_encoder
    all_ln_names = []
    for name, mod in ve.named_modules():
        if isinstance(mod, torch.nn.LayerNorm) or 'layer_norm' in name:
            if name in source_stats.means:
                all_ln_names.append(name)
    
    all_ln_names.sort()
    n_target = max(1, int(len(all_ln_names) * ratio))
    target_names = set(all_ln_names[:n_target])
    
    hooks = []
    for name in all_ln_names:
        if name in target_names:
            hook = AdaINHook(source_stats.means[name], source_stats.stds[name])
            mod = dict(ve.named_modules())[name]
            hooks.append(mod.register_forward_hook(hook))
    
    print(f"[AdaIN] Applied to {len(hooks)}/{len(all_ln_names)} layers (ratio={ratio})")
    return hooks
