"""
Fourier Domain Adaptation (FDA) for "source domain center" alignment.

The key insight: amplitude spectrum captures domain/style information while
phase spectrum preserves content/structure. By swapping the low-frequency
amplitude between test image and source domain, we can make the test image
"look like" it came from the source domain — at the PIXEL level, completely
bypassing the feature-space issues of AdaIN/whitening.

This is training-free, model-agnostic, and operates on raw pixels.

Reference: "Fourier Domain Adaptation for Medical Image Analysis" (FDA, 2020)
           "Source Free Domain Adaptation with Fourier Style Mining" (MIA 2022)
           "Curriculum-Based Augmented Fourier Domain Adaptation" (2023)
"""
import torch
import torch.fft as fft
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple


def _to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> torch float tensor [C,H,W] in [0,1]."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _to_pil(tensor: torch.Tensor) -> Image.Image:
    """torch [C,H,W] -> PIL."""
    arr = (tensor.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def fft_amplitude_swap(test_img: Image.Image, source_amplitude: torch.Tensor,
                       beta: float = 0.1, L: Optional[float] = None) -> Image.Image:
    """
    Fourier Domain Adaptation: swap low-frequency amplitude.

    Args:
        test_img: PIL image to adapt (target domain)
        source_amplitude: amplitude spectrum [C,H,W] from source domain
        beta: fraction of frequency spectrum to swap (default 0.1 = 10%)
        L: radius of low-frequency region (in pixels); if None, computed from beta

    Returns:
        Domain-adapted PIL image
    """
    t = _to_tensor(test_img).to(source_amplitude.device)
    C, H, W = t.shape

    # FFT
    f = fft.fftn(t, dim=(-2, -1))
    f_shift = fft.fftshift(f)
    amp = f_shift.abs()
    phase = f_shift.angle()

    # Resize source amplitude to match test image dimensions
    source_amp_resized = torch.nn.functional.interpolate(
        source_amplitude.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
    )[0]

    # Create low-frequency mask
    if L is None:
        L = int(min(H, W) * beta)
    cy, cx = H // 2, W // 2

    # Use resized source amplitude
    source_amp = source_amp_resized.to(t.device)

    # Swap low-frequency region
    y_slice = slice(cy - L, cy + L)
    x_slice = slice(cx - L, cx + L)
    amp[:, y_slice, x_slice] = source_amp[:, y_slice, x_slice]

    # Reconstruct
    f_new = amp * torch.exp(1j * phase)
    f_new_shift = fft.ifftshift(f_new)
    img_new = fft.ifftn(f_new_shift, dim=(-2, -1)).real
    img_new = img_new.clamp(0, 1)

    return _to_pil(img_new)


def compute_source_amplitude(images: List[Image.Image], device: str = "cuda") -> torch.Tensor:
    """
    Compute averaged amplitude spectrum from multiple source images.
    This creates the "source domain center" in frequency space.

    Args:
        images: list of PIL images from source domain
        device: torch device

    Returns:
        Average amplitude spectrum [C,H,W], resized to common size
    """
    # Use first image size as reference
    common_size = images[0].size
    amps = []
    for img in images:
        img_resized = img.resize(common_size)
        t = _to_tensor(img_resized).to(device)
        f = fft.fftn(t, dim=(-2, -1))
        f_shift = fft.fftshift(f)
        amps.append(f_shift.abs())

    # Average
    avg_amp = torch.stack(amps).mean(0)
    return avg_amp


def compute_source_amplitude_from_kb(kb_root: str, modality: str,
                                      num_samples: int = 100,
                                      img_size: Tuple[int, int] = (224, 224),
                                      device: str = "cuda") -> torch.Tensor:
    """
    Compute source amplitude spectrum from a KB of images.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../retrieval"))
    from kb_builder import load_kb
    from medshift.retrieval.clip_retriever import default_image_loader

    entries = load_kb(modality, kb_root=kb_root)
    loader = default_image_loader(kb_root, modality)

    images = []
    for entry in entries[:num_samples]:
        img = loader(entry)
        if img is not None:
            img = img.resize(img_size)
            images.append(img)
            if len(images) >= num_samples:
                break

    if not images:
        print(f"[FDA] No images found for {modality}")
        # Return a fallback: uniform amplitude
        return torch.ones(3, img_size[1], img_size[0], device=device)

    print(f"[FDA] Computing source amplitude from {len(images)} {modality} images")
    return compute_source_amplitude(images, device=device)


class FDAPreprocessor:
    """
    Applies Fourier Domain Adaptation to test images before VLM inference.

    Usage:
        fda = FDAPreprocessor(kb_root, "xray")
        adapted_img = fda(test_img, beta=0.1)
        # Use adapted_img with VLM
    """

    def __init__(self, kb_root: str, modality: str, beta: float = 0.1,
                 num_source: int = 100, device: str = "cuda"):
        self.source_amp = compute_source_amplitude_from_kb(
            kb_root, modality, num_samples=num_source, device=device)
        self.beta = beta

    def __call__(self, img: Image.Image, beta: Optional[float] = None) -> Image.Image:
        """Apply FDA to a single test image."""
        b = beta if beta is not None else self.beta
        return fft_amplitude_swap(img, self.source_amp, beta=b)

    def __repr__(self):
        return f"FDAPreprocessor(beta={self.beta}, source_amp={self.source_amp.shape})"
