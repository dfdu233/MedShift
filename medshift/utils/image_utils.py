"""
图像处理工具函数
"""
import numpy as np
import cv2
import torch
from PIL import Image, ImageFilter, ImageEnhance
from typing import Tuple, Optional
import io


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """PIL Image -> numpy array (H, W, C) uint8"""
    return np.array(img.convert("RGB"))


def numpy_to_pil(img: np.ndarray) -> Image.Image:
    """numpy array (H, W, C) -> PIL Image"""
    return Image.fromarray(img.astype(np.uint8))


def apply_gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma校正"""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def apply_contrast_adjustment(img: np.ndarray, factor: float) -> np.ndarray:
    """对比度调整"""
    pil_img = numpy_to_pil(img)
    enhancer = ImageEnhance.Contrast(pil_img)
    enhanced = enhancer.enhance(factor)
    return pil_to_numpy(enhanced)


def apply_brightness_adjustment(img: np.ndarray, factor: float) -> np.ndarray:
    """亮度调整"""
    pil_img = numpy_to_pil(img)
    enhancer = ImageEnhance.Brightness(pil_img)
    enhanced = enhancer.enhance(factor)
    return pil_to_numpy(enhanced)


def apply_gaussian_blur(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """高斯模糊"""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def apply_gaussian_noise(img: np.ndarray, std: float) -> np.ndarray:
    """高斯噪声"""
    noise = np.random.normal(0, std * 255, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def apply_histogram_shift(img: np.ndarray, shift: int = 30) -> np.ndarray:
    """直方图平移"""
    shifted = img.astype(np.int16) + shift
    return np.clip(shifted, 0, 255).astype(np.uint8)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """自适应直方图均衡化 (CLAHE)"""
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(img)


def apply_jpeg_compression(img: np.ndarray, quality: int = 50) -> np.ndarray:
    """JPEG压缩模拟"""
    pil_img = numpy_to_pil(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    return pil_to_numpy(compressed)


def apply_adaptive_equalization(img: np.ndarray) -> np.ndarray:
    """自适应均衡化"""
    return apply_clahe(img, clip_limit=3.0, grid_size=(4, 4))


def apply_illumination_correction(img: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """光照校正"""
    # 使用对数变换增强暗区
    img_float = img.astype(np.float32) / 255.0
    corrected = np.log1p(img_float * 100) / np.log1p(100)
    corrected = np.clip(corrected * 255, 0, 255).astype(np.uint8)
    # 再做gamma校正
    return apply_gamma_correction(corrected, gamma)


def apply_stain_normalization(img: np.ndarray, target_mean: Optional[np.ndarray] = None) -> np.ndarray:
    """
    简化的染色归一化 (适用于病理图像)
    基于颜色统计量的标准化
    """
    img_float = img.astype(np.float32)
    mean = img_float.mean(axis=(0, 1))
    std = img_float.std(axis=(0, 1)) + 1e-6

    if target_mean is None:
        # 使用标准医学图像的典型均值
        target_mean = np.array([150.0, 130.0, 120.0])
    target_std = np.array([50.0, 50.0, 50.0])

    normalized = (img_float - mean) / std * target_std + target_mean
    return np.clip(normalized, 0, 255).astype(np.uint8)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算两幅图像的SSIM"""
    from skimage.metrics import structural_similarity
    if len(img1.shape) == 3:
        return structural_similarity(img1, img2, channel_axis=2, data_range=255)
    return structural_similarity(img1, img2, data_range=255)


def compute_histogram_distance(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算直方图距离 (卡方距离)"""
    dist = 0.0
    for c in range(min(img1.shape[2], 3) if len(img1.shape) == 3 else 1):
        if len(img1.shape) == 3:
            h1 = cv2.calcHist([img1], [c], None, [64], [0, 256]).flatten()
            h2 = cv2.calcHist([img2], [c], None, [64], [0, 256]).flatten()
        else:
            h1 = cv2.calcHist([img1], [0], None, [64], [0, 256]).flatten()
            h2 = cv2.calcHist([img2], [0], None, [64], [0, 256]).flatten()
        h1 = h1 / (h1.sum() + 1e-10)
        h2 = h2 / (h2.sum() + 1e-10)
        dist += np.sum((h1 - h2) ** 2 / (h1 + h2 + 1e-10))
    return dist / (min(img1.shape[2], 3) if len(img1.shape) == 3 else 1)


def generate_domain_perturbation(img: np.ndarray, method: str, rng: np.random.RandomState,
                                  config=None) -> np.ndarray:
    """
    生成域扰动视图
    Args:
        img: 输入图像 (H, W, C) uint8
        method: 扰动方法名
        rng: 随机数生成器
        config: DomainShiftConfig
    Returns:
        扰动后的图像
    """
    if config is None:
        from medshift.config import DomainShiftConfig
        config = DomainShiftConfig()

    if method == "gamma":
        gamma = rng.uniform(*config.gamma_range)
        return apply_gamma_correction(img, gamma)
    elif method == "contrast":
        factor = rng.uniform(*config.contrast_range)
        return apply_contrast_adjustment(img, factor)
    elif method == "brightness":
        factor = rng.uniform(*config.brightness_range)
        return apply_brightness_adjustment(img, factor)
    elif method == "blur":
        kernel = rng.choice(range(config.blur_kernel_range[0], config.blur_kernel_range[1] + 1, 2))
        return apply_gaussian_blur(img, int(kernel))
    elif method == "noise":
        std = rng.uniform(*config.noise_std_range)
        return apply_gaussian_noise(img, std)
    elif method == "histogram_shift":
        shift = rng.randint(-30, 30)
        return apply_histogram_shift(img, shift)
    elif method == "clahe":
        clip = rng.uniform(1.5, 4.0)
        return apply_clahe(img, clip_limit=clip)
    elif method == "jpeg_compress":
        quality = rng.randint(*config.jpeg_quality_range)
        return apply_jpeg_compression(img, quality)
    elif method == "stain_normalization":
        return apply_stain_normalization(img)
    elif method == "illumination_correction":
        gamma = rng.uniform(1.0, 1.5)
        return apply_illumination_correction(img, gamma)
    elif method == "adaptive_equalization":
        return apply_adaptive_equalization(img)
    else:
        raise ValueError(f"Unknown perturbation method: {method}")
