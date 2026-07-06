"""Knowledge Bank builder: source center + memory bank"""
import numpy as np
from PIL import Image
from typing import List, Optional


def compute_source_center(vlm, images: List[Image.Image]) -> np.ndarray:
    """Compute source domain feature center from a list of images."""
    features = []
    for img in images:
        feat = vlm.extract_visual_features(img)
        if hasattr(feat, "cpu"):
            feat = feat.cpu().numpy()
        features.append(feat.flatten() if feat.ndim > 1 else feat)
    return np.mean(features, axis=0)


def build_memory_bank(vlm, samples: List[dict], image_key: str = "image_path",
                      max_entries: int = 200) -> List[dict]:
    """Build memory bank from (image, question, answer) samples."""
    bank = []
    for i, item in enumerate(samples):
        if len(bank) >= max_entries:
            break
        img = item.get(image_key)
        if img is None:
            continue
        if isinstance(img, str):
            from PIL import Image
            img = Image.open(img).convert("RGB")
        feat = vlm.extract_visual_features(img)
        if hasattr(feat, "cpu"):
            feat = feat.cpu().numpy()
        bank.append({
            "feature": feat.flatten() if feat.ndim > 1 else feat,
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
        })
    return bank


def retrieve(query_feat: np.ndarray, bank: List[dict], top_k: int = 3,
             threshold: float = 0.5) -> List[dict]:
    """Retrieve top-k similar entries from memory bank."""
    qf_n = query_feat / (np.linalg.norm(query_feat) + 1e-10)
    scored = []
    for entry in bank:
        ef = entry["feature"]
        ef_n = ef / (np.linalg.norm(ef) + 1e-10)
        sim = float(np.dot(qf_n, ef_n))
        scored.append((sim, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"sim": s, "question": e["question"], "answer": e["answer"]}
            for s, e in scored[:top_k] if s > threshold]
