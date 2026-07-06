"""Knowledge Bank builder: multi-modal memory bank + retrieval"""
import json
import os
import numpy as np
from PIL import Image
from typing import List, Optional, Dict, Any, Tuple


# Lazy-loaded sentence encoder
_encoder = None

def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _encoder


def load_kb(modality: str, kb_root: str = None) -> List[dict]:
    """Load knowledge bank for a given modality."""
    if kb_root is None:
        kb_root = os.path.join(os.path.dirname(__file__), "../../data/knowledge_bases")
    path = os.path.join(kb_root, modality, "metadata.json")
    if not os.path.exists(path):
        print(f"[WARN] KB not found: {path}")
        return []
    with open(path) as f:
        entries = json.load(f)
    return entries


def build_text_index(entries: List[dict], modality: str = "", kb_root: str = None) -> Tuple[np.ndarray, List[str], List[dict]]:
    """Build semantic embedding index for retrieval, with file caching."""
    if not entries:
        return np.array([]), [], []
    
    # Check for cached embeddings
    cache_dir = os.path.join(kb_root or os.path.join(os.path.dirname(__file__), "../../data/knowledge_bases"), modality)
    cache_path = os.path.join(cache_dir, "embeddings.npy")
    cache_texts_path = os.path.join(cache_dir, "texts.npy")
    
    if os.path.exists(cache_path) and os.path.exists(cache_texts_path):
        embeddings = np.load(cache_path)
        texts = np.load(cache_texts_path, allow_pickle=True).tolist()
        return embeddings, texts, entries
    
    texts = []
    for e in entries:
        q = e.get("question", "").strip()
        a = e.get("answer", "").strip()
        texts.append(f"{q} {a}")
    
    encoder = _get_encoder()
    embeddings = encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Cache to disk
    try:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, embeddings)
        np.save(cache_texts_path, np.array(texts, dtype=object))
    except Exception as e:
        print(f"[WARN] Could not cache embeddings: {e}")
    
    return embeddings, texts, entries


def retrieve_text(query: str, index_data: tuple, top_k: int = 3, threshold: float = 0.2) -> List[dict]:
    """Retrieve top-k similar entries by sentence embedding similarity."""
    embeddings, texts, entries = index_data
    if len(embeddings) == 0:
        return []
    
    encoder = _get_encoder()
    query_emb = encoder.encode([query], show_progress_bar=False, convert_to_numpy=True)
    
    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity(query_emb, embeddings).flatten()
    top_indices = np.argsort(scores)[::-1]
    
    results = []
    for idx in top_indices:
        if scores[idx] < threshold:
            break
        if len(results) >= top_k:
            break
        results.append({
            "sim": float(scores[idx]),
            "question": entries[idx].get("question", ""),
            "answer": entries[idx].get("answer", ""),
            "source": entries[idx].get("source", ""),
        })
    return results


def retrieve_by_modality(query: str, modality: str, kb_root: str = None, 
                          top_k: int = 3) -> List[dict]:
    """Retrieve top-k from a specific modality KB."""
    entries = load_kb(modality, kb_root)
    if not entries:
        return []
    index_data = build_text_index(entries)
    return retrieve_text(query, index_data, top_k)


def kb_stats(kb_root: str = None) -> Dict[str, int]:
    """Print KB statistics."""
    if kb_root is None:
        kb_root = os.path.join(os.path.dirname(__file__), "../../data/knowledge_bases")
    stats = {}
    for mod in ["xray", "ct", "mri", "pathology"]:
        entries = load_kb(mod, kb_root)
        stats[mod] = len(entries)
    return stats
