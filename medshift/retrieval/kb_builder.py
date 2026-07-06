"""Knowledge Bank builder: multi-modal memory bank + retrieval"""
import json
import os
import numpy as np
from PIL import Image
from typing import List, Optional, Dict, Any


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


def build_text_index(entries: List[dict]) -> list:
    """Build text-based index (question embeddings placeholder)."""
    # For now, just use questions as-is for fallback matching
    from sklearn.feature_extraction.text import TfidfVectorizer
    questions = [e.get("question", "") for e in entries]
    if not questions or not any(q.strip() for q in questions):
        return []
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(questions)
    return [vectorizer, tfidf_matrix, questions, entries]


def retrieve_text(query: str, text_index: list, top_k: int = 3, threshold: float = 0.1) -> List[dict]:
    """Retrieve top-k similar entries by TF-IDF similarity."""
    if not text_index:
        return []
    vectorizer, tfidf_matrix, questions, entries = text_index
    query_vec = vectorizer.transform([query])
    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
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
    text_index = build_text_index(entries)
    return retrieve_text(query, text_index, top_k)


def kb_stats(kb_root: str = None) -> Dict[str, int]:
    """Print KB statistics."""
    if kb_root is None:
        kb_root = os.path.join(os.path.dirname(__file__), "../../data/knowledge_bases")
    stats = {}
    for mod in ["xray", "ct", "mri", "pathology"]:
        entries = load_kb(mod, kb_root)
        stats[mod] = len(entries)
    return stats
