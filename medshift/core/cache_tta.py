"""
Cache-based Test-Time Adaptation (Cache-TTA)
=============================================
Uses the existing text-embedding KB to calibrate output logits at test time.

Core idea:
  1. Build modality-specific cache from KB text embeddings (already computed)
  2. At test time, for each question:
     a. Embed the query with sentence-BERT
     b. Retrieve top-k similar QA pairs from cache
     c. Build a calibration distribution favoring retrieved answer tokens
     d. Interpolate: l' = (1-λ)·l + λ·l_cache
  3. No vision encoder forward needed — text-based retrieval only

Reference: TDA (CVPR'24), Ultra-Light TTA (2025)
"""
import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class TextFeatureCache:
    """
    Modality-specific cache of text embeddings + QA pairs.
    Uses the existing sentence-BERT embeddings from kb_builder.
    """

    def __init__(self, modality: str, cache_root: str = None):
        self.modality = modality
        self.cache_root = cache_root or "/root/autodl-tmp/MedShift/data/knowledge_bases"
        self.embeddings = None   # np.ndarray (N, 384)
        self.entries = []        # List[dict] {question, answer, source}
        self._loaded = False
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        return self._encoder

    def load(self) -> bool:
        """Load pre-computed embeddings + entries from KB."""
        from medshift.retrieval.kb_builder import load_kb

        # Load metadata (QA pairs)
        self.entries = load_kb(self.modality, self.cache_root)
        if not self.entries:
            print(f"[Cache-{self.modality}] No entries found")
            return False

        # Load cached embeddings
        embed_path = os.path.join(self.cache_root, self.modality, "embeddings.npy")
        if not os.path.exists(embed_path):
            print(f"[Cache-{self.modality}] No embeddings found at {embed_path}")
            return False

        self.embeddings = np.load(embed_path)
        print(f"[Cache-{self.modality}] Loaded {len(self.entries)} entries, "
              f"embeddings shape={self.embeddings.shape}")
        self._loaded = True
        return True

    def retrieve(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[dict]:
        """Retrieve top-k similar entries by cosine similarity to query."""
        if not self._loaded or self.embeddings is None:
            return []

        encoder = self._get_encoder()
        query_emb = encoder.encode([query], show_progress_bar=False, convert_to_numpy=True)
        sims = cosine_similarity(query_emb, self.embeddings).flatten()
        top_idx = np.argsort(sims)[::-1]

        results = []
        for idx in top_idx:
            if sims[idx] < threshold:
                break
            if len(results) >= top_k:
                break
            results.append({
                "sim": float(sims[idx]),
                "question": self.entries[idx].get("question", ""),
                "answer": self.entries[idx].get("answer", ""),
                "source": self.entries[idx].get("source", ""),
            })
        return results


class CacheTTADecoder:
    """
    Calibrates output logits using cache-derived knowledge.

    l' = (1 - lam) * l_original + lam * l_cache

    l_cache is a distribution that boosts token IDs found in
    retrieved answer texts.
    """

    def __init__(self, cache: TextFeatureCache, tokenizer,
                 lam: float = 0.3, top_k: int = 5):
        self.cache = cache
        self.tokenizer = tokenizer
        self.lam = lam
        self.top_k = top_k

    def compute_cache_logits(self, query: str, vocab_size: int,
                             device: str = "cuda") -> Optional[torch.Tensor]:
        """
        Build a calibration logits distribution from retrieved cache entries.
        Boosts answer tokens weighted by retrieval similarity.
        """
        retrieved = self.cache.retrieve(query, self.top_k)
        if not retrieved:
            return None

        boost = torch.zeros(1, vocab_size, device=device)
        n_tokens = 0
        for r in retrieved:
            answer_text = r.get("answer", "")
            if not answer_text:
                continue
            tokens = self.tokenizer.encode(answer_text, add_special_tokens=False)
            for tid in tokens:
                if 0 <= tid < vocab_size:
                    boost[0, tid] += r["sim"]
                    n_tokens += 1

        if n_tokens == 0:
            return None

        # Normalize: mean-center so boosted tokens stand out
        boost = boost / max(n_tokens, 1)
        boost = boost - boost.mean()
        return boost

    def calibrate(self, logits: torch.Tensor, query: str) -> torch.Tensor:
        """l' = (1-lam)*l + lam * l_cache"""
        cache_logits = self.compute_cache_logits(
            query, logits.shape[-1], logits.device
        )
        if cache_logits is None:
            return logits

        return (1 - self.lam) * logits + self.lam * cache_logits
