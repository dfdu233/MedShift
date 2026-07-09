"""
Line 2: BiomedCLIP-based multimodal retrieval for medical VLM RAG.

Replaces the text-only sentence-BERT retrieval (medshift/retrieval/kb_builder.py)
that caused the "visual blind spot" failure (FAILURE_ANALYSIS §3: same question
+ different image -> identical retrieval). BiomedCLIP encodes BOTH image and
text in a shared medical-aligned space, so retrieval is image-aware.

This module:
  - encodes KB images with BiomedCLIP image encoder -> KB image embeddings
  - at query time, encodes the test image -> retrieves nearest KB images/QA
  - returns evidence for prompt injection (rag_engine.build_rag_prompt)
  - dedupes contradictory entries (same Q, opposite A) per FAILURE_ANALYSIS §3

Also provides the retrieval backend for MA-RAG multi-round agentic RAG
(medshift/core/ma_rag.py), which turns candidate conflicts into new queries.
"""
import os
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple
import torch


class BiomedCLIPRetriever:
    """Image-aware medical retrieval using BiomedCLIP.

    Usage:
        r = BiomedCLIPRetriever.from_kb("xray")
        r.build_index()                 # encode KB images (cached)
        ev = r.retrieve(test_image, question, top_k=5)
        prompt = build_rag_prompt(question, ev)
    """

    # BiomedCLIP expects 224x224, mean/std per its config
    IMAGE_SIZE = 224
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, model, preprocess, tokenizer, device="cuda"):
        self.model = model  # open_clip model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.image_embeddings: Optional[np.ndarray] = None  # (N, D)
        self.entries: List[dict] = []

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, device: str = "cuda",
                         local_dir: Optional[str] = None) -> "BiomedCLIPRetriever":
        import open_clip
        repo = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        if local_dir:
            # load from local snapshot path
            model, _, preprocess = open_clip.create_model_and_transforms(
                f"hf-hub:{repo}" if not os.path.isdir(local_dir) else local_dir,
                pretrained=local_dir if os.path.isdir(local_dir) else None,
            )
            tokenizer = open_clip.get_tokenizer(repo)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(
                f"hf-hub:{repo}")
            tokenizer = open_clip.get_tokenizer(repo)
        model = model.to(device)
        return cls(model, preprocess, tokenizer, device=device)

    @classmethod
    def from_kb(cls, modality: str, kb_root: Optional[str] = None,
                device: str = "cuda") -> "BiomedCLIPRetriever":
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from retrieval.kb_builder import load_kb
        if kb_root is None:
            kb_root = os.path.join(os.path.dirname(__file__), "../../data/knowledge_bases")
        r = cls.from_pretrained(device=device)
        entries = load_kb(modality, kb_root=kb_root)
        r.entries = entries
        return r

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        from torchvision import transforms
        img = image.convert("RGB").resize((self.IMAGE_SIZE, self.IMAGE_SIZE),
                                          Image.BILINEAR)
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().float().numpy().flatten()

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        tok = self.tokenizer([text]).to(self.device)
        feat = self.model.encode_text(tok)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().float().numpy().flatten()

    # ------------------------------------------------------------------
    # Index build (cached)
    # ------------------------------------------------------------------
    def build_index(self, image_loader, cache_path: Optional[str] = None,
                    force: bool = False):
        """Encode all KB images. image_loader(entry)->PIL.Image or None."""
        if cache_path and os.path.exists(cache_path) and not force:
            self.image_embeddings = np.load(cache_path)
            print(f"[BiomedCLIP] loaded cached index: {self.image_embeddings.shape}")
            return
        feats = []
        n = 0
        for entry in self.entries:
            img = image_loader(entry)
            if img is None:
                feats.append(np.zeros(self._feat_dim(), dtype=np.float32))
                continue
            feats.append(self.encode_image(img).astype(np.float32))
            n += 1
        self.image_embeddings = np.stack(feats).astype(np.float32)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, self.image_embeddings)
        print(f"[BiomedCLIP] encoded {n}/{len(self.entries)} KB images "
              f"-> {self.image_embeddings.shape}")

    def _feat_dim(self) -> int:
        # BiomedCLIP vit_base_patch16_224 -> 512-dim shared space
        return 512

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, image: Image.Image, question: str, top_k: int = 5,
                 text_weight: float = 0.3,
                 dedup_contradictions: bool = True) -> List[dict]:
        """Image-aware retrieval.

        Combines image similarity with question-text similarity (image_weight=
        1-text_weight). Dedupes contradictory same-question entries.

        Returns list of {sim, question, answer, source, img_name}.
        """
        if self.image_embeddings is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        q_img = self.encode_image(image)
        q_txt = self.encode_text(question)
        img_sims = self.image_embeddings @ q_img
        txt_sims = self.image_embeddings @ q_txt
        sims = (1 - text_weight) * img_sims + text_weight * txt_sims

        order = np.argsort(-sims)
        results = []
        seen_q = {}
        for idx in order:
            e = self.entries[idx]
            q = e.get("question", "").strip().lower()
            a = e.get("answer", "").strip().lower()
            # contradiction dedup: keep only one answer per normalized question
            if dedup_contradictions:
                qk = _normalize_question(q)
                if qk in seen_q:
                    # skip same-question different-answer (the contradiction)
                    if seen_q[qk] != a:
                        continue
                else:
                    seen_q[qk] = a
            results.append({
                "sim": float(sims[idx]),
                "question": e.get("question", ""),
                "answer": e.get("answer", ""),
                "source": e.get("source", ""),
                "img_name": e.get("img_name", ""),
            })
            if len(results) >= top_k:
                break
        return results


def _normalize_question(q: str) -> str:
    import re
    q = re.sub(r'[^\w\s]', '', q.lower())
    q = re.sub(r'\s+', ' ', q).strip()
    return q


# ---------------------------------------------------------------------------
# Default image loader matching existing KB metadata (img_name)
# ---------------------------------------------------------------------------

def default_image_loader(kb_root: str, modality: str):
    def _load(entry):
        img_name = entry.get("img_name", "")
        if not img_name:
            return None
        candidates = [
            os.path.join("/root/autodl-tmp/MedHEval/images/Slake", img_name),
            os.path.join("/root/autodl-tmp/MedHEval/images/VQA-RAD", img_name),
            os.path.join("/root/autodl-tmp/MedHEval/images/IU-Xray", img_name),
            os.path.join(kb_root, modality, img_name),
            os.path.join(kb_root, "pathology", img_name),
        ]
        for c in candidates:
            if os.path.exists(c):
                try:
                    return Image.open(c).convert("RGB")
                except Exception:
                    return None
        return None
    return _load
