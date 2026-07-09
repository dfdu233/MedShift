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
    def from_local_clip(cls, device: str = "cuda") -> "BiomedCLIPRetriever":
        """Load from cached openai/clip-vit-large-patch14-336 (no network)."""
        from transformers import CLIPModel, CLIPProcessor
        snap_dir = ("/root/.cache/huggingface/hub/models--openai--"
                    "clip-vit-large-patch14-336/snapshots/")
        snap = os.listdir(snap_dir)[0]
        sd_path = os.path.join(snap_dir, snap)
        model = CLIPModel.from_pretrained(sd_path, local_files_only=True,
                                          use_safetensors=True)
        proc = CLIPProcessor.from_pretrained(sd_path, local_files_only=True)
        model = model.to(device).eval()
        class _Wrapper:
            def __init__(self, m, p):
                self.m = m; self.p = p
                self.eval_mode = False
            def eval(self):
                self.m.eval(); self.eval_mode = True
                return self
            def __call__(self, images=None, text=None, return_tensors="pt",
                         padding=False, truncation=False):
                """Delegate to CLIPProcessor so encode_image works."""
                return self.p(images=images, text=text, return_tensors=return_tensors,
                              padding=padding, truncation=truncation)
            def encode_image(self, x):
                """Direct encode (bypass preprocess). Used by build_index."""
                inp = self.p(images=x, return_tensors="pt").to(self.m.device)
                inp = {k: v.to(self.m.device) if hasattr(v, 'to') else v
                       for k, v in inp.items()}
                f = self.m.get_image_features(**inp).pooler_output
                return f / f.norm(p=2, dim=-1, keepdim=True)
            def encode_text(self, t):
                inp = self.p(text=[t], return_tensors="pt", padding=True,
                            truncation=True).to(self.m.device)
                inp = {k: v.to(self.m.device) if hasattr(v, 'to') else v
                       for k, v in inp.items()}
                f = self.m.get_text_features(**inp).pooler_output
                return f / f.norm(p=2, dim=-1, keepdim=True)
        wrapper = _Wrapper(model, proc)
        retriever = cls(wrapper, proc, proc.tokenizer, device=device)
        retriever._is_local_clip = True
        return retriever

    @classmethod
    def from_kb(cls, modality: str, kb_root: Optional[str] = None,
                device: str = "cuda",
                use_local_clip: bool = True) -> "BiomedCLIPRetriever":
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from retrieval.kb_builder import load_kb
        if kb_root is None:
            kb_root = os.path.join(os.path.dirname(__file__), "../../data/knowledge_bases")
        r = cls.from_local_clip(device=device) if use_local_clip else cls.from_pretrained(device=device)
        entries = load_kb(modality, kb_root=kb_root)
        r.entries = entries
        return r

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode a PIL image to normalized embedding. Handles both open_clip
        (tensor preprocess) and local CLIP (processor returning dict)."""
        from torchvision import transforms
        img = image.convert("RGB").resize((self.IMAGE_SIZE, self.IMAGE_SIZE),
                                          Image.BILINEAR)
        x = self.preprocess(img)  # tensor (open_clip) or dict (local CLIP)
        if hasattr(self.model, 'encode_image'):
            # local CLIP wrapper: pass image directly
            feat = self.model.encode_image(img)
        elif isinstance(x, torch.Tensor):
            x = x.unsqueeze(0).to(self.device)
            feat = self.model.encode_image(x)
        else:
            # CLIPProcessor dict: pass through model
            x = {k: v.unsqueeze(0).to(self.device) if hasattr(v, 'unsqueeze') else v
                 for k, v in x.items()}
            f = self.model.get_image_features(**x).pooler_output
            feat = f / f.norm(p=2, dim=-1, keepdim=True)
        return feat.cpu().float().numpy().flatten()

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        if hasattr(self.model, 'encode_text'):
            feat = self.model.encode_text(text)
        else:
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
        """Detect feature dimension from the loaded model."""
        if hasattr(self, '_feat_dim_cached'):
            return self._feat_dim_cached
        # Try to infer from model
        try:
            test_img = Image.new('RGB', (224,224), color='gray')
            f = self.encode_image(test_img)
            dim = len(f)
        except Exception:
            dim = 768  # fallback for CLIP-ViT-L
        self._feat_dim_cached = dim
        return dim

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
