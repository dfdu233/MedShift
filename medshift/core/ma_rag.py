"""
Line 2 core: MA-RAG - Multi-round Agentic RAG (conflict -> consensus).

ICML 2026: "From Conflict to Consensus: Boosting Medical Reasoning via
Multi-Round Agentic RAG". Reported +6.8 pts avg on 7 medical QA benchmarks
(for text LLMs). Here adapted for medical VLM (image + question).

Key adaptation of the conflict-as-signal principle to fix the documented
RAG failure (FAILURE_ANALYSIS §3: contradictory KB entries "cancel out" in
logits-boost). MA-RAG turns that cancellation into a *proactive signal*:

  1. Sample N candidate answers from the VLM (high-temp decode or
     evidence-conditioned variants).
  2. Detect semantic conflict among candidates (NLI / embedding disagreement).
  3. For each conflict, build a disambiguating retrieval query and fetch
     BiomedCLIP evidence.
  4. Re-generate with the disambiguating evidence; iterate up to M rounds.
  5. Return consensus (majority / evidence-supported) answer.

This keeps the VLM frozen: it only re-prompts and re-retrieves. It uses
BiomedCLIPRetriever (clip_retriever.py) as the backend.
"""
import os
import re
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from PIL import Image


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'^(a|an|the)\s+', '', s)
    s = re.sub(r'[.,;:!?"\'()]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _short(raw: str) -> str:
    if not raw:
        return ""
    raw = re.sub(r'<.*?>', '', raw, flags=re.DOTALL).strip()
    for p in ["the organ shown", "the answer is", "based on the image,",
              "the image shows", "this image shows:", "in this image,",
              "answer:", "a:"]:
        if raw.lower().startswith(p.lower()):
            raw = raw[len(p):].strip()
    return raw.rstrip('.').strip()[:50]


def detect_conflict(candidates: List[str]) -> Optional[Tuple[str, str]]:
    """Return (a, b) of the most conflicting candidate pair, or None.

    Conflict := normalized answers disagree. For binary/MC, this is exact;
    for OE, we flag when no candidate is a substring of another.
    """
    normed = [(c, _normalize(_short(c))) for c in candidates if c]
    for i in range(len(normed)):
        for j in range(i + 1, len(normed)):
            a, na = normed[i]
            b, nb = normed[j]
            if na != nb and na and nb:
                # stronger conflict if both short (binary/MC style)
                if len(na) <= 4 and len(nb) <= 4:
                    return (a, b)
                if na not in nb and nb not in na:
                    return (a, b)
    return None


def build_disambiguation_query(question: str, a: str, b: str) -> str:
    """Build a retrieval query to disambiguate two conflicting answers."""
    return (f"Which is correct for this image: \"{a}\" or \"{b}\"? "
            f"Question: {question}")


class MARagPipeline:
    """Multi-round agentic RAG around a frozen VLM.

    Args:
        vlm: object with .generate(image, prompt, max_new_tokens, temperature)
        retriever: BiomedCLIPRetriever (image-aware)
        n_candidates: how many candidates to sample per round
        max_rounds: agentic refinement rounds
        temperature: sampling temperature for candidate diversity
    """

    def __init__(self, vlm, retriever, n_candidates: int = 3,
                 max_rounds: int = 2, temperature: float = 0.7,
                 max_new_tokens: int = 32, top_k: int = 3):
        self.vlm = vlm
        self.retriever = retriever
        self.n_candidates = n_candidates
        self.max_rounds = max_rounds
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k

    def _gen_candidates(self, image, question, evidence=None,
                        n=None) -> List[str]:
        from medshift.retrieval.rag_engine import build_rag_prompt
        prompt = build_rag_prompt(question, evidence or [])
        outs = []
        nn = n or self.n_candidates
        for _ in range(nn):
            ans, _ = self.vlm.generate(image, prompt,
                                       max_new_tokens=self.max_new_tokens,
                                       temperature=self.temperature)
            outs.append(ans)
        return outs

    def run(self, image: Image.Image, question: str) -> Dict:
        """Execute MA-RAG. Returns {answer, rounds, candidates_history, conflicts}."""
        history = []
        conflicts = []
        # Round 0: no evidence, sample candidates
        cands = self._gen_candidates(image, question, evidence=None)
        history.append(cands)
        evidence = []
        for r in range(self.max_rounds):
            conf = detect_conflict(cands)
            if conf is None:
                break
            conflicts.append(conf)
            a, b = conf
            query = build_disambiguation_query(question, a, b)
            # image-aware retrieval for disambiguation
            ev = self.retriever.retrieve(image, query, top_k=self.top_k)
            evidence.extend(ev)
            cands = self._gen_candidates(image, question, evidence=evidence)
            history.append(cands)
        # consensus: majority vote on normalized short answer
        final = self._consensus(history[-1])
        return {
            "answer": final,
            "rounds": len(history) - 1,
            "candidates_history": history,
            "conflicts": conflicts,
            "evidence": evidence,
        }

    def _consensus(self, candidates: List[str]) -> str:
        from collections import Counter
        short = [(_short(c) or c) for c in candidates if c]
        if not short:
            return ""
        # majority on normalized; tie-break by first
        cnt = Counter(_normalize(s) for s in short)
        winner = cnt.most_common(1)[0][0]
        for s in short:
            if _normalize(s) == winner:
                return s
        return short[0]
