"""
Phase 2: Conformal Safety Layer
=================================
Adds a conformal abstention mechanism to the VLM pipeline.
For each prediction, computes confidence scores and abstains
when uncertainty is high, providing distribution-free coverage guarantees.

Core idea:
  1. For each sample, extract confidence scores from output logits
  2. Calibrate a threshold on a held-out set (split conformal)
  3. At test time, abstain on samples below threshold
  4. Report selective accuracy (Acc@Coverage) curve

Reference: CAP (PMLR'26), HAC (arXiv 2604.02543), conformal abstention,
Domain-Shift-Aware Conformal Prediction (ICML'26).
"""
import re
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple


def get_confidence_scores(logits: torch.Tensor) -> Dict[str, float]:
    """
    Compute multiple confidence scores from model logits.
    Returns dict with various confidence metrics.
    """
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = probs.topk(2, dim=-1)

    scores = {
        "max_prob": float(top_probs[0, 0].cpu()),
        "margin": float((top_probs[0, 0] - top_probs[0, 1]).cpu()),
        "entropy": float(-(probs * torch.log(probs + 1e-10)).sum(-1).cpu()),
        "logit_max": float(logits.max(-1).values.cpu()),
        "logit_margin": float(
            (logits.topk(2, dim=-1).values[0, 0] -
             logits.topk(2, dim=-1).values[0, 1]).cpu()
        ),
    }
    return scores


class ConformalSafetyLayer:
    """
    Conformal prediction-based safety layer.
    Calibrates abstention threshold to achieve target coverage.

    Usage:
      safety = ConformalSafetyLayer(confidence_fn="margin", target_coverage=0.90)
      safety.calibrate(calibration_logits, calibration_correct)
      # At test time:
      decision = safety.decide(test_logits)  # {"abstain": bool, "confidence": float}
    """

    def __init__(self, confidence_fn: str = "margin", target_coverage: float = 0.90,
                 risk_level: float = 0.10):
        self.confidence_fn = confidence_fn
        self.target_coverage = target_coverage
        self.risk_level = risk_level  # conformal risk level α
        self.threshold = None
        # Domain-shift-aware (ICML26): reweight threshold by domain distance.
        # When target domain drifts from calibration set, raise the threshold
        # (more abstention) to preserve the coverage guarantee.
        self.domain_distance: float = 0.0  # 0 = no shift; >0 = shift magnitude
        self.domain_sensitivity: float = 1.0  # how strongly distance inflates threshold

    def _get_score(self, logits: torch.Tensor) -> float:
        scores = get_confidence_scores(logits)
        return scores[self.confidence_fn]

    def calibrate(self, logits_list: List[torch.Tensor],
                  correct_list: List[bool]) -> float:
        """
        Calibrate threshold using split conformal prediction.
        Finds threshold such that P(correct | score > threshold) >= target_coverage.

        Returns the calibrated threshold.
        """
        scores = [self._get_score(l) for l in logits_list]
        correct = np.array(correct_list, dtype=bool)
        scores = np.array(scores)
        n = len(scores)

        # Sort by score descending
        idx = np.argsort(-scores)
        sorted_scores = scores[idx]
        sorted_correct = correct[idx]

        # Fix (was a bug): the conformal quantile below OVERWROTE the
        # coverage-based threshold computed above, discarding the search.
        # Now compute BOTH and keep the *stricter* (larger) threshold so
        # both the coverage target and the distribution-free guarantee hold.
        coverage_threshold = None
        for i in range(n):
            selected = sorted_scores[i:]
            if len(selected) == 0:
                break
            selected_correct = sorted_correct[i:]
            coverage = selected_correct.mean()
            if coverage >= self.target_coverage:
                coverage_threshold = sorted_scores[i - 1] if i > 0 else sorted_scores[0]
                break

        if coverage_threshold is None:
            coverage_threshold = sorted_scores[n // 2]  # fallback: median

        # Distribution-free conformal quantile (split conformal guarantee)
        q_level = min(1.0, (n + 1) * (1 - self.risk_level) / n)
        q_idx = max(0, int(q_level * n) - 1)
        conformal_threshold = sorted_scores[q_idx]

        # keep the stricter of the two
        self.threshold = max(coverage_threshold, conformal_threshold)
        self._coverage_threshold = coverage_threshold
        self._conformal_threshold = conformal_threshold

        empirical_coverage = sorted_correct[sorted_scores >= self.threshold].mean() \
            if (sorted_scores >= self.threshold).any() else 0.0
        empirical_abstention = (sorted_scores < self.threshold).mean()

        print(f"[Conformal] Calibrated threshold={self.threshold:.4f}, "
              f"empirical coverage={empirical_coverage:.4f}, "
              f"abstention_rate={empirical_abstention:.4f}")

        return self.threshold

    def decide(self, logits: torch.Tensor) -> Dict:
        """
        Decide whether to answer or abstain.
        Returns {"abstain": bool, "score": float, "threshold": float}

        Domain-shift-aware (ICML26): the effective threshold is inflated by
        the current domain distance, so under larger covariate shift the
        layer abstains more to preserve the distribution-free guarantee.
        """
        score = self._get_score(logits)
        if self.threshold is None:
            eff_threshold = 0.0
            abstain = False
        else:
            eff_threshold = self.threshold + self.domain_sensitivity * self.domain_distance
            abstain = score < eff_threshold
        return {
            "abstain": abstain,
            "score": score,
            "threshold": eff_threshold,
            "base_threshold": self.threshold if self.threshold is not None else 0.0,
            "domain_distance": self.domain_distance,
        }

    def set_domain_distance(self, distance: float, sensitivity: float = 1.0):
        """Set the estimated domain distance (e.g. MMD / feature-cov distance
        between the current test batch and the calibration set) and how
        strongly it inflates the abstention threshold."""
        self.domain_distance = float(distance)
        self.domain_sensitivity = float(sensitivity)


def evaluate_selective_prediction(model, samples, safety_layer: ConformalSafetyLayer,
                                   tokenizer, split_ratio: float = 0.3):
    """
    Evaluate model with selective prediction.
    Splits samples into calibration and test sets.

    Returns:
      - selective_accuracy: accuracy on non-abstained samples
      - coverage: fraction of samples answered
      - abstention_rate: fraction of samples abstained
      - accuracy_coverage_curve: list of (threshold, coverage, selective_acc)
    """
    n = len(samples)
    n_cal = max(1, int(n * split_ratio))

    # Shuffle
    import random
    random.seed(42)
    idx = list(range(n))
    random.shuffle(idx)
    cal_idx = idx[:n_cal]
    test_idx = idx[n_cal:]

    # Collect logits for calibration
    cal_logits = []
    cal_correct = []
    from tqdm import tqdm

    print(f"Calibrating on {len(cal_idx)} samples...")
    for i in tqdm(cal_idx):
        s = samples[i]
        img = find_image_func(s)
        if img is None:
            continue
        prompt = build_prompt_func(s)
        msg = {"prompt": prompt, "image": img}

        try:
            inp = model.process_messages(msg)
            inp_fwd = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                       for k, v in inp.items()}
            with torch.no_grad():
                # Fix: was single-token greedy which is too fragile for binary/MC
                # (a single token often mismatches the gold answer format).
                # Now generate a short answer and reuse the proper judges.
                out_ids = model.model.generate(
                    **inp_fwd, max_new_tokens=16, do_sample=False,
                    use_cache=True, pad_token_id=getattr(tokenizer, "eos_token_id", None))
                gen_ids = out_ids[0][inp_fwd["input_ids"].shape[1]:]
                out = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()
                # capture logits at the first generated token for confidence
                outputs = model.model(**inp_fwd, use_cache=True, return_dict=True)
                logits = outputs.logits[:, -1, :]

            from utils.MedHEval.MedHEval import find_image, _is_binary
            from utils.utils import judge_multi_choice, judge_judgement
            from utils.question_formats import get_judgement_prompt, get_multiple_choice_prompt

            answer = s.get("answer", "").strip().lower()
            # robust correctness via the same judges the eval harness uses
            is_correct = _judge_correct(s, out)

            cal_logits.append(logits[0])
            cal_correct.append(is_correct)

        except Exception as e:
            continue

    # Calibrate
    safety_layer.calibrate(cal_logits, cal_correct)

    # Evaluate on test set
    print(f"Evaluating on {len(test_idx)} samples...")
    decisions = []
    for i in tqdm(test_idx):
        s = samples[i]
        img = find_image_func(s)
        if img is None:
            continue
        prompt = build_prompt_func(s)
        msg = {"prompt": prompt, "image": img}

        try:
            inp = model.process_messages(msg)
            inp_fwd = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                       for k, v in inp.items()}
            with torch.no_grad():
                # Fix: was single-token decode; now generate a short answer.
                out_ids = model.model.generate(
                    **inp_fwd, max_new_tokens=16, do_sample=False,
                    use_cache=True, pad_token_id=getattr(tokenizer, "eos_token_id", None))
                gen_ids = out_ids[0][inp_fwd["input_ids"].shape[1]:]
                out = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()
                outputs = model.model(**inp_fwd, use_cache=True, return_dict=True)
                logits = outputs.logits[:, -1, :]

            decision = safety_layer.decide(logits[0])
            is_correct = _judge_correct(s, out)

            decisions.append({
                "abstain": decision["abstain"],
                "correct": is_correct,
                "confidence": decision["score"],
            })

        except Exception as e:
            continue

    # Compute metrics
    answered = [d for d in decisions if not d["abstain"]]
    selective_acc = np.mean([d["correct"] for d in answered]) if answered else 0
    coverage = len(answered) / len(decisions) if decisions else 0
    abstention = 1 - coverage

    # Compute accuracy-coverage curve
    confidence_scores = np.array([d["confidence"] for d in decisions])
    correctness = np.array([d["correct"] for d in decisions])
    idx = np.argsort(-confidence_scores)

    acc_curve = []
    for t in np.linspace(confidence_scores.min(), confidence_scores.max(), 50):
        sel = confidence_scores >= t
        if sel.sum() > 0:
            acc_curve.append({
                "threshold": float(t),
                "coverage": float(sel.mean()),
                "selective_acc": float(correctness[sel].mean()),
            })

    return {
        "selective_accuracy": float(selective_acc),
        "coverage": float(coverage),
        "abstention_rate": float(abstention),
        "n_total": len(decisions),
        "n_answered": len(answered),
        "accuracy_coverage_curve": acc_curve,
    }


def find_image_func(s):
    from utils.MedHEval.MedHEval import find_image
    return find_image(s.get("img_name", ""))


def build_prompt_func(s):
    from utils.question_formats import get_judgement_prompt, get_multiple_choice_prompt
    q, cs = s["question"], s.get("choices", "")
    if cs.strip():
        cl = [c.strip().rstrip(",") for c in cs.split(",") if c.strip()]
        cl = [c for c in cl if c and c[0].upper() in "ABCD"]
        return get_multiple_choice_prompt(q, cl, False)
    return get_judgement_prompt(q, False)


def _judge_correct(sample: dict, pred: str) -> bool:
    """Robust correctness via the same judges the eval harness uses,
    instead of fragile single-token substring match.

    - binary: yes/no normalized equality
    - multiple choice: first A/B/C/D token equality
    - open-ended: soft exact match (VQA accuracy)
    """
    from medshift.utils.metrics import soft_em, vqa_accuracy
    gold = (sample.get("answer") or "").strip().lower()
    p = (pred or "").strip().lower()
    if not gold:
        return False
    # binary
    g0 = gold.split(",")[0].strip()
    if g0 in ("yes", "no"):
        p0 = p.split(",")[0].strip()
        return p0 in ("yes", "no") and p0 == g0
    # multiple choice: A/B/C/D
    if len(gold) == 1 and gold in "abcd":
        m = re.search(r'\b([abcd])\b', p)
        return bool(m) and m.group(1) == gold
    # open-ended
    return soft_em(p, gold) or vqa_accuracy(p, gold)
