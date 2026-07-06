"""
评估指标工具
"""
import re
import json
import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter
from difflib import SequenceMatcher


def normalize_answer(answer: str) -> str:
    """标准化答案文本"""
    answer = answer.strip().lower()
    answer = re.sub(r'[^\w\s]', '', answer)
    answer = re.sub(r'\s+', ' ', answer)
    return answer.strip()


def exact_match(pred: str, gold: str) -> bool:
    """精确匹配"""
    return normalize_answer(pred) == normalize_answer(gold)


def token_f1(pred: str, gold: str) -> float:
    """Token级别F1"""
    pred_tokens = set(normalize_answer(pred).split())
    gold_tokens = set(normalize_answer(gold).split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def contains_answer(pred: str, gold: str) -> bool:
    """答案包含匹配"""
    return normalize_answer(gold) in normalize_answer(pred)


def string_similarity(s1: str, s2: str) -> float:
    """字符串相似度"""
    return SequenceMatcher(None, normalize_answer(s1), normalize_answer(s2)).ratio()


def extract_claims(text: str) -> List[str]:
    """
    从文本中提取独立的医学声明
    基于句号、分号、换行等分隔
    """
    # 按句子分隔
    sentences = re.split(r'[.;\n]+', text)
    claims = []
    for s in sentences:
        s = s.strip()
        if len(s) > 10:  # 过滤过短的片段
            claims.append(s)
    return claims if claims else [text.strip()]


def compute_claim_support_score(claim: str, evidence_texts: List[str]) -> float:
    """
    计算声明的证据支持分数
    基于关键词重叠
    """
    claim_tokens = set(normalize_answer(claim).split())
    if not claim_tokens:
        return 0.0

    max_support = 0.0
    for evidence in evidence_texts:
        evidence_tokens = set(normalize_answer(evidence).split())
        if not evidence_tokens:
            continue
        overlap = claim_tokens & evidence_tokens
        support = len(overlap) / len(claim_tokens) if claim_tokens else 0.0
        max_support = max(max_support, support)
    return max_support


def compute_multiview_consistency(claim: str, view_outputs: List[str]) -> float:
    """
    计算声明的多视图一致性分数
    在多个视图输出中是否稳定出现
    """
    if not view_outputs:
        return 0.0

    claim_normalized = normalize_answer(claim)
    consistent_count = 0
    for output in view_outputs:
        if claim_normalized in normalize_answer(output):
            consistent_count += 1
    return consistent_count / len(view_outputs)


def classify_hallucination_type(pred: str, gold: str, question: str = "") -> Dict[str, bool]:
    """
    分类幻觉类型
    """
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    return {
        "is_hallucination": not exact_match(pred, gold) and token_f1(pred, gold) < 0.3,
        "is_partial_match": token_f1(pred, gold) >= 0.3 and not exact_match(pred, gold),
        "is_correct": exact_match(pred, gold) or token_f1(pred, gold) >= 0.8,
        "is_contradictory": False,  # 需要更复杂的NLI判断
        "is_unsupported": token_f1(pred, gold) < 0.2,
    }


def compute_hallucination_rate(predictions: List[str], references: List[str]) -> float:
    """
    计算幻觉率
    """
    if not predictions:
        return 0.0
    hallucination_count = 0
    for pred, ref in zip(predictions, references):
        if token_f1(pred, ref) < 0.2 and not contains_answer(pred, ref):
            hallucination_count += 1
    return hallucination_count / len(predictions)


def soft_em(pred: str, gold: str) -> bool:
    """
    Soft Exact Match (标准化后精确匹配)
    大多数Medical VQA论文的标准做法:
    - 转小写
    - 去除标点
    - 去除冠词(a/an/the)
    - 去除多余空格
    """
    def normalize_soft(s):
        s = s.strip().lower()
        s = re.sub(r'^(a|an|the)\s+', '', s)
        s = re.sub(r'[.,;:!?"\'()]', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    return normalize_soft(pred) == normalize_soft(gold)


def vqa_accuracy(pred: str, gold: str) -> bool:
    """
    VQA Accuracy (宽松包含匹配)
    ARegionCD和大多数VQA论文使用:
    answer.lower() in prediction.lower()

    更宽松: 允许预测包含额外信息
    例如: GT="CT" Pred="CT scan of the chest" → True
    """
    gold_norm = gold.strip().lower()
    pred_norm = pred.strip().lower()
    return gold_norm in pred_norm or pred_norm in gold_norm


def bleu_score(pred: str, gold: str, n: int = 1) -> float:
    """
    BLEU-N score (n-gram精度)
    """
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1)]
    gold_ngrams = [tuple(gold_tokens[i:i+n]) for i in range(len(gold_tokens)-n+1)]

    if not pred_ngrams or not gold_ngrams:
        return 0.0

    gold_counter = Counter(gold_ngrams)
    pred_counter = Counter(pred_ngrams)

    clipped = sum(min(pred_counter[g], gold_counter[g]) for g in pred_counter)
    total = sum(pred_counter.values())

    precision = clipped / total if total > 0 else 0.0

    # Brevity penalty
    bp = min(1.0, len(pred_tokens) / max(len(gold_tokens), 1))
    return bp * precision


def rouge_l(pred: str, gold: str) -> float:
    """
    ROUGE-L (最长公共子序列)
    """
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0

    # LCS
    m, n = len(pred_tokens), len(gold_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == gold_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs_len = dp[m][n]

    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n
    return 2 * precision * recall / (precision + recall)


def meteor_score(pred: str, gold: str) -> float:
    """
    METEOR简化版 (考虑词干和同义词)
    """
    pred_tokens = set(normalize_answer(pred).split())
    gold_tokens = set(normalize_answer(gold).split())
    if not pred_tokens or not gold_tokens:
        return 0.0

    # 精确匹配
    exact_matches = pred_tokens & gold_tokens

    # 词干匹配 (简单后缀去除)
    def stem(word):
        for suffix in ['ing', 'tion', 'ment', 'ness', 'ous', 'ive', 'ly', 'ed', 'er', 'est', 's']:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word

    pred_stems = {stem(w) for w in pred_tokens}
    gold_stems = {stem(w) for w in gold_tokens}
    stem_matches = pred_stems & gold_stems

    # 综合匹配
    total_matches = len(exact_matches) + 0.5 * max(0, len(stem_matches) - len(exact_matches))

    precision = total_matches / len(pred_tokens)
    recall = total_matches / len(gold_tokens)

    if precision + recall == 0:
        return 0.0
    return precision * recall / (0.9 * precision + 0.1 * recall)


def semantic_similarity(pred: str, gold: str) -> float:
    """
    语义相似度 (基于SequenceMatcher)
    """
    return string_similarity(pred, gold)


def compute_token_confidence(token_probs: List[float]) -> Dict[str, float]:
    """
    Token-level置信度分析 (ARegionCD方法)

    Args:
        token_probs: 每个生成token的softmax概率

    Returns:
        置信度统计
    """
    if not token_probs:
        return {"mean": 0.0, "min": 0.0, "geometric": 0.0, "low_conf_ratio": 1.0}

    probs = np.array(token_probs)
    return {
        "mean": float(np.mean(probs)),
        "min": float(np.min(probs)),
        "geometric": float(np.exp(np.mean(np.log(probs + 1e-10)))),
        "low_conf_ratio": float(np.mean(probs < 0.5)),  # 低置信token比例
    }


def compute_metrics(predictions: List[str], references: List[str],
                    question_types: Optional[List[str]] = None,
                    token_probs_list: Optional[List[List[float]]] = None) -> Dict[str, Any]:
    """
    计算完整评估指标 (多维度)

    包含:
    1. EM (Exact Match) - 严格精确匹配
    2. Soft EM - 标准化后精确匹配 (主流medical VQA)
    3. VQA Acc - 宽松包含匹配 (ARegionCD)
    4. Token F1 - token级别F1
    5. BLEU-1 - unigram精度
    6. ROUGE-L - 最长公共子序列
    7. METEOR - 词干+同义词匹配
    8. Hallucination Rate - 幻觉率
    """
    metrics = {
        "total": len(predictions),
        "em": 0,
        "soft_em": 0,
        "vqa_acc": 0,
        "f1": [],
        "bleu1": [],
        "rouge_l": [],
        "meteor": [],
        "contains": 0,
        "hallucination_rate": 0.0,
    }

    for idx, (pred, ref) in enumerate(zip(predictions, references)):
        if exact_match(pred, ref):
            metrics["em"] += 1
        if soft_em(pred, ref):
            metrics["soft_em"] += 1
        if vqa_accuracy(pred, ref):
            metrics["vqa_acc"] += 1
        if contains_answer(pred, ref):
            metrics["contains"] += 1

        metrics["f1"].append(token_f1(pred, ref))
        metrics["bleu1"].append(bleu_score(pred, ref, n=1))
        metrics["rouge_l"].append(rouge_l(pred, ref))
        metrics["meteor"].append(meteor_score(pred, ref))

    n = max(len(predictions), 1)
    metrics["em_rate"] = metrics["em"] / n
    metrics["soft_em_rate"] = metrics["soft_em"] / n
    metrics["vqa_acc_rate"] = metrics["vqa_acc"] / n
    metrics["avg_f1"] = np.mean(metrics["f1"]) if metrics["f1"] else 0.0
    metrics["avg_bleu1"] = np.mean(metrics["bleu1"]) if metrics["bleu1"] else 0.0
    metrics["avg_rouge_l"] = np.mean(metrics["rouge_l"]) if metrics["rouge_l"] else 0.0
    metrics["avg_meteor"] = np.mean(metrics["meteor"]) if metrics["meteor"] else 0.0
    metrics["contains_rate"] = metrics["contains"] / n
    metrics["hallucination_rate"] = compute_hallucination_rate(predictions, references)

    # Token置信度分析 (如果有)
    if token_probs_list:
        all_confs = [compute_token_confidence(tp) for tp in token_probs_list if tp]
        if all_confs:
            metrics["token_confidence"] = {
                "mean": np.mean([c["mean"] for c in all_confs]),
                "geometric_mean": np.mean([c["geometric"] for c in all_confs]),
                "low_conf_ratio": np.mean([c["low_conf_ratio"] for c in all_confs]),
            }

    # 按问题类型分组 (如果有)
    if question_types:
        type_metrics = {}
        for qtype in set(question_types):
            indices = [i for i, t in enumerate(question_types) if t == qtype]
            type_preds = [predictions[i] for i in indices]
            type_refs = [references[i] for i in indices]
            type_em = sum(1 for p, r in zip(type_preds, type_refs) if soft_em(p, r))
            type_metrics[qtype] = {
                "count": len(indices),
                "em_rate": type_em / max(len(indices), 1),
                "f1": np.mean([token_f1(p, r) for p, r in zip(type_preds, type_refs)]),
            }
        metrics["by_question_type"] = type_metrics

    return metrics
    metrics["contains_rate"] = metrics["contains"] / max(len(predictions), 1)
    metrics["hallucination_rate"] = compute_hallucination_rate(predictions, references)

    # 按问题类型分组
    if question_types:
        for qtype in set(question_types):
            indices = [i for i, t in enumerate(question_types) if t == qtype]
            type_preds = [predictions[i] for i in indices]
            type_refs = [references[i] for i in indices]
            metrics[f"{qtype}_em"] = sum(1 for p, r in zip(type_preds, type_refs) if exact_match(p, r)) / max(len(type_preds), 1)
            metrics[f"{qtype}_f1"] = np.mean([token_f1(p, r) for p, r in zip(type_preds, type_refs)]) if type_preds else 0.0
            metrics[f"{qtype}_hallucination"] = compute_hallucination_rate(type_preds, type_refs)

    return metrics


def format_metrics(metrics: Dict[str, Any]) -> str:
    """格式化输出指标"""
    lines = [
        "=" * 60,
        "Evaluation Metrics",
        "=" * 60,
        f"Total samples:     {metrics['total']}",
        f"Exact Match:       {metrics['em_rate']:.4f} ({metrics['em']}/{metrics['total']})",
        f"Token F1:          {metrics['avg_f1']:.4f}",
        f"Contains Match:    {metrics['contains_rate']:.4f}",
        f"Hallucination Rate:{metrics['hallucination_rate']:.4f}",
        "-" * 60,
    ]
    for key in sorted(metrics.keys()):
        if key.endswith("_em") or key.endswith("_f1") or key.endswith("_hallucination"):
            lines.append(f"  {key}: {metrics[key]:.4f}")
    lines.append("=" * 60)
    return "\n".join(lines)
