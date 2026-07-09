# 算法原理

---

## 1. AdaIN (Adaptive Instance Normalization)

### 原理
将测试图像在 ViT 每层的特征统计量（mean/std）对齐到源域训练集的特征分布。

### 核心公式

```
对于当前层特征 f:
  f_norm = (f - μ(f)) / σ(f)       # 标准化当前特征
  f_adain = f_norm × σ_src + μ_src  # 反标准化到源域分布
```

其中 μ_src, σ_src 是源域中所有图像在该层的特征均值和标准差。

### 源域统计量定义

```
源域中心 = {
  layer_0:  {mean: (1152,), std: (1152,)},
  layer_1:  {mean: (1152,), std: (1152,)},
  ...
  layer_54: {mean: (1152,), std: (1152,)},
  post_layernorm: {mean: (1152,), std: (1152,)}
}  # 共 55 层
```

Vision Encoder 共 27 层 Transformer，每层 2 个 LayerNorm + 1 个 post_layernorm = 55 个 LN 层。每层特征维度 1152。

### 实现方式

```python
# Forward hook on vision encoder LayerNorm nodes
class AdaINHook:
    def __call__(self, module, input, output):
        feat = output.float()
        feat_mean = feat.mean(dim=...)   # 当前特征统计量
        feat_std  = feat.std(dim=...) + 1e-5
        result = (feat - feat_mean) / feat_std * σ_src + μ_src
        return result.to(orig_dtype)
```

### 统计量计算方法

- **旧版** (Phase 0 前): 30 张 KB 图像, 55 层全对齐, 运行均值近似
- **新版** (Phase 0 后): 500 张 KB 图像, 仅前 1/3 层 (18/55), Welford 在线算法

### 参数

| 参数 | 旧版 | 新版 |
|------|------|------|
| 样本数 | ~30 | 500 |
| 对齐层数 | 55 (全部) | 18 (前1/3) |
| 统计量算法 | 运行均值 | Welford 在线 |

### 失效原因

1. 扰动了 ViT 后期语义层的特征分布（后期层编码语义信息而非域信息）
2. 30 张统计量不足以代表完整的源域分布
3. Hard AdaIN 无条件执行，无法区分"域偏移"和"语义差异"

---

## 2. VCD (Visual Contrastive Decoding)

### 来源
CVPR 2024 Highlight — "Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding"

### 原理
对比原始图像和噪声图像的模型输出分布，抑制语言先验偏差。

### 核心公式

```
l_vcd = (1 + α) · l(v, x) - α · l(v', x)
```

其中:
- l(v, x): 原始图像 v 的 logits
- l(v', x): 噪声图像 v' 的 logits
- α ∈ [0.5, 1.0]: 对比强度（原论文默认 α=1.0）

### 噪声生成

```python
def diffusion_noise(img, steps=500):
    # 逐步加噪: 每10步加 0.02 标准差高斯噪声
    arr = np.array(img) / 255.0
    for _ in range(steps // 10):
        arr = np.clip(arr + np.random.randn(*arr.shape) * 0.02, 0, 1)
    return Image.fromarray(arr * 255)
```

### LogitsProcessor 实现

```python
class VCDLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        out_noise = self.model.model(**noise_inputs)  # 二次前向
        logits_noise = out_noise.logits[:, -1, :]
        return (1 + self.alpha) * scores - self.alpha * logits_noise
```

### 两种解码模式

| 模式 | 设置 | 效果 |
|------|------|------|
| Greedy VCD | do_sample=False, t=0 | 模型输出几乎不变（验证贪心锁死） |
| Sampling VCD | do_sample=True, t=0.7, top_p=0.9 | 输出变化但质量下降 |
| Pure Sampling | do_sample=True, t=0.7, 无VCD | -10% 对比 baseline |

### 失效原因

1. **贪心锁死**: 14B 模型的 logits 排序稳定度极高, 正确/错误 token 的 logits 差距 >4。VCD 的线性对比 (1+α)l - α·l' 无法翻转如此大的排序差。需要 logits 差距数百才能改变 argmax。
2. **噪声质量**: 简单的加性高斯/扩散噪声在医学图像上可能破坏了关键诊断特征，导致噪声分支也输出合理但不同的答案。
3. **模型规模**: VCD 在 LLaVA-7B/13B 上有效，但在 Hulu-Med 14B 上无效。模型越大，logits 越稳定（robustness increase with scale）。这是新发现。

---

## 3. RAG-KI (RAG Knowledge Injection)

### 变体 A: 减法对比（原始方案）

```
l_rag = (1 + β) · l(v, x) - β · l(v, x_rag)
```

其中 x_rag 是在 prompt 中加入检索知识的增强输入。

**设计错误**: 这个公式把 RAG 上下文当作"对比负样本"去减。VCD 的减项是噪声图像（更差），而 RAG 上下文是增强输入（更好），减掉它等于惩罚正确知识。

### 变体 B: 加法实体增强（修正方案）

```python
class EntityBoostLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        boost = torch.zeros_like(scores)
        for token_id in self.entity_ids:
            boost[:, token_id] = self.boost_weight
        return scores + boost
```

将检索答案中的 token ID 做加性增强（β=1.5或2.0）。

### 检索方式

```python
# sentence-BERT (all-MiniLM-L6-v2) text embedding 余弦检索
query_emb = encoder.encode([question])
scores = cosine_similarity(query_emb, kb_embeddings)
top_idx = argsort(scores)[::-1][:top_k]
```

### 知识库

| 模态 | 条目数 | 来源 |
|------|--------|------|
| X-ray | 1,500 | SLAKE (500) + VQA-RAD (500) + IU-Xray (500) |
| Pathology | 2,002 | PathVQA (2002) |

### 失效原因（两种变体）

| 问题 | 减法变体 | 加法变体 |
|------|---------|---------|
| 方向 | 减去增强输入，方向反 | 方向对，但粒度不匹配 |
| Token 不匹配 | — | 答案 "Spinal Cord" → boost [Spinal, Cord], 但模型需要输出 "Yes"/"No" |
| 矛盾条目 | KB 中同一问题有 yes 和 no 两种答案 | 同左，boost 相互抵消 |
| 检索质量 | text-only, 无法感知图像内容 | 同左 |
| 多选问题 | — | 选项答案 A/B/C/D 与全文不匹配 |

---

## 4. Cache-based TTA (Test-Time Adaptation)

### 思路
将"源域中心"改造为 cache-based 校准：用预构建的 KB 文本特征索引 + 测试时检索 + logit 校准，避免 AdaIN 对特征层的直接扰动。

### 算法流程

```
输入图像 + 问题
    ↓
sentence-BERT 编码问题文本
    ↓
余弦检索 KB 中 top-k 相似 QA 对
    ↓
提取检索答案的 token ID
    ↓
构建校准分布: l_cache = sum(sim_i * onehot(token_i))  # 相似度加权
    ↓
均值和中心化: l_cache = l_cache / n_tokens - mean(l_cache)
    ↓
插值: l' = (1 - λ) · l_original + λ · l_cache
    ↓
解码生成
```

### Cache 数据结构

```python
class TextFeatureCache:
    embeddings: np.ndarray   # (N, 384) — sentence-BERT
    entries: List[dict]     # N 条 {question, answer, source}
```

TextFeatureCache 复用 KB 构建时的 `embeddings.npy`，无需额外 GPU 前向。

### LogitsProcessor 实现

```python
class CacheCalibrationProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        return self.decoder.calibrate(scores, self.query)

class CacheTTADecoder:
    def compute_cache_logits(self, query, vocab_size, device):
        retrieved = self.cache.retrieve(query, self.top_k)
        for r in retrieved:
            tokens = tokenizer.encode(r["answer"])
            for tid in tokens:
                boost[:, tid] += r["sim"]
        boost = boost / n_tokens - boost.mean()
        return boost

    def calibrate(self, logits, query):
        cache_logits = self.compute_cache_logits(query, ...)
        if cache_logits is None:
            return logits
        return (1 - self.lam) * logits + self.lam * cache_logits
```

### 参数实验

| λ (lam) | top_k | MM-VisHal 200 | MM-VisHal 2000 |
|---------|-------|---------------|----------------|
| — | — | 47.5% (baseline) | 47.42% (baseline) |
| 0.10 | 5 | 48.0% (+0.5%) | 47.12% (-0.3%) |
| 0.15 | 5 | 48.0% (+0.5%) | 47.12% (-0.3%) |
| 0.20 | 5 | 48.0% (+0.5%) | 47.12% (-0.3%) |
| 0.25 | 5 | 48.0% (+0.5%) | 47.22% (-0.2%) |
| 0.15 | 10 | 47.0% (-0.5%) | 46.97% (-0.5%) |

200 条样本的 +0.5% = 恰好 1 条样本翻正，全量 2000 条验证为假阳性。

### 失效原因

详见 `FAILURE_ANALYSIS.md` 第四章。

---

## 5. Conformal Safety Layer（待实现）

### 原理
为每个预测计算置信度分数，当置信度低于阈值时弃权，提供无分布假设的覆盖度保证。

### 置信度指标

```python
def get_confidence_scores(logits):
    probs = softmax(logits)
    top2 = topk(probs, 2)
    return {
        "max_prob": float(top2.probs[0]),
        "margin": float(top2.probs[0] - top2.probs[1]),
        "entropy": float(-sum(probs * log(probs))),
        "logit_margin": float(topk(logits, 2).values[0] - topk(logits, 2).values[1]),
    }
```

### 校准方法

```
1. 在 calibration set 上计算所有样本的置信度
2. 寻找阈值 T 使得 P(correct | score >= T) >= target_coverage
3. 测试时: score >= T 则回答, 否则弃权
4. 输出 selective accuracy @ coverage 曲线
```

### 参考
- CAP: Conformalized Abstention Policies (PMLR 2026)
- HAC: Hallucination-Aware Calibration (arXiv 2604.02543)
- CARE: Conformal Safety Layer for Medical Summarization (2026)

---

## 6. QLoRA DPO（待实现）

### 原理
用 Low-Rank Adaptation (LoRA) 做参数高效微调，用 Direct Preference Optimization (DPO) 让模型偏好正确回答、抑制幻觉回答。

### 量化

QLoRA: 4-bit NormalFloat (NF4) 量化基座模型权重，保持 LoRA 适配器为 float16/32。14B 模型 4-bit 后约 7-8GB。

### 偏好对构造

```
prompt + 图像 → 模型生成 → 正确回答 (chosen)
                         → 错误/幻觉回答 (rejected)
```

使用 MedHEval 的 ground truth 答案自动构造偏好对，无需人工标注。

### DPO Loss

```
L_DPO = -E[log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x)
                    - log π_θ(y_l|x) + log π_ref(y_l|x)))]
```

其中 y_w 是 chosen (正确), y_l 是 rejected (错误)。

### 参考
- OPA-DPO (CVPR 2025): On-Policy Alignment, 4.8k 数据降 13.26%
- TPR (NeurIPS 2025): Topic-level Preference Rewriting, SOTA +20%
- MMed-RAG (ICLR 2025): RAG-based preference fine-tuning
- RLHF-V (2024): Human-annotated correctional feedback
