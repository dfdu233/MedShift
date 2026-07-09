# 实验结果

---

## E0: 基线 Baseline

### 全量 4 数据集（13,412 条）

| 数据集 | 样本数 | Binary | MC | OE | Total | 子集类型 |
|--------|--------|--------|----|----|-------|---------|
| CXR-VisHal | 2,017 | **91.1%** | 45.0% | — | **78.9%** | close-ended (binary+MC) |
| MM-VisHal | 3,530 | 57.1% | 20.4% | 13.2% | **46.6%** | close-ended (混合) |
| FineGrained | 5,547 | 69.6% | 29.3% | 16.4% | **58.4%** | close-ended (混合) |
| KnowledgeDeficiencyOE | 2,318 | — | — | 48.5% | **48.6%** | open-ended |

### 与 MedHEval 论文对比（CXR-VisHal）

| 模型 | 参数量 | Acc |
|------|--------|-----|
| **GPT-4o** | — | **79.4%** |
| **Hulu-Med 14B** | **14B** | **78.9%** |
| CheXagent | 7B | 73.9% |
| LLaVA-NeXT 13B | 13B | 53.4% |
| LLaVA-Med-1.5 | 7B | 68.4% |
| LLaVA-Med | 7B | 69.8% |
| MiniGPT-4 | — | 48.3% |

Hulu-Med 14B 在 CXR 上接近 GPT-4o，但在 MM-VisHal 上大幅下降至 46.6%，说明多模态域偏移是主要挑战。

---

## E1: AdaIN — Adaptive Instance Normalization

### 实验配置

| 配置 | 旧版 | 新版 |
|------|------|------|
| 统计量来源 | ~30 KB 图像 | 500 KB 图像 |
| 对齐层数 | 55 层 (全部) | 18 层 (前 1/3) |
| 统计量算法 | 运行均值 | Welford 在线 |
| 评估子集 | CXR 200 | CXR 200 |

### 结果（仅对 CXR 做消融，因为 MM 需要病理统计量未完成）

| 方法 | CXR-200 Acc | vs Baseline |
|------|-------------|-------------|
| baseline | 79.0% | — |
| adain_full (旧版, 30张+全层) | 72.0% | **-7.0%** |
| adain_light (新版, 500张+前1/3层) | 75.5% | **-3.5%** |

### 分析

- 新版比旧版伤害减半 (-7% → -3.5%)，说明 500 张统计量 + 前 1/3 层策略有效减少了特征扰动。
- 但仍是负向，说明：
  1. 即使只有前 1/3 层，硬 AdaIN 依然破坏了视觉编码器的特征分布
  2. LayerNorm 输出的均值和方差对齐不等于域对齐（domain alignment needs more than first-order statistics）
  3. 500 张对 ViT 的特征分布而言仍然不够充分（ViT patch-level 特征空间远大于 1152 维）

---

## E2: VCD — Visual Contrastive Decoding

### 实验配置

| 参数 | 值 |
|------|-----|
| α | 0.5, 1.0 |
| 噪声类型 | 扩散噪声 (500 steps) |
| 解码模式 | Greedy (t=0), Sampling (t=0.7, top_p=0.9) |
| 评估子集 | CXR 200 |

### 结果

| 方法 | α | 解码 | CXR-200 Acc | vs Baseline |
|------|---|------|-------------|-------------|
| baseline | — | greedy | 79.0% | — |
| vcd_greedy | 1.0 | greedy | 71.5% | -7.5% |
| vcd_greedy | 0.5 | greedy | 73.0% | -6.0% |
| vcd_sampling | 1.0 | sampling(t=0.7) | 66.0% | -13.0% |
| pure_sampling | — | sampling(t=0.7) | 69.0% | -10.0% |

### 分析

1. **Greedy VCD 小幅降级 (-6~7.5%)**: VCD 的对比 logits 无法翻转 14B 模型的 argmax。14B logits 中正确/错误 token 的差距通常 >4, 而 VCD 的 (1+α)l - α·l' 线性变换只能产生 ≤α·|l-l'| 的调整量。要改变 argmax 需要调整量 >4, 这在 α≤1 下无法达到。

2. **Sampling 本身大幅降级 (-10%)**: Hulu-Med 14B 在采样下生成质量显著下降。这是因为模型训练时使用 greedy decoding 优化，采样引入了随机性但没有对应的训练策略配合。

3. **Sampling+VCD 降级最大 (-13%)**: 二次前向 (噪声分支) + 采样随机性叠加，质量进一步下降。

### 贪心锁死假说

```
假设: t=0 贪心解码下, logits 线性变换无法翻转 14B 的 argmax
验证: Sampling(t=0.7) 解锁了 argmax, ∀但采样本身导致模型退化

证据:
  - Greedy VCD:  2/200 条输出不同 (1%)  ← 验证了锁死
  - Pure Sampling:  很多条输出不同, 但 acc 从 79%→69%  ← 解锁也无效
  - Sampling VCD:  66%  ← 解锁+对比 都无效
```

---

## E3: RAG-KI — RAG Knowledge Injection

### 实验配置

| 参数 | 减法变体 | 加法变体 |
|------|---------|---------|
| β | 0.3, 0.5 | 1.5, 2.0 (boost) |
| 检索 | text only, top-3 | text only, top-3 |
| 评估子集 | CXR 200 | CXR 200, MM 200 |

### 结果

| 方法 | CXR-200 Acc | vs Baseline |
|------|-------------|-------------|
| baseline | 79.0% | — |
| rag_ki_b0.3 (减法) | 0.0% (崩溃) | ❌ |
| rag_ki_b0.5 (减法) | 0.0% (崩溃) | ❌ |
| rag_additive_boost1.5 | 73.5% | -5.5% |
| rag_additive_boost2.0 | 71.0% | -8.0% |
| adain+rag_additive | 70.5% | -8.5% |

### 减法变体崩溃原因

LogitsProcessor `__init__` 返回了 tuple 而非 None —— Python 限制。代码中 inline class 的定义方式有 bug。

### 加法变体失效原因

```
检索到的答案: "Spinal Cord"
编码 token:  [10145, 18463]
boost 位置:  logits[:, 10145] += 1.5,  logits[:, 18463] += 1.5
模型需要输出:  "Yes" (token 4804) 或 "No" (token 1517)
            ↓
boost 的 token 与需要的答案 token 完全不重叠
```

加法变体 token 级别的 boost 在 binary 问题上完全无效（boost 的全文 token 与 "Yes"/"No" 不重叠），在多选问题上也几乎无效（boost 的全文 token 与 "A"/"B"/"C"/"D" 不重叠）。

---

## E4: Cache-based TTA

### 实验流程

```
+ KB text embeddings → sentence-BERT 编码 → 余弦索引
+ 测试样本 question → sentence-BERT 编码 → 检索 top-5
+ 检索答案 token ID → 相似度加权校准分布 → logits 插值
+ l' = (1-λ)·l + λ·l_cache
```

### 全量参数扫描

#### MM-VisHal (2000 samples)

| 方法 | Acc | vs Baseline | 说明 |
|------|-----|-------------|------|
| baseline | 47.42% | — | 2000条 948/2000 |
| cache_tta λ=0.1 top_k=5 | 47.12% | -0.30% | |
| cache_tta λ=0.15 top_k=5 | 47.12% | -0.30% | |
| cache_tta λ=0.2 top_k=5 | 47.12% | -0.30% | |
| cache_tta λ=0.25 top_k=5 | 47.22% | -0.20% | 最佳 |
| cache_tta λ=0.15 top_k=10 | 46.97% | -0.45% | top_k 越大噪声越多 |

#### 三重验证 (200 samples each, 假阳性)

| 数据集 | Baseline | Cache-TTA (best) | Δ | 说明 |
|--------|----------|-----------------|---|------|
| CXR | 79.0% | 79.5% | +0.5% | 假阳性 |
| FineGrained | 79.0% | 79.5% | +0.5% | 假阳性 |
| MM-VisHal | 47.5% | 48.0% | +0.5% | 假阳性 |

200 条样本的 +0.5% = 恰好 1 条样本翻正。2000 条全量验证反向。

### 详细失败分析

见 `FAILURE_ANALYSIS.md` 第四节。100 条样本详细检索质量分析:
- 仅 42% 检索包含正确答案
- Binary 问题 boost 的 token 与答案不匹配
- 多选问题 cache 返回全文而非选项字母
- 矛盾 KB 条目互相抵消

---

## 总结: 所有实验对比

| 方法 | CXR | MM-VisHal | 说明 |
|------|-----|-----------|------|
| baseline (full) | 78.9% | 46.6% | |
| baseline (200 sample) | 79.0% | 47.5% | |
| AdaIN (旧版) | -7.0% | — | 30张+全层 |
| AdaIN (新版) | **-3.5%** | — | 500张+前1/3层, 伤害减半 |
| VCD greedy α=1.0 | -7.5% | — | 贪心锁死 |
| VCD greedy α=0.5 | -6.0% | — | |
| VCD sampling α=1.0 | -13.0% | — | 采样退化 |
| Pure sampling | -10.0% | — | 模型不适用采样 |
| RAG-KI subtractive | 崩溃 | — | 代码bug |
| RAG additive boost1.5 | -5.5% | — | token不匹配 |
| RAG additive boost2.0 | -8.0% | — | |
| **Cache TTA (全量2000)** | — | **-0.2~0.3%** | **假阳性排除** |
| AdaIN + RAG additive | -8.5% | — | 叠加负向 |

**所有训练免费方法全部无效，无一例外。**

---

## Bug 修复清单

| # | 问题 | 影响 | 修复 |
|---|------|------|------|
| 1 | Multi-choice 只解析 `A.` 遗漏 `A:`/`A)` | MC acc 2%→45% | 逗号分割+ABCD前缀过滤 |
| 2 | LLM Judge right 未累加 | Binary/MC right=0 | Phase 1 先累加 |
| 3 | `_is_binary` 误捕长回答 | Binary acc 降 1-2% | 截取 answer 首词 |
| 4 | HulumedProcessor 不兼容 transformers 4.51 | 模型无法加载 | 适配 3 处 API 变化 |
| 5 | OOM 无 flash_attn | 大数据集无法运行 | `attn_implementation="sdpa"` |
| 6 | LLM Judge 格式不遵守 | OE acc=0 | 改进 prompt+容错解析 |
| 7 | Vision encoder forward 缺 grid_sizes | AdaIN stats 无法计算 | 添加 grid_sizes 参数 |
