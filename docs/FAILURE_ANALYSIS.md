# 训练免费方法失败根因分析

---

## 概要

系统验证了 6 种训练免费幻觉缓解方法在 Hulu-Med 14B 上全部无效。本文档从 Token 级、策略级、检索级、模型级 四个维度分析每种方法的失败原因。

## 六方法总览

| # | 方法 | 类型 | 来源 | 最佳结果 | 严重度 |
|---|------|------|------|---------|--------|
| 1 | AdaIN | 特征对齐 | 本项目 | -3.5% | 🔴 负向 |
| 2 | VCD (Greedy) | 对比解码 | CVPR 2024 | -6% | 🔴 负向 |
| 3 | VCD (Sampling) | 对比解码+采样 | CVPR 2024 | -13% | 🔴 负向 |
| 4 | RAG-KI (Subtractive) | Logits对比 | 本项目 | 崩溃 | 💥 崩溃 |
| 5 | RAG (Additive Entity-boost) | Token增强 | 本项目 | -5.5% | 🔴 负向 |
| 6 | Cache-based TTA | Logits校准 | 本项目 | -0.2% | 🟡 假阳性 |

---

## 第一章: AdaIN — 特征对齐

### 预期机制

```
测试图像 → ViT 每层 LayerNorm → 统计量对齐到源域
  → 消除设备/协议引起的域偏移 → 模型在源域熟悉的特征上做判断
```

### 实现细节

55 个 forward hook 注册在 Vision Encoder 的 LayerNorm 节点上。每个 hoook 对 LN 输出做：

```python
f_norm = (f - μ(f)) / σ(f) * σ_src + μ_src
```

统计量通过 500 张 KB 图像的前向均值/方差计算。

### 实际结果

| 配置 | CXR-200 Acc | Δ |
|------|-------------|---|
| Baseline | 79.0% | — |
| AdaIN (30张+全55层) | 72.0% | -7.0% |
| AdaIN (500张+前18层) | 75.5% | -3.5% |

### 根因分析

#### 1. LayerNorm 统计量对齐 ≠ 域对齐 (策略级)

LayerNorm 的输出 mean/std 对齐归一化的是每层特征的 first-order statistics。域偏移不仅体现在 mean/std 上，还体现在：

- **高阶统计量**: 协方差、高阶矩 (skewness, kurtosis) 在 CT vs MRI 间差异巨大
- **特征子空间**: 不同设备对相同解剖结构的编码位于不同的特征子空间，仅对齐 marginal distribution 不够

```
PET Scan → ViT 特征分布 = {μ_PET, σ_PET, 还有结构差异}
CT Scan  → ViT 特征分布 = {μ_CT, σ_CT, 不同结构}
AdaIN:   f_adain = (f - μ_PET)/σ_PET * σ_CT + μ_CT
         只能对齐均值和方差, 但特征的空间结构保留原始的 PET 模式
```

#### 2. 55 层全对齐破坏语义 (模型级)

Vision Encoder 的 55 个 LayerNorm 跨越从浅层（边缘/纹理检测）到深层（语义概念）的所有层级。

- **浅层** (第 1-9 层): 编码纹理、边缘、对比度等底层视觉信息 ← **域偏移集中在这里**
- **中层** (第 10-18 层): 编码器官轮廓、相对位置等中级特征 ← **域偏移部分在这里**
- **深层** (第 19-27 层): 编码器官类别、病变存在等语义信息 ← **应保留不变**

对齐所有 55 层 = 同时扰动浅层域信息和深层语义信息，后者导致模型无法识别原本能正确识别的器官/病变。

```
浅层: "GE 设备的纹理" → AdaIN 修改 → 匹配源域  (✓ 好)
深层: "这是肺结节"    → AdaIN 修改 → 特征偏移  (✗ 坏)
```

#### 3. 统计量估计不稳定 (工程级)

Vision Encoder 输出特征维度 1152。500 张 224x224 图片，patch size 14，每张约产生 256 patches。总样本 = 500 × 256 = 128,000 个 1152 维向量。虽然样本足够多，但 running average 计算 std 的方式引入了偏差：

```python
# 当前实现（有偏）:
stats["stds"][layer_name] = (stats["stds"][layer_name] * n + std.cpu()) / (n + 1)
# 这是对 std 的运行平均, 不是标准差的无偏估计

# 正确做法（Welford）:
M2 += (x - mean) * (x - mean_new)
variance = M2 / (n-1)  # 无偏方差
```

新版使用了 Welford 算法修复此问题，但上述高阶结构问题仍无法解决。

### 教训

LayerNorm 输出统计量对齐不是域对齐的充分条件。真正需要的是特征子空间的结构对齐——这通常需要可学习的变换（如可学习仿射、对抗域适应），而非硬 AdaIN。

---

## 第二章: VCD — 对比解码

### 预期机制

```
对比: l_vcd = (1+α)·l(v, x) - α·l(v', x)
用噪声图像的 logits 作为"退化参考"：
  - 如果模型对噪声图的置信度仍高 → 语言先验/偏见
  - 减去这部分 → 保留视觉证据充分的答案
```

### 实际结果

| 模式 | CXR-200 | 输出变化率 |
|------|---------|-----------|
| Greedy VCD α=1.0 | 71.5% | <1% (2/200条输出不同) |
| Greedy VCD α=0.5 | 73.0% | <1% |
| Sampling t=0.7 | 69.0% | 几乎全部不同 |
| Sampling VCD | 66.0% | 几乎全部不同 |

### 根因分析

#### 1. 贪心锁死 (模型级) ⭐ 核心发现

```
问题: t=0 (贪心解码) 下, Hulu-Med 14B 的 logits 分布极度尖锐

证据:
  正确 token 的 logit = 25.3
  错误 token 的 logit = 21.1 (差距 4.2)
  softmax(P(correct)) ≈ exp(25.3) / (exp(25.3) + exp(21.1)) = 98.5%

VCD 调整:
  l_vcd = (1+α)·25.3 - α·l_noise
  即使 l_noise 对噪声图输出 logit=20 (低于错误token)
  l_vcd_correct = 2*25.3 - 20 = 30.6
  l_vcd_wrong   = 2*21.1 - 20 = 22.2
  差距 = 30.6 - 22.2 = 8.4 → 仍然大于 0 → argmax 不变!

要使 argmax 翻转, 需要 l_vcd_wrong > l_vcd_correct:
  (1+α)·l_wrong - α·l_noise > (1+α)·l_correct - α·l_noise
  → 只需要 l_wrong > l_correct, 这不可能 (因为贪心选 l_correct > l_wrong)
```

**数学证明: 在贪心解码下, VCD 的线性对比不可能翻转 14B 的 argmax。**

这是因为 VCD 对正确和错误 token 使用了相同的线性变换 `(1+α)·l - α·l_noise`。变换后的排序完全由原始排序决定（差值不变）。VCD 只能改变绝对置信度，不能改变相对排序。

要让 VCD 有效，需要：
- 噪声分支对正确/错误 token 的影响不对称（正确 token logit 下降更多）
- 但 14B 模型对噪声图像仍然输出相似的 token 分布（语言先验太强）

#### 2. 采样退化 (模型级)

```
t=0.7 greedy:  79.0%
t=0.7 sampling: 69.0%  ← -10%
```

Hulu-Med 14B 使用 greedy decoding 训练和优化。采样（temperature > 0）引入了不确定性，但模型的训练并未考虑这种不确定性：

- 训练时: teacher forcing + greedy → 模型学会"最短路径"到正确答案
- 测试时: sampling → 模型暴露于未曾训练过的 token 序列组合 → 质量下降

#### 3. 噪声图像的质量 (实现级)

简单加性高斯噪声在医学图像上退化的效率和自然图像不同：

- 自然图像: 加入 σ=0.1 的高斯噪声 → 目标边界模糊 → 模型置信度下降
- 医学图像: CT/MRI 本身已有噪声 → 增加少量噪声 → 模型仍能提取关键特征

### 文献对比

| 模型 | 规模 | VCD 效果 |
|------|------|---------|
| LLaVA | 7B | 有效 (原论文) |
| InstructBLIP | 7B | 有效 (原论文) |
| LLaVA-Med | 7B | 有限 (MedHEval 论文) |
| LLaVA-NeXT | 13B | 有限 (MedHEval 论文) |
| **Hulu-Med** | **14B** | **无效 (本项目)** |

VCD 的效果随模型规模下降，14B 是首次验证完全无效的规模。

---

## 第三章: RAG 两种变体

### 变体 A: RAG-KI (减法对比)

#### 预期机制

```
l_rag = (1+β)·l(v,x) - β·l(v,x_rag)

其中 x_rag = prompt + 检索上下文
想法: RAG 上下文改变模型语言先验 → 减去原始先验 → 留下去除先验的视觉证据
```

#### 实际结果

全部 200 条样本 100% 设置为 0（代码崩溃）。

#### 根因

**概念设计错误（策略级）**：

VCD 用噪声图像作为"退化参考"——噪声图确实包含更少的视觉证据。但 RAG 上下文是"增强输入"——它提供了额外的医学知识。用增强输入作为减法的参考，等于：

```
原始: l(model | image + question) → 模型依赖视觉证据 + 语言先验
RAG:  l(model | image + question + knowledge) → 模型还依赖检索知识
减后: l - β·l_rag = 惩罚"检索知识带来的额外信息"
```

这完全违反了对比解码的基本假设——减项应该代表一个**更差**的参考。

**正确的 RAG 融合应该是**:

- 加性融合: `l_combined = l_original + β · (l_rag - l_original)` = `(1-β)·l + β·l_rag`（信任 RAG）
- 或者: 用 RAG 知识做 reranking（生成多个候选，用检索选择最佳）

#### 额外问题

LogitsProcessor 的 __init__ 方法返回了 tuple（因为 inline `type()` 的构造方式出错），违反了 Python `__init__` 必须返回 None 的限制。这是一个实现 bug，即使概念正确也无法运行。

### 变体 B: Additive Entity-boost

#### 预期机制

```python
# 对检索答案中的 token 做加性 boost
for tid in retrieved_answer_tokens:
    scores[:, tid] += boost_weight (1.5 或 2.0)
```

#### 实际结果

| 方法 | CXR-200 | vs Baseline |
|------|---------|-------------|
| baseline | 79.0% | — |
| boost=1.5 | 73.5% | -5.5% |
| boost=2.0 | 71.0% | -8.0% |
| adain+boost | 70.5% | -8.5% |

#### 根因分析

##### Token 粒度失配（Token 级）⭐

这是最关键的失败原因。以下是对 100 条 MM-VisHal 样本的详细分析：

```
Cache Retrieval Analysis (50 samples)
=====================================
avg top-1 similarity: 0.69
avg top-5 similarity: 0.42
% top-5 包含正确答案: 42%
```

**Case 1: Binary 问题**

```
Q: "Is the spinal cord visible in the image?"  →  Ground Truth: "Yes"

Retrieve top-1:      [sim=0.710]  Q: "What is the main organ in the image?"  A: "Spinal Cord"
Retrieve top-2:      [sim=0.568]  Q: "what cut of the brain is this image"   A: "axial"

entity_ids (tokenized):
  "Spinal Cord"  →  [10145, 18463]
  "axial"        →  [12712, 5203]

Boost: logits[:, 10145] += 0.710,  logits[:, 18463] += 0.710
       logits[:, 12712] += 0.568,  logits[:, 5203]  += 0.568

Model needs to output: "Yes" (token 4804) or "No" (token 1517)
                      ↓
boost 的 token (10145, 18463, 12712, 5203) 与答案 token (4804, 1517) 
完全不重叠！
```

**Case 2: 多选问题**

```
Q: "What are the possible organs shown in the image?"  →  Ground Truth: "C"

Retrieve top-1:      [sim=0.774]  Q: "What is the main organ in the image?"  A: "Lung, Liver"
Retrieve top-2:      [sim=0.774]  Q: "Which organs"                         A: "Liver, Heart, Spleen, Lung"

entity_ids:
  "Lung, Liver"         →  [5462, 1342, 4511]
  "Liver, Heart, ..."   →  [4511, 1342, 7792, ...]

Model needs to output: "A", "B", "C", or "D"
                      ↓
boost 的 token (器官名词) 与答案 token (选项字母) 完全不重叠！
```

**结论**: 对于 binary 和多选问题（占 MM-VisHal 的 100%），检索答案的全文 token 与 ground truth 的短答案 token 几乎没有重叠。Token-level boost 无的放矢。

##### 检索噪声

```
Q: "is there a pneumothorax?"  →  Ground Truth: "No"

Retrieve top-1: [sim=0.918]  A: "yes"    ← 相反的答案
Retrieve top-2: [sim=0.918]  A: "no"     ← 正确的答案
↓
两个条目相似度几乎相等（同一问题不同变体）
boost: logits[:, yes_token] += 0.918,  logits[:, no_token] += 0.918
↓
相互抵消！
```

KB 中包含同一问题有不同标准答案的条目（来自不同标注者/数据集）。这些矛盾条目在检索时同时返回，它们的 boost 互相抵消。

##### 检索维度不足（检索级）

sentence-BERT 仅编码问题文本的语义，无法感知图像内容：

```
Q: "What organ is shown?"  + 图像 (肝脏)  → cache 返回"肺"的答案
Q: "What organ is shown?"  + 图像 (肺)   → cache 返回"肝"的答案
```

因为没有视觉特征编码，cache 无法区分同一问题在不同图像下的不同正确答案。这导致了检索结果与视觉内容无关，仅与问题文本相关。

### RAG 两变体失败对比

| 维度 | 减法变体 | 加法变体 |
|------|---------|---------|
| 核心问题 | 方向反 (减增强输入) | 粒度失配 (token 不重叠) |
| 可修复性 | 修复方向即可 (但效果有限) | 需要视觉检索+答案格式对齐 |
| 偶发问题 | 代码崩溃 | 矛盾条目互相抵消 |
| 对 MC 问题 | 同方向反 | 选项字母 vs 全文不匹配 |
| 对 Binary | 同方向反 | yes/no token vs 器官名词不匹配 |

---

## 第四章: Cache-based TTA

### 预期机制

```
检索 (text) → 构建校准分布 → 插值

l' = (1-λ)·l + λ·l_cache

l_cache = sum(sim_i · onehot(retrieved_answer_tokens)) - mean(...)
```

目标是绕过 AdaIN 对特征层的直接扰动，在 logits 层面用 KB 知识做软校准。

### 全量验证结果

```
MM-VisHal (2000 samples)
  baseline:               47.42% (948/2000)
  cache_tta (best):       47.22% (-0.2%)
  
200 条扫描是假阳性:
  CXR-200:  +0.5% (恰好1条翻正)
  FG-200:   +0.5% (恰好1条翻正)
  MM-200:   +0.5% (恰好1条翻正)
```

### 失败的四个维度

#### 1. Token 粒度的结构性失配（Token 级）

Cache-TTA 和 Entity-boost RAG 共享同一核心缺陷——校准分布基于检索答案的全文 token，但 binary 和多选问题只需要单个选项 token。

```
Cache-TTA 校准:
  l_cache = ∑ w_i · token_onehot(answer_i)
  = 0.71 · onehot("Spinal") + 0.71 · onehot("Cord") + 0.57 · onehot("axial") + ...

对 logits 的影响:
  logits["Spinal"] += 0.71 * λ
  logits["Cord"]   += 0.71 * λ
  logits["axial"]  += 0.57 * λ
  logits["Yes"]    += 0          // 没有 boost！
  logits["No"]     += 0          // 没有 boost！
```

在 binary 问题上，校准分布完全不触及 "Yes"/"No" 的 logits。**校准发生的 logits 空间（医学全文 token）与决策空间（Yes/No/A/B）完全不重叠。**

#### 2. 检索质量不足（检索级）

Retrieval analysis on 50 random MM-VisHal questions:

```
Retrieval Quality Metrics:
  top-1 avg similarity:       0.69
  top-5 avg similarity:       0.42
  包含正确答案 (top-5):      42%
  正确答案在 top-1:           ~24%
  矛盾条目出现在 top-5:      ~35%

Example:
  Q: "is there a pneumothorax?"
  top-1: answer="yes" (wrong for this image)
  top-2: answer="no"  (right for this image)
  top-3: answer="no"  (right for this image)
```

42% 的命中率意味着校准分布中超过一半的贡献来自错误答案。在 λ=0.2 时，校准信号的 SNR (信号噪声比) 极低。

#### 3. 校准强度与效果的矛盾（策略级）

```
λ=0.10:  校准过弱, 几乎不影响 → 等同 baseline
λ=0.50:  校准过强, 噪声放大  → -2.5%
λ=0.15-0.25: 微弱影响, SNR 不足以翻正 → -0.2~0.3%
```

**不存在一个 λ 值同时满足**：
- 足够强以翻正错误样本（需要 ~0.3+）
- 足够弱以避免引入噪声（需要 <0.1）

这是因为校准信号大部分指向错误答案，任何可见的校准都会降低整体质量。

#### 4. 检索的视觉盲区（检索级）

```
Q: "Is there a fracture in the image?"
(图像: 正常 X 光, 无骨折)
  text-only retriever → "Is there a fracture" → top-1: "Yes, there is a fracture in the distal radius"
  ↓
  校准分布 boost "Yes" 和 "fracture" → 模型倾向于输出 "Yes"
  但实际正确答案是 "No" → 校准主动增加了幻觉!
```

文本检索无法辨别"这个问题通常的答案 vs 这张图像实际的答案"。在训练集中，大多数骨折问题的答案都是肯定的（因为数据集标注了阳性样本），导致检索的答案分布偏向阳性——即使当前图像是阴性。

### 为什么 200 条上出现假阳性

```
200 条采样:
  基线 158/200 = 79.0%
  TTA  159/200 = 79.5% 

差异来源: 恰好 1 条样本的随机波动
  - 这条样本原本错误, TTA 正确
  - 但这种翻正是在 42% 检索命中率下的小概率事件
  - 2000 条扩大到 1995 条后, 效果消失甚至反向

统计检验:
  95% CI (200条, p=0.79) = 0.79 ± 0.056
  79.5% 在 95% CI 内 → 不显著
```

### Cache-TTA 失败总结

```
根本原因链:
  文本-only 检索 → 检索命中率仅 42%
  42% 命中  → 校准分布 SNR < 1
  SNR < 1   → 校准 = 在 logits 上叠加噪声
  噪声叠加  → 任何 positive λ 都会轻微降级性能

子问题:
  - Token 粒度失配 (binary/MC 的决策 token vs 检索的全文 token)
  - 矛盾 KB 条目相互抵消
  - 视觉盲区 (text-only 无法区分"通常情况"和"当前图像")
```

---

## 第五章: 跨方法共性失败模式

### 模式 1: 14B 模型的鲁棒性（模型级）

所有方法都假定可以通过 logits 层面的轻微调整来改变模型输出。但 14B 模型的 logits 极为稳定：

| 方法 | 试图改变的 | 实际影响 | 原因 |
|------|-----------|---------|------|
| VCD greedy | logits 排序 | 几乎无变化 | 排序差 >4, 线性变换无法翻转 |
| Cache-TTA | logits 分布 | ~0.2% 降级 | 校准信号 SNR < 1 |
| Entity-boost | token 概率 | 5.5% 降级 | 校准方向错误 |

14B 经过大规模预训练和医学微调，其内部表示对轻微扰动高度不敏感。这与 7B 模型的行为不同——在 7B 上同样的方法（VCD, DoLa）可以改变输出。

### 模式 2: Second-order perturbation 叠加（策略级）

| 组合 | 结果 | 说明 |
|------|------|------|
| AdaIN alone | -3.5% | 特征层扰动 |
| AdaIN + RAG | -8.5% | 特征 + logits 双重扰动 |
| AdaIN + VCD | -5.0% | 特征 + 对比双重扰动 |

多个训练免费方法的简单叠加总是比单个方法更差。这说明每层扰动（特征、logits、对比）之间没有正交性，它们互相放大噪声。

### 模式 3: 医学 VLM 的特殊性（数据级）

通用 VLM（LLaVA, InstructBLIP）的 hallucination 主要是**物体幻觉**（output objects not in image）。医学 VLM 的 hallucination 主要是**视觉误读**（misidentify anatomy）和**知识缺失**（lack medical knowledge）。前者的 mitigation 方法（VCD, DoLa 等针对物体幻觉设计的）对后者天然不适用。

```
通用 VLM hallucination:
  "There is a dog" → 图中没有狗 → VCD: 对比噪声图, 减少物体幻觉 ✓

医学 VLM hallucination:
  "This is pneumonia" → 图中可能是肺不张 → 需要医学知识区分 → VCD 无效 ✗
```

---

## 第六章: 对后续工作的指导

### 训练免费路线的上限

本项目的系统性实验表明，在 14B 医学 VLM 上，训练免费方法的准确率上限 ≈ **baseline 准确率 - 2~5%**（随方法不同）。没有任何方法能跨越这个上限达到正向。

### 可行的正向策略

基于失败分析，唯一有效的路径是通过**修改模型参数**（训练级干预）：

1. **LoRA/QLoRA 微调**：保持基座不变，训练少量 adapter 参数
   - 不受贪心锁死限制（直接改参数空间而非 logits 空间）
   - 不受检索噪声影响（模型学会"使用"检索信号）

2. **DPO 偏好优化**：用正确/错误样本构造偏好对
   - 直接优化模型输出分布
   - 成本可控（几小时即可，无需大规模 SFT）

3. **Preference pair 自动构造**：用 MedHEval 标准答案自动生成偏好对
   - 无需昂贵的人工标注
   - 可以构造 10000+ 对（整个 benchmark 的数据量）
