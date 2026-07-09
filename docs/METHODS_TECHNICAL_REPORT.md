# MedShift+ 技术报告：医学 VLM 幻觉缓解方法

---

## 目录

1. [已测试方法 (Phase 0-1)](#1-已测试方法-phase-0-1)
2. [新提出方法 (Phase 2 v2)](#2-新提出方法-phase-2-v2)
3. [ICML26 论文映射与具体实现](#3-icml26-论文映射与具体实现)
4. [实现状态与已知问题](#4-实现状态与已知问题)
5. [实验验证纪律](#5-实验验证纪律)
6. [附录：代码结构](#6-附录代码结构)

---

## 1. 已测试方法 (Phase 0-1)

### 1.1 AdaIN — Adaptive Instance Normalization

| 属性 | 值 |
|------|-----|
| **类型** | 特征层域对齐 |
| **操作空间** | ViT 编码器 LayerNorm 输出 (一阶统计量) |
| **核心公式** | `f_adain = (f - μ(f)) / σ(f) * σ_src + μ_src` |
| **代码** | `medshift/core/adain.py` |
| **最好结果** | CXR-200: **-3.5%** (vs baseline 79.0%) |

**已证失效根因** (FAILURE_ANALYSIS §1):
1. LayerNorm mean/std 对齐 ≠ 域对齐。域偏移不仅体现在一阶矩,还含协方差、高阶矩、特征子空间结构差异。
2. 全 55 层对齐扰动深层(19-27)语义特征(如器官类别、病变存在)。即使只对齐前 1/3 层(18/55),仍破坏语义。
3. 统计量计算方式误差(Welford 修复了一阶估计偏差,但高阶问题仍无法解决)。
4. Anisotropic Modality Align (2026) 从理论上证实:全局平移后跨模态残差是**各向异性低维子空间**(各向异性比 Ar=28.6,有效维 deff/d=0.284),一阶统计量只能消掉一小部分。

### 1.2 VCD — Visual Contrastive Decoding

| 属性 | 值 |
|------|-----|
| **类型** | 对比解码 (logits 线性对比) |
| **操作空间** | decoder logits |
| **核心公式** | `l_vcd = (1+α)·l(v,x) - α·l(v',x)` (v'=噪声图) |
| **代码** | `medshift/core/vcd_rag.py` |
| **最好结果** | Greedy VCD α=1.0: **-7.5%**; Sampling VCD: **-13.0%** |

**已证失效根因** (FAILURE_ANALYSIS §2):
1. **贪心锁死(核心发现)**:Hulu-Med 14B 的 logits 分布极度尖锐,正确/错误 token 差距 >4。VCD 的线性对比 `(1+α)l - α·l'` 是仿射变换,无法改变 argmax 排序。
2. **数学证明**:VCD 对所有 token 施加相同的线性变换 → 排序不变 → argmax 不变。要让 argmax 翻转需 `l_wrong > l_correct`,这不可能(因为贪心选 `l_correct > l_wrong`)。
3. Sampling 本身使模型退化(t=0.7 sampling: -10%):Hulu-Med 用 greedy 训练优化,采样引入模型未曾训练的 token 序列。
4. 医学噪声的特殊性:CT/MRI 本身含噪,简单高斯噪声不足以使可靠 logits 下降。

### 1.3 RAG-KI (Subtractive)

| 属性 | 值 |
|------|-----|
| **类型** | 对比解码 (RAG 增强作为减项) |
| **核心公式** | `l_rag = (1+β)·l(v,x) - β·l(v,x_rag)` |
| **结果** | **崩溃 (100% 设置为 0)** |

**已证失效根因**:
1. **方向反(概念设计错误)**:VCD 用噪声图(更差的参考),RAG-KI 用增强输入(更好的参考)做减法 → 惩罚正确知识。
2. 代码 bug: `__init__` 返回了 tuple(违反 Python 限制)。
3. 即使修复方向,logits 线性对比仍受贪心锁死限制。

### 1.4 RAG (Additive Entity-Boost)

| 属性 | 值 |
|------|-----|
| **类型** | Token 级加法增强 |
| **操作空间** | decoder logits (token select) |
| **核心公式** | `scores[:, token_id] += boost_weight` |
| **代码** | `run_ablation_phase0.py` 内联 |
| **最好结果** | boost=1.5: **-5.5%**; boost=2.0: **-8.0%** |

**已证失效根因** (FAILURE_ANALYSIS §3):
1. **Token 粒度失配(核心)**:检索答案 "Spinal Cord" → boost token [10145, 18463];但 binary 问题要输出 "Yes"(4804)/"No"(1517),完全不重叠。MC 问题输出 A/B/C/D,同样不重叠。
2. **检索噪声**:top-5 仅 42% 含正确答案;~35% 含同一问题不同答案的矛盾条目,boost 互相抵消。
3. **视觉盲区**:text-only(sentence-BERT)检索,无法区分同一问题在不同图像下的不同正确答案。
4. **多源标注矛盾**:KB 中同一问题有 yes/no 两种答案(不同标注者/数据集)。

### 1.5 Cache-based TTA

| 属性 | 值 |
|------|-----|
| **类型** | Logits 级校准 |
| **操作空间** | decoder logits (插值) |
| **核心公式** | `l' = (1-λ)·l + λ·l_cache` |
| **代码** | `medshift/core/cache_tta.py` |
| **最好结果** | λ=0.25 top_k=5: **-0.2% (全量 2000,假阳性)** |

**已证失效根因** (FAILURE_ANALYSIS §4):
1. **Token 粒度失配**:校准分布基于检索答案全文 token,binary/MC 决策 token(Yes/No/A/B)不被触及。
2. **检索质量不足**:42% top-5 含正确答案;矛盾条目 ~35%;SNR < 1。
3. **校准强度困境**:不存在一个 λ 同时足够强(翻正错误样本需 >0.3)又足够弱(避免噪声 <0.1)。
4. **视觉盲区**:text-only 检索无法区分"通常答案 vs 当前图像实际答案"。
5. **200 条假阳性**:+0.5% ≈ 1 条随机翻正,全量 2000 条反向降级。

### 1.6 跨方法共性失败模式

| 模式 | 描述 |
|------|------|
| **贪心锁死** | 14B logits 差 >4,任何线性 logits 干预无法翻转 argmax |
| **二阶扰动叠加** | 特征+logits+对比多法叠加总是更差 |
| **医学特殊性** | 通用 VLM 是"物体幻觉",医学 VLM 是"视觉误读+知识缺失",前者的缓解方法(DoLa/VCD)对后者不适用 |
| **200 条假阳性** | 200 条 ±0.5% 在 95% CI 内 → 必须全量验证 |

---

## 2. 新提出方法 (Phase 2 v2)

### 2.1 Line 1: 特征高阶白化 (替换 AdaIN)

#### 理论依据

- **Anisotropic Modality Align** (arXiv 2605.07825, May 2026): 证明了跨模态对齐后残差是**各向异性低维子空间**,主流一阶对齐只能消最小部分。
- **Feature-Whitening** (Roy, CVPR 2019): 层特定协方差矩阵白化对齐分布,无需专用 loss。

#### 核心公式

```
AdaIN (已失效): f' = (f - μ_test) / σ_test * σ_src + μ_src   ← 一阶仅mean+std
Whitening:     f' = μ_src + W @ (f - μ_src)                  ← 二阶协方差
               W = V_top_r @ diag(1/√λ_top_r) @ V_top_r^T
               其中V, λ = eig(Σ_src),截断至top-r主子空间

可选MMD门控: shift_score = ‖f_test_mean - μ_src‖_W > threshold → 跳过浅偏移层
```

#### 关键设计参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `top_r` | 32 | 主子空间维度(1152->32,保留谱能量~95%) |
| `ridge` | 1e-3 | 特征值正则化,防止协方差病态 |
| `gate_threshold` | 0.0 | MMD 门控阈值(0=始终白化;>0=仅高偏移层) |
| `num_shallow` | 9 | 对齐浅层数(深 18 层含语义,不动) |
| `stat_samples` | 500 | 源域统计量图像数 |

#### 与 AdaIN 对比

| 维度 | AdaIN (v1) | Whitening (v2) |
|------|-----------|----------------|
| 统计量阶数 | 一阶 (mean, std) | 二阶 (covariance, eigendecomp) |
| 对齐方式 | 逐元素缩放 | 全协方差白化 + 子空间截断 |
| 语义保护 | 仅减少层数(18/55) | 浅层白化 + MMD 自适应门控 |
| 理论不足 | 一阶统计量≠域对齐 | 沿各向异性残差主方向对齐 |
| 统计量估计 | Welford online (已修复) | Welford batch cov (向量化) |
| 预估增益 | -3.5% ~ -7.0% (全负) | 目标: +1-3% (待验证) |

#### 实现位置

`medshift/core/whitening.py`:
- `_CovAccumulator`: Welford 在线协方差累积器(向量化 batch update)
- `compute_source_cov_hook()`: 在 ViT 各层 LN 注册收集 hook
- `WhitenngHook.__call__`: 推理时在 LN 输出后做白化变换
- `apply_whitening_to_shallow_layers`: 仅注册前 N 层

#### 已知问题

- **hook 路径错误**:目前 `apply_whitening_to_shallow_layers` 写的是 `model.model.model.vision_encoder`,实际应为 `model.model.vision_encoder`（HulumedQwen3ForCausalLM 结构）。需在下一版修正。

### 2.2 Line 2: MA-RAG + BiomedCLIP 多轮冲突共识 RAG (替换 text-only RAG+token-boost)

#### 理论依据

- **MA-RAG** (ICML 2026 poster): "From Conflict to Consensus" — 多轮 agentic RAG,用语义冲突驱动新检索,最终共识。
- **MMed-RAG** (ICLR 2025): domain-aware retrieval + RAG-based preference tuning。
- **Binary Retrieval-Augmented Reward** (ICML 2026): 二元判别器评分候选。

#### 与已失效 RAG 的对比

| 维度 | Old RAG (v1) | MA-RAG (v2) |
|------|-------------|-------------|
| 检索特征 | text-only (sentence-BERT) | BiomedCLIP 图像编码 + 文本(可加权) |
| KB 条目矛盾处理 | boost 互相抵消(负向) | 冲突→新检索信号(正向) |
| 答案生成 | token 级 boost(粒度失配) | prompt 级注入(模型自然生成) |
| 多轮迭代 | 单轮 | N 候选→冲突检测→迭代检索→共识 |
| 候选重排 | 无 | 多数投票(可扩展为二元验证 reranker) |

#### 架构

```
测试图像+问题
  │
  ├─[Round 0] VLM 采样 N 候选 (temperature=0.7)
  │
  ├─[Conflict Detect] 候选间语义冲突(归一化答案精确匹配 + embedding 距离)
  │
  ├─[如果存在冲突]
  │   └─ Disambiguation Query → BiomedCLIP 检索 → 证据注入 prompt → VLM 重生成
  │   └─ 迭代至 M 轮(M=2) 或冲突解决
  │
  └─[Consensus] 多数投票(归一化短答案)
      可选增强: 二元 BERT 验证器重排(小辅助模块,无 VLM 训练)
```

#### 代码组件

| 文件 | 类/函数 | 功能 |
|------|---------|------|
| `medshift/retrieval/clip_retriever.py` | `BiomedCLIPRetriever` | CLIP 图像+文本编码、构建索引、检索、矛盾去重 |
| `medshift/core/ma_rag.py` | `MARagPipeline` | 多轮 agent 循环:候选生成→冲突检测→检索→共识 |
| `medshift/core/ma_rag.py` | `detect_conflict` | 候选间冲突检测 |
| `medshift/retrieval/kb_builder.py` | — | 已有 KB 加载器,复用 metadata.json |

#### 关键设计参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_candidates` | 3 | 每轮候选数 |
| `max_rounds` | 2 | 最大迭代轮数 |
| `top_k` | 5 | 每次检索 KB 条目数 |
| `text_weight` | 0.3 | 检索时文本嵌入权重(1-权重=图像权重) |
| `dedup_contradictions` | True | 同 Q 不同 A 只保留首个 |

#### 增强计划

1. **冲突检测增强**:当前用归一化精确匹配 → 改为 embedding 余弦距离阈值(使用已缓存的 all-MiniLM-L6-v2)。
2. **证据短语抽取 (借鉴 LGDEA)**:用本地 small LLM 从 KB 答案中抽取出医学证据短语(如 "right lower lobe consolidation"),使检索更精细。
3. **二元验证 reranker**:训练一个小 BERT 分类器(在 KB 对+MedHEval 训练集上,非 VLM),评分候选与检索证据的匹配度,用于重排。

### 2.3 Line 3: MEDA 医学激活编辑 (替换 VCD logits 对比)

#### 理论依据

- **MEDA** (ICML 2026 poster): "Medical-Oriented Activation Editing" — QMS 按查询语义定向 steer 激活 + 模态 steering vector,推理时编辑激活,不修改模型参数。
- **PAI** (ECCV 2024): 放大图像 token 注意力权重改 hidden states 而非 logits → 绕过贪心锁。

#### 核心机制

```
在 decoder 层残差流编辑激活:
  h' = h + α · ⟨h, ground⟩ · ground  -  β · ⟨h, halluc⟩ · halluc

ground_dir = mean(正确答案时的激活) - mean(错误答案时的激活)  [归一化]
halluc_dir = mean(错误答案时的激活) - mean(正确答案时的激活)  [归一化]
```

方向构造:对 KB 中每条 QA 对,做两次前向传递:
- **grounded**: (图像 + 正确问题 → 正确答案) → 捕获 decoder 层激活
- **hallucinated**: (图像 + 正确问题 → 错误答案) → 捕获 decoder 层激活

VLM 全程冻结,仅通过 hook 读取激活。

#### 与 VCD 对比

| 维度 | VCD (v1) | MEDA (v2) |
|------|---------|-----------|
| 操作空间 | logits (线性) | hidden states (非线性) |
| 绕过贪心锁 | ❌ (线性变换不变排序) | ✅ (改残差激活) |
| 医学先验 | 无 | QMS 查询定向 + 模态 steering vector |
| 构造方式 | 噪声图像 | Contrastive activation from KB |
| 计算量 | 2×前向/生成步 | 离线建造 steering vector(一次) + 推理前注册 hook |
| 论文效果 | -6%~-13% (本项目) | ~10% factuality 提升 (IU-Xray, MEDA 原文) |

#### 实现位置

`medshift/core/meda.py`:
- `SteeringVector`: per-layer (ground_dir, halluc_dir) 容器
- `MEDAHook.__call__`: 在 decoder block 输出后编辑残差流
- `build_steering_from_kb`: 用 KB 条目通过对比激活构造方向
- `_make_wrong_answer`: 构造错误答案(对 OE 问题从 KB 随机采样不同答案)
- `apply_meda`: 注册到中间 40%-80% 的 decoder 层

#### 增强计划 (QMS Router)

当前 steering vector 是 per-layer 单一方向。论文 MEDA 的 QMS 要求 per-query 动态路由:根据用户问题中出现的医学实体选择不同的 steer 方向。改进:

```
Query → NER (医学实体抽取) → 选择对应 steering vector
  e.g. "Is there a pneumothorax?" → ["pneumothorax"] → pneumothorax-specific steering
```

需构造多组 steering vector(每医学概念一组),加 NER 路由器。这增加了 KB 要求(需要按医学概念分组的 QA 对)。

### 2.4 横切: Domain-Shift-Aware Conformal Safety Layer

#### 理论依据

- **Domain-Shift-Aware Conformal Prediction** (ICML 2026 poster): 当目标域与校准集存在偏移时,重加权校准样本,提高阈值(更保守地 abstain)。
- **CAP** (PMLR 2026): Conformalized Abstention Policies。
- **HAC** (arXiv 2604.02543): Hallucination-Aware Calibration。

#### 修复的 Bug

原 `conformal_safety.py`:
1. **阈值覆盖 bug (línea 92-102)**: coverage-search 计算的阈值被 conformal 分位数**直接覆盖**→ 搜索结果丢弃。已修复:取两者最大值(更严阈值)。
2. **单 token 判对脆弱性 (línea 181)**: `out = tokenizer.decode(next_token)` 单 token 子串匹配极不可靠。已修复:生成完整短回答 + 复用 judge 函数。
3. **缺乏域感知**:已加 `set_domain_distance()`,当检测到域偏移时自动抬高阈值(更倾向 abstain)。

#### 接口

```python
safety = ConformalSafetyLayer(confidence_fn="margin", target_coverage=0.90)
safety.calibrate(calibration_logits, calibration_correct)
safety.set_domain_distance(mmd_distance)  # e.g. from WhiteningHook._shift_score
decision = safety.decide(test_logits)
# {"abstain": True/False, "score": 0.8, "threshold": 1.2, ...}
```

---

## 3. ICML26 论文映射与具体实现

### 3.1 映射总表

| ICML 2026 论文 | 映射到 MedShift 组件 | 实现状态 | 代码位置 |
|---|---|---|---|
| **MEDA** (poster 63562) | Line 3: 医学激活编辑 | ✅ 已实现,待修 path + 加 QMS | `medshift/core/meda.py` |
| **MA-RAG** (poster 63972) | Line 2: 多轮冲突共识 RAG | ✅ 已实现,待增强冲突检测 | `medshift/core/ma_rag.py` |
| **BiomedCLIP-PubMedBERT** (HF model) | Line 2: 检索编码器 | ⚠️ 下载中,CLIP 检索器代码就绪 | `medshift/retrieval/clip_retriever.py` |
| **LGDEA** (poster 65736) | Line 2: 证据短语抽取(借鉴) | 📋 待实现(小辅助模块) | — |
| **Binary RAR** (poster 65671) | Line 2: 二元验证重排(改造,非RL) | 📋 待实现(小辅助模块) | — |
| **Domain-Shift-Aware Conformal** (poster 63691) | 横切安全层 | ✅ 已修复+升级 | `medshift/core/conformal_safety.py` |
| **Anisotropic Modality Align** (arXiv) | Line 1 理论基础 | ✅ 已实现 | `medshift/core/whitening.py` |
| **See-Act-Adapt** (poster 65692) | Line 1 MMD 门控(借鉴) | ✅ 已内建 | `whitening.py:101` |

### 3.2 具体映射细节

#### MEDA → meda.py (✅ 已实现)

**论文原文**: QMS (Query-decisive Manifestation Steering) + 医学先验引导的激活编辑。

**实现对照**:
- `MEDAHook` 使用 `ground_dir` / `halluc_dir` 做投影编辑 → 对应 QMS 的核心数学(沿医学证据方向 push,away from hallucination)
- `build_steering_from_kb` 用 KB 条目通过对比激活构造方向 → 对应"利用医学先验/标注数据引导方向"
- 缺陷:当前是单一方向,无 per-query 动态路由 → 需加 QMS Router(见 2.3 增强计划)

#### MA-RAG → ma_rag.py (✅ 已实现)

**论文原文**: 采样 N 候选 → 语义冲突检测 → 冲突驱动新检索 → 迭代 → 共识。

**实现对照**:
- `_gen_candidates` → 温度采样 N 候选(对应"生成多个候选项")
- `detect_conflict` → 归一化答案精确比较(对应"语义冲突检测")
- `build_disambiguation_query` → 冲突→新检索查询(对应"将冲突转化为检索信号")
- 多轮迭代 → `max_rounds` 循环(对应"多轮 refine")
- `_consensus` → 多数投票(对应"共识答案")

**缺陷**:冲突检测目前是精确匹配,论文要求更强冲突识别。需加 embedding 距离阈值。

#### BiomedCLIP → clip_retriever.py (⚠️ 下载中)

**论文原文**: 用 Microsoft BiomedCLIP(PMC-15M 预训练)做图文对齐检索。

**实现对照**:
- 图像编码: `BiomedCLIPRetriever.encode_image` → CLIP image encoder forward
- 文本编码: `BiomedCLIPRetriever.encode_text` → CLIP text encoder forward
- 检索:余弦相似度(图像+文本加权)
- 矛盾去重: `_normalize_question` + `seen_q` 字典

**状态**:权重 0.78GB 文件 `open_clip_pytorch_model.bin` 正在通过 hf-mirror.com 下载(CDN 断流导致需多次 resume),代码已就绪。

#### LGDEA → clip_retriever.py 扩展 (📋 待实现)

**论文原文**: 用 LLM 从诊断报告中抽证据短语 → 做 evidence-level 对齐。

**适配约束**: 原论文需重训练 → 你不允许 VLM 训练。改造为:用本地已缓存 small LLM(如 dfdu233/OneReason-0.8B-LoRA)做前向推理抽取证据短语,不训练任何参数。

**实现计划**:
1. 加载 OneReason-0.8B(小模型,在 hf_hub_cache 中已缓存)
2. 对每条 KB 答案,前向抽取 evidence phrases
3. 存储为 KB 附加字段
4. 检索时匹配 evidence phrases 而非全文

#### Binary RAR → ma_rag.py `_consensus` 扩展 (📋 待实现)

**论文原文**: 二元判别器(是否与证据一致)→ RL 微调 VLM → 减少 ~40% 幻觉。

**适配约束**: RL 微调 VLM 违规。改造为:在 KB+MedHEval 训练集上训练一个小 BERT 分类器(非 VLM,小辅助模块),评分每条候选与检索证据的一致性,用于重排而非 RL。

**实现计划**:
1. 在 `medshift/retrieval/` 下实现 `BinaryVerifier` (BERT-based, 2 分类)
2. 训练数据构造:KB 中答案与对应检索证据 = positive;同 Q 换错误答案 = negative
3. `ma_rag.py` 的 `_consensus` 改为: `reranker_consensus(candidates, evidence)` 用 BinaryVerifier 评分重排

---

## 4. 实现状态与已知问题

### 4.1 状态总览

| 模块 | 状态 | 行数 | 已验证 |
|------|------|------|--------|
| `whitening.py` | ✅ 完整实现 | 305 | 语法 ✓,待加载模型后测 hook 注册 |
| `clip_retriever.py` | ✅ 完整实现 | 250 | 语法 ✓,待下载完成后测编码 |
| `ma_rag.py` | ✅ 完整实现 | 155 | 语法 ✓,需增强冲突检测 |
| `meda.py` | ✅ 完整实现 | 334 | 语法 ✓,待加载模型后测 hook 注册 |
| `conformal_safety.py` | ✅ 修复+升级 | 335 | 语法 ✓,逻辑 fix 正确性待模型验证 |
| `run_medshift_v2.py` | ✅ 完整实现 | 230 | 语法 ✓,需模型加载后运行 |
| `adain.py` | ✅ 浅层 only 升级 | — | 已有实验结果(500张+前1/3层=-3.5%) |

### 4.2 已知问题

#### 高优先级: hook 路径修正

whitening.py 和 meda.py 中 `apply_whitening_to_shallow_layers`(w-hitening.py:283) 和 `apply_meda`(meda.py:304) 使用 `model.model.model.vision_encoder` / `model.model.model`,但 Hulu-Med 实际结构为:

```
实际: HulumedQwen3ForCausalLM
  .model = HulumedQwen3Model (继承 HulumedMetaModel + Qwen3Model)
    .model = Qwen3Model       <- 等两个层次? 需加载后确认
    .vision_encoder = vision encoder
```

run_ablation_phase0.py 使用 `init_llm()` 加载,返回的 `model` 可能是已包装的。**需加载 Hulu-Med 后打印模块树确认**后统一修正。

#### 中优先级

1. **QMS Router**: meda.py 缺失 per-query 动态路由,目前是单一方向。
2. **冲突检测增强**:ma_rag.py 精确匹配改为 embedding 距离阈值。
3. **CLIP_tokenizer**: clip_retriever.py 的 `open_clip.get_tokenizer` 在 torch 2.4 加载 `.bin` 时可能报错 → 需转为 safetensors 或用 `weights_only=False` 手动加载。

---

## 5. 实验验证纪律

从 FAILURE_ANALYSIS 吸取的教训,严格执行:

### 5.1 全量验证

| 数据集 | 类型 | 样本数 | Baseline |
|--------|------|--------|----------|
| CXR-VisHal | close-ended (binary+MC) | 2,017 | 78.9% |
| MM-VisHal | close-ended (混合) | 3,530 | 46.6% |
| FineGrained | close-ended (混合) | 5,547 | 58.4% |
| KnowledgeDeficiencyOE | open-ended | 2,318 | 48.6% |

**规则**:只信全量。200 条子集仅用于快速 sanity check,不得作为结论。

### 5.2 统计显著性

配对 bootstrap 95% CI (B=2000)。Δ 需在 CI 外才算有效。

### 5.3 正交消融

单线(whitening / ma-rag / meda) → 两两(3 种组合) → 三线(all) → 三线+conformal。验证不触发二阶扰动叠加负向。

### 5.4 按子任务拆分

binary / MC / OE 分别报。因为 token 失配只在 binary+MC 发生,OE 的 root cause 不同。

### 5.5 Per-sample 统计

记录每样本 helped (翻错→对) / hurt (对→错) / unchanged,盯净增。

---

## 6. 附录：代码结构

```
MedShift/
├── medshift/
│   ├── core/
│   │   ├── adain.py            (v1) AdaIN + apply_n_layers
│   │   ├── whitening.py        (v2) 特征高阶白化 ← Line 1
│   │   ├── vcd_rag.py          (v1) VCD + RAG-KI + Joint
│   │   ├── ma_rag.py           (v2) MA-RAG 多轮冲突共识 ← Line 2
│   │   ├── meda.py             (v2) MEDA 医学激活编辑 ← Line 3
│   │   ├── cache_tta.py        (v1) Cache-based TTA
│   │   └── conformal_safety.py (v2) Domain-shift-aware conformal ← 横切
│   ├── retrieval/
│   │   ├── kb_builder.py       (v1) 知识库加载 + sentence-BERT 索引
│   │   ├── clip_retriever.py   (v2) BiomedCLIP 检索 ← Line 2
│   │   └── rag_engine.py       (v1) RAG prompt 构建
│   ├── models/
│   │   └── vlm_wrapper.py      (v1) VLM 封装
│   ├── utils/
│   │   ├── metrics.py          评估指标
│   │   └── image_utils.py      图像工具
│   └── config.py               配置
├── docs/
│   ├── FAILURE_ANALYSIS.md     v1 失败根因分析
│   ├── EXPERIMENTS.md          v1 实验结果
│   ├── ALGORITHMS.md           v1 算法原理
│   ├── README.md               项目介绍
│   └── METHODS_TECHNICAL_REPORT.md ← 本文档
├── experiments/
│   ├── run_pathvqa.py          (v2 雏形) 视觉特征检索 + prompt 注入
│   └── run_radiology.py        (v2 雏形) 同上(SLAKE+VQA-RAD)
└── data/
    └── knowledge_bases/        KB 数据
        ├── xray/               放射学(1,500 条)
        └── pathology/          病理学(2,002 条)

Hulu-Med/MedUniEval/
├── run_medshift_v2.py          (v2) 统一实验入口(三线+横切+验证纪律)
├── run_ablation_phase0.py      (v1) Phase 0 消融
├── run_phase1_tta.py           (v1) Phase 1 TTA
└── utils/MedHEval/MedHEval.py  评测框架(MedHEval harness)
```
