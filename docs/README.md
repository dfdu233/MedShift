# MedShift+ — Medical VLM Hallucination Mitigation

## 项目目标

缓解医学 VLM（Hulu-Med 14B）在测试域的幻觉问题。核心思路是将输入图像对齐到源域（训练域）的特征分布中心，同时以 RAG 辅助提升知识准确性。

## 模型

| 属性 | 值 |
|------|-----|
| 模型 | Hulu-Med 14B (ZJU-AI4H/Hulu-Med-14B) |
| 基座 | Qwen3-14B (MoE, 128 experts, 40 layers) |
| Vision Encoder | HulumedVisionEncoder (27 layers, 1152 hidden, ViT patch 14) |
| 精度 | bfloat16 |
| Attention | SDPA (PyTorch 原生) |
| GPU | RTX 4090 (49GB) 单卡 |

## Benchmark

| 数据集 | 类型 | 样本数 | Baseline Acc | 幻觉类型 |
|--------|------|--------|-------------|---------|
| CXR-VisHal | close-ended (binary+MC) | 2,017 | **78.9%** | 视觉误读 (CXR 单模态) |
| MM-VisHal | close-ended (binary+MC) | 3,530 | **46.6%** | 视觉误读 (多模态混合) |
| FineGrained | close-ended (mixed) | 5,547 | **58.4%** | 视觉误读 (细粒度) |
| KnowledgeDeficiencyOE | open-ended | 2,318 | **48.6%** | 知识缺失 |
| **合计** | | **13,412** | | |

模型在 CXR (78.9%) 上接近 GPT-4o (79.4%)，但在 MM-VisHal 上大幅下降 (46.6%)，说明域偏移严重影响多模态场景。

## 核心思路演进

```
原始方案: 源域中心(AdaIN) + VCD + RAG-KI → 三层对齐
               ↓ 全部降级 baseline
修正方案: AdaIN(500张+前1/3层) + Cache-based TTA + Additive RAG
               ↓ 全量验证仍无效
当前结论: 训练免费方法在 14B 上全部无效 → 需训练级干预 (QLoRA DPO)
```

## 核心结论

**所有训练免费方法在 Hulu-Med 14B 上全部无效。** 系统验证了 6 种方法：

| # | 方法 | 来源 | 最佳效果 | 结论 |
|---|------|------|---------|------|
| 1 | AdaIN | 本项目 | -3.5% (CXR) | 特征扰动破坏语义 |
| 2 | VCD (Greedy) | CVPR 2024 | -2~8% (CXR) | 贪心锁死,logits 无法翻转 |
| 3 | VCD (Sampling) | CVPR 2024 | -13% (CXR) | 模型在采样下退化 |
| 4 | RAG-KI (Subtractive) | 本项目 | 崩溃 (CXR) | 方向反: 减增强输入 |
| 5 | RAG (Additive Entity-boost) | 本项目 | -5.5% (CXR) | Token 粒度不匹配 |
| 6 | Cache-based TTA | 本项目 | -0.2% (MM 全量) | 检索不可用,假阳性 |

**后续唯一可行路径**: QLoRA 4-bit + DPO（训练级干预）

## 代码位置

| 模块 | 路径 | 行数 |
|------|------|------|
| 评估框架 | `Hulu-Med/MedUniEval/utils/MedHEval/MedHEval.py` | 271 |
| 消融脚本 (Phase 0) | `Hulu-Med/MedUniEval/run_ablation_final.py` | 202 |
| 消融脚本 (Phase 0 修正) | `Hulu-Med/MedUniEval/run_ablation_phase0.py` | 442 |
| Phase 1 Cache-TTA | `Hulu-Med/MedUniEval/run_phase1_tta.py` | 296 |
| 知识库构建 | `MedShift/medshift/retrieval/kb_builder.py` | 114 |
| RAG 引擎 | `MedShift/medshift/retrieval/rag_engine.py` | 19 |
| AdaIN 实现 | `MedShift/medshift/core/adain.py` | 201 |
| VCD+RAG 联合 | `MedShift/medshift/core/vcd_rag.py` | 186 |
| Cache-TTA 核心 | `MedShift/medshift/core/cache_tta.py` | 166 |
| Conformal Safety | `MedShift/medshift/core/conformal_safety.py` | 210 |
| 配置 | `MedShift/medshift/config.py` | 117 |
| 基线结果 | `Hulu-Med/MedUniEval/eval_results_medheval_fix/` | 4 JSON |
| 消融结果 | `experiments/ablation_final/summary.txt` | 14 行 |
| Phase 1 结果 | `experiments/phase1_tta/MM_summary.txt` | 6 行 |

## 文档结构

| 文件 | 内容 |
|------|------|
| `ALGORITHMS.md` | 所有算法的原理、公式、实现细节 |
| `EXPERIMENTS.md` | 完整实验结果表格与对比 |
| `FAILURE_ANALYSIS.md` | 6 种训练免费方法的失败根因详细分析 |
