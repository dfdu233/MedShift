"""
MedShift: Training-Free Mitigation of Domain-Shift-Induced Hallucinations
in Medical Vision-Language Models

基于推理时输入校准和输出约束，减轻由域偏移触发的医学幻觉。
四阶段方法:
  1. 域偏移风险估计 (Domain Shift Risk Estimation)
  2. 视觉流形校准 (Visual Manifold Calibration)
  3. 证据约束生成 (Evidence-Constrained Generation)
  4. Claim级验证与重写 (Claim-Level Verification & Rewriting)
"""

__version__ = "0.1.0"
