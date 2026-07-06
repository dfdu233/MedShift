"""
MedShift 全局配置
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DomainShiftConfig:
    """Stage 1: 域偏移风险估计配置"""
    num_perturbation_views: int = 8          # K: 语义保持扰动视图数
    perturbation_types: List[str] = field(default_factory=lambda: [
        "gamma", "contrast", "brightness", "blur",
        "noise", "histogram_shift", "clahe", "jpeg_compress"
    ])
    gamma_range: tuple = (0.7, 1.5)         # gamma校正范围
    contrast_range: tuple = (0.8, 1.3)      # 对比度扰动范围
    brightness_range: tuple = (0.85, 1.2)   # 亮度扰动范围
    blur_kernel_range: tuple = (3, 7)       # 高斯模糊核大小
    noise_std_range: tuple = (0.01, 0.05)   # 高斯噪声标准差
    jpeg_quality_range: tuple = (40, 90)    # JPEG压缩质量
    vfi_high_threshold: float = 0.15        # VFI高风险阈值
    vfi_medium_threshold: float = 0.08      # VFI中风险阈值


@dataclass
class ManifoldCalibrationConfig:
    """Stage 2: 视觉流形校准配置"""
    num_candidates: int = 6                  # 候选校准图像数量
    calibration_methods: List[str] = field(default_factory=lambda: [
        "histogram_matching", "clahe", "gamma_correction",
        "stain_normalization", "illumination_correction", "adaptive_equalization"
    ])
    manifold_weight: float = 0.4             # 流形接近度权重
    semantic_weight: float = 0.4             # 语义保持权重
    structure_weight: float = 0.2            # 结构保持权重
    similarity_top_k: int = 5               # memory bank检索top-k


@dataclass
class EvidenceGenerationConfig:
    """Stage 3: 证据约束生成配置"""
    max_findings: int = 8                    # 最大findings数量
    require_evidence: bool = True            # 是否要求证据支持
    structured_output: bool = True           # 是否要求结构化输出
    temperature: float = 0.1                 # 生成温度
    max_new_tokens: int = 512                # 最大生成token数


@dataclass
class ClaimVerificationConfig:
    """Stage 4: Claim级验证与重写配置"""
    visual_support_threshold: float = 0.5    # 视觉支持阈值
    multiview_threshold: float = 0.6         # 多视图一致性阈值
    retrieval_threshold: float = 0.4         # 检索支持阈值
    high_confidence_threshold: float = 0.7   # 高置信度阈值
    medium_confidence_threshold: float = 0.4  # 中置信度阈值
    r_shift_penalty_weight: float = 0.3      # 域偏移惩罚权重
    uncertainty_prefix: str = "Possible but uncertain:"  # 不确定性前缀


@dataclass
class CCDConfig:
    """
    校准对比解码 (CCD) 配置

    对应论文: Calibrated Contrastive Decoding for Mitigating Domain-Shift Hallucination

    核心公式:
        logits_ccd_t = (1-α)·l_t + α·l*_t - β·|l_t - l*_t|

    其中:
        α ∈ [0,1]: 校准 logits 的插值权重 (α=0 → 原始, α=1 → 校准)
        β ∈ [0,∞): 两个 logits 分歧的惩罚强度
        γ ∈ [0,∞): 熵阈值, 超过则触发 early stopping

    参考 VCD (CVPR 2024): 使用硬对比 l - λ·l*
    CCD 改进: (1-α)·l + α·l* - β·|l - l*| (软插值 + 分歧惩罚)
    """
    alpha: float = 0.3                    # CCD α: 校准logits插值权重
    beta: float = 1.0                     # CCD β: 分歧惩罚强度
    entropy_threshold: float = 2.5        # 熵阈值 γ: 超过则提前终止
    dynamic_alpha_beta: bool = True       # 是否使用不确定性自适应版本
    early_exit: bool = True               # 是否启用熵早停
    vocab_size: int = 32000               # 词汇表大小(从模型自动获取)


@dataclass
class MedShiftConfig:
    """MedShift 总配置"""
    # 模型配置
    model_path: str = ""                     # VLM模型路径
    model_name: str = "Phi-3.5-vision"       # 模型名称
    device: str = "cuda"                     # 设备
    dtype: str = "float16"                   # 模型精度
    max_memory_gb: float = 10.0              # 最大GPU显存使用

    # 数据配置
    data_dir: str = "data/slake"             # 数据目录
    split: str = "test"                      # 数据集划分
    max_samples: Optional[int] = None        # 最大样本数（调试用）
    image_dir: str = ""                      # 图像目录

    # Memory bank配置
    memory_bank_path: str = ""               # 源域memory bank路径
    use_memory_bank: bool = True             # 是否使用memory bank

    # 各阶段配置
    domain_shift: DomainShiftConfig = field(default_factory=DomainShiftConfig)
    manifold_calibration: ManifoldCalibrationConfig = field(default_factory=ManifoldCalibrationConfig)
    evidence_generation: EvidenceGenerationConfig = field(default_factory=EvidenceGenerationConfig)
    claim_verification: ClaimVerificationConfig = field(default_factory=ClaimVerificationConfig)

    # 输出配置
    output_dir: str = "results"              # 输出目录
    save_details: bool = True                # 是否保存详细结果
    log_interval: int = 10                   # 日志间隔
