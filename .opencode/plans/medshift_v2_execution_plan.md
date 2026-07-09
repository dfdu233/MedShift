# MedShift+ v2 执行计划

## 当前状态(已完成)
- [x] pip install open_clip_torch
- [x] 本地 CLIP (`openai/clip-vit-large-patch14-336`) 已转 safetensors,验证通过(428M,768维)
- [x] Line 1: `medshift/core/whitening.py` — 特征高阶白化(协方差+top-r 子空间)
- [x] Line 2: `medshift/core/ma_rag.py` — MA-RAG 多轮冲突共识
- [x] Line 3: `medshift/core/meda.py` — MEDA 医学激活编辑
- [x] Line 2: `medshift/retrieval/clip_retriever.py` — CLIP 检索器(需加本地 CLIP fallback)
- [x] 横切: `medshift/core/conformal_safety.py` — 修阈值覆盖 bug + 单 token 判对 + 域感知
- [x] `Hulu-Med/MedUniEval/run_medshift_v2.py` — 统一实验入口
- [x] git commit + push 完成

## 待执行步骤(按顺序)

### Step 1: 更新 clip_retriever.py 添加本地 CLIP fallback
**文件**: `medshift/retrieval/clip_retriever.py`
在 `from_pretrained` 方法后添加 `from_local_clip` 类方法:
```python
@classmethod
def from_local_clip(cls, device="cuda") -> "BiomedCLIPRetriever":
    from transformers import CLIPModel, CLIPProcessor
    snap_dir = "/root/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/"
    snap = os.listdir(snap_dir)[0]
    sd_path = os.path.join(snap_dir, snap)
    model = CLIPModel.from_pretrained(sd_path, local_files_only=True, use_safetensors=True)
    proc = CLIPProcessor.from_pretrained(sd_path, local_files_only=True)
    model = model.to(device)
    class _Wrapper:
        def __init__(self, m, p): self.m = m; self.p = p
        @torch.no_grad()
        def encode_image(self, x):
            inp = self.p(images=x, return_tensors="pt").to(self.m.device)
            f = self.m.get_image_features(**inp).pooler_output
            return f / f.norm(p=2, dim=-1, keepdim=True)
        @torch.no_grad()
        def encode_text(self, t):
            inp = self.p(text=t, return_tensors="pt", padding=True, truncation=True).to(self.m.device)
            f = self.m.get_text_features(**inp).pooler_output
            return f / f.norm(p=2, dim=-1, keepdim=True)
    wrapper = _Wrapper(model, proc)
    retriever = cls(wrapper, proc, proc.tokenizer, device=device)
    retriever._is_local_clip = True
    return retriever
```
更新 `from_kb` 改用 `from_local_clip`:
```python
r = cls.from_local_clip(device=device)  # 原为 from_pretrained
```

### Step 2: 更新 run_medshift_v2.py 使用本地 CLIP
**文件**: `Hulu-Med/MedUniEval/run_medshift_v2.py`
将 `cli` 部分的 `BiomedCLIPRetriever.from_kb` 改为 `from_local_clip` + `from_kb`:
```python
if "marag" in args.line:
    retriever = BiomedCLIPRetriever.from_local_clip(device="cuda")
    entries = load_kb(modality, kb_root=KB_ROOT)
    retriever.entries = entries
    ...
```

### Step 3: 加载 Hulu-Med 模型,验证 hook 层命名
```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=./datas python -c "
import sys; sys.path.insert(0, '.')
from LLMs import init_llm
m = init_llm()  # 用默认 ModelArgs
ve = m.model.model.vision_encoder
for name, mod in ve.named_modules():
    if 'layer_norm' in name or isinstance(mod, torch.nn.LayerNorm):
        print('LN:', name)
decoder = m.model.model
for name, mod in decoder.named_modules():
    if name.startswith('layers.') and name.count('.') == 1:
        print('DecLayer:', name)
        if len([n for n,_ in mod.named_modules()]) > 0: break  # 只打一个
"
```
验证 meda.py 中的 `name.startswith("layers.")` 是否 match。

### Step 4: 构造 source_cov.json(源域协方差)
```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=./datas python -c "
import sys; sys.path.insert(0, '.'); sys.path.insert(0, '/root/autodl-tmp/MedShift')
from medshift.core.whitening import compute_source_cov_from_kb, save_source_cov
from LLMs import init_llm
m = init_llm()
stats = compute_source_cov_from_kb(m, '/root/autodl-tmp/MedShift/data/knowledge_bases', 'xray', num_samples=500)
save_source_cov(stats, '/root/autodl-tmp/MedShift/data/knowledge_bases', 'xray')
"
```
对 pathvqa KB 也跑一次。

### Step 5: 构造 steering_vector.json(MEDA 方向向量)
```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=./datas python -c "
import sys; sys.path.insert(0, '.'); sys.path.insert(0, '/root/autodl-tmp/MedShift')
from medshift.core.meda import build_steering_from_kb, SteeringVector
from LLMs import init_llm
m = init_llm()
steer = build_steering_from_kb(m, '/root/autodl-tmp/MedShift/data/knowledge_bases', 'xray', num_samples=200)
steer.save('/root/autodl-tmp/MedShift/data/knowledge_bases/xray/steering_vector.json')
"
```

### Step 6: 构建 CLIP 图像索引(缓存)
```bash
CUDA_VISIBLE_DEVICES=0 python -c "
import sys; sys.path.insert(0, '/root/autodl-tmp/MedShift')
from medshift.retrieval.clip_retriever import BiomedCLIPRetriever, default_image_loader
KB_ROOT='/root/autodl-tmp/MedShift/data/knowledge_bases'
for mod in ['xray','pathology']:
    r = BiomedCLIPRetriever.from_local_clip()
    from medshift.retrieval.kb_builder import load_kb
    r.entries = load_kb(mod, kb_root=KB_ROOT)
    loader = default_image_loader(KB_ROOT, mod)
    cache = f'{KB_ROOT}/{mod}/clip_image_embeddings.npy'
    r.build_index(loader, cache, force=True)
"
```

### Step 7: 跑 baseline(全量)
```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=./datas python run_medshift_v2.py --dataset CXR --line baseline --n_samples 0
CUDA_VISIBLE_DEVICES=0 HF_HOME=./datas python run_medshift_v2.py --dataset MM --line baseline --n_samples 0
CUDA_VISIBLE_DEVICES=0 HF_HOME=./datas python run_medshift_v2.py --dataset FG --line baseline --n_samples 0
# Knowledge OE 14B 已确认可运行
CUDA_VISIBLE_DEVICES=0 HF_HOME=./datas python run_medshift_v2.py --dataset Knowledge --line baseline --n_samples 0
```
每个约 4–8 小时(取决于样本数)。CXR(2017)最快,MM(3530)最慢。

### Step 8: 跑三线消融(全量)
```bash
for line in whitening marag meda whitening+marag marag+meda whitening+meda all; do
    for ds in CXR MM FG Knowledge; do
        CUDA_VISIBLE_DEVICES=0 HF_HOME=./datas python run_medshift_v2.py --dataset $ds --line $line --n_samples 0
    done
done
```

### Step 9: 横切 Conformal
```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=./datas python -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'/root/autodl-tmp/MedShift')
from medshift.core.conformal_safety import evaluate_selective_prediction, ConformalSafetyLayer
# 加载模型和样本,调 evaluate_selective_prediction
"
```

### Step 10: 提交 github
```bash
cd /root/autodl-tmp/MedShift
git add -A && git commit -m "MedShift+ v2: local CLIP retriever, execution fixes"
git push origin main
```

## 关键风险
1. **torch 2.4 + Hulu-Med**: 已有 run_ablation_phase0.py 成功运行先例,兼容性已验证
2. **MEDA steering**: `build_steering_from_kb` 需要 VLM 读 activation through forward;每样本两次前向(正确+错误),200 样本约 5–10 分钟
3. **MA-RAG 算力**: 2 轮 × 3 候选 = 6× 前向/样本;全量 MM(3530) 约 8–12 小时
4. **OOM**: 29B 模型需约 16GB;12-layer forward hook 额外 ~2GB;RTX 4090 24GB 应够

## 验证纪律
- **只信全量**(n_samples=0);200 条 = 假阳性(FAILURE_ANALYSIS §4)
- **per-sample helped/hurt 拆分**:内置在 run_medshift_v2.py 输出
- **配对 bootstrap 95% CI**:内置在 summary 输出
- **按子任务 binary/MC/OE 分报**:内置 `by_task` 字段

## 样本量策略更新(基于 FAILURE_ANALYSIS §4 教训)

### 关键原则
- **200 条 → 假阳性**:95% CI 内单条随机翻正,不做任何结论
- **全量 13,412 条**:最终结论的唯一可靠依据
- **中间检查**:3,000–5,000 条子集可做 sanity check,但不停止

### Step 4 修正:source_cov.json 样本量

| 模态 | KB 图像数 | 特征向量数(×256 patches) | 1152维协方差需求 |
|------|----------|------------------------|----------------|
| X-ray | 1,500 | 384,000 | ✅ 远超 dim²(1.3M) |
| Pathology | 2,002 | 512,512 | ✅ 远超 |

原来 500 条够用,**使用全量 KB 1,500+ 张**以获得最稳的协方差估计。
```bash
# 改为全量:
stats = compute_source_cov_from_kb(m, KB_ROOT, 'xray', num_samples=1500)
```

### Step 5 修正:steering_vector.json 样本量

MEDA 方向向量需要**正确 vs 错误激活对比**。当前 `_make_wrong_answer` 仅支持 binary/MC(占~70%),OE 问题被跳过。

| 问题类型 | 占比 | 当前能否构造错误答案 | 建议 |
|---------|------|-------------------|------|
| binary(Yes/No) | ~35% | ✅ 直接反转 | 全量使用 |
| MC(A/B/C/D) | ~35% | ✅ 旋转字母 | 全量使用 |
| OE(开放式) | ~30% | ❌ 当前跳过 | 需新增:用 LLM judge(或随机采样 KB 中其他答案作为干扰) |

**方案**:对 OE 问题,用同一个 KB 内不同问题的答案作为负样本(随机采样),扩大可用样本到全量。

```python
# 在 _make_wrong_answer 中新增:
def _make_wrong_answer(correct, kb_answers=None):
    c = correct.strip().lower()
    if c in ("yes", "no"): return "no" if c == "yes" else "yes"
    if len(c) == 1 and c in "abcd": return {"a":"b","b":"a","c":"d","d":"c"}[c]
    # OE: 从 KB 随机采一个不同的答案作为弱负样本
    if kb_answers:
        import random
        others = [a for a in kb_answers if a.strip().lower() != c]
        return random.choice(others) if others else None
    return None
```

这样 steering vector 可用样本从 binary/MC 的~70% 扩展到全部 KB(~1,500-2,000)。

### Step 7–8 实验:全量运行(不变)

| 数据集 | 全量 |
|--------|------|
| CXR-VisHal | 2,017 |
| MM-VisHal | 3,530 |
| FineGrained | 5,547 |
| KnowledgeDeficiencyOE | 2,318 |
| **合计** | **13,412** |

每条线跑一次约 12–24 小时(MA-RAG 最慢),建议在无 GPU 竞争时夜间跑。

### 消融组合全量

| 配置 | n_samples | 预计时间(全量) |
|------|----------|-------------|
| baseline | 全量 | 4h |
| whitening | 全量 | 4h |
| marag | 全量 | 12h |
| meda | 全量 | 4h |
| whitening+marag | 全量 | 12h |
| meda+marag | 全量 | 12h |
| whitening+meda | 全量 | 4h |
| all(whitening+meda+marag) | 全量 | 12h |

约 64 小时总 GPU 时间。可按单行 `for line in baseline whitening marag meda; do CUDA... python run_medshift_v2.py --dataset CXR --line $line --n_samples 0; done` 分批逐步跑,可中断(断点续存 Json)。

### 显著性判定标准

- **有显著提升**:全量 Δ > 1.0% 且 95% CI 不重叠
- **有显著降级**:全量 Δ < -1.0% 且 95% CI 不重叠
- **可考虑**:全量 Δ > 0.5% 但 CI 有重叠 → 需在其它数据集复现
- **放弃**:200 条 Δ 再大都不可信(已证实假阳性)

### 验证纪律汇总
- 全量(0) = 真实结论 | 200 条 = 不可信
- per-sample helped/hurt 拆分看净增量(非原始 acc)
- 按 binary/MC/OE 分别报(根因不同)
- 配对 bootstrap 95% CI 为显著性门槛
