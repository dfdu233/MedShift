# MedShift

Training-free domain generalization for medical VLM hallucination mitigation via retrieval-augmented decoding.

## Results Summary

| Method | Domain | N | EM | Δ |
|--------|--------|---:|---:|--:|
| Baseline | Radiology | 950 | 46.4% | — |
| ★ RAG | Radiology | 950 | **60.6%** | **+14.2%** |
| VQA-RAD (OOD) | Radiology | 451 | **67.8%** | **+24.6%** |
| SLAKE (ID) | Radiology | 499 | 54.1% | +4.8% |
| RAG on PathVQA | Pathology | 500 | 36.4% | -3.4% |

## Structure

```
medshift/          Core library (VLM wrapper, RAG engine, CCD)
experiments/       Experiment scripts
results/           Source center and memory bank
reports/           PDF reports
```

## Quick Start

```bash
pip install -r requirements.txt
python experiments/run_radiology.py
python experiments/run_pathvqa.py
```

## Citation

If you use this code, please cite our work.
