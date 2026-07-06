"""PathVQA: Baseline vs RAG only (fastest comparison)"""
import os, sys, json, time, io
import numpy as np
import torch
import pandas as pd
from PIL import Image
sys.path.insert(0, '.')
from medshift.models.vlm_wrapper import MedicalVLMWrapper
from medshift.utils.metrics import soft_em

def short(raw):
    if not raw: return ""
    import re
    raw = re.sub(r'<think>.*?</think>','',raw,flags=re.DOTALL).strip()
    for p in ["The organ shown","The answer is","Based on the image,","The image shows",
              "This image shows:","In this image,","Answer:","A:"]:
        if raw.lower().startswith(p.lower()): raw=raw[len(p):].strip()
    return raw.rstrip('.')[:50].strip() if len(raw)>30 else raw.rstrip('.').strip()

def load_img(row):
    img=row['image']
    if isinstance(img,dict) and 'bytes' in img:
        return Image.open(io.BytesIO(img['bytes'])).convert("RGB")
    return None

vlm=MedicalVLMWrapper(model_path='/home/dubw/.cache/huggingface/hub/models--ZJU-AI4H--Hulu-Med-14B',device='cuda',dtype='float16')
vlm.load()

# KB from validation (200)
kb=[]
for _,row in pd.read_parquet('/home/dubw/data/path-vqa/data/validation-00000-of-00003-90a5518d26493b67.parquet').iterrows():
    if len(kb)>=200: break
    img=load_img(row)
    if img:
        f=vlm.extract_visual_features(img)
        if isinstance(f,torch.Tensor): f=f.cpu().numpy()
        kb.append({"q":row["question"],"a":row["answer"],"feat":f.flatten()})
print(f"KB: {len(kb)}",flush=True)

# Test data (all 2239)
tests=[]
for fn in ['test-00000-of-00003-e9adadb4799f44d3.parquet','test-00001-of-00003-7ea98873fc919813.parquet']:
    for _,row in pd.read_parquet(f'/home/dubw/data/path-vqa/data/{fn}').iterrows():
        if len(tests)>=500: break
        img=load_img(row)
        if img: tests.append({"img":img,"q":row["question"],"a":row["answer"]})
print(f"Test: {len(tests)}",flush=True)

# Eval
PT="Answer concisely based on the image: {}"
results=[]
for i,item in enumerate(tests):
    if i>=500: break
    img,q,gt=item["img"],item["q"],item["a"]
    prompt=PT.format(q)

    orig_ans,_=vlm.generate(img,prompt,max_new_tokens=32,temperature=0.0)
    orig_s=short(orig_ans)

    qf=vlm.extract_visual_features(img)
    if isinstance(qf,torch.Tensor): qf=qf.cpu().numpy()
    qf=qf.flatten(); qf_n=qf/(np.linalg.norm(qf)+1e-10)
    sims=[(float(np.dot(e["feat"]/(np.linalg.norm(e["feat"])+1e-10),qf_n)),e) for e in kb]
    sims.sort(key=lambda x:x[0],reverse=True)
    ev=[f"Similar case (sim={s:.2f}): Q=\"{e['q']}\" A=\"{e['a']}\"" for s,e in sims[:3] if s>0.5]
    rag_p=f"{prompt}\n\nReference similar cases:\n"+"\n".join(ev[:3]) if ev else prompt
    rag_ans,_=vlm.generate(img,rag_p,max_new_tokens=32,temperature=0.0)
    rag_s=short(rag_ans)

    results.append({"orig_s":orig_s,"orig_em":soft_em(orig_s,gt),"rag_s":rag_s,"rag_em":soft_em(rag_s,gt),"gt":gt})
    if (i+1)%100==0:
        n=i+1; be=sum(1 for r in results if r["orig_em"]); re=sum(1 for r in results if r["rag_em"])
        print(f"  [{i+1}] Base={be/n:.1%} RAG={re/n:.1%} (Δ={re/n-be/n:+.1%})",flush=True)

n=len(results)
be=sum(1 for r in results if r["orig_em"]); re=sum(1 for r in results if r["rag_em"])
print(f"\nPATHVQA (N={n}): Baseline={be/n:.1%} RAG={re/n:.1%} (Δ={re/n-be/n:+.1%})",flush=True)

ts=time.strftime("%Y%m%d_%H%M%S")
os.makedirs("results/pathvqa_ablation",exist_ok=True)
with open(f"results/pathvqa_ablation/pathvqa_rag_{ts}.json","w") as f:
    json.dump({"n":n,"baseline_em":be/n,"rag_em":re/n},f)
print(f"Saved.",flush=True)
