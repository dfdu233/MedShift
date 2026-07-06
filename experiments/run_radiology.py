"""Radiology ablation: Baseline vs RAG on SLAKE + VQA-RAD."""
import os, sys, json, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from medshift.models.vlm_wrapper import MedicalVLMWrapper
from medshift.data.slake_loader import load_slake_dataset, load_image
from medshift.utils.metrics import soft_em


def extract(img, vlm):
    f = vlm.extract_visual_features(img)
    if isinstance(f, __import__("torch").Tensor):
        f = f.cpu().numpy()
    return f.flatten() if f.ndim > 1 else f


def short(raw):
    if not raw:
        return ""
    import re
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    for p in ["The organ shown", "The answer is", "Based on the image,",
              "The image shows", "This image shows:", "In this image,", "Answer:", "A:"]:
        if raw.lower().startswith(p.lower()):
            raw = raw[len(p):].strip()
    return raw.rstrip(".")[:50].strip() if len(raw) > 30 else raw.rstrip(".").strip()


def build_kb(vlm, max_kb=200):
    """Build memory bank from SLAKE train."""
    print(f"Building KB ({max_kb} entries)...", flush=True)
    from medshift.data.slake_loader import load_slake_dataset, load_image
    train = load_slake_dataset("data/slake", split="train", max_samples=max_kb)
    kb = []
    for it in train:
        img = load_image(it["image_path"])
        if img:
            feat = extract(img.convert("RGB"), vlm)
            kb.append({"q": it["question"], "a": it["answer"], "feat": feat})
    return kb


def main():
    # Load
    print("Loading VLM...", flush=True)
    vlm = MedicalVLMWrapper(
        model_path="/home/dubw/.cache/huggingface/hub/models--ZJU-AI4H--Hulu-Med-14B",
        device="cuda", dtype="float16"
    )
    vlm.load()

    # Build KB
    kb = build_kb(vlm)
    print(f"KB: {len(kb)} entries", flush=True)

    # Load test data
    tests = []
    # SLAKE
    for it in load_slake_dataset("data/slake", split="test", max_samples=300):
        img = load_image(it["image_path"])
        if img:
            tests.append({"ds": "SLAKE", "img": img.convert("RGB"),
                          "q": it["question"], "a": it["answer"]})
    # VQA-RAD
    import pickle
    with open("data/vqa_rad_test.pkl", "rb") as f:
        for it in pickle.load(f):
            img = it["image"].convert("RGB") if hasattr(it["image"], "convert") else \
                  Image.open(it["image_path"]).convert("RGB")
            tests.append({"ds": "VQA-RAD", "img": img,
                          "q": it["question"], "a": it["answer"]})
    print(f"Test: {len(tests)} samples", flush=True)

    # Evaluate
    PT = "Answer concisely based on the image: {}"
    results = []
    for i, item in enumerate(tests):
        if i >= 200:
            break
        img, q, gt = item["img"], item["q"], item["a"]
        prompt = PT.format(q)

        # Baseline
        orig_ans, _ = vlm.generate(img, prompt, max_new_tokens=32, temperature=0.0)
        orig_s = short(orig_ans)

        # RAG
        qf = extract(img, vlm)
        qf_n = qf / (np.linalg.norm(qf) + 1e-10)
        sims = [(float(np.dot(e["feat"] / (np.linalg.norm(e["feat"]) + 1e-10), qf_n)), e)
                for e in kb]
        sims.sort(key=lambda x: x[0], reverse=True)
        ev = [f"Similar case (sim={s:.2f}): Q=\"{e['q']}\" A=\"{e['a']}\""
              for s, e in sims[:3] if s > 0.5]
        rag_p = f"{prompt}\n\nReference similar cases:\n" + "\n".join(ev[:3]) if ev else prompt
        rag_ans, _ = vlm.generate(img, rag_p, max_new_tokens=32, temperature=0.0)

        results.append({
            "ds": item["ds"], "gt": gt,
            "orig_s": orig_s, "orig_em": soft_em(orig_s, gt),
            "rag_s": short(rag_ans), "rag_em": soft_em(short(rag_ans), gt),
        })
        if (i + 1) % 50 == 0:
            n = i + 1
            be = sum(1 for r in results if r["orig_em"])
            re_ = sum(1 for r in results if r["rag_em"])
            print(f"  [{i+1}/{len(tests)}] Base={be/n:.1%} RAG={re_/n:.1%}", flush=True)

    # Report
    n = max(len(results), 1)
    be = sum(1 for r in results if r["orig_em"]) / n
    re_ = sum(1 for r in results if r["rag_em"]) / n
    print(f"\nRadiology ablation (N={n}):")
    print(f"  Baseline: {be:.1%}")
    print(f"  RAG:      {re_:.1%} (Δ={re_-be:+.1%})")

    # Save
    os.makedirs("results", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = {"n": n, "baseline_em": be, "rag_em": re_}
    with open(f"results/radiology_{ts}.json", "w") as f:
        json.dump(out, f)
    print(f"Saved: results/radiology_{ts}.json")


if __name__ == "__main__":
    main()
