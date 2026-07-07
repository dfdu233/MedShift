"""
VCD + RAG-KI + AdaIN Joint Framework
=====================================
VCD:      l_vcd = (1+α)·l(v,x) - α·l(v',x)      v'=噪声图像
RAG-KI:   l_rag = (1+β)·l(v,x) - β·l(v,x_rag)    x_rag=RAG增强prompt
AdaIN:    特征层源域对齐
Joint:    l_final = γ·l_vcd + (1-γ)·l_rag
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList
from typing import Optional


def add_gaussian_noise_to_image(image: Image.Image, noise_std: float = 0.1) -> Image.Image:
    """Add Gaussian noise to PIL image. Returns noisy PIL image."""
    img_array = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    noise = np.random.randn(*img_array.shape) * noise_std
    noisy = np.clip(img_array + noise, 0, 1)
    noisy_img = Image.fromarray((noisy * 255).astype(np.uint8))
    return noisy_img


class VCDLogitsProcessor(LogitsProcessor):
    """VCD contrastive decoding: contrast original vs noise-distorted image logits."""
    
    def __init__(self, model, inputs_orig, inputs_noise, alpha=0.5):
        self.model = model
        self.inputs_orig = inputs_orig
        self.inputs_noise = inputs_noise
        self.alpha = alpha
    
    def __call__(self, input_ids, scores):
        with torch.no_grad():
            noise_inputs = {
                k: v.to(scores.device) if isinstance(v, torch.Tensor) else v
                for k, v in self.inputs_noise.items()
            }
            out_noise = self.model.model(**noise_inputs, use_cache=True, return_dict=True)
            logits_noise = out_noise.logits[:, -1, :]
        
        l_vcd = (1 + self.alpha) * scores - self.alpha * logits_noise
        return l_vcd


class RAGKILogitsProcessor(LogitsProcessor):
    """RAG Knowledge Injection: contrast original vs RAG-enhanced prompt logits."""
    
    def __init__(self, model, inputs_orig, inputs_rag, beta=0.3):
        self.model = model
        self.inputs_orig = inputs_orig
        self.inputs_rag = inputs_rag
        self.beta = beta
    
    def __call__(self, input_ids, scores):
        with torch.no_grad():
            rag_inputs = {
                k: v.to(scores.device) if isinstance(v, torch.Tensor) else v
                for k, v in self.inputs_rag.items()
            }
            out_rag = self.model.model(**rag_inputs, use_cache=True, return_dict=True)
            logits_rag = out_rag.logits[:, -1, :]
        
        l_rag = (1 + self.beta) * scores - self.beta * logits_rag
        return l_rag


class JointVCDRAGLogitsProcessor(LogitsProcessor):
    """Joint VCD + RAG-KI fusion."""
    
    def __init__(self, model, inputs_orig, inputs_noise, inputs_rag, 
                 alpha=0.5, beta=0.3, gamma=0.6):
        self.model = model
        self.inputs_orig = inputs_orig
        self.inputs_noise = inputs_noise
        self.inputs_rag = inputs_rag
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def __call__(self, input_ids, scores):
        with torch.no_grad():
            # Forward noise branch
            noise_inputs = {
                k: v.to(scores.device) if isinstance(v, torch.Tensor) else v
                for k, v in self.inputs_noise.items()
            }
            out_noise = self.model.model(**noise_inputs, use_cache=True, return_dict=True)
            logits_noise = out_noise.logits[:, -1, :]
            
            # Forward RAG branch
            rag_inputs = {
                k: v.to(scores.device) if isinstance(v, torch.Tensor) else v
                for k, v in self.inputs_rag.items()
            }
            out_rag = self.model.model(**rag_inputs, use_cache=True, return_dict=True)
            logits_rag = out_rag.logits[:, -1, :]
        
        l_vcd = (1 + self.alpha) * scores - self.alpha * logits_noise
        l_rag_ki = (1 + self.beta) * scores - self.beta * logits_rag
        
        l_joint = self.gamma * l_vcd + (1 - self.gamma) * l_rag_ki
        return l_joint


def process_messages_with_noise(model, messages):
    """Process messages with added noise for VCD."""
    inputs = model.process_messages(messages)
    if "image" in messages and messages["image"] is not None:
        noisy_img = add_gaussian_noise_to_image(messages["image"])
        noisy_messages = {**messages, "image": noisy_img}
        inputs_noise = model.process_messages(noisy_messages)
    elif "pixel_values" in inputs:
        # Fallback: add noise directly to pixel_values
        inputs_noise = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs_noise["pixel_values"] = inputs["pixel_values"] + torch.randn_like(inputs["pixel_values"]) * 0.05
    else:
        inputs_noise = inputs
    return inputs, inputs_noise


def generate_with_vcd_rag(model, messages_orig, messages_rag=None, 
                           alpha=0.5, beta=0.3, gamma=0.6, 
                           use_vcd=True, use_rag=True, use_adain=False,
                           max_new_tokens=64, adain_hooks=None):
    """
    Generate with VCD, RAG-KI, and optional AdaIN.
    
    Args:
        model: HuluMed model
        messages_orig: original {prompt, image}
        messages_rag: RAG-enhanced {prompt, image} (same image, different prompt)
        alpha: VCD strength
        beta: RAG-KI strength  
        gamma: VCD vs RAG-KI weight in joint mode
        use_vcd: enable VCD
        use_rag: enable RAG-KI
        use_adain: enable AdaIN hooks
        max_new_tokens: max generation length
        adain_hooks: list of hook handles (applied before forward)
    """
    if use_adain and adain_hooks:
        pass  # Hooks should already be registered on model
    
    # Process messages (original and noise)
    inputs_orig, inputs_noise = process_messages_with_noise(model, messages_orig)
    
    # Process RAG messages if provided
    inputs_rag = None
    if messages_rag is not None and use_rag:
        inputs_rag, _ = process_messages_with_noise(model, messages_rag)
    
    device = inputs_orig["input_ids"].device
    
    # Create processor list
    processors = LogitsProcessorList()
    
    if use_vcd and use_rag and inputs_rag is not None:
        processors.append(JointVCDRAGLogitsProcessor(
            model, inputs_orig, inputs_noise, inputs_rag,
            alpha=alpha, beta=beta, gamma=gamma
        ))
    elif use_vcd:
        processors.append(VCDLogitsProcessor(
            model, inputs_orig, inputs_noise, alpha=alpha
        ))
    elif use_rag and inputs_rag is not None:
        processors.append(RAGKILogitsProcessor(
            model, inputs_orig, inputs_rag, beta=beta
        ))
    
    # Generate
    with torch.no_grad():
        output_ids = model.model.generate(
            **{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_orig.items()},
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            logits_processor=processors,
            pad_token_id=model.tokenizer.eos_token_id,
        )
    
    output = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return output
