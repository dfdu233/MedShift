"""
CCD: Calibrated Contrastive Decoding
Formula: l_ccd = (1-α)·l_orig + α·l_cal - β·|l_orig - l_cal|

Usage:
  decoder = CCDDecoder(alpha=0.3, beta=1.0)
  output = decoder.generate(model, messages_orig, messages_cal)
"""
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList


class CCDLogitsProcessor(LogitsProcessor):
    """CCD logits processor for use with HuggingFace generate()."""
    
    def __init__(self, model, messages_orig, messages_cal, alpha=0.3, beta=1.0):
        self.model = model
        self.messages_orig = messages_orig
        self.messages_cal = messages_cal
        self.alpha = alpha
        self.beta = beta
        self.inputs_orig = model.process_messages(messages_orig) if messages_orig else None
        self.inputs_cal = model.process_messages(messages_cal) if messages_cal else None
        self.cal_logits = None  # computed once per step
        self.step = 0
    
    def __call__(self, input_ids, scores):
        self.step += 1
        # Only apply CCD if we have a calibration branch
        if self.inputs_cal is None or self.inputs_orig is None:
            return scores
        
        # Forward calibration branch
        with torch.no_grad():
            cal_inputs = {k: v.to(scores.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in self.inputs_cal.items()}
            cal_outputs = self.model.model(
                **cal_inputs,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            logits_cal = cal_outputs.logits[:, -1, :]  # (1, vocab_size)
        
        # CCD formula
        logits_ccd = (1 - self.alpha) * scores + self.alpha * logits_cal - self.beta * torch.abs(scores - logits_cal)
        return logits_ccd


class CCDDecoder:
    def __init__(self, alpha=0.3, beta=1.0):
        self.alpha = alpha
        self.beta = beta
    
    def generate(self, model, messages_orig, messages_cal, max_new_tokens=64):
        """Generate with CCD using HuggingFace generate() + custom logits processor."""
        
        # If no calibration, just do normal generation
        if messages_cal is None or messages_cal == messages_orig:
            return model.generate_output(messages_orig), []
        
        # Create CCD logits processor
        ccd_processor = CCDLogitsProcessor(
            model, messages_orig, messages_cal, 
            alpha=self.alpha, beta=self.beta
        )
        
        # Use original inputs for main generation
        inputs = model.process_messages(messages_orig)
        device = inputs["input_ids"].device
        
        # Set up logits processors
        logits_processor = LogitsProcessorList([ccd_processor])
        
        # Generate with model
        with torch.no_grad():
            output_ids = model.model.generate(
                **{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()},
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                logits_processor=logits_processor,
                pad_token_id=model.tokenizer.eos_token_id,
            )
        
        output = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output, []
    
    def generate_baseline(self, model, messages, max_new_tokens=64):
        """Normal (non-CCD) generation for baseline comparison."""
        return model.generate_output(messages), []
