"""
医学视觉语言模型 (VLM) 封装
支持 Phi-3.5-vision-instruct 和其他医学VLM
"""
import os
import re
import json
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class MedicalVLMWrapper:
    """
    医学VLM统一封装
    支持:
      - Microsoft Phi-3.5-vision-instruct
      - LLaVA系列
      - 其他HuggingFace VLM
    """

    def __init__(self, model_path: str, device: str = "cuda",
                 dtype: str = "float16", max_memory_gb: float = 10.0):
        self.model_path = model_path
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.max_memory_gb = max_memory_gb
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.model_type = self._detect_model_type()

    def _detect_model_type(self) -> str:
        """检测模型类型"""
        config_path = Path(self.model_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            model_type = config.get("model_type", "").lower()
            if "phi3" in model_type or "phi" in model_type:
                return "phi3v"
            if "hulumed" in model_type or "hulu" in model_type:
                return "hulumed"
            if "llava" in model_type:
                return "llava"
        # 从目录名推断
        name = Path(self.model_path).name.lower()
        if "phi" in name:
            return "phi3v"
        if "hulu" in name:
            return "hulumed"
        if "llava" in name:
            return "llava"
        return "generic"

    def load(self):
        """加载模型和处理器"""
        print(f"[VLM] 加载模型: {self.model_path}")
        print(f"[VLM] 检测到模型类型: {self.model_type}")

        if self.model_type == "phi3v":
            self._load_phi3v()
        elif self.model_type == "llava":
            self._load_llava()
        elif self.model_type == "hulumed":
            self._load_hulumed()
        else:
            self._load_generic()

        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"[VLM] 加载完成 | 参数量: {param_count / 1e9:.2f}B | 设备: {self.device}")

    def _load_phi3v(self):
        """加载Phi-3.5-vision-instruct模型"""
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            num_crops=4,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            _attn_implementation="eager",
        )

        if self.device == "cpu":
            self.model = self.model.to("cpu")

        self.model.eval()

    def _load_llava(self):
        """加载LLaVA模型 (支持LLaVA-Med等自定义架构)"""
        from transformers import AutoTokenizer

        # 检测是否是自定义LLaVA架构 (如LLaVA-Med)
        config_path = Path(self.model_path) / "config.json"
        is_custom = False
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            arch = config.get("architectures", [""])[0]
            model_type = config.get("model_type", "")
            if "LlavaMistral" in arch or model_type == "llava_mistral":
                is_custom = True

        if is_custom:
            # LLaVA-Med: 手动构建config + 权重映射 + accelerate分片
            self._load_llava_med(config)
        else:
            # 标准LLaVA
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                device_map="auto" if self.device == "cuda" else None,
            )
            self.model.eval()

    def _load_llava_med(self, config: dict):
        """加载LLaVA-Med-v1.5-Mistral-7B (需要权重映射)"""
        import re
        from transformers import (LlavaForConditionalGeneration, LlavaConfig,
                                  MistralConfig, CLIPVisionConfig, AutoTokenizer)
        from safetensors.torch import load_file

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)

        # 构建与checkpoint匹配的config
        llm_config = MistralConfig(
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            hidden_act=config["hidden_act"],
            max_position_embeddings=config["max_position_embeddings"],
            rms_norm_eps=config["rms_norm_eps"],
            rope_theta=config["rope_theta"],
            vocab_size=config["vocab_size"],
            tie_word_embeddings=False,
        )
        vision_config = CLIPVisionConfig(
            hidden_size=1024, intermediate_size=4096,
            num_hidden_layers=24, num_attention_heads=16,
            image_size=336, patch_size=14,
        )
        self.image_token_id = config["vocab_size"]  # 32000
        self.num_image_patches = (336 // 14) ** 2  # 576

        llava_config = LlavaConfig(
            vision_config=vision_config, text_config=llm_config,
            image_token_index=self.image_token_id,
            projector_type=config.get("mm_projector_type", "mlp2x_gelu"),
        )

        # Step 1: from_pretrained puts model on GPU with device_map
        print("[LLaVA-Med] Loading model with device_map=auto...")
        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path, config=llava_config,
            torch_dtype=self.dtype, device_map="auto",
            ignore_mismatched_sizes=True
        )

        # Step 2: Map weights
        print("[LLaVA-Med] Mapping weights...")
        def mk(k):
            k = re.sub(r'^model\.layers\.', 'model.language_model.layers.', k)
            k = re.sub(r'^model\.vision_tower\.vision_tower\.vision_model\.encoder\.', 'model.vision_tower.encoder.', k)
            k = re.sub(r'^model\.vision_tower\.vision_tower\.vision_model\.', 'model.vision_tower.', k)
            k = re.sub(r'^model\.mm_projector\.0\.', 'model.multi_modal_projector.linear_1.', k)
            k = re.sub(r'^model\.mm_projector\.2\.', 'model.multi_modal_projector.linear_2.', k)
            k = re.sub(r'^model\.embed_tokens\.', 'model.language_model.embed_tokens.', k)
            k = re.sub(r'^model\.norm\.', 'model.language_model.norm.', k)
            return k

        sd = {}
        for f in sorted(Path(self.model_path).glob('model-*.safetensors')):
            d = load_file(str(f))
            for k, v in d.items():
                sd[mk(k)] = v.to(dtype=self.dtype)

        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[LLaVA-Med] Weights: Missing={len(missing)}, Unexpected={len(unexpected)}")

        # Step 3: Resize embedding for image token
        model.resize_token_embeddings(config["vocab_size"] + 1)

        model.eval()
        self.model = model
        self.processor = None
        print(f"[LLaVA-Med] Ready! Params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    def _load_generic(self):
        """通用模型加载"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        self.model.eval()

    def _load_hulumed(self):
        """加载Hulu-Med-14B模型 (基于Qwen3的医学VLM)"""
        from transformers import AutoProcessor, AutoModelForCausalLM

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        self.model.eval()

    def extract_visual_features(self, image: Image.Image) -> torch.Tensor:
        """
        提取图像的视觉特征 (用于域偏移估计和memory bank检索)

        Args:
            image: PIL Image

        Returns:
            视觉特征张量 (1, hidden_dim)
        """
        if self.model_type == "phi3v":
            return self._extract_phi3v_features(image)
        elif self.model_type == "llava":
            return self._extract_llava_features(image)
        elif self.model_type == "hulumed":
            return self._extract_hulumed_features(image)
        else:
            return self._extract_generic_features(image)

    def _extract_phi3v_features(self, image: Image.Image) -> torch.Tensor:
        """提取Phi-3.5V的视觉特征"""
        # 使用processor处理图像获取pixel_values
        messages = [
            {"role": "user", "content": "<|image_1|>\nDescribe this medical image briefly."}
        ]
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(prompt, [image], return_tensors="pt")

        # 将输入移到正确的设备
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            # 获取图像嵌入
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                # 通过前向传播获取hidden states
                outputs = self.model(**inputs, output_hidden_states=True)
                # 使用最后一层hidden state的平均池化作为视觉特征
                last_hidden = outputs.hidden_states[-1]
                # 找到图像token的位置
                vision_mask = inputs.get("image_attention_mask",
                                         torch.ones_like(inputs["input_ids"], dtype=torch.bool))
                features = last_hidden.mean(dim=1)  # 简单平均池化
            else:
                outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]
                features = last_hidden.mean(dim=1)

        return features

    def _extract_llava_features(self, image: Image.Image) -> torch.Tensor:
        """提取LLaVA的视觉特征"""
        # 获取vision tower
        if hasattr(self.model, 'vision_tower'):
            vision_tower = self.model.vision_tower
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_tower'):
            vision_tower = self.model.model.vision_tower
        else:
            return self._extract_generic_features(image)

        # 获取image processor
        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"]
        elif hasattr(self, 'image_processor') and self.image_processor is not None:
            inputs = self.image_processor(image, return_tensors="pt")
            pixel_values = inputs["pixel_values"]
        else:
            # CLIP-style manual preprocessing
            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize((336, 336)),
                T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711]),
            ])
            pixel_values = transform(image).unsqueeze(0)

        device = next(vision_tower.parameters()).device
        dtype = next(vision_tower.parameters()).dtype
        pixel_values = pixel_values.to(device, dtype=dtype)

        with torch.no_grad():
            outputs = vision_tower(pixel_values=pixel_values)
            if hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state[:, 0, :]  # CLS token
            elif isinstance(outputs, torch.Tensor):
                features = outputs[:, 0, :]
            else:
                features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0][:, 0, :]

        return features

    def _extract_generic_features(self, image: Image.Image) -> torch.Tensor:
        """通用视觉特征提取 (基于模型hidden states)"""
        messages = [{"role": "user", "content": "Describe this image."}]
        prompt = str(messages)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

    def _extract_hulumed_features(self, image: Image.Image) -> torch.Tensor:
        """提取Hulu-Med的视觉特征"""
        device = next(self.model.parameters()).device

        # 统一resize到256x256，避免大图像patch数超过token限制
        image = image.resize((256, 256), Image.BILINEAR)

        # 分步处理: 先图像后文本
        image_inputs = self.processor.process_images([image])
        
        # 转换numpy为tensor, 确保dtype一致
        import numpy as np
        for k, v in image_inputs.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                image_inputs[k] = v.to(dtype=self.dtype)
            elif isinstance(v, torch.Tensor):
                image_inputs[k] = v
        
        messages = [{"role": "user", "content": "<image>\nDescribe this medical image briefly."}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text_inputs = self.processor.process_text(text, image_inputs, return_tensors="pt", truncation=True, max_length=512)
        
        inputs = {**text_inputs, **image_inputs}

        if hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        else:
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            features = last_hidden.mean(dim=1)

        return features

    def generate(self, image: Image.Image, prompt: str,
                 max_new_tokens: int = 256, temperature: float = 0.1,
                 system_prompt: Optional[str] = None) -> Tuple[str, List[float]]:
        """
        基于图像和prompt生成回答

        Args:
            image: PIL Image
            prompt: 文本prompt/问题
            max_new_tokens: 最大生成token数
            temperature: 生成温度
            system_prompt: 系统提示

        Returns:
            (generated_text, token_confidences)
        """
        if self.model_type == "phi3v":
            return self._generate_phi3v(image, prompt, max_new_tokens,
                                         temperature, system_prompt)
        elif self.model_type == "llava":
            return self._generate_llava(image, prompt, max_new_tokens,
                                         temperature, system_prompt)
        elif self.model_type == "hulumed":
            return self._generate_hulumed(image, prompt, max_new_tokens,
                                           temperature, system_prompt)
        else:
            return self._generate_generic(image, prompt, max_new_tokens,
                                            temperature, system_prompt)

    def _generate_phi3v(self, image: Image.Image, prompt: str,
                         max_new_tokens: int, temperature: float,
                         system_prompt: Optional[str] = None) -> Tuple[str, List[float]]:
        """Phi-3.5V推理"""
        if system_prompt is None:
            system_prompt = (
                "You are a medical imaging expert. Analyze the medical image carefully "
                "and provide accurate, evidence-based answers. Only state findings that "
                "are clearly supported by the image. If uncertain, express uncertainty."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<|image_1|>\n{prompt}"}
        ]

        prompt_text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(prompt_text, [image], return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        generation_args = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else None,
            "do_sample": temperature > 0,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_args)

        # 解码生成文本
        gen_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        text = self.processor.decode(gen_ids, skip_special_tokens=True).strip()

        # 计算token置信度
        confidences = []
        if hasattr(outputs, 'scores') and outputs.scores:
            for i, score in enumerate(outputs.scores):
                if i < len(gen_ids):
                    probs = torch.softmax(score[0], dim=-1)
                    conf = probs[gen_ids[i]].item()
                    confidences.append(conf)

        return text, confidences

    def _generate_llava(self, image: Image.Image, prompt: str,
                         max_new_tokens: int, temperature: float,
                         system_prompt: Optional[str] = None) -> Tuple[str, List[float]]:
        """LLaVA推理 (支持标准LLaVA和LLaVA-Med)"""
        # 找到模型第一个参数的设备 (对于多GPU, 可能是cpu或accelerate的meta)
        try:
            first_param = next(self.model.parameters())
            device = first_param.device
        except StopIteration:
            device = torch.device("cuda:0")

        if self.processor is not None:
            # 标准LLaVA
            full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
            inputs = self.processor(full_prompt, image, return_tensors="pt")
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
        else:
            # LLaVA-Med: 使用正确的image token处理
            image_token_id = getattr(self, 'image_token_id', None)
            num_patches = getattr(self, 'num_image_patches', 576)

            if image_token_id is not None:
                # 构建input_ids: [BOS] USER: [IMG*N] \nprompt\nASSISTANT:
                text_before = self.tokenizer.encode('USER: ', add_special_tokens=True, return_tensors='pt')
                text_after = self.tokenizer.encode(f'\n{prompt}\nASSISTANT:', add_special_tokens=False, return_tensors='pt')
                img_tokens = torch.full((1, num_patches), image_token_id, dtype=text_before.dtype)
                input_ids = torch.cat([text_before, img_tokens, text_after], dim=1)
            else:
                full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
                input_ids = self.tokenizer.encode(full_prompt, return_tensors='pt')

            # 图像预处理 (CLIP-style)
            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize((336, 336)),
                T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711]),
            ])
            pixel_values = transform(image).unsqueeze(0).to(dtype=self.dtype)

            # 找到模型设备
            lm_device = next(p.device for n, p in self.model.named_parameters()
                           if 'language_model' in n)

            # ★ FIX: 生成 attention_mask 并设置 pad_token_id
            attention_mask = torch.ones_like(input_ids)
            inputs = {
                "input_ids": input_ids.to(lm_device),
                "attention_mask": attention_mask.to(lm_device),
                "pixel_values": pixel_values.to(lm_device),
            }
            # 设置 pad_token_id
            if self.tokenizer.pad_token_id is not None:
                self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            else:
                self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                return_dict_in_generate=True,
                output_scores=True,
            )

        gen_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        if self.processor is not None:
            text = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
        else:
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        confidences = []
        if hasattr(outputs, 'scores') and outputs.scores:
            for i, score in enumerate(outputs.scores):
                if i < len(gen_ids):
                    probs = torch.softmax(score[0], dim=-1)
                    conf = probs[gen_ids[i]].item()
                    confidences.append(conf)

        return text, confidences

    def _generate_hulumed(self, image: Image.Image, prompt: str,
                           max_new_tokens: int, temperature: float,
                           system_prompt: Optional[str] = None) -> Tuple[str, List[float]]:
        """Hulu-Med推理 (基于Qwen3-VL, 直接使用tokenizer+image_processor)"""
        device = next(self.model.parameters()).device

        if system_prompt is None:
            system_prompt = (
                "You are a medical imaging expert. Analyze the medical image carefully "
                "and provide accurate, evidence-based answers."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<image>\n{prompt}"}
        ]

        # Step 0: 统一resize到256x256，避免大图像patch数超过token限制
        image = image.resize((256, 256), Image.BILINEAR)

        # Step 1: 处理图像
        image_inputs = self.processor.process_images([image])
        
        # 将numpy array转为tensor, 并确保dtype一致
        for k, v in image_inputs.items():
            import numpy as np
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                image_inputs[k] = v.to(dtype=self.dtype)
            elif isinstance(v, torch.Tensor):
                image_inputs[k] = v
        
        # Step 2: 获取chat template文本
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Step 3: 处理文本 (手动调用process_text, 避免__call__的list问题)
        text_inputs = self.processor.process_text(text, image_inputs, return_tensors="pt", truncation=True, max_length=2048)
        
        # Step 4: 合并inputs
        inputs = {**text_inputs, **image_inputs}

        if hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        else:
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
            )

        # Hulu-Med使用inputs_embeds, generate只返回生成的token
        # 检查output长度是否包含输入
        input_len = inputs["input_ids"].shape[1] if isinstance(inputs, dict) else inputs.input_ids.shape[1]
        if outputs.shape[1] > input_len:
            gen_ids = outputs[0][input_len:]
        else:
            gen_ids = outputs[0]
        text_out = self.processor.decode(gen_ids, skip_special_tokens=True).strip()

        return text_out, []

    def _generate_generic(self, image: Image.Image, prompt: str,
                            max_new_tokens: int, temperature: float,
                            system_prompt: Optional[str] = None) -> Tuple[str, List[float]]:
        """通用文本模型推理 (无图像输入)"""
        full_prompt = f"Medical Question: {prompt}\nAnswer:"
        inputs = self.tokenizer(full_prompt, return_tensors="pt",
                                truncation=True, max_length=512)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                return_dict_in_generate=True,
                output_scores=True,
            )

        gen_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        confidences = []
        if hasattr(outputs, 'scores') and outputs.scores:
            for i, score in enumerate(outputs.scores):
                if i < len(gen_ids):
                    probs = torch.softmax(score[0], dim=-1)
                    conf = probs[gen_ids[i]].item()
                    confidences.append(conf)

        return text, confidences

    def generate_structured_findings(self, image: Image.Image,
                                      question: str,
                                      evidence: List[str] = None,
                                      max_findings: int = 8) -> List[Dict[str, str]]:
        """
        生成结构化findings (Stage 3)

        Returns:
            List of dicts with keys: anatomy, observation, presence, visual_evidence, confidence
        """
        system_prompt = (
            "You are a medical imaging expert. Analyze the image and provide structured findings. "
            "For each finding, specify:\n"
            "- anatomy: the anatomical structure\n"
            "- observation: what you observe\n"
            "- presence: present/absent/uncertain\n"
            "- visual_evidence: specific visual features supporting this finding\n"
            "- confidence: high/medium/low\n"
            "Only include findings clearly supported by the image. "
            "Output as JSON array."
        )

        evidence_text = ""
        if evidence:
            evidence_text = "\nReference evidence from similar cases:\n"
            for i, ev in enumerate(evidence[:5]):
                evidence_text += f"  {i+1}. {ev}\n"

        prompt = (
            f"Question: {question}\n"
            f"{evidence_text}\n"
            f"Provide up to {max_findings} structured medical findings in JSON format:\n"
            f'[{{"anatomy": "...", "observation": "...", "presence": "...", '
            f'"visual_evidence": "...", "confidence": "..."}}]'
        )

        text, confidences = self.generate(
            image, prompt, max_new_tokens=512, temperature=0.1,
            system_prompt=system_prompt
        )

        # 解析JSON
        findings = self._parse_findings_json(text)
        return findings

    def _parse_findings_json(self, text: str) -> List[Dict[str, str]]:
        """解析结构化findings JSON"""
        # 尝试提取JSON数组
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            try:
                findings = json.loads(json_match.group())
                if isinstance(findings, list):
                    return findings
            except json.JSONDecodeError:
                pass

        # 回退: 将文本按行解析为findings
        findings = []
        lines = text.strip().split('\n')
        current = {}
        for line in lines:
            line = line.strip()
            if not line:
                if current:
                    findings.append(current)
                    current = {}
                continue
            # 尝试解析key-value
            kv_match = re.match(r'[-*]?\s*(\w+):\s*(.+)', line)
            if kv_match:
                key = kv_match.group(1).lower().replace(' ', '_')
                value = kv_match.group(2).strip()
                current[key] = value

        if current:
            findings.append(current)

        return findings if findings else [{"observation": text, "confidence": "low"}]
