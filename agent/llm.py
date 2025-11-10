from __future__ import annotations

import os
from typing import List, TypedDict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-Coder-7B-Instruct"


class ChatMessage(TypedDict):
    role: str
    content: str


class LLMChat:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # reduce tokenizer parallelism warnings and potential deadlocks
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        self.model_name: Optional[str] = None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=hf_token,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype="auto",
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                token=hf_token,
            )
            # Disable sliding-window attention if present and pick stable attn impl for CPU/sdpa
            cfg = self.model.config
            if hasattr(cfg, "use_sliding_window"):
                cfg.use_sliding_window = False
            if hasattr(cfg, "sliding_window"):
                cfg.sliding_window = None
            if hasattr(cfg, "attn_implementation"):
                # For CPU/macOS prefer eager to avoid sdpa+SWA warning
                if self.device == "cpu":
                    cfg.attn_implementation = "eager"
            self.model_name = model_name
        except OSError as e:
            raise OSError(
                f"Failed to load model '{model_name}'. "
                f"Set MODEL_NAME to a valid HF model id and ensure you have access "
                f"(huggingface-cli login or HUGGING_FACE_HUB_TOKEN). Original error: {e}"
            )
        # model_name is set if no exception was raised above

        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True if temperature and temperature > 0 else False,
        )

    @torch.inference_mode()
    def invoke(self, messages: List[ChatMessage]) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Ensure inputs are on the same device as the model (handles mps/cpu/cuda and accelerate offload)
        model_device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        output_ids = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            generation_config=self.generation_config,
        )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


