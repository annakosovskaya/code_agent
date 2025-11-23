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
        temperature: float = 0.2,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
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
            no_repeat_ngram_size=6,
            do_sample=True if temperature and temperature > 0 else False,
        )

    @torch.inference_mode()
    def invoke(self, messages: List[ChatMessage]) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Ensure inputs are on the same device as the model (handles mps/cpu/cuda and accelerate offload)
        model_device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        # Heuristic: adjust budget when we expect a full code payload
        recent_text = " ".join(m["content"] for m in messages[-3:] if isinstance(m.get("content"), str))
        wants_action_input = ("Action Input:" in recent_text) or ("ONLY the corrected function" in recent_text)
        gen_kwargs = {}
        if wants_action_input:
            gen_kwargs["max_new_tokens"] = max(self.generation_config.max_new_tokens or 0, 2048)
            gen_kwargs["max_time"] = 360
            gen_kwargs["min_new_tokens"] = 128
        else:
            gen_kwargs["max_time"] = 120
            gen_kwargs["min_new_tokens"] = 64
        output_ids = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
            **gen_kwargs,
        )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


