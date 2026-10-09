from typing import Any, cast
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenLLM:
    """Wrapper around Qwen/Qwen3-0.6B for text generation."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Load tokenizer and model weights into memory once."""
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
        )
        self._model = cast(Any, model).to(self._device)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate response given system instructions and user prompt."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self._tokenizer([text],
                                       return_tensors="pt").to(self._device)
        input_len = model_inputs.input_ids.shape[1]

        with torch.no_grad():
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        answer_tokens = generated_ids[0][input_len:]
        decoded = self._tokenizer.decode(
            answer_tokens, skip_special_tokens=True
        )
        if isinstance(decoded, list):
            decoded = " ".join(decoded)
        return str(decoded).strip()
