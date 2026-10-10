"""Qwen LLM wrapper for grounded answer generation."""

from typing import Any, cast
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen:
    """Wrapper around Qwen/Qwen3-0.6B for context-augmented text generation."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Initialize the tokenizer and model weights on the target device."""
        self.model_name = model_name

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.dtype = (
            torch.float16 if self.device in ["cuda", "mps"] else torch.float32
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=self.dtype,
        )
        self.model: Any = cast(Any, model).to(self.device)
        self.model.eval()

    def build_rag_messages(
        self, question: str, chunks: list[str]
    ) -> list[dict[str, str]]:
        """Construct system and user messages containing retrieved context."""
        if isinstance(chunks, str):
            chunks = [chunks]

        docs = "\n\n".join(
            f'<document id="{i + 1}">\n{chunk.strip()}\n</document>'
            for i, chunk in enumerate(chunks)
        )

        user_content = f"""<context>
{docs}
</context>

Question: {question}"""

        system_instruction = (
            "You are a helpful assistant. Answer the user's question STRICTLY "
            "based on the provided context in the <context> tag. "
            "Do not use external knowledge or invent facts. "
            'If the answer cannot be found in the context, reply: '
            '"I do not have enough information to answer this question."'
        )

        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content},
        ]

    def generate(
        self,
        question: str,
        chunks: list[str] | None = None,
        max_new_tokens: int = 128,
    ) -> str:
        """Generate a natural language answer grounded in context."""
        if chunks:
            messages = self.build_rag_messages(question, chunks)
        else:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer directly.",
                },
                {"role": "user", "content": question},
            ]

        try:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        text_prompt = str(raw_prompt)
        if "<think>" not in text_prompt:
            text_prompt += "<think>\n</think>\n"

        inputs = self.tokenizer(
            text_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
            )

        input_length = inputs.input_ids.shape[1]
        response_tokens = outputs[0][input_length:]

        decoded = self.tokenizer.decode(
            response_tokens, skip_special_tokens=True
        )
        if isinstance(decoded, list):
            decoded = " ".join(decoded)

        if (
            self.device == "mps"
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "empty_cache")
        ):
            torch.mps.empty_cache()

        return str(decoded).strip()
