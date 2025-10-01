import os
import json
from openai import OpenAI
from typing import Any, Optional
from adapters.logger_adapter import LoggerAdapter
from provider.base import BaseProvider
from utils.checkpointer import CheckpointerRegister


class ChatGPTProvider(BaseProvider):
    def __init__(
        self,
        logger: LoggerAdapter,
        checkpointerRegister: Optional[CheckpointerRegister] = None,
        default_model: str = "openai/gpt-5-nano",
    ) -> None:
        self._logger = logger
        self._provider = "ChatGPT"
        self._checkpointerRegister = checkpointerRegister
        self._default_model = default_model
        self._last_used_model: Optional[str] = None

    @property
    def name(self):
        return self._provider

    @property
    def last_used_model(self) -> Optional[str]:
        return self._last_used_model

    def chat(self, system_prompt: str, message: str, **kwargs: Any) -> Any:
        """
        Send a user message to OpenAI and return the assistant reply.
        Accepts optional kwargs: model, temperature, max_tokens.
        If OPENAI_API_KEY is not set, returns a simple local fallback response.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # fallback for offline / test environments
            self._last_used_model = self._default_model
            response = f"[ChatGPT fallback - no API key found] Received message: {message}"
            if kwargs:
                response += f" with additional parameters: {kwargs}"
            return response

        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

        model = kwargs.pop("model", self._default_model)
        temperature = kwargs.pop("temperature", 0.2)
        max_tokens = kwargs.pop("max_tokens", None)

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
                temperature=temperature,
                **({"max_tokens": max_tokens} if max_tokens is not None else {}),
            )

            # Print raw response for debugging. If the SDK object has a to_dict()
            # method, use that for a cleaner, serializable output; otherwise use repr().
            try:
                raw = resp.to_dict() if hasattr(resp, "to_dict") else repr(resp)
            except Exception:
                raw = repr(resp)
            self._logger.info(f"RAW CHATGPT RESPONSE: {raw}")
            self._last_used_model = model
            self._logger.info(
                f"ChatGPTProvider executed with model '{model}'"
            )
            if self._checkpointerRegister is not None:
                self._checkpointerRegister.addCost(json.dumps(raw))

            # Primary extraction: attribute-style
            try:
                choices = getattr(resp, "choices", None)
                if not choices:
                    choices = resp.get("choices") if isinstance(resp, dict) else None # type: ignore

                if choices and len(choices) > 0: # type: ignore
                    first = choices[0] # type: ignore
                    msg = getattr(first, "message", None) # type: ignore
                    if msg is not None:
                        content = getattr(msg, "content", None)
                        if content:
                            return content.strip()

                    text = getattr(first, "text", None) # type: ignore
                    if text:
                        return text.strip()

                    if isinstance(first, dict):
                        text = first.get("message", {}).get("content") or first.get("text") # type: ignore
                        if text:
                            return text.strip() # type: ignore

            except Exception as e:
                return f"[ChatGPT error] extraction failure: {e}"

            return "[ChatGPT error] unexpected response structure or empty reply"
        except Exception as e:
            self._last_used_model = model
            return f"[ChatGPT error] {str(e)}"