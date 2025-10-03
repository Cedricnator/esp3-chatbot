import json
import os
from openai import OpenAI
from typing import Any, Optional
from provider.base import BaseProvider
from adapters.logger_adapter import LoggerAdapter
from utils.checkpointer import CheckpointerRegister

class DeepSeekProvider(BaseProvider):
    def __init__(
        self,
        logger: LoggerAdapter,
    checkpointerRegister: Optional[CheckpointerRegister] = None,
        default_model: str = "deepseek/deepseek-chat-v3.1:free",
    ) -> None:
        self._logger = logger
        self._provider = "DeepSeek"
        self._checkpointerRegister = checkpointerRegister
        self._default_model = default_model
        self._last_used_model: Optional[str] = None

    @property
    def name(self) -> str:
        return self._provider

    @property
    def last_used_model(self) -> Optional[str]:
        return self._last_used_model
    
    def chat(self, system_prompt: str, message: str, **kwargs: Any):
        """
        Send a user message to OpenAI and return the assistant reply.
        Accepts optional kwargs: model, temperature, max_tokens, and any
        additional fields will be ignored for now.
        If DEEPSEEK_API_KEY is not set, returns a simple local fallback response.
        """
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            # fallback for offline / test environments
            self._last_used_model = self._default_model
            response = f"[DeepSeek fallback - no API key found] Received message: {message}"
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
            self._last_used_model = model
            self._logger.info(
                f"DeepSeekProvider executed with model '{model}'"
            )
            if self._checkpointerRegister is not None:
                self._checkpointerRegister.addCost(json.dumps(raw))

            # Primary extraction
            try:
                choices = getattr(resp, "choices", None)
                if not choices:
                    # minimal fallback: try mapping access if attribute missing
                    choices = resp.get("choices") if isinstance(resp, dict) else None

                if choices and len(choices) > 0:
                    first = choices[0]
                    # prefer .message.content
                    msg = getattr(first, "message", None)
                    if msg is not None:
                        content = getattr(msg, "content", None)
                        if content:
                            return content.strip()

                    # next prefer .text on the choice
                    text = getattr(first, "text", None)
                    if text:
                        return text.strip()

                    # last resort: if first is a dict, try nested lookup
                    if isinstance(first, dict):
                        text = first.get("message", {}).get("content") or first.get("text")
                        if text:
                            return text.strip()

            except Exception as e:
                # If extraction fails, include exception in the returned message
                return f"[DeepSeek error] extraction failure: {e}"

            return "[DeepSeek error] unexpected response structure or empty reply"
        except Exception as e:
            self._last_used_model = model
            return f"[DeepSeek error] {str(e)}"