import os
from openai import OpenAI
from typing import Any

from provider.base import BaseProvider


class ChatGPTProvider(BaseProvider):
    def __init__(self) -> None:
        self._provider = "ChatGPT"

    @property
    def name(self):
        return self._provider

    def chat(self, message: str, **kwargs: Any) -> Any:
        """
        Send a user message to OpenAI and return the assistant reply.
        Accepts optional kwargs: model, temperature, max_tokens.
        If OPENAI_API_KEY is not set, returns a simple local fallback response.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # fallback for offline / test environments
            response = f"[ChatGPT fallback - no API key found] Received message: {message}"
            if kwargs:
                response += f" with additional parameters: {kwargs}"
            return response

        client = OpenAI(api_key=api_key)

        model = kwargs.pop("model", "gpt-5-nano")
        temperature = kwargs.pop("temperature", 0.2)
        max_tokens = kwargs.pop("max_tokens", 256)

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are ChatGPT, a helpful assistant."},
                    {"role": "user", "content": message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Print raw response for debugging. If the SDK object has a to_dict()
            # method, use that for a cleaner, serializable output; otherwise use repr().
            try:
                raw = resp.to_dict() if hasattr(resp, "to_dict") else repr(resp)
            except Exception:
                raw = repr(resp)
            print("RAW CHATGPT RESPONSE:", raw)

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
            return f"[ChatGPT error] {str(e)}"