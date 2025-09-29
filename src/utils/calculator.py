import json
import ast
from typing import Any, Dict


class Calculator:
    @staticmethod
    def _ensure_dict(raw: Any) -> Dict:
        """Try to convert raw input (str or dict-like) into a dict.

        Accepts a dict, a JSON string, or a Python literal string. Returns an
        empty dict if parsing fails.
        """
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            # Try JSON first
            try:
                return json.loads(raw)
            except Exception:
                pass
            # Fallback to python literal parsing
            try:
                return ast.literal_eval(raw)
            except Exception:
                return {}
        return {}

    @staticmethod
    def calculate_total_tost(
        raw: Any,
        prompt_cost_per_1k_usd: float = 0.002,
        completion_cost_per_1k_usd: float = 0.002,
    ) -> float:
        """Calculate total cost (USD) from a model response object or string.

        Parameters
        - raw: dict or str containing the response. Expected to include a
          'usage' object with keys 'prompt_tokens' and 'completion_tokens'.
        - prompt_cost_per_1k_usd: cost in USD per 1000 prompt tokens (default 0.002 USD)
        - completion_cost_per_1k_usd: cost in USD per 1000 completion tokens (default 0.002 USD)

        Returns
        - total cost in USD (float, rounded to 2 decimals)
        """
        obj = Calculator._ensure_dict(raw)
        usage = obj.get("usage") if isinstance(obj, dict) else None

        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")

        # fall back to top-level keys if not present under usage
        if prompt_tokens is None:
            prompt_tokens = obj.get("prompt_tokens")
        if completion_tokens is None:
            completion_tokens = obj.get("completion_tokens")
        if total_tokens is None:
            total_tokens = obj.get("total_tokens")

        # If we only have total_tokens, assume a 50/50 split
        if prompt_tokens is None and completion_tokens is None and total_tokens is not None:
            prompt_tokens = completion_tokens = int(round(total_tokens / 2))

        # Ensure numeric values
        try:
            prompt_tokens = int(prompt_tokens) if prompt_tokens is not None else 0
        except Exception:
            prompt_tokens = 0
        try:
            completion_tokens = int(completion_tokens) if completion_tokens is not None else 0
        except Exception:
            completion_tokens = 0

        # per-token prices (USD)
        prompt_per_token = prompt_cost_per_1k_usd / 1000.0
        completion_per_token = completion_cost_per_1k_usd / 1000.0

        total_usd = prompt_tokens * prompt_per_token + completion_tokens * completion_per_token
        return round(float(total_usd), 6)