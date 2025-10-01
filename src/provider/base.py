from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseProvider(ABC):
    @abstractmethod
    def chat(self, system_prompt: str,  message: str, **kwargs: Any) -> Any:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        return ""

    @property
    def last_used_model(self) -> Optional[str]:
        """Optional hint about the last model string used by the provider."""
        return None