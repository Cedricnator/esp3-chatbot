from abc import ABC, abstractmethod
from typing import Any

class BaseProvider(ABC):
    @abstractmethod
    def chat(self, message: str, **kwargs: Any) -> Any:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        return ""