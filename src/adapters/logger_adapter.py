from abc import ABC, abstractmethod

class LoggerAdapter(ABC):
	@abstractmethod
	def info(self, message: str) -> None:
		"""Registra un mensaje de nivel INFO."""
		pass

	@abstractmethod
	def warning(self, message: str) -> None:
		"""Registra un mensaje de nivel WARNING."""
		pass

	@abstractmethod
	def error(self, message: str) -> None:
		"""Registra un mensaje de nivel ERROR."""
		pass

	@abstractmethod
	def debug(self, message: str) -> None:
		"""Registra un mensaje de nivel DEBUG."""
		pass
