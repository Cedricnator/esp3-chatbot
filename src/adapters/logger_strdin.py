from adapters.logger_adapter import LoggerAdapter
import logging

class ColorFormatter(logging.Formatter):
	"""Formatter que aplica colores ANSI según el nivel de log."""

	RESET = "\033[0m"
	COLORS = {
		logging.DEBUG: "\033[36m",    # Cyan
		logging.INFO: "\033[32m",     # Verde
		logging.WARNING: "\033[33m",  # Amarillo
		logging.ERROR: "\033[31m",    # Rojo
		logging.CRITICAL: "\033[41m", # Fondo rojo
	}

	def format(self, record: logging.LogRecord) -> str:
		original_levelname = record.levelname
		color = self.COLORS.get(record.levelno)
		if color:
			record.levelname = f"{color}{original_levelname}{self.RESET}"
		try:
			return super().format(record)
		finally:
			record.levelname = original_levelname

class LoggerStdin(LoggerAdapter):
	"""Clase para gestionar el logging de la aplicación."""

	def __init__(self, name: str, filename: str) -> None:
		self.logger = logging.getLogger(name)
		self.logger.setLevel(logging.DEBUG)
		self._verify_handlers(filename)

	def _verify_handlers(self, filename: str) -> None:
		"""Verifica si el logger ya tiene handlers para evitar duplicados."""
		if not self.logger.hasHandlers():
			fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
			datefmt = "%Y-%m-%d %H:%M:%S"
			formatter = logging.Formatter(fmt, datefmt=datefmt)
			color_formatter = ColorFormatter(fmt, datefmt=datefmt)

			# Handler Archivo
			file_handler = self._file_handler(filename, formatter)

			# Handler Consola
			console_handler = logging.StreamHandler()
			console_handler.setLevel(logging.DEBUG)
			console_handler.setFormatter(color_formatter)

			# Agregar Handlers al Logger
			self.logger.addHandler(file_handler)
			self.logger.addHandler(console_handler)

	def _file_handler(
		self, filename: str, formatter: logging.Formatter
	) -> logging.FileHandler:
		"""Crea un handler para escribir logs en un archivo."""
		file_handler = logging.FileHandler(filename, encoding="utf-8")
		file_handler.setLevel(logging.DEBUG)
		file_handler.setFormatter(formatter)
		return file_handler

	def info(self, message: str) -> None:
		"""Registra un mensaje de nivel INFO."""
		self.logger.info(message)

	def warning(self, message: str) -> None:
		"""Registra un mensaje de nivel WARNING."""
		self.logger.warning(message)

	def error(self, message: str) -> None:
		"""Registra un mensaje de nivel ERROR."""
		self.logger.error(message)

	def debug(self, message: str) -> None:
		"""Registra un mensaje de nivel DEBUG."""
		self.logger.debug(message)
