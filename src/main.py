import argparse
from dotenv import load_dotenv
from provider.chat_gpt import ChatGPTProvider
from provider.deepseek import DeepSeekProvider
from adapters.logger_strdin import LoggerStdin
from adapters.logger_adapter import LoggerAdapter

load_dotenv()

class Main:
    def __init__(self, logger: LoggerAdapter) -> None:
        self._logger = logger

    def run(self):
        provider = ""
        message = ""

        parser = argparse.ArgumentParser(
            description="Call DeepSeek provider with a message"
        )
        parser.add_argument(
            "-m", "--message", help="Message to send to the provider", default=None
        )
        parser.add_argument(
            "-p", "--provider", help="Select the provider to send the message", default=None
        )
        args = parser.parse_args()

        if args.message:
            message = args.message
        else:
            message = input("Enter message for DeepSeek: ").strip()
            if not message:
                self._logger.warning("No message provided, exiting.")
                return

        if args.provider:
            provider: str = args.provider
            provider = provider.lower()

        if provider == "deepseek":
            deepseek_logger = LoggerStdin("deepseek_logger", "logs/deepseek.log")
            deepseek_provider = DeepSeekProvider(deepseek_logger)
            response = deepseek_provider.chat(message)  # type: ignore
            self._logger.info("\nDeepSeek response:")
            self._logger.info(response)  # type: ignore
        elif provider == "chatgpt":
            chatgpt_logger = LoggerStdin("chatgpt_logger", "logs/chatgpt.log")
            chatgpt_provider = ChatGPTProvider(chatgpt_logger)
            response = chatgpt_provider.chat(message)
            self._logger.info("\nChatGpt response:")
            self._logger.info(response)
        else:
            self._logger.warning("Invalid provider, please select deepseek or chatgpt")
            return


if __name__ == "__main__":
    main_logger = LoggerStdin("main", "logs/main.log")
    main = Main(main_logger)
    main.run()
