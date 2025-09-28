from dotenv import load_dotenv

import argparse

from provider.chat_gpt import ChatGPTProvider
from provider.deepseek import DeepSeekProvider

load_dotenv()


def main():
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
            print("No message provided, exiting.")
            return

    if args.provider:
        provider: str = args.provider
        provider = provider.lower()

    if provider == "deepseek":
        # provider = ChatGPTProvider()
        deepseek_provider = DeepSeekProvider()
        response = deepseek_provider.chat(message)  # type: ignore
        print("\nDeepSeek response:")
        print(response)  # type: ignore
    elif provider == "chatgpt":
        chatgpt_provider = ChatGPTProvider()
        response = chatgpt_provider.chat(message)
        print("\nChatGpt response:")
        print(response)
    else:
        print("Invalid provider, please select deepseek or chatgpt")
        return


if __name__ == "__main__":
    main()
