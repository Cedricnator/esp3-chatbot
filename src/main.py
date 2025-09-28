from dotenv import load_dotenv

import argparse

from provider.chat_gpt import ChatGPTProvider

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Call DeepSeek provider with a message")
    parser.add_argument('-m', '--message', help='Message to send to DeepSeek', default=None)
    args = parser.parse_args()

    if args.message:
        message = args.message
    else:
        message = input('Enter message for DeepSeek: ').strip()
        if not message:
            print('No message provided, exiting.')
            return

    provider = ChatGPTProvider()
    response = provider.chat(message)
    print('\nChatGPT response:')
    print(response)

if __name__ == '__main__':
    main()