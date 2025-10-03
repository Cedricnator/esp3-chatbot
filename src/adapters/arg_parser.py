from adapters.arg_adapter import ArgAdapter
import argparse

class ArgParser(ArgAdapter):
  def parse(self):
    parser = argparse.ArgumentParser(
      description="Call DeepSeek provider with a message"
    )
    parser.add_argument(
      "-m", "--message", help="Message to send to the provider", default=None, type=str
    )
    parser.add_argument(
      "-p", "--provider", help="Select the provider to send the message", default=None, type=str
    )
    parser.add_argument(
      "-r", "--rag", help="Load FAISS data given an input", default=False, type=bool
    )
    parser.add_argument(
      "-k", "--topk", help="Indicate wich topk use to the rag", default=5, type=int
    )
    parser.add_argument(
      "-e", "--evaluation", help="Evaluate the application", default=False, type=bool
    )
    return parser.parse_args()