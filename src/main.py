import argparse
from dotenv import load_dotenv
from provider.chat_gpt import ChatGPTProvider
from provider.deepseek import DeepSeekProvider
from adapters.logger_strdin import LoggerStdin
from adapters.logger_adapter import LoggerAdapter
from rag.rag_orchestrator import RAGOrchestrator
from rag.retrieve import FaissRetriever
from rag.re_ranker import CrossEncoderReranker
from rag.prompts import build_synthesis_prompt

load_dotenv()

class Main:
    def __init__(self, logger: LoggerAdapter) -> None:
        self._logger = logger

    def run(self):
        provider = ""
        message = ""
        system_prompt = "You're a helpfull asistant"

        parser = argparse.ArgumentParser(
            description="Call DeepSeek provider with a message"
        )
        parser.add_argument(
            "-m", "--message", help="Message to send to the provider", default=None
        )
        parser.add_argument(
            "-p", "--provider", help="Select the provider to send the message", default=None
        )
        parser.add_argument(
            "-r", "--rag", help="Load FAISS data given an input", default=False, type=bool
        )
        args = parser.parse_args()

        if args.message:
            message = args.message
            
        if args.rag is not False:
            self._logger.info("running rag...")
            faiss_index_path = "data/processed/index.faiss"
            chunks_path = "data/processed/chunks.parquet"
            mapping_path = "data/processed/mapping.parquet"

            retriever = FaissRetriever(faiss_index_path, chunks_path, mapping_path, self._logger)
            reranker = CrossEncoderReranker(self._logger)
            rag_orchestrator = RAGOrchestrator(retriever,self._logger, reranker)

            query = message
            self._logger.info(f"Running RAG for query: {query}")
            result = rag_orchestrator.run(query, k_retrieve=5, rerank_top_n=5, do_rewrite=True)
            self._logger.info("RAG result:")
            self._logger.info(f"{result}")
            system_prompt =  build_synthesis_prompt(result['query'], result['hints']) # type: ignore

        if args.provider:
            provider: str = args.provider
            provider = provider.lower()

        if provider == "deepseek":
            deepseek_logger = LoggerStdin("deepseek_logger", "logs/deepseek.log")
            deepseek_provider = DeepSeekProvider(deepseek_logger)
            response = deepseek_provider.chat(system_prompt, message)
            self._logger.info("\nDeepSeek response:")
            self._logger.info(response)
        elif provider == "chatgpt":
            chatgpt_logger = LoggerStdin("chatgpt_logger", "logs/chatgpt.log")
            chatgpt_provider = ChatGPTProvider(chatgpt_logger)
            response = chatgpt_provider.chat(system_prompt,message)
            self._logger.info("\nChatGpt response:")
            self._logger.info(response)
        else:
            self._logger.warning("Invalid provider, please select deepseek or chatgpt")
            return


if __name__ == "__main__":
    main_logger = LoggerStdin("main", "logs/main.log")
    main = Main(main_logger)
    main.run()
