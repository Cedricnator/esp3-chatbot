from dotenv import load_dotenv
from provider.chat_gpt import ChatGPTProvider
from provider.deepseek import DeepSeekProvider
from adapters.logger_strdin import LoggerStdin
from adapters.logger_adapter import LoggerAdapter
from rag.rag_orchestrator import RAGOrchestrator
from rag.retrieve import FaissRetriever
from rag.re_ranker import CrossEncoderReranker
from rag.prompts import build_synthesis_prompt
from adapters.arg_adapter import ArgAdapter
from adapters.arg_parser import ArgParser
from eval.evaluate import Evaluator, GoldSet, EvaluatorAgent
from utils.checkpointer import CheckpointerRegister
load_dotenv()

class Main:
    def __init__(self, logger: LoggerAdapter, arg: ArgAdapter) -> None:
        self._logger = logger
        self.parser = arg

    def run(self):
        checkpoint = CheckpointerRegister()
        provider = ""
        message = ""
        system_prompt = "You're a helpfull asistant"
        args = self.parser.parse()
        faiss_index_path = "data/processed/index.faiss"
        chunks_path = "data/processed/chunks.parquet"
        mapping_path = "data/processed/mapping.parquet"
        retriever = FaissRetriever(faiss_index_path, chunks_path, mapping_path, self._logger)
        reranker = CrossEncoderReranker(self._logger)
        rag_orchestrator = RAGOrchestrator(retriever,self._logger, reranker)
        checkpoint.setCheckpoint("setup")

        if args.message:
            message = args.message

        if args.rag is not False:
            self._logger.info("running rag...")
            query = message
            self._logger.info(f"Running RAG for query: {query}")
            result = rag_orchestrator.run(query, k_retrieve=5, rerank_top_n=5, do_rewrite=True)
            self._logger.info("RAG result:")
            self._logger.info(f"{result}")
            system_prompt =  build_synthesis_prompt(result['query'], result['hints']) # type: ignore
            checkpoint.setCheckpoint("rag")
        
        if args.evaluation is not False:
            self._logger.info("runnnig evaluation...")
            gold_set =  GoldSet("./gold_set.json", self._logger)
            deepseek_logger = LoggerStdin("deepseek_logger", "logs/deepseek.log")
            chatgpt_logger = LoggerStdin("chatgpt_logger", "logs/chatgpt.log")
            # d_provider = DeepSeekProvider(deepseek_logger, checkpoint)
            evaluator_agent = EvaluatorAgent(self._logger, gold_set)
            rag_orchestrator = RAGOrchestrator(retriever, self._logger, reranker) # type: ignore

            # Evaluate DeepSeek
            # deepseek_evaluator = Evaluator(
            #     gold_set,                
            #     d_provider,
            #     evaluator_agent,
            #     rag_orchestrator,
            # )
            # deepseek_evaluator.run()

            # Evaluate ChatGPT
            c_provider = ChatGPTProvider(chatgpt_logger, checkpoint)
            chatgpt_evaluator = Evaluator(
                gold_set,                
                c_provider,
                evaluator_agent,
                rag_orchestrator,
            )
            chatgpt_evaluator.run()

            checkpoint.setCheckpoint("evaluation")
            return

        if args.provider:
            provider: str = args.provider
            provider = provider.lower()

        if provider == "deepseek":
            deepseek_logger = LoggerStdin("deepseek_logger", "logs/deepseek.log")
            deepseek_provider = DeepSeekProvider(deepseek_logger, checkpoint)
            response = deepseek_provider.chat(system_prompt, message)
            self._logger.info("\nDeepSeek response:")
            self._logger.info(response)
            checkpoint.setCheckpoint("deepseek-provider")
            checkpoint.save()
        elif provider == "chatgpt":
            chatgpt_logger = LoggerStdin("chatgpt_logger", "logs/chatgpt.log")
            chatgpt_provider = ChatGPTProvider(chatgpt_logger, checkpoint)
            response = chatgpt_provider.chat(system_prompt,message)
            self._logger.info("\nChatGpt response:")
            self._logger.info(response)
            checkpoint.setCheckpoint("chatgpt-provider")
            checkpoint.save()
        else:
            self._logger.warning("Invalid provider, please select deepseek or chatgpt")
            return

if __name__ == "__main__":
    main_logger = LoggerStdin("main", "logs/main.log")
    argsparser = ArgParser()
    main = Main(main_logger, argsparser)
    main.run()
