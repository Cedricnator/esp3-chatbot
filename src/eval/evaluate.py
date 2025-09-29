from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from adapters.logger_adapter import LoggerAdapter
from adapters.logger_strdin import LoggerStdin
from openai import OpenAI
from provider.deepseek import DeepSeekProvider
from rag.prompts import build_synthesis_prompt
from rag.rag_orchestrator import RAGOrchestrator
from rag.re_ranker import CrossEncoderReranker
from rag.retrieve import FaissRetriever

class GoldSet:
    def __init__(self, path: str, logger: LoggerAdapter) -> None:
        self._path = path
        self.logger = logger
        self._cache: Optional[Dict[str, Any]] = None

    def find_gold_set(self) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache

        candidates: List[Path] = []
        configured = Path(self._path)
        candidates.append(configured)
        if not configured.is_absolute():
            candidates.append(Path("data") / self._path)
            candidates.append(Path(__file__).resolve().parent / self._path)

        last_error: Optional[Exception] = None
        for option in candidates:
            try:
                if option.exists():
                    payload = json.loads(option.read_text(encoding="utf-8"))
                    structured = self._normalize_payload(payload)
                    self._cache = structured
                    self.logger.info(f"Gold set loaded from {option}")
                    return structured
            except Exception as exc:
                last_error = exc
                self.logger.error(
                    f"Unexpected error loading gold set from {option}: {exc}"
                )

        if last_error:
            raise last_error
        raise FileNotFoundError(
            f"Could not locate gold_set JSON using candidates: {candidates}"
        )

    @staticmethod
    def _normalize_payload(payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict):
            return cast(Dict[str, Any], payload)
        if isinstance(payload, list):
            normalized_items: List[Dict[str, Any]] = []
            for raw_item in cast(List[Any], payload):
                if not isinstance(raw_item, dict):
                    raise ValueError(
                        "Cada elemento del gold set debe ser un diccionario"
                    )
                normalized_items.append(cast(Dict[str, Any], raw_item))
            return {"items": normalized_items}
        raise ValueError(
            "Gold set JSON must be a dict with an 'items' key or a list of items"
        )


class EvaluatorAgent:
    def __init__(self, logger: LoggerAdapter, gold_set: GoldSet) -> None:
        self.gold = gold_set
        self.logger = logger

    def evaluator(self, chat_response: str, example: Dict[str, Any]) -> str:
        reference_block = self._build_reference_block(example)

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            self.logger.warning(
                "DEEPSEEK_API_KEY not set; returning fallback evaluation output"
            )
            return (
                "[Evaluator fallback] Missing DEEPSEEK_API_KEY; unable to score"
                f". Modelo respondió: {chat_response[:200]}..."
            )

        try:
            client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        except Exception as exc:
            self.logger.error(f"Cannot initialise evaluator client: {exc}")
            return f"[Evaluator error] {exc}"

        system_prompt = (
            "Eres un evaluador experto. Usa la referencia y la rúbrica entregadas para "
            "calificar la respuesta del modelo. Devuelve un JSON compacto con las "
            "claves 'score' (0-100), 'notes', 'EM' y 'citas'."
        )

        user_message = (
            "Referencia oficial y rúbrica:\n"
            f"{reference_block}\n\n"
            "Respuesta del modelo a evaluar:\n"
            f"{chat_response}\n\n"
            "Evalúa la respuesta considerando la rúbrica. Si hay información faltante en la "
            "referencia, indícalo en las notas."
        )

        temperature = 0.1
        max_tokens = 250

        try:
            resp = client.chat.completions.create(
                model="deepseek/deepseek-chat-v3.1:free",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            evaluation_text = self._extract_content(resp)
            try:
                raw = resp.to_dict() if hasattr(resp, "to_dict") else repr(resp)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error(f"{exc}")
                raw = repr(resp)

            raw_str = (
                json.dumps(raw, ensure_ascii=False, indent=2)
                if isinstance(raw, (dict, list))
                else str(raw)
            )
            self.logger.info(f"RAW evaluator response: {raw_str}")
            if evaluation_text:
                self.logger.info(f"Evaluation summary: {evaluation_text}")
            return evaluation_text or raw_str
        except Exception as exc:
            self.logger.error(f"Error in evaluator: {exc}")
            return f"[Evaluator error] {exc}"

    @staticmethod
    def _extract_content(resp: Any) -> str:
        try:
            choices = getattr(resp, "choices", None)
            if not choices and isinstance(resp, dict):
                resp_dict = cast(Dict[str, Any], resp)
                choices = resp_dict.get("choices")
            if not choices:
                return ""

            choices_list = cast(List[Any], choices)
            if not choices_list:
                return ""

            first = choices_list[0]
            message = getattr(first, "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                if content:
                    return str(content).strip()

            text = getattr(first, "text", None)
            if text:
                return str(text).strip()

            if isinstance(first, dict):
                first_dict = cast(Dict[str, Any], first)
                nested_message_obj = first_dict.get("message", {})
                nested_message = (
                    cast(Dict[str, Any], nested_message_obj)
                    if isinstance(nested_message_obj, dict)
                    else {}
                )
                nested_content = nested_message.get("content")
                nested = nested_content if nested_content is not None else first_dict.get("text")
                if nested is not None:
                    return str(nested).strip()
        except Exception:  # pragma: no cover - defensive guard
            return ""
        return ""

    @staticmethod
    def _build_reference_block(example: Dict[str, Any]) -> str:
        lines: List[str] = []

        query = example.get("query")
        if query:
            lines.append(f"Pregunta: {query}")

        expected = example.get("expected_answer")
        if expected:
            lines.append("Respuesta esperada:")
            lines.append(str(expected))

        rubric_raw = example.get("rubric")
        rubric_items: List[str] = []
        if isinstance(rubric_raw, list):
            for raw_item in cast(List[Any], rubric_raw):
                if raw_item is None:
                    continue
                rubric_items.append(str(raw_item))
        if rubric_items:
            lines.append("Rúbrica:")
            for item in rubric_items:
                lines.append(f"- {item}")

        references_raw = example.get("references")
        references: List[Dict[str, Any]] = []
        if isinstance(references_raw, list):
            for ref in cast(List[Any], references_raw):
                if isinstance(ref, dict):
                    references.append(cast(Dict[str, Any], ref))
        if references:
            lines.append("Referencias:")
            for ref in references:
                title = str(ref.get("title") or ref.get("doc_id") or "")
                page = ref.get("page")
                snippet = ref.get("snippet")
                parts: List[str] = []
                if title:
                    parts.append(title)
                if page is not None and str(page):
                    parts.append(f"p. {page}")
                if snippet:
                    parts.append(str(snippet))
                if parts:
                    lines.append("- " + " — ".join(parts))

        return "\n".join(lines)


class Evaluator:
    def __init__(
        self,
        gold_set: GoldSet,
        deepseek_provider: DeepSeekProvider,
        evaluator_agent: EvaluatorAgent,
        rag_orchestrator: RAGOrchestrator,
    ) -> None:
        self._gold = gold_set
        self._deepseek_provider = deepseek_provider
        self.evaluator_agent = evaluator_agent
        self.rag_orchestrator = rag_orchestrator
        self.logger = gold_set.logger

    def run(
        self,
        k_retrieve: int = 5,
        rerank_top_n: int = 5,
        do_rewrite: bool = True,
    ) -> List[Dict[str, Any]]:
        try:
            gold_data = self._gold.find_gold_set()
        except Exception as exc:
            self.logger.error(f"Unable to load gold set: {exc}")
            return []

        items_raw = gold_data.get("items")
        if not isinstance(items_raw, list):
            self.logger.error("Gold set format invalid: 'items' must be a list")
            return []

        items: List[Dict[str, Any]] = []
        for example_raw in cast(List[Any], items_raw):
            if isinstance(example_raw, dict):
                items.append(cast(Dict[str, Any], example_raw))
            else:
                self.logger.warning("Skipping malformed gold example (not a mapping)")

        evaluations: List[Dict[str, Any]] = []
        for example in items:

            example_id = example.get("id", "unknown")
            query = example.get("query")
            if not query:
                self.logger.warning(
                    f"Skipping gold example {example_id}: missing query"
                )
                continue

            self.logger.info(f"Evaluating example {example_id}: {query}")
            try:
                rag_output = self.rag_orchestrator.run(
                    query,
                    k_retrieve=k_retrieve,
                    rerank_top_n=rerank_top_n,
                    do_rewrite=do_rewrite,
                )
            except Exception as exc:
                self.logger.error(
                    f"RAG orchestration failed for {example_id}: {exc}"
                )
                evaluations.append(
                    {
                        "id": example_id,
                        "query": query,
                        "error": f"RAG failure: {exc}",
                        "expected_answer": example.get("expected_answer"),
                    }
                )
                continue

            rewritten_query = rag_output.get("query", query)
            hints_raw = rag_output.get("hints", [])
            hints: List[Dict[str, Any]] = []
            if isinstance(hints_raw, list):
                for hint_raw in cast(List[Any], hints_raw):
                    if isinstance(hint_raw, dict):
                        hints.append(cast(Dict[str, Any], hint_raw))

            system_prompt = build_synthesis_prompt(str(rewritten_query), hints)
            answer = self._deepseek_provider.chat(
                system_prompt,
                query,
                temperature=0.2,
                max_tokens=512,
            )
            self.logger.info(f"Model answer for {example_id}: {answer}")

            evaluation_result = self.evaluator_agent.evaluator(str(answer), example)

            record: Dict[str, Any] = {
                "id": example_id,
                "query": query,
                "rewritten_query": rewritten_query,
                "expected_answer": example.get("expected_answer"),
                "rubric": example.get("rubric"),
                "references": example.get("references"),
                "answer": answer,
                "rag_hints": hints,
                "evaluation": evaluation_result,
            }
            evaluations.append(record)

        output_path = Path("data/eval_results.json")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(evaluations, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.logger.info(f"Evaluation batch saved to {output_path}")
        except Exception as exc:
            self.logger.error(f"Unable to persist evaluation results: {exc}")

        return evaluations


def main() -> None:
    evaluator_logger = LoggerStdin("evaluate", "logs/evaluate.log")
    gold_set = GoldSet("gold_set.json", evaluator_logger)
    evaluator_agent = EvaluatorAgent(evaluator_logger, gold_set)

    faiss_index_path = "data/processed/index.faiss"
    chunks_path = "data/processed/chunks.parquet"
    mapping_path = "data/processed/mapping.parquet"

    retriever = FaissRetriever(
        faiss_index_path,
        chunks_path,
        mapping_path,
        evaluator_logger,
    )
    reranker = CrossEncoderReranker(evaluator_logger)
    rag_orchestrator = RAGOrchestrator(retriever, evaluator_logger, reranker)
    deepseek_provider = DeepSeekProvider(evaluator_logger)

    evaluator = Evaluator(
        gold_set,
        deepseek_provider,
        evaluator_agent,
        rag_orchestrator,
    )
    evaluator.run()


if __name__ == "__main__":
    main()