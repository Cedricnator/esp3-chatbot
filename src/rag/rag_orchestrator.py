from typing import Any, Dict, List, Optional

from rag.retrieve import FaissRetriever
from rag.re_ranker import CrossEncoderReranker
from adapters.logger_adapter import LoggerAdapter


class RAGOrchestrator:
  def __init__(
    self,
    retriever: FaissRetriever,
    logger: LoggerAdapter,
    reranker: Optional[CrossEncoderReranker] = None,
  ) -> None:
    self.retriever = retriever
    self.reranker = reranker
    self.logger = logger

  def run(
    self,
    query: str,
    k_retrieve: int,
    rerank_top_n: int = 5,
    do_rewrite: bool = False,
  ) -> Dict[str, Any]:
    rewritten_query = query
    if do_rewrite:
      try:
        self.logger.info("rewriting query...")
        rewritten_query = self.retriever.rewrite_query(query) 
        self.logger.info(f"rewrite: {rewritten_query}") 
      except Exception:
        rewritten_query = query

    df_ret = self.retriever.search(rewritten_query, top_k=k_retrieve)
    candidates: List[Dict[str, Any]] = []
    if not df_ret.empty:
      rows = df_ret.to_dict(orient="records")
      candidates = [
        {str(key): value for key, value in row.items()}
        for row in rows
      ]

    ranked = self._rank_candidates(rewritten_query, candidates, rerank_top_n)

    return {
      "query": rewritten_query,
      "hints": ranked,
    }

  def _rank_candidates(
    self,
    query: str,
    candidates: List[Dict[str, Any]],
    rerank_top_n: int,
  ) -> List[Dict[str, Any]]:
    if not candidates:
      return []

    if self.reranker is None:
      return self._top_by_score(candidates, rerank_top_n)

    try:
      return self.reranker.rerank(query, candidates, top_n=rerank_top_n)
    except Exception:
      return self._top_by_score(candidates, rerank_top_n)

  @staticmethod
  def _top_by_score(
    candidates: List[Dict[str, Any]], rerank_top_n: int
  ) -> List[Dict[str, Any]]:
    return sorted(
      candidates,
      key=lambda x: x.get("__score", 0.0),
      reverse=True,
    )[:rerank_top_n]