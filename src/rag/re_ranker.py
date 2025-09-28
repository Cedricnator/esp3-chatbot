from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
from adapters.logger_adapter import LoggerAdapter

class CrossEncoderReranker:
  def __init__(self, logger: LoggerAdapter, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    # model returns relevance scores for (query, candidate) pairs
    self.model = CrossEncoder(model_name)
    self.logger = logger

  def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
    """
    candidates: list of dicts with key 'text' at least (and metadata)
    returns new list sorted by cross-encoder score (desc), limited to top_n
    """
    self.logger.info("Reranking ....")
    pairs = [(query, c["text"]) for c in candidates]
    scores = self.model.predict([list(p) for p in pairs]) # type: ignore
    for c, s in zip(candidates, scores): # pyright: ignore[reportUnknownArgumentType]
        c["__rerank_score"] = float(s)
    candidates_sorted = sorted(candidates, key=lambda x: x["__rerank_score"], reverse=True)
    return candidates_sorted[:top_n]
