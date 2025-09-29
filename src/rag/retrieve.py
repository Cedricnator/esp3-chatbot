import os
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast
from adapters.logger_adapter import LoggerAdapter
from openai import OpenAI
from sentence_transformers import SentenceTransformer

class FaissRetriever:
	def __init__(self, indx_path: str, chunks_path: str, mapping_path: str, logger: LoggerAdapter) -> None:
		self._logger = logger
		self._model = SentenceTransformer("all-MiniLM-L6-v2")
		self.index_path = Path(indx_path)
		self.chunks_path = Path(chunks_path)
		self.mapping_path = Path(mapping_path)
		self.index = self.load_faiss_index(self.index_path) 
		self.chunks, self.mapping = self.load_chunks_and_mapping(self.chunks_path, self.mapping_path)  # type: ignore
		self._position_to_chunk = self._build_position_lookup()

	def rewrite_query(self, query: str) -> str:
		"""query rewriting/clarification using an LLM provider; fallback = identity."""
		api_key = os.getenv("DEEPSEEK_API_KEY")
		client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
		system_prompt = """
			You are an advanced retrieval specialist. Rewrite the user query into a single,
			self-contained sentence that emphasizes key entities, expands acronyms, and adds relevant
			synonyms so it becomes highly informative for semantic search. Remove pronouns and vague
			references while keeping the intent unchanged. Write the query in the spanish language.
  	"""
		response = client.chat.completions.create(
			model="deepseek/deepseek-chat-v3.1:free",
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": query}
			],
			max_tokens=80,
			temperature=0.1,
		)
		choices = getattr(response, "choices", None)
		if choices is not None:
			first = choices[0]
		else:
			return query
		msg = getattr(first, "message", None)
		if msg is not None:
			content = getattr(msg, "content", None)
			if content:
				return content.strip()
		# fallback to .text
		text = getattr(first, "text", None) 
		if text:
			return text.strip()
		# fallback to identity
		return query

	def load_chunks_and_mapping(self, chunks_path: str, mapping_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
		self._logger.info("Loading chunks and mapping...")
		chunks = pd.read_parquet(chunks_path)
		mapping = pd.read_parquet(mapping_path)
		return chunks, mapping

	def load_faiss_index(self, index_path: Path) -> faiss.Index:
		self._logger.info("Loading faiss index...")
		idx = faiss.read_index(str(index_path))
		return idx 

	def _build_position_lookup(self) -> Dict[int, str]:
		if {"position", "chunk_id"}.issubset(set(self.mapping.columns)):
			lookup: Dict[int, str] = {}
			for _, record in self.mapping.iterrows(): 
				try:
					position = int(record["position"])
					chunk = str(record["chunk_id"])
				except (KeyError, ValueError, TypeError):
					continue
				lookup[position] = chunk
			return lookup
		return {}

	def _embed_query(self, query: str) -> Any:
		emb_raw = self._model.encode( 
			[query], 
   		convert_to_numpy=True, 
     	show_progress_bar=False
		)
		emb = np.asarray(emb_raw, dtype=np.float32)
		if emb.ndim == 1:
			emb = emb.reshape(1, -1)
		return emb

	def _chunk_from_position(self, position: int) -> dict[str, Any]:
		if 0 <= position < len(self.chunks):
			return self.chunks.iloc[position].to_dict()
		chunk_id = self._position_to_chunk.get(position)
		if not chunk_id:
			raise KeyError(position)
		row = self.chunks[self.chunks["chunk_id"] == chunk_id]
		if row.empty:
			raise KeyError(position)
		return row.iloc[0].to_dict()

	def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
		q_emb = self._embed_query(query)
		distances, indices = self.index.search(q_emb, top_k) # type: ignore
		scores = cast(List[float], distances[0].tolist()) 
		positions = [int(x) for x in cast(List[int], indices[0].tolist())] 
		rows = []
		for score, pos in zip(scores, positions): 
			if pos < 0:
				continue
			try:
				chunk_row = self._chunk_from_position(pos)
			except KeyError:
				continue
			row = { 
				**chunk_row,
				"__score": float(score),
				"__position": int(pos),
			}
			rows.append(row)
		return pd.DataFrame(rows)