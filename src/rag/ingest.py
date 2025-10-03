import os
import re
from collections import Counter
from pathlib import Path
import sys
import faiss # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text

# pdfminer.extract_text supports page_numbers argument
from pdfminer.pdfpage import PDFPage # type: ignore
from typing import Any, Dict, Optional, cast
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from dotenv import load_dotenv

try:
    from adapters.logger_adapter import LoggerAdapter
    from adapters.logger_strdin import LoggerStdin
except ModuleNotFoundError:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from adapters.logger_adapter import LoggerAdapter
    from adapters.logger_strdin import LoggerStdin

class Ingestor:
    def __init__(self, csv_path: str, output_dir: str, logger: LoggerAdapter) -> None:
        self._csv_path = Path(csv_path).resolve()
        self._output_dir = Path(output_dir).resolve()
        self._logger = logger
        
    def extract_pages_text(self, pdf_path: str) -> list[str]:
        """Return list of page texts (1-indexed pages -> index 0 = page 1)"""

        pages: list[str] = []
        # Get total pages by iterating once
        with open(pdf_path, "rb") as f:
            for _ in PDFPage.get_pages(f):
                pages.append("")
        total = len(pages)
        for i in range(total):
            try:
                text: Optional[str] = extract_text(pdf_path, page_numbers=[i]) 
                pages[i] = text or ""  # ensure we store a string, not None
            except Exception:
                # If extraction fails for a page (e.g. images-only), keep empty string
                pages[i] = ""
        return pages


    def detect_repeated_header_footer(
        self,
        page_texts: list[str],
        head_lines: int = 3,
        tail_lines: int = 3,
        sample_pages: int = 10,
    ):
        """Heuristic: look for lines that repeat across many pages in the first/last N lines.
        Returns (header_pattern, footer_pattern) regex strings (may be None).
        """
        sample = page_texts
        if len(page_texts) > sample_pages:
            # sample evenly
            idxs = np.linspace(0, len(page_texts) - 1, sample_pages, dtype=int)
            sample = [page_texts[i] for i in idxs] 

        headers = []
        footers = []
        for p in sample: 
            if not p:
                continue  # skip empty / None pages
            lines = [line.strip() for line in p.splitlines() if line.strip()]
            if not lines:
                continue
            headers.append("\n".join(lines[:head_lines]))
            footers.append("\n".join(lines[-tail_lines:]))

        def common_pattern(strings, threshold_ratio=0.4):
            if not strings:
                return None
            cnt = Counter(strings)
            common, freq = cnt.most_common(1)[0]
            if freq / len(strings) >= threshold_ratio:
                # escape regex special chars, but allow digits (page numbers) variability
                # replace runs of digits with \d+
                esc = re.escape(common)
                esc = re.sub(r"\\\d\+", r"\\d\+", esc)
                esc = re.sub(r"\\\d{1,}", r"\\d+", esc)
                # also collapse variable whitespace
                esc = re.sub(r"\\\s\+", r"\\s+", esc)
                return esc
            return None

        header_pat = common_pattern(headers)
        footer_pat = common_pattern(footers)
        return header_pat, footer_pat


    def clean_page_text(
        self,
        text: str, header_pat: Optional[str] = None, footer_pat: Optional[str] = None
    ):
        if not text:
            return ""
        s = text
        # remove header
        if header_pat:
            try:
                s = re.sub(r"(?m)^" + header_pat + r"\s*\n?", "", s)
            except re.error:
                pass
        if footer_pat:
            try:
                s = re.sub(r"(?m)\n?\s*" + footer_pat + r"$", "", s)
            except re.error:
                pass
        # remove multiple blank lines, normalize spaces
        s = re.sub(r"\r", "\n", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        s = re.sub(r"[ \t]{2,}", " ", s)
        # trim lines
        s = "\n".join([ln.strip() for ln in s.splitlines() if ln.strip()])
        return s


    def simple_sentence_split(self, text: str) -> list[str]:
        # naive sentence splitter
        if not text:
            return []
        # protect abbreviations (very naive)
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        # fallback to line-based
        if len(sentences) == 1:
            sentences = [ln for ln in text.splitlines() if ln.strip()]
        return [s.strip() for s in sentences if s.strip()]


    def chunk_text(
        self, text: str, max_chars: int = 1000, overlap: int = 200
    ) -> list[tuple[str, int, int]]:
        sentences = self.simple_sentence_split(text)
        chunks: list[tuple[str, int, int]] = []
        cur = ""
        cur_len = 0
        start_idx = 0
        for sent in sentences:
            if cur_len + len(sent) + 1 <= max_chars:
                if cur:
                    cur += " " + sent
                else:
                    cur = sent
                cur_len = len(cur)
            else:
                chunks.append((cur, start_idx, start_idx + cur_len))
                # start new chunk with overlap
                # compute overlap in characters from end of cur
                overlap_text = cur[-overlap:] if overlap and overlap < len(cur) else cur
                cur = overlap_text + " " + sent
                start_idx = start_idx + cur_len - len(overlap_text)
                cur_len = len(cur)
        if cur:
            chunks.append((cur, start_idx, start_idx + cur_len))
        # give chunk ids and order
        return chunks


    def slugify(self, text: str) -> str:
        if not text:
            return "unknown"
        slug = re.sub(r"[^\w]+", "-", text.lower())
        slug = re.sub(r"-+", "-", slug).strip("-")
        return slug or "unknown"


    def process_row(
        self,
        row: pd.Series,
        base_dir: Path,
        doc_index: int,
        max_chars: int = 1000,
        overlap: int = 200,
    ) -> list[dict[str, Any]]:
        raw_path = str(row.get("path", "")).strip()
        if not raw_path:
            raise ValueError("Missing 'path' value in metadata row")

        pdf_path = Path(raw_path)
        if not pdf_path.is_absolute():
            pdf_path = (base_dir / pdf_path).resolve()

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        page_texts = self.extract_pages_text(str(pdf_path))
        header_pat, footer_pat = self.detect_repeated_header_footer(page_texts)

        title = str(row.get("nombre", "")).strip()
        doc_slug = self.slugify(title) or f"doc-{doc_index}"
        doc_id = f"{doc_index:03d}-{doc_slug}"

        metadata = {
            "source_path": str(pdf_path),
            "title": title,
            "url": str(row.get("url", "")).strip(),
            "fecha": str(row.get("fecha", "")).strip(),
            "vigencia": str(row.get("vigencia", "")).strip(),
        }

        rows: list[dict[str, Any]] = []
        for page_number, page_text in enumerate(page_texts, start=1):
            cleaned = self.clean_page_text(page_text, header_pat, footer_pat)
            if not cleaned:
                continue

            for chunk_idx, (chunk_text_value, start_char, end_char) in enumerate(
                self.chunk_text(cleaned, max_chars=max_chars, overlap=overlap)
            ):
                if not chunk_text_value:
                    continue

                chunk_id = f"{doc_id}_p{page_number:03d}_c{chunk_idx:03d}"
                rows.append(
                    {
                        **metadata,
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "page": page_number,
                        "start_char": start_char,
                        "end_char": end_char,
                        "text": chunk_text_value,
                    }
                )

        return rows


    def process_csv(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        max_chars: int = 1000,
        overlap: int = 200,
        index_type: str = "ip",
        push_to_qdrant: bool = False,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        qdrant_collection: str = "esp3_chunks",
        qdrant_prefer_grpc: bool = False,
        qdrant_recreate: bool = False,
        qdrant_batch_size: int = 256,
    ) -> dict[str, str]:
        output_dir_path = self._output_dir
        output_dir_path.mkdir(parents=True, exist_ok=True)

        df_meta = pd.read_csv(self._csv_path)
        base_dir = self._csv_path.parent

        all_rows: list[dict[str, Any]] = []
        for idx, (_, row) in enumerate(df_meta.iterrows()):
            all_rows.extend(
                self.process_row(
                    row,
                    base_dir=base_dir,
                    doc_index=idx,
                    max_chars=max_chars,
                    overlap=overlap,
                )
            )

        df = pd.DataFrame(all_rows)
        if df.empty:
            raise RuntimeError("No text extracted from PDFs")

        df = df.reset_index(drop=True)
        df["position"] = df.index.astype(int)

        model = SentenceTransformer(model_name)
        texts: list[str] = df["text"].astype(str).tolist()
        embeddings = model.encode(
            texts, convert_to_numpy=True, show_progress_bar=True
        )

        dim = embeddings.shape[1]
        if index_type.lower() in ("ip", "indexflatip"):
            index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(embeddings)
        else:
            index = faiss.IndexFlatL2(dim)
        index.add(embeddings)  # type: ignore

        index_path = output_dir_path / "index.faiss"
        faiss.write_index(index, str(index_path))

        chunks_path = output_dir_path / "chunks.parquet"
        df.to_parquet(chunks_path, index=False)

        mapping = pd.DataFrame(
            {"chunk_id": df["chunk_id"].tolist(), "position": df["position"].tolist()}
        )
        mapping_path = output_dir_path / "mapping.parquet"
        mapping.to_parquet(mapping_path, index=False)

        self._logger.info(f"Saved FAISS index to: {index_path}")
        self._logger.info(f"Saved chunks to: {chunks_path}")
        self._logger.info(f"Saved mapping to: {mapping_path}")

        if push_to_qdrant:
            if qdrant_url is None:
                raise ValueError("Qdrant URL must be provided when push_to_qdrant=True")
            self._logger.info(
                f"Uploading {len(df)} chunk embeddings to Qdrant collection '{qdrant_collection}' at {qdrant_url}"
            )
            self.push_to_qdrant(
                embeddings=embeddings,
                df=df,
                collection_name=qdrant_collection,
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
                prefer_grpc=qdrant_prefer_grpc,
                recreate_collection=qdrant_recreate,
                batch_size=qdrant_batch_size,
            )

        results: dict[str, str] = {
            "index_path": str(index_path),
            "chunks_path": str(chunks_path),
            "mapping_path": str(mapping_path),
        }

        if push_to_qdrant:
            results["qdrant_collection"] = qdrant_collection
            results["qdrant_url"] = qdrant_url or ""

        return results

    def push_to_qdrant(
        self,
        embeddings: np.ndarray,
        df: pd.DataFrame,
        collection_name: str,
        qdrant_url: str,
        qdrant_api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        recreate_collection: bool = False,
        batch_size: int = 256,
    ) -> None:
        if QdrantClient is None or PointStruct is None or VectorParams is None or Distance is None:
            raise ImportError(
                "qdrant-client is required to push embeddings. Install it or disable push_to_qdrant."
            )

        client_cls = cast(Any, QdrantClient)
        point_struct_cls = cast(Any, PointStruct)
        vector_params_cls = cast(Any, VectorParams)
        distance_enum = cast(Any, Distance)

        client = client_cls(
            url=qdrant_url,
            api_key=qdrant_api_key,
            prefer_grpc=prefer_grpc,
        )

        vector_size = int(embeddings.shape[1])
        self._ensure_qdrant_collection(
            client=client,
            collection_name=collection_name,
            vector_size=vector_size,
            recreate=recreate_collection,
            vector_params_cls=vector_params_cls,
            distance_enum=distance_enum,
        )

        total_rows = len(df)
        if total_rows == 0:
            self._logger.warning("No chunks available to push to Qdrant; skipping upload.")
            return

        embeddings = embeddings.astype(np.float32, copy=False)
        records = df.to_dict(orient="records")
        effective_batch_size = max(int(batch_size), 1)

        for start in range(0, total_rows, effective_batch_size):
            end = min(start + effective_batch_size, total_rows)
            batch_vectors = embeddings[start:end].tolist()
            batch_payloads = records[start:end]

            points = []
            for idx, payload in enumerate(batch_payloads):
                safe_payload: Dict[str, Any] = {str(key): value for key, value in payload.items()}
                point_id = safe_payload.get("position")
                if point_id is None:
                    point_id = start + idx
                try:
                    numeric_id = int(point_id)
                except (ValueError, TypeError):
                    self._logger.warning(
                        f"Invalid Qdrant point id '{point_id}' derived from chunk, using fallback index {start + idx}."
                    )
                    numeric_id = int(start + idx)

                point = point_struct_cls(
                    id=numeric_id,
                    vector=batch_vectors[idx],
                    payload=safe_payload,
                )
                points.append(point)

            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )

            self._logger.info(
                f"Upserted {end}/{total_rows} chunks into Qdrant collection '{collection_name}'"
            )

    def _ensure_qdrant_collection(
        self,
        client: Any,
        collection_name: str,
        vector_size: int,
        recreate: bool,
        vector_params_cls: Any,
        distance_enum: Any,
    ) -> None:
        if recreate:
            self._logger.warning(
                f"Recreating Qdrant collection '{collection_name}'; existing points will be removed."
            )
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=vector_params_cls(size=vector_size, distance=distance_enum.COSINE),
            )
            return

        if not client.collection_exists(collection_name):
            self._logger.info(
                f"Creating Qdrant collection '{collection_name}' with vector size {vector_size}"
            )
            client.create_collection(
                collection_name=collection_name,
                vectors_config=vector_params_cls(size=vector_size, distance=distance_enum.COSINE),
            )
            return

        existing_size = self._extract_vector_size(client, collection_name)
        if existing_size is not None and existing_size != vector_size:
            raise ValueError(
                (
                    f"Qdrant collection '{collection_name}' expects vector size {existing_size}, "
                    f"but embeddings have size {vector_size}. Set QDRANT_RECREATE_COLLECTION=true "
                    "to rebuild the collection or adjust the embedding model."
                )
            )

    def _extract_vector_size(self, client: Any, collection_name: str) -> Optional[int]:
        try:
            info = client.get_collection(collection_name)
        except Exception as exc:  # pragma: no cover - network call
            self._logger.warning(
                f"Unable to inspect Qdrant collection '{collection_name}': {exc}"
            )
            return None

        config = getattr(info, "config", None)
        params = getattr(config, "params", None)
        vectors = getattr(params, "vectors", None)

        if vectors is None:
            return None

        size_attr = getattr(vectors, "size", None)
        if size_attr is not None:
            try:
                return int(size_attr)
            except (TypeError, ValueError):
                return None

        to_dict = getattr(vectors, "to_dict", None)
        if callable(to_dict):
            try:
                data = to_dict()
                raw_size = data.get("size") if isinstance(data, dict) else None
                if raw_size is not None:
                    return int(raw_size)
            except Exception:
                return None

        if isinstance(vectors, dict):
            raw_size = vectors.get("size")
            if raw_size is not None:
                try:
                    return int(raw_size)
                except (TypeError, ValueError):
                    return None

        return None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    load_dotenv()
    ingest_logger = LoggerStdin("ingest", "logs/ingest.log")
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "sources.csv"
    output_dir = project_root / "data" / "processed"
    ingestor = Ingestor(csv_path=str(csv_path), output_dir=str(output_dir), logger=ingest_logger)

    push_to_qdrant = _env_flag("INGEST_PUSH_TO_QDRANT", default=False)
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "esp3_chunks")
    qdrant_recreate = _env_flag("QDRANT_RECREATE_COLLECTION", default=False)
    qdrant_prefer_grpc = _env_flag("QDRANT_USE_GRPC", default=False)
    qdrant_batch_size_str = os.getenv("QDRANT_BATCH_SIZE", "256")

    try:
        qdrant_batch_size = int(qdrant_batch_size_str)
    except ValueError:
        ingest_logger.warning(
            f"Invalid QDRANT_BATCH_SIZE='{qdrant_batch_size_str}', fallback to 256"
        )
        qdrant_batch_size = 256

    results = ingestor.process_csv(
        push_to_qdrant=push_to_qdrant,
        qdrant_url=qdrant_url if push_to_qdrant else None,
        qdrant_api_key=qdrant_api_key,
        qdrant_collection=qdrant_collection,
        qdrant_prefer_grpc=qdrant_prefer_grpc,
        qdrant_recreate=qdrant_recreate,
        qdrant_batch_size=qdrant_batch_size,
    )
    ingest_logger.info(f"Done: {results}")