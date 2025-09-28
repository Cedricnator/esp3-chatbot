import re
from collections import Counter
from pathlib import Path
import faiss # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text  # type: ignore

# pdfminer.extract_text supports page_numbers argument
from pdfminer.pdfpage import PDFPage # type: ignore
from typing import Any, Optional

class Ingestor:
    def __init__(self, csv_path: str, output_dir: str) -> None:
        self._csv_path = Path(csv_path).resolve()
        self._output_dir = Path(output_dir).resolve()
        
    def extract_pages_text(self, pdf_path: str) -> list[str]:
        """Return list of page texts (1-indexed pages -> index 0 = page 1)"""

        pages: list[str] = []
        # Get total pages by iterating once
        with open(pdf_path, "rb") as f:
            for _ in PDFPage.get_pages(f):  # type: ignore
                pages.append("")  # initialize with empty string (never None)
        total = len(pages)
        for i in range(total):
            try:
                text: Optional[str] = extract_text(pdf_path, page_numbers=[i])  # pyright: ignore[reportUnknownVariableType]
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
            sample = [page_texts[i] for i in idxs]  # type: ignore

        headers = []
        footers = []
        for p in sample:  # type: ignore
            if not p:
                continue  # skip empty / None pages
            lines = [line.strip() for line in p.splitlines() if line.strip()]  # type: ignore
            if not lines:
                continue
            headers.append("\n".join(lines[:head_lines]))  # type: ignore
            footers.append("\n".join(lines[-tail_lines:]))  # type: ignore

        def common_pattern(strings, threshold_ratio=0.4):  # type: ignore
            if not strings:
                return None
            cnt = Counter(strings)  # type: ignore
            common, freq = cnt.most_common(1)[0]  # type: ignore
            if freq / len(strings) >= threshold_ratio:  # type: ignore
                # escape regex special chars, but allow digits (page numbers) variability
                # replace runs of digits with \d+
                esc = re.escape(common)  # type: ignore
                esc = re.sub(r"\\\d\+", r"\\d\+", esc)  # type: ignore # no-op if none
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
    ) -> dict[str, str]:
        output_dir_path = self._output_dir
        output_dir_path.mkdir(parents=True, exist_ok=True)

        df_meta = pd.read_csv(self._csv_path)  # type: ignore[call-overload]
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

        model = SentenceTransformer(model_name)
        texts: list[str] = df["text"].astype(str).tolist()
        embeddings = model.encode(  # type: ignore[call-overload]
            texts, convert_to_numpy=True, show_progress_bar=True
        )

        dim = embeddings.shape[1]
        if index_type.lower() in ("ip", "indexflatip"):
            index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(embeddings)  # type: ignore[arg-type]
        else:
            index = faiss.IndexFlatL2(dim)
        index.add(embeddings)  # type: ignore[arg-type]

        index_path = output_dir_path / "index.faiss"
        faiss.write_index(index, str(index_path))  # type: ignore[call-overload]

        chunks_path = output_dir_path / "chunks.parquet"
        df.to_parquet(chunks_path, index=False)

        mapping = pd.DataFrame(
            {"chunk_id": df["chunk_id"].tolist(), "position": list(range(len(df)))}
        )
        mapping_path = output_dir_path / "mapping.parquet"
        mapping.to_parquet(mapping_path, index=False)

        print(f"Saved FAISS index to: {index_path}")
        print(f"Saved chunks to: {chunks_path}")
        print(f"Saved mapping to: {mapping_path}")

        return {
            "index_path": str(index_path),
            "chunks_path": str(chunks_path),
            "mapping_path": str(mapping_path),
        }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "sources.csv"
    output_dir = project_root / "data" / "processed"
    ingestor = Ingestor(str(csv_path), str(output_dir))

    results = ingestor.process_csv()
    print("Done:", results)