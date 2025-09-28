from typing import List, Dict, Any

def basic_system_prompt():
    return "You are ChatGPT, a helpful assistant."

def format_citation(meta: Dict[str, Any]) -> str:
    """
    Given a metadata row (dict), try to produce a citation string like:
      - [Autor, 2020] if 'author' and 'fecha' (year) exist
      - else [Title, pX] if 'title' and 'page' exist
      - else [doc_id, chunk_id]
    """
    # Prefer author+year if present
    author = meta.get("author") or meta.get("autor") or None
    fecha = meta.get("fecha") or meta.get("date") or None
    title = meta.get("title") or meta.get("nombre") or None
    page = meta.get("page") or meta.get("pagina") or None

    if author and fecha:
        # try extracting year
        year = None
        try:
            year = str(fecha).strip()[:4]
            int(year)
        except Exception:
            year = None
        if year:
            return f"[{author}, {year}]"
    if title and page:
        return f"[{title}, p{page}]"
    if title:
        return f"[{title}]"
    if meta.get("doc_id") and meta.get("page"):
        return f"[{meta.get('doc_id')}, p{meta.get('page')}]"
    return f"[{meta.get('chunk_id', meta.get('doc_id', 'source'))}]"

def build_synthesis_prompt(
    query: str,
    retrieved: List[Dict[str, Any]],
    instruction: str = "",
    max_context_tokens: int = 2500,
) -> str:
    """
    Build a prompt that places the retrieved chunks and instructs the LLM to answer citing sources.
    retrieved: list of dicts (must contain 'text' and metadata)
    instruction: additional instructions for the LLM (tone, length, policy)
    Returns a string prompt for the LLM.
    """
    header = (
        "Eres un asistente experto que responde haciendo RAG (Retrieval-Augmented Generation).\n"
        "Usa únicamente la información provista abajo en los fragmentos cuando prepares la respuesta. "
        "Siempre incluye citas inline en el formato [Autor, año] o [Título, sección/página] según esté disponible.\n"
        "Si la respuesta no puede ser inferida de los fragmentos, di honestamente: 'No tengo suficiente información en las fuentes provistas.'\n\n"
    )
    if instruction:
        header += instruction.strip() + "\n\n"

    # Concat retrieved documents but guard token/char budget (simple char-based truncation)
    parts = []
    accumulated = 0
    max_chars = max_context_tokens * 4  # rough approximation chars vs tokens
    for i, r in enumerate(retrieved):
        meta = r.copy()
        text = meta.pop("text", "")
        citation = format_citation(meta)
        entry = f"=== SOURCE {i + 1} {citation} ===\n{text}\n"
        if accumulated + len(entry) > max_chars:
            # truncate this entry
            remaining = max(0, max_chars - accumulated - 200)
            if remaining <= 0:
                break
            entry = entry[:remaining] + "\n...[TRUNCATED]\n"
            parts.append(entry) # type: ignore
            break
        parts.append(entry) # type: ignore
        accumulated += len(entry)

    sources_block = "\n\n".join(parts) # type: ignore

    prompt = (
        header
        + "DOCUMENTOS RECUPERADOS (usa sólo estas fuentes):\n\n"
        + sources_block
        + "\n\n"
        + f"Pregunta del usuario: {query}\n\n"
        + "RESPUESTA (incluye citas inline para cada afirmación que venga de una fuente; ejemplo: 'La ley X dice... [Ministerio de Salud, p. 12]'):\n"
    )
    return prompt
