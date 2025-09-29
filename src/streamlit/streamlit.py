"""Streamlit frontend for the ESP3 chatbot."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
PROJECT_ROOT = CURRENT_FILE.parents[2]
for candidate in (SRC_DIR, PROJECT_ROOT):
	candidate_str = str(candidate)
	if candidate_str not in sys.path:
		sys.path.insert(0, candidate_str)

import streamlit as st

from adapters.logger_strdin import LoggerStdin
from provider.chat_gpt import ChatGPTProvider
from provider.deepseek import DeepSeekProvider
from rag.prompts import build_synthesis_prompt
from rag.rag_orchestrator import RAGOrchestrator
from rag.re_ranker import CrossEncoderReranker
from rag.retrieve import FaissRetriever
from utils.checkpointer import CheckpointerRegister


# ---------------------------------------------------------------------------
# Configuration & Theme
# ---------------------------------------------------------------------------

st.set_page_config(
	page_title="ESP3 Chatbot",
	page_icon="🤖",
	layout="wide",
	menu_items={
		"Get Help": "https://github.com/Cedricnator/esp3-chatbot",
		"Report a bug": "https://github.com/Cedricnator/esp3-chatbot/issues",
		"About": "Asistente RAG que responde preguntas sobre los reglamentos de la UFRO.",
	},
)

THEME_COLORS = {
	"background": "#0f172a",
	"surface": "#111827",
	"card": "#1f2937",
	"accent": "#0ea5e9",
	"accent_soft": "rgba(14,165,233,0.12)",
	"text_primary": "#f9fafb",
	"text_secondary": "#cbd5f5",
}


def inject_custom_css() -> None:
	st.markdown(
		f"""
		<style>
			body {{
				background: radial-gradient(circle at top left, #0f172a 0%, #020617 55%);
				color: {THEME_COLORS['text_primary']};
			}}
			section.main > div {{
				padding-top: 2rem;
			}}
			.stApp header {{
				background: transparent;
			}}
			.stSidebar {{
				background: linear-gradient(180deg, #020617 0%, #0b1120 100%);
				border-right: 1px solid rgba(148, 163, 184, 0.12);
			}}
			.chat-container {{
				background: {THEME_COLORS['surface']};
				border-radius: 24px;
				padding: 24px 28px;
				box-shadow: 0 18px 45px rgba(8, 51, 68, 0.35);
			}}
			.stChatMessage[data-testid="user"] {{
				background: {THEME_COLORS['card']};
				border: 1px solid rgba(148, 163, 184, 0.15);
			}}
			.stChatMessage[data-testid="assistant"] {{
				background: {THEME_COLORS['accent_soft']};
				border: 1px solid rgba(14,165,233,0.3);
			}}
			.stSelectbox label,
			.stRadio label,
			.stTextInput label {{
				font-weight: 600;
				color: {THEME_COLORS['text_secondary']};
				letter-spacing: 0.01em;
			}}
			.stButton > button {{
				background: linear-gradient(135deg, #0284c7, #0ea5e9);
				color: white;
				border-radius: 10px;
				font-weight: 600;
				border: none;
				transition: transform 0.12s ease, box-shadow 0.12s ease;
			}}
			.stButton > button:hover {{
				transform: translateY(-1px);
				box-shadow: 0 12px 34px rgba(14,165,233,0.4);
			}}
		</style>
		""",
		unsafe_allow_html=True,
	)


inject_custom_css()


# ---------------------------------------------------------------------------
# Utilities & Providers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_rag_stack(logger: LoggerStdin) -> Tuple[FaissRetriever, RAGOrchestrator]:
	index_path = "data/processed/index.faiss"
	chunks_path = "data/processed/chunks.parquet"
	mapping_path = "data/processed/mapping.parquet"
	retriever = FaissRetriever(index_path, chunks_path, mapping_path, logger)
	reranker = CrossEncoderReranker(logger)
	orchestrator = RAGOrchestrator(retriever, logger, reranker)
	return retriever, orchestrator


def call_provider(
	provider_name: str,
	system_prompt: str,
	user_message: str,
	logger: LoggerStdin,
	checkpointer: CheckpointerRegister,
) -> str:
	if provider_name == "chatgpt":
		provider = ChatGPTProvider(logger)
		return str(provider.chat(system_prompt, user_message))
	provider = DeepSeekProvider(logger, checkpointer)
	return str(provider.chat(system_prompt, user_message))


def run_rag_pipeline(
	query: str,
	logger: LoggerStdin,
	top_k: int = 6,
	rerank_top_n: int = 4,
) -> Dict[str, Any]:
	_, orchestrator = get_rag_stack(logger)
	return orchestrator.run(query, k_retrieve=top_k, rerank_top_n=rerank_top_n, do_rewrite=True)


def append_to_history(role: str, content: str, metadata: Dict[str, Any] | None = None) -> None:
	if "history" not in st.session_state:
		st.session_state.history = []
	st.session_state.history.append({
		"role": role,
		"content": content,
		"meta": metadata or {},
	})


def render_message(message: Dict[str, Any]) -> None:
	role = message.get("role", "assistant")
	meta = message.get("meta", {})

	avatar = "👤" if role == "user" else "🤖"
	with st.chat_message(role, avatar=avatar):
		if role == "assistant" and "sources" in meta and meta["sources"]:
			cols = st.columns([0.72, 0.28])
			with cols[0]:
				st.markdown(message["content"], unsafe_allow_html=True)
			with cols[1]:
				with st.container(border=True):
					st.caption("Fuentes sugeridas")
					for idx, source in enumerate(meta["sources"], start=1):
						title = source.get("title") or source.get("doc_id") or f"Fragmento #{idx}"
						page = source.get("page")
						snippet = source.get("text") or source.get("snippet")
						st.markdown(
							f"**{title}**{f' · p. {page}' if page else ''}\n\n"
							f"<span style='color:#94a3b8'>{snippet[:280]}…</span>",
							unsafe_allow_html=True,
						)
		else:
			st.markdown(message["content"], unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar  (Provider selection, settings)
# ---------------------------------------------------------------------------


logger = LoggerStdin("streamlit", "logs/streamlit.log")
checkpointer = CheckpointerRegister()

with st.sidebar:
	st.image(
		"https://upload.wikimedia.org/wikipedia/commons/thumb/3/39/Emblem_of_the_Universidad_de_La_Frontera.svg/240px-Emblem_of_the_Universidad_de_La_Frontera.svg.png",
		caption="Universidad de La Frontera",
		use_column_width=True,
	)

	st.markdown(
		"""
		### ⚙️ Configuración
		Ajusta el proveedor de LLM y explora respuestas basadas en los reglamentos institucionales.
		"""
	)

	provider = st.selectbox(
		"Proveedor LLM",
		options=["deepseek", "chatgpt"],
		index=0,
		help="Elige el modelo que responderá a tus preguntas.",
	)

	st.divider()
	st.markdown(
		"""
		#### Conoce las reglas
		Este asistente usa RAG (Retrieval-Augmented Generation) sobre reglamentos académicos, financieros y de convivencia de la UFRO.
		Las respuestas incluyen fragmentos citados para transparentar el origen.
		"""
	)

	if st.button("🔄 Reiniciar conversación", use_container_width=True):
		st.session_state.history = []


# ---------------------------------------------------------------------------
# Hero Section
# ---------------------------------------------------------------------------


hero_left, hero_right = st.columns([0.58, 0.42], gap="large")
with hero_left:
	st.markdown(
		"""
		<h1 style="font-size:3rem;font-weight:800;line-height:1.1;margin-bottom:0.4em;">
			Asistente experto en reglamentos UFRO
		</h1>
		<p style="font-size:1.1rem;color:#cbd5f5;line-height:1.6;">
			Consulta artículos clave de los reglamentos de Régimen de Estudios, Admisión, 
			Finanzas y Convivencia. El sistema combina búsqueda semántica y modelos de lenguaje
			avanzados para entregar respuestas fundamentadas y fáciles de citar.
		</p>
		""",
		unsafe_allow_html=True,
	)

	st.markdown(
		"""
		<div style="display:flex;gap:12px;">
			<div style="padding:10px 16px;background:rgba(14,165,233,0.15);border-radius:999px;color:#38bdf8;font-weight:600;">
				🔒 Datos institucionales
			</div>
			<div style="padding:10px 16px;background:rgba(56,189,248,0.08);border-radius:999px;color:#bae6fd;font-weight:600;">
				📚 Citas automáticas
			</div>
		</div>
		""",
		unsafe_allow_html=True,
	)

with hero_right:
	st.markdown(
		"""
		<div style="background:linear-gradient(180deg, rgba(14,165,233,0.35) 0%, rgba(14,165,233,0) 100%);
					padding:32px;border-radius:24px;height:100%;display:flex;flex-direction:column;justify-content:center;">
			<h3 style="color:#e0f2fe;margin-bottom:12px;">¿Qué puedes preguntar?</h3>
			<ul style="color:#cbd5f5;line-height:1.8;margin:0;padding-left:18px;font-size:1rem;">
				<li>Criterios de aprobación y escalas de notas.</li>
				<li>Requisitos de admisión especial por vía técnica.</li>
				<li>Beneficios y obligaciones financieras.</li>
				<li>Protocolos de convivencia y resolución de conflictos.</li>
			</ul>
		</div>
		""",
		unsafe_allow_html=True,
	)


st.markdown("---")


# ---------------------------------------------------------------------------
# Chat Experience
# ---------------------------------------------------------------------------


st.markdown("### 💬 Conversa con ESP3")

if "history" not in st.session_state:
	append_to_history(
		"assistant",
		"Hola 👋 Soy tu asistente UFRO. Pregúntame por artículos específicos o reglas y te responderé citando las fuentes pertinentes.",
	)

with st.container():
	st.markdown('<div class="chat-container">', unsafe_allow_html=True)
	for msg in st.session_state.history:
		render_message(msg)
	st.markdown('</div>', unsafe_allow_html=True)


# User Input
user_prompt = st.chat_input("Escribe tu consulta sobre los reglamentos de la universidad")

if user_prompt:
	append_to_history("user", user_prompt)

	with st.spinner("Consultando reglamentos y preparando respuesta..."):
		rag_result = run_rag_pipeline(user_prompt, logger)
		hints = rag_result.get("hints", [])
		rewritten_query = rag_result.get("query", user_prompt)
		synthesis_prompt = build_synthesis_prompt(rewritten_query, hints)
		response_text = call_provider(provider, synthesis_prompt, user_prompt, logger, checkpointer)

	append_to_history(
		"assistant",
		response_text,
		metadata={"sources": hints, "provider": provider},
	)



# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------


st.markdown(
	"""
	<div style="margin-top:2.5rem;text-align:center;color:#64748b;font-size:0.9rem;">
		Construido por el equipo ESP3 · Basado en RAG · {year}
	</div>
	""".format(year=os.getenv("CURRENT_YEAR", "2025")),
	unsafe_allow_html=True,
)

