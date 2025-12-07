from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal
from dotenv import load_dotenv
import logging
import asyncio

from provider.chat_gpt import ChatGPTProvider
from provider.deepseek import DeepSeekProvider
from adapters.logger_strdin import LoggerStdin
from rag.rag_orchestrator import RAGOrchestrator
from rag.retrieve import FaissRetriever
from rag.re_ranker import CrossEncoderReranker
from rag.prompts import build_synthesis_prompt
from utils.checkpointer import CheckpointerRegister

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Recursos compartidos de la aplicación
shared_resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestión del ciclo de vida de la aplicación.
    Código antes del yield se ejecuta al inicio, después del yield al cerrar.
    """
    # Startup: Inicializar componentes
    logger.info("🚀 Aplicación iniciándose...")
    logger.info("Inicializando componentes del chatbot...")
    
    # Configurar rutas
    faiss_index_path = "data/processed/index.faiss"
    chunks_path = "data/processed/chunks.parquet"
    mapping_path = "data/processed/mapping.parquet"
    
    # Inicializar RAG
    rag_logger = LoggerStdin("rag_logger", "logs/api_rag.log")
    retriever = FaissRetriever(
        faiss_index_path,
        chunks_path,
        mapping_path,
        rag_logger
    )
    reranker = CrossEncoderReranker(rag_logger)
    rag_orchestrator = RAGOrchestrator(
        retriever,
        rag_logger,
        reranker
    )
    
    # Loggers para proveedores
    deepseek_logger = LoggerStdin("deepseek_api_logger", "logs/api_deepseek.log")
    chatgpt_logger = LoggerStdin("chatgpt_api_logger", "logs/api_chatgpt.log")
    
    # Guardar en recursos compartidos
    shared_resources["rag_orchestrator"] = rag_orchestrator
    shared_resources["deepseek_logger"] = deepseek_logger
    shared_resources["chatgpt_logger"] = chatgpt_logger
    shared_resources["initialized"] = True
    
    logger.info("✓ Componentes inicializados correctamente")
    
    yield  # La aplicación se ejecuta aquí
    
    # Shutdown: Limpiar recursos
    logger.info("Aplicación cerrándose...")
    shared_resources.clear()
    logger.info("✓ Recursos liberados correctamente")

# Crear aplicación FastAPI con lifespan
app = FastAPI(
    title="Chatbot Universitario API",
    description="API REST para interactuar con el chatbot de información universitaria usando RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic para validación
class ChatRequest(BaseModel):
    """Modelo de solicitud para el endpoint de chat."""
    message: str = Field(..., description="Mensaje del usuario", min_length=1, max_length=2000)
    provider: Optional[Literal["chatgpt", "deepseek"]] = Field(
        default="chatgpt",
        description="Proveedor de LLM a utilizar (opcional, por defecto chatgpt)"
    )
    use_rag: Optional[bool] = Field(
        default=True,
        description="Si se debe usar RAG para recuperar contexto (opcional, por defecto True)"
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Número de documentos a recuperar con RAG"
    )

class ChatResponse(BaseModel):
    """Modelo de respuesta del endpoint de chat."""
    response: str = Field(..., description="Respuesta generada por el chatbot")
    provider: str = Field(..., description="Proveedor utilizado")
    rag_used: bool = Field(..., description="Indica si se usó RAG")
    context_chunks: Optional[int] = Field(None, description="Número de chunks de contexto usados")

class HealthResponse(BaseModel):
    """Modelo de respuesta para health check."""
    status: str
    service: str
    version: str

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """
    Health check endpoint.
    Verifica que la API esté funcionando correctamente.
    """
    return HealthResponse(
        status="healthy",
        service="Chatbot Universitario API",
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check detallado.
    Verifica que todos los componentes estén inicializados.
    """
    if not shared_resources.get("initialized", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Componentes no inicializados"
        )
    
    return HealthResponse(
        status="healthy",
        service="Chatbot Universitario API",
        version="1.0.0"
    )

@app.post("/ask", response_model=ChatResponse, tags=["Ask"])
async def chat(request: ChatRequest):
    """
    Endpoint principal para interactuar con el chatbot.
    
    Envía un mensaje y recibe una respuesta generada por el LLM,
    opcionalmente usando RAG para recuperar contexto relevante.
    
    Args:
        request: Objeto ChatRequest con el mensaje y configuración
        
    Returns:
        ChatResponse con la respuesta del chatbot y metadatos
        
    Raises:
        HTTPException: Si ocurre un error durante el procesamiento
    """
    try:
        logger.info(f"Recibida solicitud: mensaje='{request.message[:50]}...', provider={request.provider}, rag={request.use_rag}")
        
        # Preparar checkpoint
        checkpoint = CheckpointerRegister()
        system_prompt = "Eres un asistente útil especializado en información universitaria."
        context_chunks = None
        
        # Ejecutar RAG si está habilitado
        if request.use_rag:
            logger.info(f"Ejecutando RAG con top_k={request.top_k}")
            rag_orchestrator = shared_resources["rag_orchestrator"]
            # Ejecutar RAG en thread separado para no bloquear
            rag_result = await asyncio.to_thread(
                rag_orchestrator.run,
                request.message,
                k_retrieve=request.top_k,
                rerank_top_n=5,
                do_rewrite=True
            )
            system_prompt = build_synthesis_prompt(rag_result['query'], rag_result['hints'])
            context_chunks = len(rag_result['hints']) if 'hints' in rag_result else 0
            logger.info(f"RAG completado, {context_chunks} chunks recuperados")
        
        # Seleccionar proveedor y generar respuesta
        if request.provider == "deepseek":
            deepseek_logger = shared_resources["deepseek_logger"]
            provider = DeepSeekProvider(deepseek_logger, checkpoint)
            # Ejecutar llamada al LLM en thread separado
            response_text = await asyncio.to_thread(
                provider.chat,
                system_prompt,
                request.message
            )
        elif request.provider == "chatgpt":
            chatgpt_logger = shared_resources["chatgpt_logger"]
            provider = ChatGPTProvider(chatgpt_logger, checkpoint)
            # Ejecutar llamada al LLM en thread separado
            response_text = await asyncio.to_thread(
                provider.chat,
                system_prompt,
                request.message
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Proveedor no válido: {request.provider}"
            )
        
        logger.info(f"Respuesta generada exitosamente con {request.provider}")
        
        return ChatResponse(
            response=response_text,
            provider=request.provider or "chatgpt",
            rag_used=request.use_rag if request.use_rag is not None else True,
            context_chunks=context_chunks
        )
        
    except FileNotFoundError as e:
        logger.error(f"Error: Archivo no encontrado - {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Componentes RAG no disponibles. Asegúrate de ejecutar la ingesta de datos primero."
        )
    except Exception as e:
        logger.error(f"Error procesando solicitud: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor: {str(e)}"
        )

@app.post("/ask/simple", tags=["Ask"])
async def chat_simple(message: str, provider: str = "deepseek"):
    """
    Endpoint simplificado para chat rápido.
    
    Acepta parámetros de query string para facilitar pruebas.
    
    Args:
        message: Mensaje del usuario
        provider: Proveedor LLM (chatgpt o deepseek)
        
    Returns:
        Respuesta del chatbot como texto plano
    """
    if not message or len(message.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El mensaje no puede estar vacío"
        )
    
    if provider not in ["chatgpt", "deepseek"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Proveedor debe ser 'chatgpt' o 'deepseek'"
        )
    
    # Cast explícito para satisfacer el type checker
    provider_typed: Literal["chatgpt", "deepseek"] = provider  # type: ignore
    
    request = ChatRequest(
        message=message,
        provider=provider_typed,
        use_rag=True,
        top_k=5
    )
    
    result = await chat(request)
    return {"response": result.response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=33201)
