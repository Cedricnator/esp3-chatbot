from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal
from dotenv import load_dotenv
import logging

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

# Crear aplicación FastAPI
app = FastAPI(
    title="Chatbot Universitario API",
    description="API REST para interactuar con el chatbot de información universitaria usando RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar orígenes exactos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic para validación
class ChatRequest(BaseModel):
    """Modelo de solicitud para el endpoint de chat."""
    message: str = Field(..., description="Mensaje del usuario", min_length=1, max_length=2000)
    provider: Literal["chatgpt", "deepseek"] = Field(
        default="deepseek",
        description="Proveedor de LLM a utilizar"
    )
    use_rag: bool = Field(
        default=True,
        description="Si se debe usar RAG para recuperar contexto"
    )
    top_k: int = Field(
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

# Inicializar componentes (singleton pattern)
class ChatbotComponents:
    """Componentes singleton del chatbot."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def initialize(self):
        """Inicializar componentes una sola vez."""
        if self._initialized:
            return
        
        logger.info("Inicializando componentes del chatbot...")
        
        # Configurar rutas
        self.faiss_index_path = "data/processed/index.faiss"
        self.chunks_path = "data/processed/chunks.parquet"
        self.mapping_path = "data/processed/mapping.parquet"
        
        # Inicializar RAG
        rag_logger = LoggerStdin("rag_logger", "logs/api_rag.log")
        self.retriever = FaissRetriever(
            self.faiss_index_path,
            self.chunks_path,
            self.mapping_path,
            rag_logger
        )
        self.reranker = CrossEncoderReranker(rag_logger)
        self.rag_orchestrator = RAGOrchestrator(
            self.retriever,
            rag_logger,
            self.reranker
        )
        
        # Loggers para proveedores
        self.deepseek_logger = LoggerStdin("deepseek_api_logger", "logs/api_deepseek.log")
        self.chatgpt_logger = LoggerStdin("chatgpt_api_logger", "logs/api_chatgpt.log")
        
        self._initialized = True
        logger.info("✓ Componentes inicializados correctamente")

# Instancia global de componentes
components = ChatbotComponents()

@app.on_event("startup")
async def startup_event():
    """Inicializar componentes al arrancar la aplicación."""
    components.initialize()

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
    if not components._initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Componentes no inicializados"
        )
    
    return HealthResponse(
        status="healthy",
        service="Chatbot Universitario API",
        version="1.0.0"
    )

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
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
            rag_result = components.rag_orchestrator.run(
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
            provider = DeepSeekProvider(components.deepseek_logger, checkpoint)
            response_text = provider.chat(system_prompt, request.message)
        elif request.provider == "chatgpt":
            provider = ChatGPTProvider(components.chatgpt_logger, checkpoint)
            response_text = provider.chat(system_prompt, request.message)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Proveedor no válido: {request.provider}"
            )
        
        logger.info(f"Respuesta generada exitosamente con {request.provider}")
        
        return ChatResponse(
            response=response_text,
            provider=request.provider,
            rag_used=request.use_rag,
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

@app.post("/chat/simple", tags=["Chat"])
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
