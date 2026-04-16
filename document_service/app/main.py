import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from app import extractor, storage, utils
from app import llm_engine
from app.models import DocumentCreateResponse, DocumentQueryAndChatResponse, DocumentRetrieveResponse, DocumentQueryRequest, DocumentQueryResponse, DocumentQueryAndChatRequest, DocumentQueryMultiRequest, DocumentQueryMultiResponse, DocumentQueryMultiChatRequest, DocumentQueryMultiChatResponse
import logging
from app.rag_engine import RagService
from app.infra.embeddings_local import LocalHuggingFaceEmbeddings
from app.infra.vector_chroma import LocalChromaIndex
from app.config import settings
from app.llm_engine import short_summary, map_summary_chunk, reduce_summaries, summarize_with_map_reduce, generate_answer, optimize_chat_history

# Configurar logging desde config.json (server.log_level)
_log_level_str = (settings._config.get("server", {}) or {}).get("log_level", "INFO")
_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}
logging.basicConfig(
    level=_LEVELS.get(str(_log_level_str).upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("doc_service")

try:
    emb_conf = settings.embeddings
    embedder = LocalHuggingFaceEmbeddings(
        model_name=emb_conf.get("model_name", "BAAI/bge-m3"),
        cache_dir=emb_conf.get("cache_dir", "models_cache")
    )

    vec_conf = settings.vector_db
    vector_store = LocalChromaIndex(
        db_path=vec_conf.get("path", "db/chroma_db"),
        collection_name=vec_conf.get("collection_name", "documents_rag")
    )
    
    rag_conf = settings.rag
    rag_service = RagService(
        embedder=embedder, 
        vector_store=vector_store,
        chunk_size=rag_conf.get("chunk_size", 500),
        overlap=rag_conf.get("chunk_overlap", 50),
        n_results_default=rag_conf.get("search_results_default", 3)
    )

except Exception as e:
    logger.error(f"Error fatal iniciando servicios de IA: {e}")
    raise e

app = FastAPI(title="document_service", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    # Log de server config (host/port solo informativo; los gestiona uvicorn/docker)
    _srv = settings._config.get("server", {}) or {}
    logger.info(f"Iniciando BD | Server config host={_srv.get('host','0.0.0.0')} port={_srv.get('port',8000)} level={_log_level_str}")
    storage.init_db()
    logger.info("BD iniciada correctamente")

ALLOWED_EXT = set(settings.storage.get("allowed_extensions", [".pdf", ".txt", ".docx",".png", ".jpg", ".jpeg"]))

def validate_filename(filename: str):
    if not filename:
        raise HTTPException(status_code=400, detail="Filename missing")
    if not any(filename.lower().endswith(ext) for ext in ALLOWED_EXT):
        raise HTTPException(status_code=400, detail=f"Unsupported file type; allowed: {ALLOWED_EXT}")

#Endpoint para listar todos los documentos
@app.get("/documents")
def list_docs():
    logger.info("Listando documentos")
    return storage.list_documents()

#Endpoint de salud
@app.get("/health")
def health():
    return {"status": "ok", "service": "document_service"}

#Endpoint para subir un documento
@app.post("/documents", response_model=DocumentCreateResponse, status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...)
):
    filename = file.filename
    logger.info(f"Recibiendo archivo: {filename}")

    validate_filename(filename)
    
    file_bytes = await file.read()
    
    file_hash = await asyncio.to_thread(utils.compute_file_hash, file_bytes)
    
    existing = await asyncio.to_thread(storage.find_document_by_hash, file_hash)
    if existing:
        logger.warning(f"Se ha intentado subir un archivo duplicado: {file_hash}")
        raise HTTPException(
            status_code=409,
            detail=f"Document already exists (document_id={existing['document_id']})"
        )

    logger.info("Iniciando extraccion de texto")

    if filename.lower().endswith(".pdf"):
        text = await asyncio.to_thread(extractor.extract_text_from_pdf_bytes, file_bytes)
    elif filename.lower().endswith(".docx"):
        text = await asyncio.to_thread(extractor.extract_text_from_docx_bytes, file_bytes)
    elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
        text = await asyncio.to_thread(extractor.extract_text_from_image_bytes, file_bytes)
    else:
        text = await asyncio.to_thread(extractor.extract_text_from_txt_bytes, file_bytes)

    if not text:
        text = ""

    logger.info(f"Texto extraido correctamente. Longitud: {len(text)}")
    
    logger.info("Generando resumen corto y largo")


    summary_long_txt = await asyncio.to_thread(
        summarize_with_map_reduce, 
        text=text,
        chunk_size=15000
    )

    logger.info("Generando resumen corto basado en el largo")
    summary_short_txt = await asyncio.to_thread(short_summary, summary_long_txt)

    document_id = utils.generate_document_id()
    file_path = await asyncio.to_thread(storage.save_file_to_disk, filename, file_bytes)
    
    created_at = await asyncio.to_thread(
        storage.save_document_record,
        document_id=document_id, 
        filename=filename, 
        file_path=file_path, 
        file_hash=file_hash, 
        text=text,
        summary_short=summary_short_txt, 
        summary_long=summary_long_txt    
    )

    background_tasks.add_task(rag_service.index_document, document_id, text)
    background_tasks.add_task(rag_service.index_summary, document_id, summary_short_txt, filename)

    return DocumentCreateResponse(
        document_id=document_id,
        filename=filename,
        text_snippet=text[:500],
        created_at=created_at,
        summary_short=summary_short_txt,
        summary_long=summary_long_txt
    )

#Endpoint para obtener un documento por ID
@app.get("/documents/{document_id}", response_model=DocumentRetrieveResponse)
def get_document(document_id: str):

    logger.info(f"Buscando documento con ID: {document_id}")

    doc = storage.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    logger.info("Documento encontrado correctamente")

    return DocumentRetrieveResponse(
        document_id=doc["document_id"],
        filename=doc["filename"],
        full_text=doc["text"],
        created_at=doc["created_at"],
        summary_short=doc["summary_short"],
        summary_long=doc["summary_long"]
    )

#Endpoint para consultar un documento (soporta chat opcional)
@app.post("/documents/{document_id}/query", response_model=DocumentQueryAndChatResponse)
async def query_document(body: DocumentQueryAndChatRequest, document_id: str = None):
    """
    Query a document with RAG. Supports both:
    - Simple query (chat_context = None or [])
    - Conversational query (chat_context with history)
    """
    is_global = not document_id or document_id.lower() == "all"
    results = []
    long_summary_text = "" 

    # RAG Search
    if is_global:
        logger.info(f"Búsqueda GLOBAL: {body.query}")
        results = await asyncio.to_thread(rag_service.search, "all", body.query)
        long_summary_text = "Búsqueda Global: Esta consulta se realiza sobre toda la base de conocimientos disponible."
    else:
        logger.info(f"Consultando documento {document_id}: {body.query}")
        doc = storage.get_document(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        long_summary_text = doc.get("summary_long", "")
        results = await asyncio.to_thread(rag_service.search, document_id, body.query)

    llm_answer_text = ""
    
    if body.llm_answer:
        if results or body.chat_context or is_global:
            try:
                # Optimizar historial si es muy largo
                chat_context = body.chat_context or []
                if chat_context:
                    TOKEN_LIMIT_THRESHOLD = 3000
                    CHARS_PER_TOKEN = 3.5
                    CHAR_LIMIT = int(TOKEN_LIMIT_THRESHOLD * CHARS_PER_TOKEN)
                    
                    current_history_chars = sum(
                        len(turn.get("user", "") or "") + len(turn.get("bot", "") or "")
                        for turn in chat_context
                    )
                    
                    if current_history_chars > CHAR_LIMIT:
                        logger.info(f"Historial pesado (~{int(current_history_chars/4)} tokens). Resumiendo...")
                        chat_context = await asyncio.to_thread(
                            llm_engine.optimize_chat_history,
                            chat_context
                        )
                
                # Generar respuesta (con o sin chat)
                llm_answer_text = await asyncio.to_thread(
                    llm_engine.generate_answer,
                    body.query,
                    long_summary_text,
                    results,
                    chat_context if chat_context else None
                )
            except Exception as e:
                logger.error(f"Error generando LLM response: {e}")
                llm_answer_text = "Error generando respuesta automática."
        else:
            llm_answer_text = "No encontré información suficiente en el documento para responder."

    return DocumentQueryAndChatResponse(results=results, llm_answer=llm_answer_text)

#Endpoint para borrar un documento por ID
@app.delete("/documents/{document_id}")
def delete_document(document_id: str):

    logger.info(f"Borrando documento con ID: {document_id}")

    deleted = storage.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    
    rag_service.delete_document_index(document_id)
    
    logger.info("Documento borrado correctamente")
                
    return {"status": "deleted", "document_id": document_id}

# ============================================================================
# ENDPOINTS PARA SESIONES MULTI-DOCUMENTO
# ============================================================================

@app.post("/sessions/{session_id}/query_multi", response_model=DocumentQueryMultiChatResponse)
async def query_multi_documents(body: DocumentQueryMultiChatRequest, session_id: str):
    """
    Multi-document search with semantic routing. Supports:
    - Simple query (chat_context = None or [])
    - Conversational query (chat_context with history)
    
    Flow:
    1. Router: Semantic search in summaries → Best document
    2. Primary: Search chunks in best document
    3. Secondary: Search in other documents (score > 0.3)
    4. LLM: Generate answer with best context
    """
    if not body.document_ids:
        raise HTTPException(status_code=400, detail="Se requiere al menos un document_id")
    
    logger.info(f"Multi-search sesión {session_id}: {len(body.document_ids)} docs")
    
    # Ejecutar búsqueda multi-documento
    search_results = await asyncio.to_thread(rag_service.search_multi_document, body.document_ids, body.query)

    best_doc_id = search_results.get("best_doc")
    best_chunks = search_results.get("best_chunks", [])
    results_by_doc = search_results.get("results_by_doc", {})

    # Construir lista combinada de chunks (prioriza best_doc, luego resto) respetando presupuesto ya aplicado en RAG
    combined_chunks = []
    if results_by_doc:
        # Orden: best_doc primero, luego otros por clave
        ordered_docs = [d for d in [best_doc_id] if d] + [d for d in sorted(results_by_doc.keys()) if d != best_doc_id]
        for d in ordered_docs:
            chunks_list = results_by_doc.get(d, [])
            if chunks_list:
                combined_chunks.extend(chunks_list)
    
    # Obtener metadata del documento más relevante
    best_filename = None
    if best_doc_id:
        best_doc_meta = storage.get_document(best_doc_id)
        best_filename = best_doc_meta.get("filename") if best_doc_meta else None
    
    llm_answer_text = ""
    if combined_chunks:
        # Usar resumen del mejor documento para contexto
        summary_long = ""
        if best_doc_id:
            best_doc_meta = storage.get_document(best_doc_id)
            summary_long = best_doc_meta.get("summary_long", "") if best_doc_meta else ""
        
        # Chat context (opcional)
        chat_context = body.chat_context if hasattr(body, 'chat_context') else None
            
        try:
            llm_answer_text = await asyncio.to_thread(
                llm_engine.generate_answer,
                body.query,
                summary_long,
                combined_chunks,
                chat_context
            )
        except Exception as e:
            logger.error(f"Error generando respuesta LLM: {e}")
            llm_answer_text = "Error generando respuesta."
    
    return DocumentQueryMultiChatResponse(
        best_document_id=best_doc_id or "unknown",
        best_document_filename=best_filename,
        results_by_document=results_by_doc,
        best_chunks=best_chunks,
        llm_answer=llm_answer_text
    )


# Endpoint para chat general (sin documentos, solo LLM)
@app.post("/sessions/{session_id}/chat_general", response_model=DocumentQueryAndChatResponse)
async def chat_general(body: DocumentQueryAndChatRequest, session_id: str):
    """
    Chat general sin documentos. Solo consulta LLM sin RAG.
    """
    logger.info(f"Chat general (sin docs) para sesión {session_id}")
    
    llm_answer_text = ""
    
    try:
        # Llamar LLM sin contexto de documentos
        llm_answer_text = await asyncio.to_thread(
            llm_engine.generate_answer,
            body.query,
            "",  # Sin resumen
            [],  # Sin chunks
            body.chat_context if body.chat_context else []
        )
    except Exception as e:
        logger.error(f"Error en LLM general: {e}")
        llm_answer_text = "Error generando respuesta."
    
    return DocumentQueryAndChatResponse(
        results=[],
        llm_answer=llm_answer_text
    )


