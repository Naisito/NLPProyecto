import logging
import uuid
import json
import os
import asyncio
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn

from client_service import upload_file_to_backend, get_backend_response, get_backend_response_multi, get_backend_response_general

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] AGENT: %(message)s")
logger = logging.getLogger("demo_agent")

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "sessions.json")

app = FastAPI(title="Demo Document Agent", version="2.0.0")


class Session(BaseModel):
    session_id: str
    document_ids: List[str] = []  # Lista de IDs de documentos (multi-documento)
    filenames: Dict[str, str] = {}  # {document_id: filename}
    chat_history: List[Dict] = []

class CreateSessionResponse(BaseModel):
    session_id: str
    message: str

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    used_chunks: List[str] = []
    results_by_document: Dict[str, List[str]] = {}


def load_sessions() -> Dict[str, Session]:
    if not os.path.exists(DATA_FILE):
        return {}
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {k: Session(**v) for k, v in data.items()}
    except Exception as e:
        logger.error(f"Error cargando sesiones: {e}")
        return {}

def save_sessions():
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        data_to_save = {k: v.model_dump() for k, v in sessions_db.items()}
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error guardando sesiones: {e}")

sessions_db: Dict[str, Session] = load_sessions()

#Endpoint para listar todas las sesiones
@app.get("/sessions")
def list_all_sessions():
    """NUEVO: Devuelve una lista resumen de todas las sesiones con múltiples documentos."""
    summary = []
    for s in sessions_db.values():
        filenames_list = [s.filenames.get(doc_id, f"Doc {doc_id[:8]}") for doc_id in s.document_ids]
        filenames_str = ", ".join(filenames_list) if filenames_list else "Sin documentos"
        
        summary.append({
            "session_id": s.session_id,
            "filenames": filenames_str,
            "document_count": len(s.document_ids),
            "messages_count": len(s.chat_history)
        })
    return summary

#Endpoint para crear una nueva sesión
@app.post("/sessions", response_model=CreateSessionResponse)
def create_session():
    s_id = str(uuid.uuid4())
    new_session = Session(session_id=s_id)
    sessions_db[s_id] = new_session
    save_sessions()
    logger.info(f"Nueva sesión creada: {s_id}")
    return CreateSessionResponse(session_id=s_id, message="Sesión creada.")

#Endpoint para borrar una sesión
@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """NUEVO: Borra la sesión y actualiza el JSON."""
    global sessions_db
    if session_id in sessions_db:
        del sessions_db[session_id]
        save_sessions()
        logger.info(f"Sesión eliminada: {session_id}")
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")

# Endpoint para subir un documento a una sesión
@app.post("/sessions/{session_id}/upload_document")
async def upload_document(session_id: str, file: UploadFile = File(...)):
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    file_bytes = await file.read()
    try:
        real_doc_id = await upload_file_to_backend(file_bytes, file.filename)
        if not real_doc_id:
             raise HTTPException(500, "Error recuperando ID del documento.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error backend: {str(e)}")
    
    # Agregar documento a la lista si no está ya
    session = sessions_db[session_id]
    if real_doc_id not in session.document_ids:
        session.document_ids.append(real_doc_id)
    
    # Registrar el filename
    session.filenames[real_doc_id] = file.filename
    save_sessions()
    
    return {"status": "success", "document_id": real_doc_id, "filename": file.filename, "total_docs": len(session.document_ids)}

#Endpoint para chatear dentro de una sesión
@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(session_id: str, body: ChatRequest):
    global sessions_db 
    if session_id not in sessions_db:
        sessions_db = load_sessions()
        if session_id not in sessions_db:
             raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    session = sessions_db[session_id]
    if not session.document_ids:
        raise HTTPException(status_code=400, detail="Primero sube un documento.")
    
    query = body.query
    
    sanitized_history = []
    for msg in session.chat_history:
        sanitized_history.append({
            "user": msg.get("user", ""),
            "bot": msg.get("bot", "")
        })

    try:
        backend_data = await get_backend_response_multi(
            session_id=session_id,
            document_ids=session.document_ids, 
            query=query,
            chat_context=sanitized_history 
        )
    except Exception as e:
        logger.error(f"Error backend: {e}")
        raise HTTPException(status_code=500, detail="Error consultando servicio.")

    real_answer = backend_data.get("llm_answer", "")
    chunks = backend_data.get("best_chunks", [])
    results_by_doc = backend_data.get("results_by_document", {})
    best_doc_id = backend_data.get("best_document_id", "")
    best_filename = backend_data.get("best_document_filename", "")

    if not real_answer: real_answer = "No se encontró información."

    session.chat_history.append({
        "user": query, 
        "bot": real_answer, 
        "chunks": chunks,
        "results_by_document": results_by_doc,
        "source_doc": best_doc_id,
        "source_filename": best_filename
    })
    save_sessions()

    return ChatResponse(answer=real_answer, used_chunks=chunks, results_by_document=results_by_doc)

# Endpoint para obtener detalles de una sesión específica
@app.get("/sessions/{session_id}")
def get_session_details(session_id: str):
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    return sessions_db[session_id]

# Endpoint para chat general (sin documentos)
@app.post("/sessions/{session_id}/chat_general", response_model=ChatResponse)
async def chat_general(session_id: str, body: ChatRequest):
    """Chat general sin documentos. Solo LLM, sin RAG."""
    global sessions_db
    if session_id not in sessions_db:
        sessions_db = load_sessions()
        if session_id not in sessions_db:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    session = sessions_db[session_id]
    query = body.query
    
    sanitized_history = []
    for msg in session.chat_history:
        sanitized_history.append({
            "user": msg.get("user", ""),
            "bot": msg.get("bot", "")
        })

    try:
        backend_data = await get_backend_response_general(
            session_id=session_id,
            query=query,
            chat_context=sanitized_history
        )
    except Exception as e:
        logger.error(f"Error en chat general: {e}")
        raise HTTPException(status_code=500, detail="Error consultando servicio.")

    real_answer = backend_data.get("llm_answer", "")

    if not real_answer:
        real_answer = "No se pudo generar una respuesta."

    session.chat_history.append({
        "user": query, 
        "bot": real_answer,
        "chunks": [],
        "source_doc": None,
        "source_filename": None
    })
    save_sessions()

    return ChatResponse(answer=real_answer, used_chunks=[])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)