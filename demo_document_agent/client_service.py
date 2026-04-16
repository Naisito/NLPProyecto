import httpx
import logging
import os
import re 

DOC_SERVICE_URL = os.getenv("DOCUMENT_SERVICE_URL", "http://document_service:8000")

logger = logging.getLogger("demo_agent")

async def upload_file_to_backend(file_bytes, filename):
    """
    Sube el archivo. Si ya existe (409), extrae el ID real del mensaje de error del backend
    usando Regex para soportar UUIDs.
    """
    url = f"{DOC_SERVICE_URL}/documents"
    files = {'file': (filename, file_bytes, 'application/pdf')}
    
    logger.info(f"Enviando archivo {filename} a {url}...")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(url, files=files)
            
            if response.status_code == 201:
                data = response.json()
                real_id = data["document_id"]
                logger.info(f"Archivo nuevo creado. ID: {real_id}")
                return real_id
            
            elif response.status_code == 409:
                error_detail = response.json().get("detail", "")
                logger.warning(f"El backend dice que ya existe: {error_detail}")
                
                match = re.search(r"document_id=([a-zA-Z0-9_\-]+)", error_detail)
                
                if match:
                    real_id = match.group(1)
                    logger.info(f"ID recuperado del error 409: {real_id}")
                    return real_id
                else:
                    logger.error("No se pudo extraer el ID del mensaje de error 409")
                    return None

            else:
                logger.error(f"Error inesperado del backend: {response.status_code} - {response.text}")
                response.raise_for_status()
                
        except Exception as e:
            logger.error(f"Excepción conectando con document_service: {e}")
            raise e

async def get_backend_response(document_id: str, query: str, chat_context: list):
    """
    Solicita al backend que busque chunks Y genere la respuesta con su LLM interno.
    Retorna el JSON completo con 'results' y 'llm_answer'.
    
    Usa endpoint consolidado /documents/{id}/query (soporta chat opcional).
    """
    url = f"{DOC_SERVICE_URL}/documents/{document_id}/query"
    
    payload = {
        "query": query,
        "llm_answer": True,
        "chat_context": chat_context
    }
    
    logger.info(f"Consultando backend (RAG + LLM) para doc: {document_id}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload)
        
        if response.status_code == 404:
            logger.error(f"Error 404: Documento {document_id} no encontrado.")
            return {
                "results": [], 
                "llm_answer": "Error: El documento no existe en la base de datos."
            }
            
        response.raise_for_status()
        
        return response.json()


async def get_backend_response_general(session_id: str, query: str, chat_context: list):
    """
    Solicita chat general al backend sin documentos.
    Devuelve respuesta del LLM con historial.
    """
    url = f"{DOC_SERVICE_URL}/sessions/{session_id}/chat_general"
    
    payload = {
        "query": query,
        "llm_answer": True,
        "chat_context": chat_context
    }
    
    logger.info(f"Consultando backend chat general para sesión {session_id}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload)
        
        if response.status_code == 404:
            logger.error(f"Error 404 en chat general")
            return {
                "llm_answer": "Error: Sesión no encontrada."
            }
            
        response.raise_for_status()
        
        return response.json()


async def get_backend_response_multi(session_id: str, document_ids: list, query: str, chat_context: list):
    """
    Solicita búsqueda multi-documento al backend con routing automático.
    El backend determina automáticamente cuál documento es más relevante.
    Retorna el JSON con 'best_document_id', 'best_chunks', 'llm_answer', etc.
    
    Usa endpoint consolidado /sessions/{id}/query_multi (soporta chat opcional).
    """
    url = f"{DOC_SERVICE_URL}/sessions/{session_id}/query_multi"
    
    payload = {
        "query": query,
        "document_ids": document_ids,
        "llm_answer": True,
        "chat_context": chat_context
    }
    
    logger.info(f"Consultando backend multi-documento: {len(document_ids)} documentos para query: '{query}'")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload)
        
        if response.status_code == 404:
            logger.error(f"Error 404 en búsqueda multi-documento")
            return {
                "best_chunks": [], 
                "llm_answer": "Error: No se encontraron documentos."
            }
            
        response.raise_for_status()
        
        return response.json()
