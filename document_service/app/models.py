from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, List, Optional

class DocumentCreateResponse(BaseModel):
    document_id: str
    filename: str
    text_snippet: str = Field(..., description="Primeros 500 caracteres del texto extraído")
    created_at: datetime
    summary_short: str
    summary_long: str

class DocumentRetrieveResponse(BaseModel):
    document_id: str
    filename: str
    full_text: str
    created_at: datetime
    summary_short: str
    summary_long: str

class DocumentQueryRequest(BaseModel):
    query: str
    llm_answer: bool

class DocumentQueryResponse(BaseModel):
    results: List[str]
    llm_answer: str

class DocumentQueryAndChatRequest(BaseModel):
    query: str
    llm_answer: bool
    chat_context: List[Dict[str, str]] = []

class DocumentQueryAndChatResponse(BaseModel):
    results: List[str]
    llm_answer: str

# Nuevos modelos para multi-documento
class DocumentQueryMultiRequest(BaseModel):
    """Búsqueda en múltiples documentos con routing automático."""
    query: str
    document_ids: List[str]  # IDs de documentos en sesión
    llm_answer: bool = False

class DocumentQueryMultiResponse(BaseModel):
    """Respuesta de búsqueda multi-documento."""
    best_document_id: str
    best_document_filename: Optional[str] = None
    results_by_document: Dict[str, List[str]]
    best_chunks: List[str]
    llm_answer: str = ""

class DocumentQueryMultiChatRequest(BaseModel):
    """Búsqueda multi-documento con chat history."""
    query: str
    document_ids: List[str]
    llm_answer: bool = False
    chat_context: List[Dict[str, str]] = []

class DocumentQueryMultiChatResponse(BaseModel):
    """Respuesta de búsqueda multi-documento con chat."""
    best_document_id: str
    best_document_filename: Optional[str] = None
    results_by_document: Dict[str, List[str]]
    best_chunks: List[str]
    llm_answer: str = ""