# 🤖 Demo Document Agent

**Servicio de Orquestación y Gestión de Sesiones Multi-Documento**

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Arquitectura](#-arquitectura)
- [Endpoints](#-endpoints)
- [Flujos Detallados](#-flujos-detallados)
- [Modelos de Datos](#-modelos-de-datos)
- [Configuración](#-configuración)
- [Ejemplos de Uso](#-ejemplos-de-uso)

---

## 🎯 Descripción

El **Demo Document Agent** actúa como **capa de orquestación** entre el cliente (GUI) y el Document Service (RAG backend). Sus responsabilidades principales son:

- ✅ **Gestión de Sesiones:** Crear, listar, eliminar sesiones
- ✅ **Tracking N:M:** Una sesión puede tener múltiples documentos
- ✅ **Persistencia:** Historial de chat guardado en `sessions.json`
- ✅ **Routing:** Dirige peticiones al backend apropiado
- ✅ **Error Handling:** Maneja duplicados (409) y errores de conexión

---

## 🏗️ Arquitectura

```
┌────────────────────────────────────────────────────────────┐
│                    DEMO DOCUMENT AGENT                      │
│                        (Port 8001)                          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                     main.py                          │  │
│  │                                                       │  │
│  │  • FastAPI application                               │  │
│  │  • Session management endpoints                      │  │
│  │  • Request validation (Pydantic)                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                 │
│                            ↓                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                 client_service.py                     │  │
│  │                                                       │  │
│  │  • HTTP client (httpx)                               │  │
│  │  • Backend communication                             │  │
│  │  • Error handling (409 duplicate detection)          │  │
│  │  • Timeout management                                │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                 │
│                            ↓                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               data/sessions.json                      │  │
│  │                                                       │  │
│  │  {                                                    │  │
│  │    "session-uuid": {                                 │  │
│  │      "session_id": "uuid",                           │  │
│  │      "document_ids": ["doc1", "doc2"],               │  │
│  │      "filenames": {"doc1": "file.pdf", ...},         │  │
│  │      "chat_history": [                               │  │
│  │        {                                             │  │
│  │          "user": "question",                         │  │
│  │          "bot": "answer",                            │  │
│  │          "chunks": [...],                            │  │
│  │          "source_doc": "doc1",                       │  │
│  │          "source_filename": "file.pdf"               │  │
│  │        }                                             │  │
│  │      ]                                               │  │
│  │    }                                                  │  │
│  │  }                                                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       │ HTTP REST
                       │
                       ↓
         ┌──────────────────────────────┐
         │   DOCUMENT SERVICE           │
         │   (Port 8000)                │
         │                              │
         │   • Document upload          │
         │   • RAG search               │
         │   • LLM generation           │
         └──────────────────────────────┘
```

---

## 🔌 Endpoints

### **Resumen**

| Método | Endpoint | Descripción | Autenticación |
|--------|----------|-------------|---------------|
| GET | `/sessions` | Lista todas las sesiones | No |
| POST | `/sessions` | Crea nueva sesión | No |
| GET | `/sessions/{id}` | Obtiene detalles de sesión | No |
| DELETE | `/sessions/{id}` | Elimina sesión | No |
| POST | `/sessions/{id}/upload_document` | Sube documento | No |
| POST | `/sessions/{id}/chat` | Chat multi-documento | No |
| POST | `/sessions/{id}/chat_general` | Chat sin documentos | No |

---

## 🔄 Flujos Detallados

### **Endpoint 1: `GET /sessions`**

**Descripción:** Lista todas las sesiones con información resumida

```
┌─────────────┐
│   CLIENT    │
└──────┬──────┘
       │
       │ GET /sessions
       ↓
┌─────────────────────────────────────────────────────────┐
│  AGENT: GET /sessions                                    │
│                                                           │
│  1. Load sessions from file                              │
│     ┌──────────────────────────────────────────┐        │
│     │  load_sessions()                         │        │
│     │    │                                      │        │
│     │    ├─→ Check if sessions.json exists     │        │
│     │    │                                      │        │
│     │    ├─→ Read JSON file                    │        │
│     │    │                                      │        │
│     │    └─→ Parse into Session objects        │        │
│     │         {                                 │        │
│     │           "abc-123": Session(...),        │        │
│     │           "def-456": Session(...)         │        │
│     │         }                                 │        │
│     └──────────────────────────────────────────┘        │
│                                                           │
│  2. Build summary for each session                       │
│     ┌──────────────────────────────────────────┐        │
│     │  For each session:                       │        │
│     │    │                                      │        │
│     │    ├─→ Extract filenames from            │        │
│     │    │    document_ids using filenames{}   │        │
│     │    │                                      │        │
│     │    ├─→ Count documents                   │        │
│     │    │                                      │        │
│     │    └─→ Count messages in chat_history    │        │
│     │                                           │        │
│     │  Example output:                         │        │
│     │  [                                        │        │
│     │    {                                      │        │
│     │      "session_id": "abc-123",            │        │
│     │      "filenames": "doc1.pdf, doc2.txt",  │        │
│     │      "document_count": 2,                │        │
│     │      "messages_count": 5                 │        │
│     │    },                                     │        │
│     │    {                                      │        │
│     │      "session_id": "def-456",            │        │
│     │      "filenames": "Sin documentos",      │        │
│     │      "document_count": 0,                │        │
│     │      "messages_count": 3                 │        │
│     │    }                                      │        │
│     │  ]                                        │        │
│     └──────────────────────────────────────────┘        │
│                                                           │
│  3. Return summary array                                 │
│                                                           │
└───────────────────────┬─────────────────────────────────┘
                        │
                        │ Response 200 OK
                        │ [
                        │   {session_id, filenames, 
                        │    document_count, messages_count},
                        │   ...
                        │ ]
                        ↓
                 ┌─────────────┐
                 │   CLIENT    │
                 └─────────────┘
```

---

### **Endpoint 2: `POST /sessions`**

**Descripción:** Crea una nueva sesión vacía

```
┌─────────────┐
│   CLIENT    │
└──────┬──────┘
       │
       │ POST /sessions
       ↓
┌─────────────────────────────────────────────────────────┐
│  AGENT: POST /sessions                                   │
│                                                           │
│  1. Generate unique session ID                           │
│     ┌──────────────────────────────────────────┐        │
│     │  import uuid                             │        │
│     │  session_id = str(uuid.uuid4())          │        │
│     │                                           │        │
│     │  Example: "3efeb038-d4db-4c5d-8a92-..."  │        │
│     └──────────────────────────────────────────┘        │
│                                                           │
│  2. Create new Session object                            │
│     ┌──────────────────────────────────────────┐        │
│     │  new_session = Session(                  │        │
│     │    session_id = "3efeb038-...",          │        │
│     │    document_ids = [],                    │        │
│     │    filenames = {},                        │        │
│     │    chat_history = []                     │        │
│     │  )                                        │        │
│     └──────────────────────────────────────────┘        │
│                                                           │
│  3. Add to in-memory database                            │
│     ┌──────────────────────────────────────────┐        │
│     │  sessions_db[session_id] = new_session   │        │
│     └──────────────────────────────────────────┘        │
│                                                           │
│  4. Persist to disk                                      │
│     ┌──────────────────────────────────────────┐        │
│     │  save_sessions()                         │        │
│     │    │                                      │        │
│     │    ├─→ Create data/ directory            │        │
│     │    │                                      │        │
│     │    ├─→ Convert Session objects to dict   │        │
│     │    │                                      │        │
│     │    └─→ Write to data/sessions.json       │        │
│     │         with indent=4, ensure_ascii=False│        │
│     └──────────────────────────────────────────┘        │
│                                                           │
│  5. Log creation                                         │
│     logger.info(f"Nueva sesión creada: {session_id}")   │
│                                                           │
└───────────────────────┬─────────────────────────────────┘
                        │
                        │ Response 200 OK
                        │ {
                        │   "session_id": "3efeb038-...",
                        │   "message": "Sesión creada."
                        │ }
                        ↓
                 ┌─────────────┐
                 │   CLIENT    │
                 └─────────────┘
```

---

### **Endpoint 3: `POST /sessions/{id}/upload_document`**

**Descripción:** Sube un documento y lo asocia a la sesión

```
┌─────────────┐
│   CLIENT    │
└──────┬──────┘
       │
       │ POST /sessions/{id}/upload_document
       │ Form-data: file = documento.pdf
       ↓
┌─────────────────────────────────────────────────────────┐
│  AGENT: POST /sessions/{id}/upload_document              │
│                                                           │
│  1. Validate session exists                              │
│     ┌──────────────────────────────────────────┐        │
│     │  if session_id not in sessions_db:       │        │
│     │    raise HTTPException(404, "Not found") │        │
│     └──────────────────────────────────────────┘        │
│                                                           │
│  2. Read file bytes                                      │
│     file_bytes = await file.read()                       │
│                                                           │
│  3. Upload to backend                                    │
│     ┌──────────────────────────────────────────┐        │
│     │  upload_file_to_backend(                 │        │
│     │    file_bytes, filename                  │        │
│     │  )                                        │        │
│     │    │                                      │        │
│     │    │ POST http://document_service:8000/documents │
│     │    │ files = {'file': (filename, bytes)} │        │
│     │    │                                      │        │
│     │    ↓                                      │        │
│     │  Response handling:                      │        │
│     │                                           │        │
│     │  Case A: Success (201)                   │        │
│     │    ┌────────────────────────────┐        │        │
│     │    │ Extract document_id        │        │        │
│     │    │ from response JSON         │        │        │
│     │    └────────────────────────────┘        │        │
│     │                                           │        │
│     │  Case B: Duplicate (409)                 │        │
│     │    ┌────────────────────────────┐        │        │
│     │    │ Parse error message:       │        │        │
│     │    │ "...document_id=abc-123"   │        │        │
│     │    │         ↓                  │        │
│     │    │ Regex extract ID:          │        │        │
│     │    │ r"document_id=([a-zA-Z0-9_\-]+)" │  │
│     │    │         ↓                  │        │
│     │    │ Return existing ID         │        │        │
│     │    └────────────────────────────┘        │        │
│     │                                           │        │
│     │  Case C: Error (other)                   │        │
│     │    ┌────────────────────────────┐        │        │
│     │    │ raise HTTPException(500)   │        │        │
│     │    └────────────────────────────┘        │        │
│     └──────────────────────────────────────────┘        │
│                                                           │
│  4. Update session                                       │
│     ┌──────────────────────────────────────────┐        │
│     │  session = sessions_db[session_id]       │        │
│     │                                           │        │
│     │  # Add document ID if not present        │        │
│     │  if doc_id not in session.document_ids:  │        │
│     │      session.document_ids.append(doc_id) │        │
│     │                                           │        │
│     │  # Register filename                     │        │
│     │  session.filenames[doc_id] = filename    │        │
│     │                                           │        │
│     │  Before: {                               │        │
│     │    document_ids: [],                     │        │
│     │    filenames: {}                         │        │
│     │  }                                        │        │
│     │                                           │        │
│     │  After: {                                │        │
│     │    document_ids: ["abc-123"],            │        │
│     │    filenames: {                          │        │
│     │      "abc-123": "documento.pdf"          │        │
│     │    }                                      │        │
│     │  }                                        │        │
│     └──────────────────────────────────────────┘        │
│                                                           │
│  5. Persist to disk                                      │
│     save_sessions()                                      │
│                                                           │
└───────────────────────┬─────────────────────────────────┘
                        │
                        │ Response 200 OK
                        │ {
                        │   "status": "success",
                        │   "document_id": "abc-123",
                        │   "filename": "documento.pdf",
                        │   "total_docs": 1
                        │ }
                        ↓
                 ┌─────────────┐
                 │   CLIENT    │
                 └─────────────┘
```

---

### **Endpoint 4: `POST /sessions/{id}/chat`**

**Descripción:** Chat multi-documento con routing semántico

```
┌─────────────┐
│   CLIENT    │
└──────┬──────┘
       │
       │ POST /sessions/{id}/chat
       │ Body: { "query": "¿De qué trata el documento?" }
       ↓
┌──────────────────────────────────────────────────────────┐
│  AGENT: POST /sessions/{id}/chat                          │
│                                                            │
│  1. Validate session & documents                          │
│     ┌───────────────────────────────────────────┐        │
│     │  Load session (from memory or file)       │        │
│     │                                            │        │
│     │  if session_id not in sessions_db:        │        │
│     │      sessions_db = load_sessions()        │        │
│     │      if still not found:                  │        │
│     │          raise 404                         │        │
│     │                                            │        │
│     │  if not session.document_ids:             │        │
│     │      raise 400 "Sube un documento"        │        │
│     └───────────────────────────────────────────┘        │
│                                                            │
│  2. Prepare chat context                                  │
│     ┌───────────────────────────────────────────┐        │
│     │  sanitized_history = []                   │        │
│     │                                            │        │
│     │  for msg in session.chat_history:         │        │
│     │      sanitized_history.append({           │        │
│     │          "user": msg.get("user", ""),     │        │
│     │          "bot": msg.get("bot", "")        │        │
│     │      })                                    │        │
│     │                                            │        │
│     │  # Remove chunks, metadata for backend    │        │
│     │  # Keep only conversational turns         │        │
│     └───────────────────────────────────────────┘        │
│                                                            │
│  3. Call backend multi-doc search                         │
│     ┌───────────────────────────────────────────┐        │
│     │  get_backend_response_multi(              │        │
│     │    session_id = session_id,               │        │
│     │    document_ids = ["doc1", "doc2"],       │        │
│     │    query = "¿De qué trata?",              │        │
│     │    chat_context = sanitized_history       │        │
│     │  )                                         │        │
│     │    │                                       │        │
│     │    │ POST http://document_service:8000    │        │
│     │    │      /sessions/{id}/query_multi      │        │
│     │    │                                       │        │
│     │    │ Body: {                               │        │
│     │    │   "query": "¿De qué trata?",         │        │
│     │    │   "document_ids": ["doc1", "doc2"],  │        │
│     │    │   "llm_answer": true,                │        │
│     │    │   "chat_context": [                  │        │
│     │    │     {"user": "...", "bot": "..."}    │        │
│     │    │   ]                                   │        │
│     │    │ }                                     │        │
│     │    │                                       │        │
│     │    │ Timeout: 60s                          │        │
│     │    │                                       │        │
│     │    ↓                                       │        │
│     │  Response: {                               │        │
│     │    "best_document_id": "doc1",            │        │
│     │    "best_document_filename": "manual.pdf",│        │
│     │    "best_chunks": [                       │        │
│     │      "Fragment A...",                     │        │
│     │      "Fragment B..."                      │        │
│     │    ],                                      │        │
│     │    "llm_answer": "El documento trata..."  │        │
│     │  }                                         │        │
│     │                                            │        │
│     │  Error handling:                          │        │
│     │    • 404 → Return error message           │        │
│     │    • Timeout → 500 "Error consultando"    │        │
│     │    • Other → Log + raise 500              │        │
│     └───────────────────────────────────────────┘        │
│                                                            │
│  4. Extract response data                                 │
│     ┌───────────────────────────────────────────┐        │
│     │  llm_answer = backend_data.get(           │        │
│     │      "llm_answer",                        │        │
│     │      "No se encontró información."        │        │
│     │  )                                         │        │
│     │                                            │        │
│     │  chunks = backend_data.get("best_chunks", []) │   │
│     │  best_doc_id = backend_data.get(          │        │
│     │      "best_document_id", ""               │        │
│     │  )                                         │        │
│     │  best_filename = backend_data.get(        │        │
│     │      "best_document_filename", ""         │        │
│     │  )                                         │        │
│     └───────────────────────────────────────────┘        │
│                                                            │
│  5. Save to chat history                                  │
│     ┌───────────────────────────────────────────┐        │
│     │  session.chat_history.append({            │        │
│     │      "user": "¿De qué trata?",            │        │
│     │      "bot": "El documento trata...",      │        │
│     │      "chunks": [                          │        │
│     │          "Fragment A...",                 │        │
│     │          "Fragment B..."                  │        │
│     │      ],                                    │        │
│     │      "source_doc": "doc1",                │        │
│     │      "source_filename": "manual.pdf"      │        │
│     │  })                                        │        │
│     │                                            │        │
│     │  save_sessions()  # Persist to disk       │        │
│     └───────────────────────────────────────────┘        │
│                                                            │
│  6. Return response                                       │
│                                                            │
└───────────────────────┬──────────────────────────────────┘
                        │
                        │ Response 200 OK
                        │ {
                        │   "answer": "El documento trata...",
                        │   "used_chunks": [
                        │     "Fragment A...",
                        │     "Fragment B..."
                        │   ]
                        │ }
                        ↓
                 ┌─────────────┐
                 │   CLIENT    │
                 └─────────────┘
```

---

### **Endpoint 5: `POST /sessions/{id}/chat_general`**

**Descripción:** Chat sin documentos (solo LLM)

```
┌─────────────┐
│   CLIENT    │
└──────┬──────┘
       │
       │ POST /sessions/{id}/chat_general
       │ Body: { "query": "¿Quién ganó el Mundial 2022?" }
       ↓
┌──────────────────────────────────────────────────────────┐
│  AGENT: POST /sessions/{id}/chat_general                  │
│                                                            │
│  1. Validate session                                      │
│     (Same as chat endpoint)                               │
│     Load from memory or file                              │
│                                                            │
│  2. Prepare sanitized history                             │
│     ┌───────────────────────────────────────────┐        │
│     │  sanitized_history = []                   │        │
│     │                                            │        │
│     │  for msg in session.chat_history:         │        │
│     │      sanitized_history.append({           │        │
│     │          "user": msg.get("user", ""),     │        │
│     │          "bot": msg.get("bot", "")        │        │
│     │      })                                    │        │
│     └───────────────────────────────────────────┘        │
│                                                            │
│  3. Call backend general chat                             │
│     ┌───────────────────────────────────────────┐        │
│     │  get_backend_response_general(            │        │
│     │    session_id = session_id,               │        │
│     │    query = "¿Quién ganó el Mundial?",     │        │
│     │    chat_context = sanitized_history       │        │
│     │  )                                         │        │
│     │    │                                       │        │
│     │    │ POST http://document_service:8000    │        │
│     │    │      /sessions/{id}/chat_general     │        │
│     │    │                                       │        │
│     │    │ Body: {                               │        │
│     │    │   "query": "¿Quién ganó...?",        │        │
│     │    │   "llm_answer": true,                │        │
│     │    │   "chat_context": [...]              │        │
│     │    │ }                                     │        │
│     │    │                                       │        │
│     │    │ Timeout: 60s                          │        │
│     │    │                                       │        │
│     │    ↓                                       │        │
│     │  Response: {                               │        │
│     │    "llm_answer": "Argentina ganó..."      │        │
│     │  }                                         │        │
│     │                                            │        │
│     │  Note: No documents involved              │        │
│     │        Pure LLM conversation               │        │
│     └───────────────────────────────────────────┘        │
│                                                            │
│  4. Extract answer                                        │
│     llm_answer = backend_data.get("llm_answer", "")      │
│     if not llm_answer:                                    │
│         llm_answer = "No se pudo generar respuesta."      │
│                                                            │
│  5. Save to chat history                                  │
│     ┌───────────────────────────────────────────┐        │
│     │  session.chat_history.append({            │        │
│     │      "user": "¿Quién ganó...?",           │        │
│     │      "bot": "Argentina ganó...",          │        │
│     │      "chunks": [],  ← Empty               │        │
│     │      "source_doc": None,                  │        │
│     │      "source_filename": None              │        │
│     │  })                                        │        │
│     │                                            │        │
│     │  save_sessions()                          │        │
│     └───────────────────────────────────────────┘        │
│                                                            │
└───────────────────────┬──────────────────────────────────┘
                        │
                        │ Response 200 OK
                        │ {
                        │   "answer": "Argentina ganó...",
                        │   "used_chunks": []
                        │ }
                        ↓
                 ┌─────────────┐
                 │   CLIENT    │
                 └─────────────┘
```

---

### **Endpoint 6: `DELETE /sessions/{id}`**

**Descripción:** Elimina una sesión

```
┌─────────────┐
│   CLIENT    │
└──────┬──────┘
       │
       │ DELETE /sessions/{id}
       ↓
┌──────────────────────────────────────────────────────────┐
│  AGENT: DELETE /sessions/{id}                             │
│                                                            │
│  1. Validate session exists                               │
│     ┌───────────────────────────────────────────┐        │
│     │  if session_id not in sessions_db:        │        │
│     │      raise HTTPException(404)             │        │
│     └───────────────────────────────────────────┘        │
│                                                            │
│  2. Delete from memory                                    │
│     ┌───────────────────────────────────────────┐        │
│     │  del sessions_db[session_id]              │        │
│     │                                            │        │
│     │  Before: {                                │        │
│     │    "abc-123": Session(...),               │        │
│     │    "def-456": Session(...)                │        │
│     │  }                                         │        │
│     │                                            │        │
│     │  After: {                                 │        │
│     │    "def-456": Session(...)                │        │
│     │  }                                         │        │
│     └───────────────────────────────────────────┘        │
│                                                            │
│  3. Persist to disk                                       │
│     save_sessions()                                       │
│                                                            │
│  4. Log deletion                                          │
│     logger.info(f"Sesión eliminada: {session_id}")       │
│                                                            │
│  Note: Documents are NOT deleted from backend            │
│        Only the session-document relationship is removed  │
│                                                            │
└───────────────────────┬──────────────────────────────────┘
                        │
                        │ Response 200 OK
                        │ {
                        │   "status": "deleted",
                        │   "session_id": "abc-123"
                        │ }
                        ↓
                 ┌─────────────┐
                 │   CLIENT    │
                 └─────────────┘
```

---

### **Endpoint 7: `GET /sessions/{id}`**

**Descripción:** Obtiene detalles completos de una sesión

```
┌─────────────┐
│   CLIENT    │
└──────┬──────┘
       │
       │ GET /sessions/{id}
       ↓
┌──────────────────────────────────────────────────────────┐
│  AGENT: GET /sessions/{id}                                │
│                                                            │
│  1. Validate session exists                               │
│     ┌───────────────────────────────────────────┐        │
│     │  if session_id not in sessions_db:        │        │
│     │      raise HTTPException(404)             │        │
│     └───────────────────────────────────────────┘        │
│                                                            │
│  2. Return full session object                            │
│     return sessions_db[session_id]                        │
│                                                            │
└───────────────────────┬──────────────────────────────────┘
                        │
                        │ Response 200 OK
                        │ {
                        │   "session_id": "abc-123",
                        │   "document_ids": [
                        │     "doc1", "doc2"
                        │   ],
                        │   "filenames": {
                        │     "doc1": "manual.pdf",
                        │     "doc2": "guide.txt"
                        │   },
                        │   "chat_history": [
                        │     {
                        │       "user": "pregunta",
                        │       "bot": "respuesta",
                        │       "chunks": [...],
                        │       "source_doc": "doc1",
                        │       "source_filename": "manual.pdf"
                        │     }
                        │   ]
                        │ }
                        ↓
                 ┌─────────────┐
                 │   CLIENT    │
                 └─────────────┘
```

---

## 📦 Modelos de Datos

### **Session**
```python
class Session(BaseModel):
    session_id: str
    document_ids: List[str] = []       # Lista de UUIDs de documentos
    filenames: Dict[str, str] = {}     # {document_id: filename}
    chat_history: List[Dict] = []      # Historial conversacional
```

**Ejemplo:**
```json
{
  "session_id": "3efeb038-d4db-4c5d-8a92-077873d12a67",
  "document_ids": [
    "5334c96d-0409-43ba-9c9a-2d3e8858c400",
    "7f2b1e8c-3d5a-4f9b-b6c2-1a8d9e4f5c6d"
  ],
  "filenames": {
    "5334c96d-0409-43ba-9c9a-2d3e8858c400": "manual_usuario.pdf",
    "7f2b1e8c-3d5a-4f9b-b6c2-1a8d9e4f5c6d": "guia_tecnica.docx"
  },
  "chat_history": [
    {
      "user": "¿Cómo instalo el software?",
      "bot": "Según el manual, debes ejecutar...",
      "chunks": ["Fragment A...", "Fragment B..."],
      "source_doc": "5334c96d-0409-43ba-9c9a-2d3e8858c400",
      "source_filename": "manual_usuario.pdf"
    },
    {
      "user": "¿Qué requisitos tiene?",
      "bot": "Los requisitos mínimos son...",
      "chunks": ["Fragment C..."],
      "source_doc": "7f2b1e8c-3d5a-4f9b-b6c2-1a8d9e4f5c6d",
      "source_filename": "guia_tecnica.docx"
    }
  ]
}
```

### **ChatRequest**
```python
class ChatRequest(BaseModel):
    query: str
```

### **ChatResponse**
```python
class ChatResponse(BaseModel):
    answer: str
    used_chunks: List[str]
```

---

## ⚙️ Configuración

### **Variables de Entorno**

```bash
# .env file
DOCUMENT_SERVICE_URL=http://document_service:8000
```

### **Docker Compose**
```yaml
demo_agent:
  build: ./demo_document_agent
  container_name: demo_agent
  ports:
    - "8001:8001"
  environment:
    - DOCUMENT_SERVICE_URL=http://document_service:8000
  volumes:
    - ./demo_document_agent/data:/app/data
  depends_on:
    - document_service
```

---

## 💡 Ejemplos de Uso

### **Python Client**

```python
import requests

BASE_URL = "http://localhost:8001"

# 1. Crear sesión
resp = requests.post(f"{BASE_URL}/sessions")
session = resp.json()
session_id = session["session_id"]
print(f"Sesión creada: {session_id}")

# 2. Upload documento
with open("documento.pdf", "rb") as f:
    files = {"file": ("documento.pdf", f, "application/pdf")}
    resp = requests.post(
        f"{BASE_URL}/sessions/{session_id}/upload_document",
        files=files
    )
    doc_info = resp.json()
    print(f"Documento subido: {doc_info['document_id']}")

# 3. Chat multi-documento
resp = requests.post(
    f"{BASE_URL}/sessions/{session_id}/chat",
    json={"query": "¿De qué trata el documento?"}
)
chat_response = resp.json()
print(f"Respuesta: {chat_response['answer']}")

# 4. Chat general (sin documentos)
resp = requests.post(
    f"{BASE_URL}/sessions/{session_id}/chat_general",
    json={"query": "¿Quién ganó el Mundial 2022?"}
)
general_response = resp.json()
print(f"Respuesta: {general_response['answer']}")

# 5. Obtener detalles de sesión
resp = requests.get(f"{BASE_URL}/sessions/{session_id}")
session_details = resp.json()
print(f"Documentos: {len(session_details['document_ids'])}")
print(f"Mensajes: {len(session_details['chat_history'])}")

# 6. Listar todas las sesiones
resp = requests.get(f"{BASE_URL}/sessions")
sessions = resp.json()
for s in sessions:
    print(f"{s['session_id']}: {s['document_count']} docs, {s['messages_count']} msgs")

# 7. Eliminar sesión
resp = requests.delete(f"{BASE_URL}/sessions/{session_id}")
print(resp.json())
```

### **cURL**

```bash
# Crear sesión
SESSION_ID=$(curl -s -X POST http://localhost:8001/sessions | jq -r '.session_id')
echo "Session ID: $SESSION_ID"

# Upload documento
curl -X POST "http://localhost:8001/sessions/$SESSION_ID/upload_document" \
  -F "file=@documento.pdf"

# Chat
curl -X POST "http://localhost:8001/sessions/$SESSION_ID/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "¿De qué trata?"}'

# Obtener detalles
curl -X GET "http://localhost:8001/sessions/$SESSION_ID" | jq .

# Eliminar
curl -X DELETE "http://localhost:8001/sessions/$SESSION_ID"
```

---

## 🐛 Error Handling

### **Duplicados (409)**
```python
# client_service.py maneja automáticamente
# Extrae el document_id del mensaje de error
# y lo devuelve sin fallar

# Ejemplo:
# Backend retorna: 409 "Document already exists (document_id=abc-123)"
# Cliente extrae: "abc-123"
# Devuelve: "abc-123"
```

### **Sesión No Encontrada (404)**
```json
{
  "detail": "Sesión no encontrada"
}
```

### **Sin Documentos (400)**
```json
{
  "detail": "Primero sube un documento."
}
```

### **Error Backend (500)**
```json
{
  "detail": "Error consultando servicio."
}
```

---

## 📊 Persistencia

### **sessions.json Structure**
```json
{
  "3efeb038-d4db-4c5d-8a92-077873d12a67": {
    "session_id": "3efeb038-d4db-4c5d-8a92-077873d12a67",
    "document_ids": [
      "5334c96d-0409-43ba-9c9a-2d3e8858c400"
    ],
    "filenames": {
      "5334c96d-0409-43ba-9c9a-2d3e8858c400": "6 - Detección y reconocimiento de Rostros.pdf"
    },
    "chat_history": [
      {
        "user": "resumeme el documento ahora",
        "bot": "El documento trata sobre...",
        "chunks": ["Fragment A...", "Fragment B..."],
        "source_doc": "5334c96d-0409-43ba-9c9a-2d3e8858c400",
        "source_filename": "6 - Detección y reconocimiento de Rostros.pdf"
      }
    ]
  }
}
```

---

## 🔗 Enlaces

- **[Main README](../README.md)** - Documentación general del sistema
- **[Document Service README](../document_service/README.md)** - Backend RAG detallado

---

## 📝 Notas

- **Stateless Sessions:** Solo persistencia en archivo JSON (no base de datos)
- **No Authentication:** Diseñado para entorno local/demo
- **File Limits:** Depende del backend (Document Service)
- **Timeouts:** 60s para queries normales, 120s para uploads

---

**Puerto:** 8001  
**Tecnologías:** FastAPI, Pydantic, httpx, JSON storage  
**Versión:** 2.0.0

*Última actualización: Diciembre 2025*
