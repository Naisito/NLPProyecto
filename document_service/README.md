# 📚 Document Service

**Backend RAG Multi-Documento con LLM y Routing Semántico**

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white)](https://openai.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B6B?style=flat)](https://www.trychroma.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Arquitectura](#-arquitectura)
- [Endpoints](#-endpoints)
- [Flujos Detallados](#-flujos-detallados)
- [LLM Engine](#-llm-engine)
- [RAG Engine](#-rag-engine)
- [Configuración](#-configuración)
- [Ejemplos de Uso](#-ejemplos-de-uso)

---

## 🎯 Descripción

El **Document Service** es el núcleo RAG (Retrieval-Augmented Generation) del sistema. Funciones principales:

- ✅ **Extracción Multi-Formato:** PDF, TXT, DOCX, imágenes (OCR con Tesseract)
- ✅ **Deduplicación SHA-256:** Evita documentos duplicados
- ✅ **Summarization Map-Reduce:** Resúmenes automáticos de alta calidad
- ✅ **RAG con ChromaDB:** Embeddings BGE-M3, búsqueda semántica vectorial
- ✅ **Routing Semántico:** Busca en el mejor documento con 2 fases (Primary → Secondary)
- ✅ **Chat General:** LLM standalone sin documentos
- ✅ **LLM Refactorizado:** 6 funciones consolidadas, config centralizado

---

## 🏗️ Arquitectura

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          DOCUMENT SERVICE                                 │
│                             (Port 8000)                                   │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                        main.py (FastAPI)                            │ │
│  │                                                                      │ │
│  │  • 8 endpoints (health, CRUD, query, multi-doc, chat)              │ │
│  │  • Request validation & error handling                             │ │
│  │  • Pydantic models                                                  │ │
│  └──────────────────────┬───────────────────────────────────────────────┘ │
│                         │                                                  │
│          ┌──────────────┴──────────────┐                                  │
│          │                             │                                  │
│  ┌───────▼──────────┐       ┌──────────▼──────────┐                      │
│  │   llm_engine.py  │       │   rag_engine.py     │                      │
│  │                  │       │                      │                      │
│  │  6 Functions:    │       │  • search()          │                      │
│  │  1. short_summary│       │  • add_document()    │                      │
│  │  2. map_summary_ │       │  • delete_doc()      │                      │
│  │     chunk        │       │  • multi_doc_search()│                      │
│  │  3. reduce_summ. │       │  • routing (2-phase) │                      │
│  │  4. summarize_   │       │                      │                      │
│  │     with_map_    │       │  Uses:               │                      │
│  │     reduce       │       │  • RecursiveCharText │                      │
│  │  5. generate_    │       │    Splitter (500ch)  │                      │
│  │     answer       │       │  • Semantic chunking │                      │
│  │  6. optimize_    │       │                      │                      │
│  │     chat_history │       └──────────┬───────────┘                      │
│  │                  │                  │                                  │
│  │  Config from:    │                  │                                  │
│  │  config.json     │                  │                                  │
│  └─────────┬────────┘                  │                                  │
│            │                            │                                  │
│            │ OpenAI API                 │                                  │
│            │ GPT-4o-mini                │                                  │
│            ↓                            ↓                                  │
│      ┌─────────┐              ┌──────────────────┐                        │
│      │ OpenAI  │              │  vector_chroma.py│                        │
│      │ Service │              │                   │                        │
│      └─────────┘              │  • ChromaDB ops   │                        │
│                               │  • Collection mgmt│                        │
│                               └────────┬──────────┘                        │
│                                        │                                   │
│                                        ↓                                   │
│                               ┌──────────────────┐                        │
│                               │ embeddings_local │                        │
│                               │                   │                        │
│                               │  • HuggingFace    │                        │
│                               │    BGE-M3         │                        │
│                               │  • Local cache    │                        │
│                               └────────┬──────────┘                        │
│                                        │                                   │
│  ┌─────────────────────────────────────┴───────────────────────────────┐ │
│  │                         STORAGE LAYER                               │ │
│  │                                                                      │ │
│  │  ┌────────────────┐  ┌──────────────┐  ┌──────────────────────┐   │ │
│  │  │  storage.py    │  │ SQLite       │  │  ChromaDB            │   │ │
│  │  │                │  │              │  │                      │   │ │
│  │  │  • save_file() │  │  documents.db│  │  chroma.sqlite3      │   │ │
│  │  │  • get_file()  │  │              │  │  + vectors           │   │ │
│  │  │  • delete_file │  │  Fields:     │  │                      │   │ │
│  │  │  • SHA-256     │  │  - id        │  │  Collection:         │   │ │
│  │  │    dedup       │  │  - filename  │  │  "documents_rag"     │   │ │
│  │  │                │  │  - hash      │  │                      │   │ │
│  │  │  data/         │  │  - summary   │  │  Embeddings:         │   │ │
│  │  │  *.pdf, *.txt  │  │  - metadata  │  │  BGE-M3 (1024-dim)   │   │ │
│  │  └────────────────┘  └──────────────┘  └──────────────────────┘   │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

<a id='-rag-engine'></a>
## 🧠 RAG Engine

Esta sección describe cómo funciona el enrutado semántico multi-documento y la asignación de fragmentos (chunks) que devuelve el backend.

- Enrutado (routing): se calcula una puntuación por documento a partir de la similitud de la consulta con los resúmenes cortos indexados. El backend registra algo como: `Routing scores: {docA: 1.0, docB: 0.7, docC: 0.4}`.
- Presupuesto global: `rag.search_results_default` en `config.json` define el máximo total de chunks a devolver (por defecto 10). Este límite es global, es decir, la suma de chunks de todos los documentos nunca supera este valor.
- Asignación proporcional: el presupuesto total se reparte entre los documentos elegibles (el mejor documento y los que superan un umbral, p. ej. ≥ 0.3) de forma proporcional a su score. El mejor documento recibe un refuerzo ligero para priorizarlo.
- Redistribución de sobrantes: si algún documento devuelve menos chunks de los asignados, el sobrante se redistribuye comenzando por el mejor documento y luego el resto, hasta completar el presupuesto total.
- Agrupación y orden: los resultados se devuelven agrupados por documento en `results_by_document`. Dentro de cada documento, los chunks se ordenan numéricamente por su índice (IDs con formato `{document_id}_{n}`), de modo que se preserva el orden natural del texto.
- Campos de salida (multi-doc):
  - `best_document_id`: documento con mayor score de routing.
  - `results_by_document`: `{document_id: [chunk_text, ...]}` agrupado y ordenado.
  - `best_chunks`: alias conveniente de `results_by_document[best_document_id]`.

Ejemplo de asignación (presupuesto = 10):

- Scores: `{A: 1.0, B: 0.7, C: 0.4}` → A, B y C son elegibles (C ≥ 0.3).
- Asignación proporcional (aprox.): A=5, B=3, C=2 (ajustada a 10 en total). Si C sólo entrega 1 chunk, el sobrante (1) se cede a A o B según disponibilidad.

Logs útiles (nivel INFO):

- `Routing scores: {...}` → puntuaciones por documento.
- `Chunk allocation plan (pre-search): {...}` → plan de asignación antes de la búsqueda.

Configuración relevante (`document_service/config.json`):

- `rag.chunk_size`: tamaño de los fragmentos indexados (caracteres).
- `rag.chunk_overlap`: solape entre fragmentos (caracteres).
- `rag.search_results_default`: tope total de chunks que se devuelven por consulta multi-documento.

Relación con la GUI:

- La interfaz agrupa las referencias por documento y muestra el nombre del archivo (no el ID). Los chunks aparecen numerados en el orden natural del documento.

## 🔌 Endpoints

### **Resumen**

| Método | Endpoint | Descripción | LLM | Vector DB |
|--------|----------|-------------|-----|-----------|
| GET | `/health` | Health check | ❌ | ❌ |
| GET | `/documents` | Lista documentos | ❌ | ❌ |
| POST | `/documents` | Upload + Map-Reduce | ✅ | ✅ |
| GET | `/documents/{id}` | Obtiene metadata | ❌ | ❌ |
| POST | `/documents/{id}/query` | RAG single-doc | ✅ | ✅ |
| DELETE | `/documents/{id}` | Elimina documento | ❌ | ✅ |
| POST | `/sessions/{id}/query_multi` | RAG multi-doc routing | ✅ | ✅ |
| POST | `/sessions/{id}/chat_general` | Chat sin docs | ✅ | ❌ |

---

## 🔄 Flujos Detallados

### **Endpoint 1: `POST /documents`**

**Descripción:** Upload documento con Map-Reduce summarization

```
┌─────────────┐
│   CLIENT    │
└──────┬──────┘
       │
       │ POST /documents
       │ Form-data: file = documento.pdf
       ↓
┌────────────────────────────────────────────────────────────────────────┐
│  BACKEND: POST /documents                                               │
│                                                                          │
│  PHASE 1: VALIDATION & EXTRACTION                                       │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  1. Validate file extension                                    │    │
│  │     • Check against: .pdf, .txt, .docx, .png, .jpg, .jpeg      │    │
│  │     • If invalid → 400 "Extension no permitida"                │    │
│  │                                                                  │    │
│  │  2. Save to temp file                                           │    │
│  │     temp_path = "data/temp/{uuid}.ext"                          │    │
│  │                                                                  │    │
│  │  3. Extract text                                                │    │
│  │     extractor.extract_text(temp_path)                           │    │
│  │                                                                  │    │
│  │     ┌─────────────────────────────────────┐                     │    │
│  │     │  • PDF  → pypdf (per-page)          │                     │    │
│  │     │  • DOCX → python-docx (paragraphs)  │                     │    │
│  │     │  • TXT  → utf-8 decode              │                     │    │
│  │     │  • IMG  → Tesseract OCR (eng+spa)   │                     │    │
│  │     └─────────────────────────────────────┘                     │    │
│  │                                                                  │    │
│  │     text_length = len(text)                                     │    │
│  │     if text_length == 0 → 400 "No se pudo extraer texto"        │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  PHASE 2: DEDUPLICATION                                                 │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  4. Calculate SHA-256 hash                                      │    │
│  │     file_hash = sha256(file_bytes).hexdigest()                  │    │
│  │                                                                  │    │
│  │  5. Check if exists                                             │    │
│  │     storage.file_exists(file_hash)                              │    │
│  │                                                                  │    │
│  │     if exists:                                                  │    │
│  │       ┌──────────────────────────────────┐                      │    │
│  │       │  existing = get_by_hash(hash)    │                      │    │
│  │       │  raise HTTPException(409,        │                      │    │
│  │       │    "Document already exists "    │                      │    │
│  │       │    f"(document_id={existing.id}")│                      │    │
│  │       └──────────────────────────────────┘                      │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  PHASE 3: MAP-REDUCE SUMMARIZATION                                      │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  6. Summarize with Map-Reduce                                   │    │
│  │     llm_engine.summarize_with_map_reduce(text, chunk_size=15000)│    │
│  │                                                                  │    │
│  │     ┌────────────────────────────────────────────────────┐      │    │
│  │     │  Step A: Detect if needs chunking                  │      │    │
│  │     │    if len(text) <= 15000:                          │      │    │
│  │     │        return llm_engine.map_summary_chunk(text)   │      │    │
│  │     │    else:                                            │      │    │
│  │     │        proceed to chunking                         │      │    │
│  │     │                                                     │      │    │
│  │     │  Step B: Split into chunks (15000 chars)           │      │    │
│  │     │    chunks = [                                      │      │    │
│  │     │      text[0:15000],                                │      │    │
│  │     │      text[15000:30000],                            │      │    │
│  │     │      ...                                            │      │    │
│  │     │    ]                                                │      │    │
│  │     │                                                     │      │    │
│  │     │  Step C: MAP - Parallel summarization              │      │    │
│  │     │    with ThreadPoolExecutor(max_workers=4):         │      │    │
│  │     │        summaries = []                              │      │    │
│  │     │        for chunk in chunks:                        │      │    │
│  │     │            summary = map_summary_chunk(chunk)      │      │    │
│  │     │            summaries.append(summary)               │      │    │
│  │     │                                                     │      │    │
│  │     │    LLM Call (each chunk):                          │      │    │
│  │     │    ┌──────────────────────────────────────┐        │      │    │
│  │     │    │ System: "Experto en síntesis"        │        │      │    │
│  │     │    │ User:                                │        │      │    │
│  │     │    │   "Resume detectando tipo:          │        │      │    │
│  │     │    │    - Narrativo: trama + personajes  │        │      │    │
│  │     │    │    - Técnico: pasos + comandos       │        │      │    │
│  │     │    │    - Legal: cláusulas formales"      │        │      │    │
│  │     │    │ Temperature: 0.2                     │        │      │    │
│  │     │    └──────────────────────────────────────┘        │      │    │
│  │     │                                                     │      │    │
│  │     │  Step D: REDUCE - Unify summaries                  │      │    │
│  │     │    combined = "\n\n".join(summaries)               │      │    │
│  │     │    final = llm_engine.reduce_summaries(combined)   │      │    │
│  │     │                                                     │      │    │
│  │     │    LLM Call:                                       │      │    │
│  │     │    ┌──────────────────────────────────────┐        │      │    │
│  │     │    │ System: "Sintetizador de metadatos"  │        │      │    │
│  │     │    │ User:                                │        │      │    │
│  │     │    │   "Unifica resúmenes eliminando      │        │      │    │
│  │     │    │    costuras (Parte 1, Parte 2).     │        │      │    │
│  │     │    │    Crea narrativa cohesiva."         │        │      │    │
│  │     │    │ Temperature: 0.3                     │        │      │    │
│  │     │    │ Max tokens: 300                      │        │      │    │
│  │     │    └──────────────────────────────────────┘        │      │    │
│  │     └────────────────────────────────────────────────────┘      │    │
│  │                                                                  │    │
│  │  7. Generate short summary (semantic tags)                      │    │
│  │     llm_engine.short_summary(text[:15000])                      │    │
│  │                                                                  │    │
│  │     ┌────────────────────────────────────────────────────┐      │    │
│  │     │ System: "Extractor de metadatos técnicos"          │      │    │
│  │     │ User:                                              │      │    │
│  │     │   "Genera perfil de MÁXIMO 40 palabras:          │      │    │
│  │     │    - Nombres propios, códigos, fechas             │      │    │
│  │     │    - Tema central + tipo documento               │      │    │
│  │     │    - NO uses 'Este texto trata de...'"            │      │    │
│  │     │ Temperature: 0.1                                  │      │    │
│  │     │                                                    │      │    │
│  │     │ Example output:                                   │      │    │
│  │     │ "Manual técnico Python. Pandas, NumPy, scikit-   │      │    │
│  │     │  learn. Machine Learning. Target: Data Scientists"│      │    │
│  │     └────────────────────────────────────────────────────┘      │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  PHASE 4: RAG INDEXING                                                  │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  8. Chunk document for RAG (500 chars, 25 overlap)             │    │
│  │     splitter = RecursiveCharacterTextSplitter(                  │    │
│  │         chunk_size=500, chunk_overlap=25                        │    │
│  │     )                                                            │    │
│  │     chunks = splitter.split_text(text)                          │    │
│  │                                                                  │    │
│  │  9. Generate embeddings                                         │    │
│  │     embeddings_model = HuggingFaceEmbeddings(                   │    │
│  │         model_name="BAAI/bge-m3",                               │    │
│  │         model_kwargs={'device': 'cpu'},                         │    │
│  │         encode_kwargs={'normalize_embeddings': True}            │    │
│  │     )                                                            │    │
│  │                                                                  │    │
│  │     For each chunk:                                             │    │
│  │       embedding_vector = model.embed(chunk)  # 1024-dim         │    │
│  │                                                                  │    │
│  │  10. Store in ChromaDB                                          │    │
│  │      rag_engine.add_document(                                   │    │
│  │          document_id, chunks, {"filename": "..."}               │    │
│  │      )                                                           │    │
│  │                                                                  │    │
│  │      ┌──────────────────────────────────────┐                   │    │
│  │      │ ChromaDB collection.add(             │                   │    │
│  │      │   ids = ["doc_uuid_chunk_0", ...],   │                   │    │
│  │      │   documents = chunks,                │                   │    │
│  │      │   embeddings = vectors,              │                   │    │
│  │      │   metadatas = [                      │                   │    │
│  │      │     {                                │                   │    │
│  │      │       "document_id": "uuid",         │                   │    │
│  │      │       "filename": "doc.pdf",         │                   │    │
│  │      │       "chunk_index": 0               │                   │    │
│  │      │     }, ...                           │                   │    │
│  │      │   ]                                  │                   │    │
│  │      │ )                                    │                   │    │
│  │      └──────────────────────────────────────┘                   │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  PHASE 5: METADATA PERSISTENCE                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  11. Save to SQLite                                             │    │
│  │      storage.save_file(                                         │    │
│  │          file_bytes,                                            │    │
│  │          filename,                                              │    │
│  │          summary=full_summary,                                  │    │
│  │          short_summary=short_summary                            │    │
│  │      )                                                           │    │
│  │                                                                  │    │
│  │      INSERT INTO documents (                                    │    │
│  │          id, filename, file_hash, summary, short_summary,       │    │
│  │          uploaded_at                                            │    │
│  │      ) VALUES (?, ?, ?, ?, ?, ?)                                │    │
│  │                                                                  │    │
│  │  12. Log success                                                │    │
│  │      logger.info(f"Documento procesado: {doc_id}")              │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │
                           │ Response 201 Created
                           │ {
                           │   "document_id": "5334c96d-...",
                           │   "filename": "documento.pdf",
                           │   "summary": "Este documento presenta...",
                           │   "summary_length": 287,
                           │   "total_chunks": 45,
                           │   "text_length": 23450
                           │ }
                           ↓
                    ┌─────────────┐
                    │   CLIENT    │
                    └─────────────┘
```

---

### **Endpoint 2: `POST /documents/{id}/query`**

**Descripción:** RAG búsqueda en documento único

```
┌─────────────┐
│   CLIENT    │
└──────┬──────┘
       │
       │ POST /documents/{id}/query
       │ Body: {
       │   "query": "¿Cómo funciona X?",
       │   "llm_answer": true,
       │   "chat_context": [...]  (optional)
       │ }
       ↓
┌────────────────────────────────────────────────────────────────────────┐
│  BACKEND: POST /documents/{id}/query                                    │
│                                                                          │
│  PHASE 1: RETRIEVAL (Vector Search)                                     │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  1. Validate document exists                                    │    │
│  │     storage.get_by_id(document_id)                              │    │
│  │     if not found → 404                                          │    │
│  │                                                                  │    │
│  │  2. Embed query                                                 │    │
│  │     query_vector = embeddings_model.embed(query)  # 1024-dim    │    │
│  │                                                                  │    │
│  │  3. Search ChromaDB                                             │    │
│  │     rag_engine.search(                                          │    │
│  │         query = "¿Cómo funciona X?",                            │    │
│  │         document_id = "5334c96d-...",                           │    │
│  │         top_k = 10                                              │    │
│  │     )                                                            │    │
│  │                                                                  │    │
│  │     ┌────────────────────────────────────────────────────┐      │    │
│  │     │ ChromaDB:                                          │      │    │
│  │     │   collection.query(                                │      │    │
│  │     │       query_embeddings = [query_vector],           │      │    │
│  │     │       n_results = 10,                              │      │    │
│  │     │       where = {"document_id": "5334c96d-..."}      │      │    │
│  │     │   )                                                │      │    │
│  │     │                                                     │      │    │
│  │     │ Cosine Similarity Ranking:                         │      │    │
│  │     │   1. chunk_42: 0.89                                │      │    │
│  │     │   2. chunk_15: 0.87                                │      │    │
│  │     │   3. chunk_8:  0.82                                │      │    │
│  │     │   ...                                               │      │    │
│  │     └────────────────────────────────────────────────────┘      │    │
│  │                                                                  │    │
│  │  Result:                                                         │    │
│  │    chunks = [                                                    │    │
│  │      "Para ejecutar X, primero debes configurar...",            │    │
│  │      "X se compone de tres módulos principales...",             │    │
│  │      "La función X() acepta los siguientes parámetros..."       │    │
│  │    ]                                                             │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  PHASE 2: GENERATION (LLM Answer)                                       │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  4. If llm_answer == true:                                      │    │
│  │                                                                  │    │
│  │     llm_engine.generate_answer(                                 │    │
│  │         query = "¿Cómo funciona X?",                            │    │
│  │         summary = doc.summary,  # From SQLite                   │    │
│  │         context = chunks,       # From vector search            │    │
│  │         chat_history = chat_context  # Optional                 │    │
│  │     )                                                            │    │
│  │                                                                  │    │
│  │     ┌────────────────────────────────────────────────────┐      │    │
│  │     │ Prompt Construction:                               │      │    │
│  │     │                                                     │      │    │
│  │     │ if chat_context:                                   │      │    │
│  │     │     # Chat mode with history                       │      │    │
│  │     │     system_msg = "Asistente con memoria           │      │    │
│  │     │                   conversacional"                  │      │    │
│  │     │                                                     │      │    │
│  │     │     messages = [                                   │      │    │
│  │     │       {"role": "system", "content": system_msg},   │      │    │
│  │     │       {"role": "user", "content": history[0].user},│      │    │
│  │     │       {"role": "assistant", "content": history[0]  │      │    │
│  │     │        .bot},                                      │      │    │
│  │     │       ...,                                         │      │    │
│  │     │       {"role": "user", "content": f"""            │      │    │
│  │     │         CONTEXTO: {summary}                        │      │    │
│  │     │         FRAGMENTOS: {chunks}                       │      │    │
│  │     │         PREGUNTA: {query}                          │      │    │
│  │     │       """}                                         │      │    │
│  │     │     ]                                              │      │    │
│  │     │     temperature = 0.3  # Chat                      │      │    │
│  │     │                                                     │      │    │
│  │     │ else:                                              │      │    │
│  │     │     # Single-shot answer mode                      │      │    │
│  │     │     system_msg = "Analista técnico. Responde      │      │    │
│  │     │                   basándote SOLO en fragmentos"    │      │    │
│  │     │                                                     │      │    │
│  │     │     messages = [                                   │      │    │
│  │     │       {"role": "system", "content": system_msg},   │      │    │
│  │     │       {"role": "user", "content": f"""            │      │    │
│  │     │         DOCUMENTO: {summary}                       │      │    │
│  │     │         FRAGMENTOS:                                │      │    │
│  │     │         {chunks[0]}                                │      │    │
│  │     │         {chunks[1]}                                │      │    │
│  │     │         ...                                        │      │    │
│  │     │         PREGUNTA: {query}                          │      │    │
│  │     │       """}                                         │      │    │
│  │     │     ]                                              │      │    │
│  │     │     temperature = 0.1  # Answer                    │      │    │
│  │     │                                                     │      │    │
│  │     │ OpenAI Call:                                       │      │    │
│  │     │   response = client.chat.completions.create(      │      │    │
│  │     │       model="gpt-4o-mini",                         │      │    │
│  │     │       messages=messages,                           │      │    │
│  │     │       temperature=temperature                      │      │    │
│  │     │   )                                                │      │    │
│  │     │                                                     │      │    │
│  │     │ Example output:                                    │      │    │
│  │     │ "Para ejecutar X, primero debes configurar el     │      │    │
│  │     │  módulo A según los fragmentos. Los parámetros    │      │    │
│  │     │  principales son..."                               │      │    │
│  │     └────────────────────────────────────────────────────┘      │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │
                           │ Response 200 OK
                           │ {
                           │   "document_id": "5334c96d-...",
                           │   "chunks": [
                           │     "Para ejecutar X, primero...",
                           │     "X se compone de tres..."
                           │   ],
                           │   "llm_answer": "Para ejecutar X..."
                           │ }
                           ↓
                    ┌─────────────┐
                    │   CLIENT    │
                    └─────────────┘
```

---

### **Endpoint 3: `POST /sessions/{id}/query_multi`**

**Descripción:** RAG multi-documento con routing semántico y respuesta unificada

```
┌─────────────┐
│   CLIENT    │
└──────┬──────┘
  │
  │ POST /sessions/{id}/query_multi
  │ Body: {
  │   "query": "¿Qué dice sobre seguridad?",
  │   "document_ids": ["doc1", "doc2", "doc3"],
  │   "llm_answer": true,
  │   "chat_context": [...]
  │ }
  ↓
┌────────────────────────────────────────────────────────────────────────┐
│  BACKEND: POST /sessions/{id}/query_multi                               │
│                                                                          │
│  PHASE 1: SEMANTIC ROUTING (RagService)                                  │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  1. Búsqueda multi-doc encapsulada                             │    │
│  │     search_results = rag_service.search_multi_document(        │    │
│  │         document_ids, query                                    │    │
│  │     )                                                          │    │
│  │                                                                  │    │
│  │     Devuelve:                                                   │    │
│  │       - best_doc: "doc2"                                       │    │
│  │       - best_chunks: [top chunks del ganador]                   │    │
│  │       - results_by_doc: {doc_id: [chunks relevantes]}          │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  PHASE 2: CONTEXT BUILDING (Priorizar ganador + resto)                   │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  2. Combinar fragmentos respetando presupuesto RAG             │    │
│  │     ordered_docs = [best_doc] + otros                           │    │
│  │     combined_chunks = []                                        │    │
│  │     for d in ordered_docs:                                      │    │
│  │         combined_chunks += results_by_doc.get(d, [])            │    │
│  │                                                                  │    │
│  │     Nota: filtros y límites (p.ej. umbral/Top-K) ya aplicados   │    │
│  │     dentro de RagService.                                       │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  PHASE 3: LLM GENERATION                                                │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  3. Contexto de documento ganador                               │    │
│  │     best_doc_meta = storage.get_document(best_doc)              │    │
│  │     summary_long = best_doc_meta.summary_long                   │    │
│  │                                                                  │    │
│  │  4. Generar respuesta (con o sin chat)                          │    │
│  │     llm_engine.generate_answer(                                 │    │
│  │         query = "¿Qué dice sobre seguridad?",                   │    │
│  │         summary = summary_long,                                  │    │
│  │         context = combined_chunks,                               │    │
│  │         chat_history = chat_context                              │    │
│  │     )                                                            │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└──────────────────────────┬─────────────────────────────────────────────┘
            │
            │ Response 200 OK
            │ {
            │   "best_document_id": "doc2",
            │   "best_document_filename": "security_manual.pdf",
            │   "results_by_document": {
            │       "doc2": ["...", "..."],
            │       "doc1": ["..."],
            │       "doc3": []
            │   },
            │   "best_chunks": [
            │     "La seguridad del sistema...",
            │     "Protocolos de cifrado: AES-256...",
            │     ...
            │   ],
            │   "llm_answer": "Según el documento más relevante..."
            │ }
            ↓
          ┌─────────────┐
          │   CLIENT    │
          └─────────────┘
```

---

### **Endpoint 4: `POST /sessions/{id}/chat_general`**

**Descripción:** Chat sin documentos (LLM puro)

```
┌─────────────┐
│   CLIENT    │
└──────┬──────┘
       │
       │ POST /sessions/{id}/chat_general
       │ Body: {
       │   "query": "¿Cuál es la capital de Francia?",
       │   "chat_context": [...]
       │ }
       ↓
┌────────────────────────────────────────────────────────────────────────┐
│  BACKEND: POST /sessions/{id}/chat_general                              │
│                                                                          │
│  NO RAG - PURE LLM CONVERSATION                                         │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  1. Optimize chat history                                       │    │
│  │     (Keep last N messages to avoid token overflow)              │    │
│  │                                                                  │    │
│  │     if len(chat_context) > KEEP_LAST_N_MESSAGES:                │    │
│  │         # Summarize old messages with LLM                       │    │
│  │         optimized = llm_engine.optimize_chat_history(           │    │
│  │             chat_context                                        │    │
│  │         )                                                        │    │
│  │                                                                  │    │
│  │         ┌────────────────────────────────────────────────┐      │    │
│  │         │ LLM Call:                                      │      │    │
│  │         │   System: "Resume conversación preservando    │      │    │
│  │         │            información relevante"              │      │    │
│  │         │   User: "Historial: {old_messages}"           │      │    │
│  │         │   Temperature: 0.3                             │      │    │
│  │         │                                                │      │    │
│  │         │ Output: "Resumen: Usuario preguntó sobre X,   │      │    │
│  │         │          asistente explicó Y..."               │      │    │
│  │         └────────────────────────────────────────────────┘      │    │
│  │                                                                  │    │
│  │         optimized_history = [                                   │    │
│  │             {"user": "...", "bot": summary},                    │    │
│  │             ...last 5 messages                                  │    │
│  │         ]                                                        │    │
│  │                                                                  │    │
│  │  2. Build conversation prompt                                   │    │
│  │                                                                  │    │
│  │     messages = [                                                │    │
│  │         {                                                        │    │
│  │             "role": "system",                                    │    │
│  │             "content": "Eres un asistente conversacional        │    │
│  │                         útil. Mantén coherencia con el         │    │
│  │                         historial."                             │    │
│  │         },                                                       │    │
│  │         {"role": "user", "content": history[0].user},           │    │
│  │         {"role": "assistant", "content": history[0].bot},       │    │
│  │         ...,                                                     │    │
│  │         {                                                        │    │
│  │             "role": "user",                                      │    │
│  │             "content": "¿Cuál es la capital de Francia?"        │    │
│  │         }                                                        │    │
│  │     ]                                                            │    │
│  │                                                                  │    │
│  │  3. LLM Call                                                     │    │
│  │                                                                  │    │
│  │     response = client.chat.completions.create(                  │    │
│  │         model="gpt-4o-mini",                                     │    │
│  │         messages=messages,                                      │    │
│  │         temperature=0.3  # Chat temperature                     │    │
│  │     )                                                            │    │
│  │                                                                  │    │
│  │     answer = response.choices[0].message.content                │    │
│  │                                                                  │    │
│  │  Example output:                                                │    │
│  │  "La capital de Francia es París. Es la ciudad más poblada     │    │
│  │   del país y un centro cultural e histórico importante."        │    │
│  │                                                                  │    │
│  │  Note: NO document context, NO vector search                    │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │
                           │ Response 200 OK
                           │ {
                           │   "llm_answer": "La capital de Francia es París..."
                           │ }
                           ↓
                    ┌─────────────┐
                    │   CLIENT    │
                    └─────────────┘
```

---

## 🤖 LLM Engine

**Archivo:** `llm_engine.py`

### **6 Funciones Consolidadas**

#### **1. `short_summary(text: str) -> str`**
- **Propósito:** Genera tags/keywords densos (40 palabras max)
- **Uso:** Metadatos para búsqueda vectorial, router semántico
- **Temperature:** 0.1 (determinista)
- **Prompt:** "Extrae: nombres propios, códigos, fechas, tema, tipo documento"

#### **2. `map_summary_chunk(text: str) -> str`**
- **Propósito:** Resume fragmento detectando tipo (Narrativo/Técnico/Legal)
- **Uso:** Map phase de Map-Reduce
- **Temperature:** 0.2 (baja variabilidad)
- **Adaptación automática:**
  - Narrativa → Trama + personajes
  - Técnico → Pasos + comandos
  - Legal → Cláusulas formales

#### **3. `reduce_summaries(summaries_text: str) -> str`**
- **Propósito:** Unifica múltiples resúmenes eliminando costuras
- **Uso:** Reduce phase de Map-Reduce
- **Temperature:** 0.3
- **Max Tokens:** 300
- **Objetivo:** Narrativa cohesiva sin "Parte 1", "Parte 2"

#### **4. `summarize_with_map_reduce(text: str, chunk_size: int = 15000) -> str`**
- **Propósito:** Orquesta Map-Reduce completo
- **Lógica:**
  - Si texto < 15000 chars → Direct summary
  - Si texto >= 15000 → Split + Map (paralelo) + Reduce
- **Paralelización:** ThreadPoolExecutor con 4 workers

#### **5. `generate_answer(...) -> str`**
- **Propósito:** Responde preguntas con RAG context (unificado)
- **Parámetros:**
  - `query`: Pregunta
  - `summary`: Resumen documento
  - `context`: Chunks relevantes
  - `chat_history`: Opcional para modo chat
- **Modos:**
  - Sin chat_history → Temperature 0.1 (answer)
  - Con chat_history → Temperature 0.3 (chat)

#### **6. `optimize_chat_history(chat_history: List[Dict]) -> List[Dict]`**
- **Propósito:** Resume historial viejo + mantiene últimos N mensajes
- **Configuración:** `KEEP_LAST_N_MESSAGES = 5` (config.json)
- **Previene:** Token overflow en conversaciones largas

---

## 🔍 RAG Engine

**Archivo:** `rag_engine.py`

### **Componentes**

#### **Embeddings**
- **Modelo:** `BAAI/bge-m3` (HuggingFace)
- **Dimensiones:** 1024
- **Normalización:** Habilitada
- **Cache:** `models_cache/`

#### **Text Splitter**
- **Tipo:** RecursiveCharacterTextSplitter
- **Chunk Size:** 500 caracteres
- **Overlap:** 25 caracteres
- **Separadores:** `["\n\n", "\n", ".", " "]`

#### **Vector Database**
- **Provider:** ChromaDB
- **Persistencia:** `db/chroma_db/`
- **Collection:** `documents_rag`
- **Metadata:**
  ```json
  {
    "document_id": "uuid",
    "filename": "file.pdf",
    "chunk_index": 0
  }
  ```

### **Métodos Principales**

#### **`search(query, document_id, top_k=10)`**
- Búsqueda en documento único
- Retorna chunks ordenados por relevancia

#### **`multi_document_search(query, document_ids, top_k_per_doc=1)`**
- Búsqueda en múltiples documentos (Primary Phase)
- Retorna 1 chunk por documento
- Usado por routing semántico

#### **`add_document(document_id, chunks, metadata)`**
- Indexa chunks con embeddings
- Almacena metadata

#### **`delete_document(document_id)`**
- Elimina todos los chunks del documento
- Filter by metadata: `{"document_id": document_id}`

---

## ⚙️ Configuración

### **config.json**

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "log_level": "INFO"
  },
  "storage": {
    "allowed_extensions": [".pdf", ".txt", ".docx", ".png", ".jpg", ".jpeg"],
    "upload_dir": "data",
    "db_path": "db/documents.db"
  },
  "rag": {
    "chunk_size": 500,
    "chunk_overlap": 25,
    "search_results_default": 10
  },
  "embeddings": {
    "provider": "huggingface",
    "model_name": "BAAI/bge-m3",
    "cache_dir": "models_cache"
  },
  "vector_db": {
    "provider": "chroma",
    "path": "db/chroma_db",
    "collection_name": "documents_rag"
  },
  "llm": {
    "openai_api_key": "sk-proj-...",
    "openai_model_name": "gpt-4o-mini",
    "temperature_map": 0.2,
    "temperature_reduce": 0.3,
    "temperature_answer": 0.1,
    "temperature_chat": 0.3,
    "max_tokens_summary": 300,
    "keep_last_n_messages": 5
  }
}
```

### **Variables Importantes**

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `chunk_size` | 500 | Tamaño de fragmentos RAG |
| `chunk_overlap` | 25 | Overlap entre chunks |
| `temperature_map` | 0.2 | LLM temp para Map phase |
| `temperature_reduce` | 0.3 | LLM temp para Reduce phase |
| `temperature_answer` | 0.1 | LLM temp para answers |
| `temperature_chat` | 0.3 | LLM temp para chat |
| `max_tokens_summary` | 300 | Max tokens en resúmenes |
| `keep_last_n_messages` | 5 | Mensajes conservados en chat |

---

## 💡 Ejemplos de Uso

### **Python Client**

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Upload documento
with open("manual.pdf", "rb") as f:
    files = {"file": ("manual.pdf", f, "application/pdf")}
    resp = requests.post(f"{BASE_URL}/documents", files=files)
    doc = resp.json()
    doc_id = doc["document_id"]
    print(f"Summary: {doc['summary'][:100]}...")

# 2. Query single document
resp = requests.post(
    f"{BASE_URL}/documents/{doc_id}/query",
    json={
        "query": "¿Cómo instalo el software?",
        "llm_answer": True
    }
)
result = resp.json()
print(f"Answer: {result['llm_answer']}")
print(f"Chunks used: {len(result['chunks'])}")

# 3. Multi-document query
resp = requests.post(
    f"{BASE_URL}/sessions/session-123/query_multi",
    json={
        "query": "¿Qué dicen sobre seguridad?",
        "document_ids": [doc_id, "other-doc-id"],
        "llm_answer": True
    }
)
multi_result = resp.json()
print(f"Best doc: {multi_result['best_document_filename']}")
print(f"Answer: {multi_result['llm_answer']}")

# 4. General chat (no documents)
resp = requests.post(
    f"{BASE_URL}/sessions/session-123/chat_general",
    json={
        "query": "¿Qué es Python?",
        "chat_context": [
            {"user": "Hola", "bot": "¡Hola! ¿En qué puedo ayudarte?"}
        ]
    }
)
chat_result = resp.json()
print(f"Chat answer: {chat_result['llm_answer']}")

# 5. List documents
resp = requests.get(f"{BASE_URL}/documents")
docs = resp.json()
for doc in docs:
    print(f"{doc['filename']}: {doc['summary'][:50]}...")

# 6. Delete document
resp = requests.delete(f"{BASE_URL}/documents/{doc_id}")
print(resp.json())
```

### **cURL**

```bash
# Upload
curl -X POST "http://localhost:8000/documents" \
  -F "file=@documento.pdf"

# Query
curl -X POST "http://localhost:8000/documents/{ID}/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "¿De qué trata?",
    "llm_answer": true
  }' | jq .

# Multi-doc
curl -X POST "http://localhost:8000/sessions/session-123/query_multi" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "¿Qué dice sobre X?",
    "document_ids": ["doc1", "doc2"],
    "llm_answer": true
  }' | jq .

# Delete
curl -X DELETE "http://localhost:8000/documents/{ID}"
```

---

## 🐛 Error Handling

### **409 Conflict - Duplicate Document**
```json
{
  "detail": "Document already exists (document_id=abc-123)"
}
```

### **400 Bad Request - Invalid File**
```json
{
  "detail": "Extension no permitida. Solo: .pdf, .txt, .docx, .png, .jpg, .jpeg"
}
```

### **400 Bad Request - No Text Extracted**
```json
{
  "detail": "No se pudo extraer texto del archivo"
}
```

### **404 Not Found**
```json
{
  "detail": "Documento no encontrado"
}
```

---

## 📊 Performance Metrics

### **Upload + Processing Time**
- PDF (10 pages): ~15-20s
  - Extraction: 2s
  - Map-Reduce: 8-12s (parallel)
  - Embedding: 3-5s
  - Storage: 1s

### **Query Response Time**
- Single-doc query: ~2-4s
  - Vector search: 0.5s
  - LLM generation: 1.5-3s

- Multi-doc query: ~3-6s
  - Primary search (3 docs): 1s
  - Secondary search: 0.5s
  - LLM generation: 2-4s

### **Storage**
- SQLite DB: ~50 KB per document (metadata only)
- ChromaDB vectors: ~500 KB per document (500-char chunks, BGE-M3)
- Raw files: Original size (stored in `data/`)

---

## 🔗 Enlaces

- **[Main README](../README.md)** - Documentación general del sistema
- **[Agent Service README](../demo_document_agent/README.md)** - Orquestador de sesiones

---

## 📝 Notas Técnicas

- **Extracción OCR:** Requiere Tesseract instalado (eng + spa)
- **GPU Support:** Embeddings en CPU por defecto (cambiar en `embeddings_local.py`)
- **Token Limits:** GPT-4o-mini max 16K tokens context
- **Concurrent Uploads:** No recomendado (SQLite single-writer)
- **Cache Models:** Primera ejecución descarga BGE-M3 (~1.2 GB)

---

**Puerto:** 8000  
**Tecnologías:** FastAPI, OpenAI GPT-4o-mini, ChromaDB, HuggingFace BGE-M3, SQLite, Tesseract  
**Versión:** 2.0.0

*Última actualización: Diciembre 2025*
