# Generador de Rutas Turísticas — Bilbao / Bizkaia

> **Proyecto NLP** · Grupo VI: Unai de León, Xabier Ballesteros, Oscar Basaguren

Sistema automático de generación de itinerarios turísticos personalizados para Bilbao y Bizkaia. Implementa un pipeline **RAG híbrido** (dense + BM25) con reranking, planificación geográfica y generación narrativa. Funciona completamente en local: no necesita claves de API ni conexión a internet.

---

## Arranque rápido

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

La primera vez descarga ~4 GB de modelos. Con Docker:

```bash
# Requiere Ollama corriendo en el host (ver instrucciones abajo)
docker-compose up
```

El frontend Streamlit arranca en `http://localhost:9001` y la API en `http://localhost:9000`.

---

## Cómo funciona

El sistema recibe preferencias del usuario (texto libre o formulario) y devuelve una ruta con narrativa generada:

```
Texto del usuario
    → Intérprete LLM (Ollama, local)
    → Recuperación híbrida (dense + BM25)
    → Reranking con cross-encoder
    → Planificación geográfica + 2-opt
    → Narrativa generada por LLM
    → Evaluación automática (7 métricas)
```

### Recuperación (RAG)

Combinación de dos recuperadores:

- **Denso**: BAAI/bge-m3 + ChromaDB. Captura similitud semántica. Bueno con paráfrasis.
- **Lexical (BM25)**: Tokenizador español, 150 stopwords, parámetros TREC (k1=1.5, b=0.75). Bueno con topónimos y términos exactos ("Gernika", "dólmen").

La fusión se hace con Reciprocal Rank Fusion (rrf_k=60). También hay modo lineal (α·dense + (1-α)·bm25). Los modos se eligen en `config.json → retrieval.mode` (`dense`, `bm25` o `hybrid`).

### Reranking

El cross-encoder BAAI/bge-reranker-v2-m3 reordena los candidatos. El score final combina semejanza vectorial (0.50), reranking (0.20), coincidencia con intereses (0.20) y diversidad de categorías (0.10). Los pesos se configuran en `config.json`.

### Planificación

Selecciona N POIs por día según ritmo (tranquilo=3, moderado=4, intenso=6), valida horarios de apertura por día de la semana y ordena geográficamente con vecino más próximo + 2-opt. Asigna franjas de mañana (09:30-14:00) y tarde (16:00-20:00).

La expansión semántica de queries busca automáticamente categorías del corpus cercanas a intereses sin mapeo explícito (ej. "surf" → playa, naturaleza).

### Generación y evaluación

El LLM local (Ollama) genera una narrativa en castellano con los datos del itinerario. El evaluador calcula 7 métricas automáticas: cobertura de preferencias, coherencia temporal, compacidad geográfica, adherencia al presupuesto, diversidad, accesibilidad y puntuación global.

El endpoint `/api/route/stream` expone el progreso de cada etapa vía Server-Sent Events (SSE), permitiendo al frontend mostrar el avance en tiempo real. Las rutas generadas se persisten en SQLite y pueden consultarse en el historial (`/api/routes/saved`).

### Persistencia de rutas

Cada ruta generada se guarda automáticamente en SQLite (`db/routes.db`). El frontend incluye una página de **Historial** donde consultar, visualizar y eliminar rutas anteriores, incluyendo el mapa interactivo de cada una.

---

## Configuración (`config.json`)

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "log_level": "INFO"
  },
  "embeddings": {
    "provider": "huggingface",
    "model_name": "BAAI/bge-m3",
    "cache_dir": "models_cache"
  },
  "reranker": {
    "model_name": "BAAI/bge-reranker-v2-m3",
    "cache_dir": "models_cache",
    "enabled": true
  },
  "vector_db": {
    "provider": "chroma",
    "path": "db/chroma_db",
    "collection_name": "pois_turisticos"
  },
  "retrieval": {
    "mode": "hybrid",
    "fusion": "rrf",
    "rrf_k": 60,
    "linear_alpha": 0.5
  },
  "bm25": {
    "k1": 1.5,
    "b": 0.75
  },
  "rag": {
    "retrieval_k": 20,
    "rerank_top_n": 12,
    "min_score_threshold": 0.0
  },
  "planner": {
    "slots": {
      "manana": {"start": "09:30", "end": "14:00"},
      "tarde":  {"start": "16:00", "end": "20:00"}
    },
    "pois_per_day": {
      "tranquilo": 3,
      "moderado": 4,
      "intenso": 6
    },
    "walking_speed_kmh": 4.5,
    "avg_travel_minutes_intra_day": 15
  },
  "llm": {
    "ollama_base_url": "http://localhost:11434",
    "ollama_model_name": "qwen3:8b",
    "request_timeout_seconds": 1800,
    "temperature_generation": 0.5,
    "temperature_interpretation": 0.1,
    "max_tokens_generation": 4000
  },
  "poi_data": {
    "path": "data/pois_bilbao_bizkaia.json"
  },
  "scoring_weights": {
    "semantic": 0.50,
    "rerank": 0.20,
    "preference_match": 0.20,
    "spatial_diversity": 0.10
  }
}
```

Sin Ollama el sistema funciona igual (retrieval + ranking + planificación), solo la narrativa será básica.

---

## Endpoints principales

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/route` | POST | Genera una ruta completa |
| `/api/route/stream` | POST | Igual pero con Server-Sent Events (streaming) |
| `/api/pois` | GET | Lista todos los POIs (filtros: `?category=museo&municipality=Bilbao`) |
| `/api/pois/{id}` | GET | Detalle de un POI |
| `/api/pois/search` | POST | Búsqueda semántica libre sobre la colección |
| `/api/routes/saved` | GET | Lista el historial de rutas generadas |
| `/api/routes/saved/{id}` | GET | Detalle de una ruta guardada |
| `/api/routes/saved/{id}` | DELETE | Elimina una ruta del historial |
| `/api/health` | GET | Estado del sistema y modelos cargados |
| `/api/stats` | GET | Estadísticas de la colección de POIs |
| `/api/admin/reindex` | POST | Re-indexa todos los POIs en ChromaDB |

---

## Corpus

1109 POIs de Bilbao (465) y Bizkaia (644) con datos de OSM, Open Data Euskadi y Wikidata. Campos: nombre, municipio, categoría, subcategoría, coordenadas, horarios por día, precio, accesibilidad, tags, texto enriquecido para indexación.

El corpus se regenera con `python scripts/expand_bilbao_corpus.py`.

---

## Evaluación del retriever

```bash
# Denso puro
python -m evaluation.eval_retrieval --gold evaluation/gold_set.json --mode dense --output results/dense.json

# BM25 solo
python -m evaluation.eval_retrieval --gold evaluation/gold_set.json --mode bm25 --output results/bm25.json

# Híbrido (RRF)
python -m evaluation.eval_retrieval --gold evaluation/gold_set.json --mode hybrid --output results/hybrid.json

# Con reranker
python -m evaluation.eval_retrieval --gold evaluation/gold_set.json --mode dense --with-reranker

# Comparativa
python -m evaluation.eval_retrieval --gold evaluation/gold_set.json --mode compare --output results/compare.json
```

El gold set tiene 40 queries anotadas en 5 grupos: interés principal (10), multi-interés (5), restricciones (5), paráfrasis (10) y adversariales (10). Métricas: Recall@k, Precision@k, MRR, NDCG@10, MAP y latencia mediana.

### Barrido de hiperparámetros

```bash
# Búsqueda rápida (muestreo reducido de combinaciones)
python -m evaluation.hyperparameter_search --gold evaluation/gold_set.json --output results/hyper_search.json --fast --top 15

# Búsqueda completa con reranker
python -m evaluation.hyperparameter_search --gold evaluation/gold_set.json --output results/hyper_search.json --top 15

# Aplicar los mejores parámetros al config.json
python -m evaluation.hyperparameter_search --gold evaluation/gold_set.json --apply-best --config-path ./config.json
```

Explora combinaciones de `k1`, `b` (BM25), `rrf_k`, `linear_alpha` (fusión), `retrieval_k`, `rerank_top_n` y el uso del cross-encoder.

### CI/CD — GitHub Actions

El proyecto incluye dos workflows de evaluación automática (disparo manual desde la pestaña Actions):

| Workflow | Descripción |
|----------|-------------|
| `eval_retrieval.yml` | Evaluación comparativa del retriever (dense, BM25, híbrido, con/sin reranker) |
| `hyper_search.yml` | Barrido de hiperparámetros + comparativa con los mejores valores |

---

## Estructura del proyecto

```
├── .github/workflows/       # CI/CD (evaluación + barrido de hiperparámetros)
├── app/                     # Backend FastAPI
│   ├── main.py              # Endpoints y ciclo de vida
│   ├── config.py            # Configuración validada (Pydantic v2)
│   ├── models.py            # Modelos de datos
│   ├── interfaces.py        # Abstracciones (EmbeddingClient, VectorIndex)
│   ├── retriever.py         # Recuperación semántica (dense)
│   ├── hybrid_retriever.py  # Fusión dense + BM25 (RRF / lineal)
│   ├── ranker.py            # Reranking compuesto
│   ├── planner.py           # Planificación geográfica + 2-opt
│   ├── generator.py         # Narrativa LLM + interpretación de preferencias
│   ├── evaluator.py         # Métricas automáticas (7 métricas)
│   ├── poi_manager.py       # Carga e indexación de POIs
│   ├── route_store.py       # Persistencia SQLite (historial de rutas)
│   └── infra/
│       ├── embeddings_local.py   # BAAI/bge-m3 via SentenceTransformers
│       ├── vector_chroma.py      # ChromaDB con normalización de scores
│       └── bm25_index.py         # Índice léxico BM25 (español)
├── frontend/
│   └── app.py               # Streamlit (4 páginas)
├── evaluation/
│   ├── gold_set.json        # 40 queries anotadas
│   ├── eval_retrieval.py    # Script de evaluación del retriever
│   ├── metrics.py           # Métricas IR (Recall, Precision, MRR, NDCG, MAP)
│   └── hyperparameter_search.py  # Barrido de hiperparámetros (BM25, RRF, reranker)
├── tests/                   # Tests unitarios y E2E
├── data/
│   └── pois_bilbao_bizkaia.json
├── config.json
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── prefetch_models.py       # Descarga anticipada de modelos HuggingFace
```

---

## Referencias

- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
- Chen et al. (2024). BGE M3-Embedding. arXiv:2402.03216.
- Cormack et al. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. TREC.
- Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
- Järvelin & Kekäläinen (2002). Cumulated Gain-Based Evaluation of IR Techniques. TOIS.
