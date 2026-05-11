# Generador de Rutas Turísticas — Bilbao / Bizkaia

Sistema automático de generación de itinerarios turísticos personalizados para Bilbao y Bizkaia. Funciona completamente en local: no necesita claves de API ni conexión a internet.

---

## Arranque rápido

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

La primera vez descarga ~4 GB de modelos. Con Docker:

```bash
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

El cross-encoder BAAI/bge-reranker-v2-m3 reordena los candidatos. El score final combina semejanza vectorial (0.30), reranking (0.35), coincidencia con intereses (0.25) y diversidad de categorías (0.10). Los pesos se configuran en `config.json`.

### Planificación

Selecciona N POIs por día según ritmo (tranquilo=3, moderado=4, intenso=6), valida horarios de apertura por día de la semana y ordena geográficamente con vecino más próximo + 2-opt. Asigna franjas de mañana (09:30-14:00) y tarde (16:00-20:00).

La expansión semántica de queries busca automáticamente categorías del corpus cercanas a intereses sin mapeo explícito (ej. "surf" → playa, naturaleza).

### Generación y evaluación

El LLM local (Ollama) genera una narrativa en castellano con los datos del itinerario. El evaluador calcula 7 métricas automáticas: cobertura de preferencias, coherencia temporal, compacidad geográfica, adherencia al presupuesto, diversidad, accesibilidad y puntuación global.

---

## Configuración (`config.json`)

```json
{
  "retrieval": {
    "mode": "hybrid",
    "fusion": "rrf",
    "rrf_k": 60,
    "linear_alpha": 0.5
  },
  "reranker": {
    "model_name": "BAAI/bge-reranker-v2-m3",
    "enabled": true
  },
  "rag": {
    "retrieval_k": 20,
    "rerank_top_n": 12
  },
  "llm": {
    "ollama_base_url": "http://localhost:11434",
    "ollama_model_name": "qwen3:8b"
  },
  "scoring_weights": {
    "semantic": 0.30,
    "rerank": 0.35,
    "preference_match": 0.25,
    "spatial_diversity": 0.10
  }
}
```

Sin Ollama el sistema funciona igual (retrieval + ranking + planificación), solo la narrativa será básica.

---

## Endpoints principales

| Endpoint | Descripción |
|----------|-------------|
| `POST /api/route` | Genera una ruta completa |
| `POST /api/route/stream` | Igual pero con Server-Sent Events |
| `GET /api/pois` | Lista todos los POIs (filtros: `?category=museo&municipality=Bilbao`) |
| `GET /api/pois/{id}` | Detalle de un POI |
| `POST /api/pois/search` | Búsqueda semántica libre |
| `GET /api/health` | Estado del sistema |
| `GET /api/stats` | Estadísticas de la colección |

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
```

El gold set tiene 40 queries anotadas en 5 grupos: interés principal (10), multi-interés (5), restricciones (5), paráfrasis (10) y adversariales (10). Métricas: Recall@k, Precision@k, MRR, NDCG@10, MAP y latencia mediana.

---

## Estructura del proyecto

```
├── app/                  # Backend FastAPI
│   ├── main.py           # Endpoints y ciclo de vida
│   ├── config.py         # Configuración validada (Pydantic v2)
│   ├── models.py         # Modelos de datos
│   ├── retriever.py      # Recuperación semántica
│   ├── hybrid_retriever.py # Fusión dense + BM25 (RRF)
│   ├── ranker.py         # Reranking compuesto
│   ├── planner.py        # Planificación geográfica + 2-opt
│   ├── generator.py      # Narrativa LLM
│   ├── evaluator.py      # Métricas automáticas
│   ├── poi_manager.py    # Carga e indexación de POIs
│   ├── route_store.py    # Persistencia SQLite
│   └── infra/
│       ├── embeddings_local.py
│       ├── vector_chroma.py
│       └── bm25_index.py
├── frontend/
│   └── app.py            # Streamlit (4 páginas)
├── evaluation/
│   ├── gold_set.json     # 40 queries anotadas
│   ├── eval_retrieval.py # Script de evaluación
│   └── metrics.py        # Métricas IR
├── tests/                # Tests unitarios y E2E
├── data/
│   └── pois_bilbao_bizkaia.json
├── config.json
└── requirements.txt
```

---

## Referencias

- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
- Chen et al. (2024). BGE M3-Embedding. arXiv:2402.03216.
- Cormack et al. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. TREC.
- Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
- Järvelin & Kekäläinen (2002). Cumulated Gain-Based Evaluation of IR Techniques. TOIS.
