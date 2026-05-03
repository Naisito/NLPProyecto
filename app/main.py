"""
API REST del Generador de Rutas Turísticas — Bilbao / Bizkaia.

Pipeline completo (automatizado):
  Solicitud → Interpretación → Recuperación RAG → Reranking →
  Planificación → Generación narrativa → Evaluación → Respuesta

Endpoints principales:
  POST /api/route          — genera una ruta turística completa
  GET  /api/pois           — lista todos los POIs con filtros opcionales
  GET  /api/pois/{id}      — detalle de un POI
  POST /api/pois/search    — búsqueda semántica libre
  GET  /api/health         — estado del sistema
  GET  /api/stats          — estadísticas de la colección
  POST /api/admin/reindex  — re-indexa todos los POIs (admin)
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.evaluator import evaluate_route
from app.generator import assemble_route, generate_narrative, interpret_preferences
from app.infra.embeddings_local import LocalHuggingFaceEmbeddings
from app.infra.vector_chroma import LocalChromaIndex
from app.models import (
    EvaluationMetrics,
    HealthResponse,
    POI,
    POIListResponse,
    POISearchRequest,
    RouteRequest,
    RouteResponse,
    TouristRoute,
    UserPreferences,
)
from app.planner import ItineraryPlanner
from app.poi_manager import POIManager
from app.ranker import POIRanker
from app.retriever import SemanticRetriever

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=settings.server.get("log_level", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("turismo_rag")

# ---------------------------------------------------------------------------
# Estado global de la aplicación
# ---------------------------------------------------------------------------
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicialización y cierre de recursos al arrancar/parar la API."""
    logger.info("Iniciando sistema RAG turístico…")
    t0 = time.time()

    # 1. Infraestructura vectorial
    embedder = LocalHuggingFaceEmbeddings(
        model_name=settings.embeddings["model_name"],
        cache_dir=settings.embeddings["cache_dir"],
    )
    vector_store = LocalChromaIndex(
        db_path=settings.vector_db["path"],
        collection_name=settings.vector_db["collection_name"],
    )

    # 2. Gestor de POIs (carga + indexación)
    poi_manager = POIManager(embedder=embedder, vector_store=vector_store)
    n = poi_manager.load_pois()
    logger.info(f"{n} POIs cargados y listos.")

    # 3. Retriever semántico
    retriever = SemanticRetriever(
        embedder=embedder,
        vector_store=vector_store,
        poi_manager=poi_manager,
        retrieval_k=settings.rag.get("retrieval_k", 20),
    )

    # 4. Ranker (cross-encoder + scoring compuesto)
    ranker = POIRanker()

    # 5. Planificador
    planner = ItineraryPlanner()

    _state.update({
        "embedder":    embedder,
        "vector_store": vector_store,
        "poi_manager": poi_manager,
        "retriever":   retriever,
        "ranker":      ranker,
        "planner":     planner,
    })

    elapsed = time.time() - t0
    logger.info(f"Sistema listo en {elapsed:.1f}s.")
    yield

    logger.info("Apagando sistema RAG turístico.")


# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Generador de Rutas Turísticas — Bilbao / Bizkaia",
    description=(
        "Sistema RAG híbrido para la generación personalizada de rutas turísticas "
        "en Bilbao y Bizkaia. Pipeline: interpretación LLM → recuperación semántica "
        "(BAAI/bge-m3) → reranking (cross-encoder multilingual) → planificación "
        "geográfica → generación narrativa (Ollama local) → evaluación automática."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _get(key: str):
    if key not in _state:
        raise HTTPException(status_code=503, detail="Sistema no inicializado.")
    return _state[key]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse, tags=["Sistema"])
def health():
    """Comprueba el estado del sistema y los modelos cargados."""
    poi_mgr: POIManager = _state.get("poi_manager")
    ranker:  POIRanker  = _state.get("ranker")
    return HealthResponse(
        status="ok" if poi_mgr and poi_mgr.is_loaded else "degraded",
        index_size=poi_mgr.total if poi_mgr else 0,
        model_loaded=True,
        reranker_loaded=ranker.reranker_loaded if ranker else False,
    )


@app.get("/api/stats", tags=["Sistema"])
def stats():
    """Estadísticas de la colección de POIs."""
    poi_mgr: POIManager = _get("poi_manager")
    return {
        "total_pois":     poi_mgr.total,
        "categories":     poi_mgr.get_categories(),
        "municipalities": poi_mgr.get_municipalities(),
        "index_vectors":  _get("vector_store").count(),
        "corpus_load":    poi_mgr.load_summary,
    }


@app.post("/api/route", response_model=RouteResponse, tags=["Rutas"])
def generate_route(request: RouteRequest):
    """
    Genera una ruta turística personalizada.

    Acepta:
    - `query`: texto libre (ej. «3 días en Bilbao con mis hijos, amantes de la naturaleza»)
    - `preferences`: objeto estructurado con las preferencias del usuario

    Si se proporcionan ambos, el sistema fusiona la query con las preferencias
    explícitas. Si sólo se indica query, el LLM la interpreta automáticamente.
    """
    t_start = time.time()

    poi_mgr:  POIManager       = _get("poi_manager")
    retriever: SemanticRetriever = _get("retriever")
    ranker:   POIRanker        = _get("ranker")
    planner:  ItineraryPlanner = _get("planner")

    # ---- 1. Interpretar preferencias ----------------------------------
    if request.query and not request.preferences:
        preferences = interpret_preferences(request.query)
    elif request.preferences:
        preferences = request.preferences
        if request.query:
            # Fusionar: las preferencias explícitas tienen prioridad
            interpreted = interpret_preferences(request.query)
            # Completar sólo los campos que no vienen explícitos
            if not preferences.interests and interpreted.interests:
                preferences.interests = interpreted.interests
            if preferences.extra_notes is None and interpreted.extra_notes:
                preferences.extra_notes = interpreted.extra_notes
    else:
        # Sin input: valores por defecto
        preferences = UserPreferences()

    logger.info(
        f"Generando ruta: scope={preferences.city_scope} "
        f"days={preferences.duration_days} "
        f"interests={preferences.interests} "
        f"pace={preferences.pace}"
    )

    # ---- 2. Recuperación semántica (RAG) ------------------------------
    # Ajuste dinámico: calcular cuántos candidatos necesitamos en función
    # del número de días solicitados y el ritmo (pois por día).
    per_day = planner.pois_per_day.get(preferences.pace, 4)
    desired_total = preferences.duration_days * per_day
    # Margen para compensar filtros y descartes posteriores
    margin = 2
    default_k = settings.rag.get("retrieval_k", 20)
    dynamic_k = min(poi_mgr.total, max(default_k, int(desired_total * margin), desired_total + 8))
    logger.info(f"Recuperando candidatos semánticos: desired={desired_total} margin={margin} k={dynamic_k}")

    candidates = retriever.retrieve(
        preferences=preferences,
        k=dynamic_k,
    )

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="No se encontraron POIs relevantes para las preferencias indicadas.",
        )

    # Construir query enriquecida para el cross-encoder
    query_for_rerank = request.query or " ".join(preferences.interests) or "turismo Bilbao"

    # ---- 3. Reranking -------------------------------------------------
    # Rerank: ajustar top_n también proporcionalmente para no recortar
    # los candidatos necesarios para todos los días.
    default_top_n = settings.rag.get("rerank_top_n", 12)
    dynamic_top_n = min(poi_mgr.total, max(default_top_n, int(desired_total * margin)))
    logger.info(f"Reranking: top_n={dynamic_top_n}")

    ranked = ranker.rank(
        candidates=candidates,
        preferences=preferences,
        query=query_for_rerank,
        top_n=dynamic_top_n,
    )

    # ---- 4. Planificación del itinerario ------------------------------
    from datetime import date
    today_weekday = date.today().weekday()   # 0=lunes

    days = planner.plan(
        ranked=ranked,
        preferences=preferences,
        start_weekday=today_weekday,
    )

    if not any(d.pois for d in days):
        raise HTTPException(
            status_code=422,
            detail="El planificador no pudo construir un itinerario válido. "
                   "Prueba a cambiar el ámbito geográfico o las fechas.",
        )

    # ---- 5. Generación narrativa ---------------------------------------
    narrative = generate_narrative(
        days=days,
        preferences=preferences,
        original_query=request.query,
    )

    # ---- 6. Ensamblado de la ruta ------------------------------------
    route = assemble_route(days=days, preferences=preferences, narrative=narrative)

    # ---- 7. Evaluación automática -------------------------------------
    evaluation = evaluate_route(
        route=route,
        preferences=preferences,
        start_weekday=today_weekday,
    )

    # ---- 8. Info de trazabilidad ------------------------------------
    retrieval_info = {
        "candidates_retrieved":   len(candidates),
        "candidates_after_rerank": len(ranked),
        "reranker_used":          ranker.reranker_loaded,
        "embedding_model":        settings.embeddings["model_name"],
        "reranker_model":         settings.reranker.get("model_name", "N/A"),
        "llm_model":              settings.llm.get("ollama_model_name"),
        "top_candidates": [
            {
                "id":    r[0].id,
                "name":  r[0].name,
                "s_score":  round(r[1], 4),
                "r_score":  round(r[2], 4),
                "final":    round(r[3], 4),
            }
            for r in ranked[:5]
        ],
    }

    return RouteResponse(
        route=route,
        evaluation=evaluation,
        retrieval_info=retrieval_info,
        execution_time_seconds=round(time.time() - t_start, 2),
    )


@app.get("/api/pois", response_model=POIListResponse, tags=["POIs"])
def list_pois(
    category: Optional[str] = Query(None, description="Filtrar por categoría"),
    municipality: Optional[str] = Query(None, description="Filtrar por municipio"),
):
    """Lista todos los POIs, con filtros opcionales."""
    poi_mgr: POIManager = _get("poi_manager")
    pois = poi_mgr.get_all()

    if category:
        pois = [p for p in pois if category.lower() in p.category.lower()]
    if municipality:
        pois = [p for p in pois if municipality.lower() in p.municipality.lower()]

    return POIListResponse(total=len(pois), pois=pois)


@app.get("/api/pois/{poi_id}", response_model=POI, tags=["POIs"])
def get_poi(poi_id: str):
    """Obtiene el detalle de un POI por su ID."""
    poi_mgr: POIManager = _get("poi_manager")
    poi = poi_mgr.get_by_id(poi_id)
    if not poi:
        raise HTTPException(status_code=404, detail=f"POI '{poi_id}' no encontrado.")
    return poi


@app.post("/api/pois/search", tags=["POIs"])
def search_pois(request: POISearchRequest):
    """Búsqueda semántica libre sobre la colección de POIs."""
    retriever: SemanticRetriever = _get("retriever")
    results = retriever.search_by_text(query=request.query, k=request.k)

    filtered = []
    for poi, score in results:
        if request.category_filter and request.category_filter.lower() not in poi.category.lower():
            continue
        if request.municipality_filter and request.municipality_filter.lower() not in poi.municipality.lower():
            continue
        filtered.append({"poi": poi, "score": round(score, 4)})

    return {"query": request.query, "total": len(filtered), "results": filtered}


@app.post("/api/admin/reindex", tags=["Admin"])
def reindex():
    """Vacía y re-indexa todos los POIs en ChromaDB. Útil tras actualizar el JSON."""
    poi_mgr: POIManager = _get("poi_manager")
    poi_mgr.reindex()
    return {"status": "ok", "indexed": poi_mgr.total}
