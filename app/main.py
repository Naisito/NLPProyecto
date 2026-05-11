import json as _json
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import settings
from app.evaluator import evaluate_route
from app.generator import assemble_route, generate_narrative, generate_narrative_stream, interpret_preferences
from app.hybrid_retriever import HybridRetriever
from app.infra.bm25_index import BM25Index
from app.route_store import delete_route, get_route, init_db, list_routes, save_route
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

logging.basicConfig(
    level=settings.server.get("log_level", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("turismo_rag")

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando sistema RAG turístico…")
    t0 = time.time()

    embedder = LocalHuggingFaceEmbeddings(
        model_name=settings.embeddings["model_name"],
        cache_dir=settings.embeddings["cache_dir"],
    )
    vector_store = LocalChromaIndex(
        db_path=settings.vector_db["path"],
        collection_name=settings.vector_db["collection_name"],
    )

    poi_manager = POIManager(embedder=embedder, vector_store=vector_store)
    n = poi_manager.load_pois()
    logger.info(f"{n} POIs cargados y listos.")

    retriever = SemanticRetriever(
        embedder=embedder,
        vector_store=vector_store,
        poi_manager=poi_manager,
        retrieval_k=settings.rag.get("retrieval_k", 20),
    )

    bm25_index = BM25Index()
    bm25_path = "db/bm25.pkl"
    bm25_poi_id_by_idx: dict = {}
    retrieval_mode = settings.retrieval.get("mode", "dense")

    if retrieval_mode in ("bm25", "hybrid"):
        if bm25_index.load(bm25_path):
            logger.info("BM25 index reutilizado desde disco.")
        else:
            all_pois = poi_manager.get_all()
            texts, mapping = HybridRetriever.build_bm25_mapping(all_pois)
            bm25_index.build(texts)
            bm25_index.persist(bm25_path)
            logger.info("BM25 index construido y persistido.")
        all_pois = poi_manager.get_all()
        bm25_poi_id_by_idx = {i: p.id for i, p in enumerate(all_pois)}

    hybrid_retriever = None
    if retrieval_mode == "hybrid":
        hybrid_retriever = HybridRetriever(
            dense_retriever=retriever,
            bm25_index=bm25_index,
            poi_id_by_bm25_idx=bm25_poi_id_by_idx,
            id_to_poi=poi_manager.get_by_id,
        )
        logger.info("HybridRetriever inicializado (dense + BM25).")

    if retrieval_mode == "bm25":
        from app.retriever import SCOPE_FILTER as _scope
        _poi_list = poi_manager.get_all()
        _poi_map = {p.id: p for p in _poi_list}
        _bm25_idx_map = {i: p.id for i, p in enumerate(_poi_list)}

        class _BM25OnlyRetriever:
            def __init__(self, bm25, idx_map, poi_map, dense_retriever):
                self.bm25 = bm25
                self._idx_map = idx_map
                self._poi_map = poi_map
                self._dense = dense_retriever

            def retrieve(self, preferences, k=20):
                from app.retriever import _build_query as _bq
                query = _bq(preferences)
                results = self.bm25.search(query, k=k * 2)
                candidates = []
                allowed = _scope.get(preferences.city_scope)
                allowed_set = set(m.lower() for m in allowed) if allowed else None
                for doc_idx, score in results:
                    poi_id = self._idx_map.get(doc_idx)
                    if poi_id is None:
                        continue
                    poi = self._poi_map.get(poi_id)
                    if poi is None:
                        continue
                    if allowed_set and poi.municipality.lower() not in allowed_set:
                        continue
                    if preferences.mobility == "reducida" and not poi.accessibility:
                        continue
                    daily_budget = preferences.budget_per_day
                    if poi.price_numeric > daily_budget * 0.8 and poi.price_numeric > 0:
                        score *= 0.7
                    if daily_budget < 30 and poi.price_numeric == 0.0:
                        score *= 1.1
                    candidates.append((poi, score))
                    if len(candidates) >= k:
                        break
                return candidates

            def search_by_text(self, query, k=10):
                results = self.bm25.search(query, k=k)
                out = []
                for doc_idx, score in results:
                    poi_id = self._idx_map.get(doc_idx)
                    if poi_id:
                        poi = self._poi_map.get(poi_id)
                        if poi:
                            out.append((poi, score))
                return out

        bm25_only = _BM25OnlyRetriever(bm25_index, _bm25_idx_map, _poi_map, retriever)
        active_retriever = bm25_only
    elif retrieval_mode == "hybrid":
        active_retriever = hybrid_retriever
    else:
        active_retriever = retriever

    ranker = POIRanker()
    planner = ItineraryPlanner()

    _state.update({
        "embedder":         embedder,
        "vector_store":     vector_store,
        "poi_manager":      poi_manager,
        "retriever":        retriever,
        "bm25_index":       bm25_index,
        "hybrid_retriever": hybrid_retriever,
        "active_retriever": active_retriever,
        "ranker":           ranker,
        "planner":          planner,
    })

    init_db()

    elapsed = time.time() - t0
    logger.info(f"Sistema listo en {elapsed:.1f}s.")
    yield
    logger.info("Apagando sistema RAG turístico.")


app = FastAPI(
    title="Generador de Rutas Turísticas — Bilbao / Bizkaia",
    description="Sistema RAG híbrido: interpretación LLM → recuperación (dense + BM25) → reranking → planificación → narrativa → evaluación.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get(key: str):
    if key not in _state:
        raise HTTPException(status_code=503, detail="Sistema no inicializado.")
    return _state[key]


# ----- Endpoints -------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse, tags=["Sistema"])
def health():
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
    t_start = time.time()

    poi_mgr:  POIManager       = _get("poi_manager")
    retriever = _get("active_retriever")
    ranker:   POIRanker        = _get("ranker")
    planner:  ItineraryPlanner = _get("planner")

    if request.query and not request.preferences:
        preferences = interpret_preferences(request.query)
    elif request.preferences:
        preferences = request.preferences
        if request.query:
            interpreted = interpret_preferences(request.query)
            if not preferences.interests and interpreted.interests:
                preferences.interests = interpreted.interests
            if preferences.extra_notes is None and interpreted.extra_notes:
                preferences.extra_notes = interpreted.extra_notes
    else:
        preferences = UserPreferences()

    logger.info(
        f"Generando ruta: scope={preferences.city_scope} "
        f"days={preferences.duration_days} "
        f"interests={preferences.interests} pace={preferences.pace}"
    )

    per_day = planner.pois_per_day.get(preferences.pace, 4)
    desired_total = preferences.duration_days * per_day
    margin = 2
    default_k = settings.rag.get("retrieval_k", 20)
    dynamic_k = min(poi_mgr.total, max(default_k, int(desired_total * margin), desired_total + 8))

    candidates = retriever.retrieve(preferences=preferences, k=dynamic_k)

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="No se encontraron POIs relevantes para las preferencias indicadas.",
        )

    query_for_rerank = request.query or " ".join(preferences.interests) or "turismo Bilbao"

    default_top_n = settings.rag.get("rerank_top_n", 12)
    dynamic_top_n = min(poi_mgr.total, max(default_top_n, int(desired_total * margin)))

    ranked = ranker.rank(
        candidates=candidates,
        preferences=preferences,
        query=query_for_rerank,
        top_n=dynamic_top_n,
    )

    from datetime import date
    today_weekday = date.today().weekday()

    days = planner.plan(ranked=ranked, preferences=preferences, start_weekday=today_weekday)

    if not any(d.pois for d in days):
        raise HTTPException(
            status_code=422,
            detail="El planificador no pudo construir un itinerario válido. Prueba a cambiar el ámbito o las fechas.",
        )

    narrative = generate_narrative(days=days, preferences=preferences, original_query=request.query)
    route = assemble_route(days=days, preferences=preferences, narrative=narrative)
    evaluation = evaluate_route(route=route, preferences=preferences, start_weekday=today_weekday)

    retrieval_info = {
        "candidates_retrieved":   len(candidates),
        "candidates_after_rerank": len(ranked),
        "reranker_used":          ranker.reranker_loaded,
        "embedding_model":        settings.embeddings["model_name"],
        "reranker_model":         settings.reranker.get("model_name", "N/A"),
        "llm_model":              settings.llm.get("ollama_model_name"),
        "top_candidates": [
            {
                "id": r[0].id, "name": r[0].name,
                "s_score": round(r[1], 4), "r_score": round(r[2], 4), "final": round(r[3], 4),
            }
            for r in ranked[:5]
        ],
    }

    exec_secs = round(time.time() - t_start, 2)
    response = RouteResponse(
        route=route,
        evaluation=evaluation,
        retrieval_info=retrieval_info,
        execution_time_seconds=exec_secs,
    )
    save_route(response.model_dump(), request.query or "", exec_secs)
    return response


@app.post("/api/route/stream", tags=["Rutas"])
def generate_route_stream(request: RouteRequest):
    def _sse(type: str, **kwargs) -> str:
        return f"data: {_json.dumps({'type': type, **kwargs}, ensure_ascii=False)}\n\n"

    def event_gen():
        t_start = time.time()
        try:
            poi_mgr:   POIManager        = _get("poi_manager")
            retriever = _get("active_retriever")
            ranker:    POIRanker         = _get("ranker")
            planner:   ItineraryPlanner  = _get("planner")

            yield _sse("status", stage="interpret", message="Interpretando preferencias...")
            if request.query and not request.preferences:
                preferences = interpret_preferences(request.query)
            elif request.preferences:
                preferences = request.preferences
                if request.query:
                    interpreted = interpret_preferences(request.query)
                    if not preferences.interests and interpreted.interests:
                        preferences.interests = interpreted.interests
                    if preferences.extra_notes is None and interpreted.extra_notes:
                        preferences.extra_notes = interpreted.extra_notes
            else:
                preferences = UserPreferences()

            per_day       = planner.pois_per_day.get(preferences.pace, 4)
            desired_total = preferences.duration_days * per_day
            margin        = 2
            default_k     = settings.rag.get("retrieval_k", 20)
            dynamic_k     = min(poi_mgr.total, max(default_k, int(desired_total * margin), desired_total + 8))

            yield _sse("status", stage="rag", message=f"Recuperando POIs en {preferences.city_scope}...")
            candidates = retriever.retrieve(preferences=preferences, k=dynamic_k)
            if not candidates:
                yield _sse("error", message="No se encontraron POIs relevantes.")
                return

            query_for_rerank = request.query or " ".join(preferences.interests) or "turismo Bilbao"
            default_top_n    = settings.rag.get("rerank_top_n", 12)
            dynamic_top_n    = min(poi_mgr.total, max(default_top_n, int(desired_total * margin)))

            yield _sse("status", stage="rerank", message="Reordenando candidatos...")
            ranked = ranker.rank(candidates=candidates, preferences=preferences, query=query_for_rerank, top_n=dynamic_top_n)

            yield _sse("status", stage="plan", message="Construyendo itinerario...")
            from datetime import date
            today_weekday = date.today().weekday()
            days = planner.plan(ranked=ranked, preferences=preferences, start_weekday=today_weekday)
            if not any(d.pois for d in days):
                yield _sse("error", message="No se pudo construir un itinerario válido.")
                return

            yield _sse("status", stage="narrative", message=f"Generando narrativa ({preferences.duration_days} día(s))...")
            narrative_chunks: list[str] = []
            for chunk in generate_narrative_stream(days=days, preferences=preferences, original_query=request.query):
                narrative_chunks.append(chunk)
                yield _sse("narrative_chunk", text=chunk)
            narrative = "".join(narrative_chunks).strip()

            yield _sse("status", stage="evaluate", message="Evaluando ruta...")
            route      = assemble_route(days=days, preferences=preferences, narrative=narrative)
            evaluation = evaluate_route(route=route, preferences=preferences, start_weekday=today_weekday)

            retrieval_info = {
                "candidates_retrieved":    len(candidates),
                "candidates_after_rerank": len(ranked),
                "reranker_used":           ranker.reranker_loaded,
                "embedding_model":         settings.embeddings["model_name"],
                "reranker_model":          settings.reranker.get("model_name", "N/A"),
                "llm_model":               settings.llm.get("ollama_model_name"),
                "top_candidates": [
                    {"id": r[0].id, "name": r[0].name, "s_score": round(r[1], 4), "r_score": round(r[2], 4), "final": round(r[3], 4)}
                    for r in ranked[:5]
                ],
            }
            exec_secs = round(time.time() - t_start, 2)
            response_obj = RouteResponse(
                route=route,
                evaluation=evaluation,
                retrieval_info=retrieval_info,
                execution_time_seconds=exec_secs,
            )
            route_data = response_obj.model_dump()
            save_route(route_data, request.query or "", exec_secs)
            yield _sse("result", data=route_data)

        except Exception as e:
            logger.exception("Error en generate_route_stream")
            yield _sse("error", message=str(e))

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/api/routes/saved", tags=["Rutas"])
def list_saved_routes(limit: int = Query(100, ge=1, le=500)):
    return list_routes(limit=limit)


@app.get("/api/routes/saved/{route_id}", tags=["Rutas"])
def get_saved_route(route_id: str):
    data = get_route(route_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Ruta no encontrada.")
    return data


@app.delete("/api/routes/saved/{route_id}", tags=["Rutas"])
def delete_saved_route(route_id: str):
    if not delete_route(route_id):
        raise HTTPException(status_code=404, detail="Ruta no encontrada.")
    return {"ok": True}


@app.get("/api/pois", response_model=POIListResponse, tags=["POIs"])
def list_pois(
    category: Optional[str] = Query(None),
    municipality: Optional[str] = Query(None),
):
    poi_mgr: POIManager = _get("poi_manager")
    pois = poi_mgr.get_all()

    if category:
        pois = [p for p in pois if category.lower() in p.category.lower()]
    if municipality:
        pois = [p for p in pois if municipality.lower() in p.municipality.lower()]

    return POIListResponse(total=len(pois), pois=pois)


@app.get("/api/pois/{poi_id}", response_model=POI, tags=["POIs"])
def get_poi(poi_id: str):
    poi_mgr: POIManager = _get("poi_manager")
    poi = poi_mgr.get_by_id(poi_id)
    if not poi:
        raise HTTPException(status_code=404, detail=f"POI '{poi_id}' no encontrado.")
    return poi


@app.post("/api/pois/search", tags=["POIs"])
def search_pois(request: POISearchRequest):
    retriever = _get("active_retriever")
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
    poi_mgr: POIManager = _get("poi_manager")
    poi_mgr.reindex()
    return {"status": "ok", "indexed": poi_mgr.total}
