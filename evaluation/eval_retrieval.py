from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Aseguramos que la raíz del proyecto está en el path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.metrics import compute_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("eval_retrieval")


# ---------------------------------------------------------------------------
# Carga del gold set
# ---------------------------------------------------------------------------

def load_gold_set(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if "queries" not in data:
        raise ValueError("El gold set no contiene la clave 'queries'.")

    queries = []
    for i, q in enumerate(data["queries"]):
        for field in ("id", "query", "relevant_poi_ids", "highly_relevant_poi_ids"):
            if field not in q:
                raise ValueError(f"Query #{i} ({q.get('id', '?')}) falta el campo '{field}'.")
        queries.append(q)

    return queries


# ---------------------------------------------------------------------------
# Inicialización de componentes RAG
# ---------------------------------------------------------------------------

def _build_retriever(k: int):
    from app.config import settings
    from app.infra.embeddings_local import LocalHuggingFaceEmbeddings
    from app.infra.vector_chroma import LocalChromaIndex
    from app.poi_manager import POIManager
    from app.retriever import SemanticRetriever

    print("[eval_retrieval] [1/4] Cargando modelo de embeddings...")
    embedder = LocalHuggingFaceEmbeddings(
        model_name=settings.embeddings["model_name"],
        cache_dir=settings.embeddings.get("cache_dir", "models_cache"),
    )

    print("[eval_retrieval] [2/4] Conectando a ChromaDB...")
    vector_store = LocalChromaIndex(
        db_path=settings.vector_db["path"],
        collection_name=settings.vector_db["collection_name"],
    )

    print("[eval_retrieval] [3/4] Cargando POIs y verificando índice vectorial...")
    poi_manager = POIManager(embedder=embedder, vector_store=vector_store)
    poi_manager.load_pois()

    print("[eval_retrieval] [4/4] Construyendo SemanticRetriever...")
    retriever = SemanticRetriever(
        embedder=embedder,
        vector_store=vector_store,
        poi_manager=poi_manager,
        retrieval_k=k,
    )
    return retriever, poi_manager


def _build_reranker():
    from app.ranker import POIRanker
    print("[eval_retrieval] Cargando modelo de reranker (cross-encoder)...")
    return POIRanker()


# ---------------------------------------------------------------------------
# Funciones de retrieval por modo
# ---------------------------------------------------------------------------

def _enrich_eval_query(query: str, retriever) -> str:
    """Construye query enriquecida imitando _build_query en producción."""
    from app.retriever import INTEREST_QUERY_MAP, _build_query, _strip_negations
    from app.models import UserPreferences

    query = _strip_negations(query.lower())
    interests = []

    for interest, expansion in INTEREST_QUERY_MAP.items():
        if interest in query:
            interests.append(interest)
            continue
        for term in expansion.split():
            if len(term) > 3 and term in query:
                interests.append(interest)
                break

    scope = "Bizkaia" if any(w in query for w in ("bizkaia", "costa", "rural", "pueblos")) else "Bilbao"

    prefs = UserPreferences(
        interests=list(set(interests)) if interests else ["turismo"],
        city_scope=scope,
    )
    retriever._ensure_category_embeddings()
    return _build_query(prefs, retriever.embedder, retriever._category_embeddings)


def retrieve_dense(query: str, k: int, retriever) -> List[str]:
    """Devuelve lista de poi_ids usando solo el retriever semántico denso."""
    query = _enrich_eval_query(query, retriever)
    results = retriever.search_by_text(query, k=k)
    return [poi.id for poi, _ in results]


def retrieve_dense_reranked(query: str, k: int, retriever, reranker) -> List[str]:
    """Dense retrieval seguido de cross-encoder reranking."""
    from app.models import UserPreferences
    query = _enrich_eval_query(query, retriever)
    candidates = retriever.search_by_text(query, k=k)
    if not candidates:
        return []
    prefs = UserPreferences()  # preferencias neutras para el reranker
    ranked = reranker.rank(candidates, prefs, query, top_n=k)
    return [poi.id for poi, _, _, _ in ranked]


def retrieve_bm25(query: str, k: int, bm25_index, poi_manager) -> List[str]:
    """Retrieval solo BM25 sparse."""
    results = bm25_index.search(query, k=k)
    poi_ids = []
    all_pois = poi_manager.get_all()
    for doc_idx, score in results:
        if doc_idx < len(all_pois):
            poi_ids.append(all_pois[doc_idx].id)
    return poi_ids


def retrieve_hybrid(
    query: str,
    k: int,
    hybrid_retriever,
    poi_manager,
) -> List[str]:
    """Retrieval híbrido (dense + BM25 con RRF o linear)."""
    query = _enrich_eval_query(query, hybrid_retriever.dense)
    results = hybrid_retriever.retrieve_raw(query, k=k)
    return [poi.id for poi, _ in results]


# ---------------------------------------------------------------------------
# Evaluación principal
# ---------------------------------------------------------------------------

def evaluate(
    queries: List[Dict],
    retrieve_fn,
    k: int = 20,
) -> Dict[str, Any]:
    """
    Itera sobre el gold set, llama a retrieve_fn para cada query y acumula métricas.

    Args:
        queries:      Lista de queries del gold set.
        retrieve_fn:  Callable(query_text, k) → List[str] de poi_ids.
        k:            Número máximo de resultados a recuperar.

    Returns:
        Dict con métricas agregadas + métricas por query.
    """
    per_query: List[Dict] = []
    ks = [5, 10, 20]

    for q in queries:
        query_text = q["query"]
        relevant = q["relevant_poi_ids"]
        highly = q["highly_relevant_poi_ids"]

        t0 = time.perf_counter()
        retrieved = retrieve_fn(query_text, k)
        latency_ms = (time.perf_counter() - t0) * 1000

        metrics = compute_all(retrieved, relevant, highly, ks=ks)
        per_query.append({
            "id": q["id"],
            "group": q.get("group", ""),
            "query": query_text,
            "retrieved_ids": retrieved[:10],  # solo guardamos top-10 para no inflar el JSON
            "metrics": metrics,
            "latency_ms": round(latency_ms, 1),
        })

    # ---- Agregación ----
    aggregate = _aggregate_metrics(per_query)

    return {
        "aggregate": aggregate,
        "per_query": per_query,
    }


def _aggregate_metrics(per_query: List[Dict]) -> Dict[str, float]:
    """Media aritmética de cada métrica sobre todas las queries, incluyendo latencia."""
    if not per_query:
        return {}

    metric_keys = list(per_query[0]["metrics"].keys())
    agg = {k: 0.0 for k in metric_keys}
    latencies = []
    n = len(per_query)

    for pq in per_query:
        for k, v in pq["metrics"].items():
            agg[k] += v
        latencies.append(pq.get("latency_ms", 0))

    result = {k: round(v / n, 4) for k, v in agg.items()}
    if latencies:
        result["latency_median_ms"] = round(sorted(latencies)[len(latencies) // 2], 1)
    return result


# ---------------------------------------------------------------------------
# Formateo de resultados
# ---------------------------------------------------------------------------

def print_markdown_table(results: Dict[str, Any], mode_label: str) -> None:
    """Imprime una tabla markdown con los resultados agregados."""
    agg = results["aggregate"]
    print(f"\n## Resultados — {mode_label}\n")
    print(f"| Métrica     | Valor  |")
    print(f"|-------------|--------|")
    for metric, value in agg.items():
        print(f"| {metric:<11} | {value:.4f} |")

    # Tabla compacta para comparación rápida
    r5  = agg.get("recall@5", 0)
    r10 = agg.get("recall@10", 0)
    mrr_val  = agg.get("mrr", 0)
    ndcg_val = agg.get("ndcg@10", 0)
    map_val  = agg.get("ap", 0)
    lat_val  = agg.get("latency_median_ms", 0)
    print(f"\n### Resumen compacto\n")
    print(f"| Configuración | Recall@5 | Recall@10 | MRR    | NDCG@10 | MAP    | Latencia (ms) |")
    print(f"|---------------|----------|-----------|--------|---------|--------|---------------|")
    print(f"| {mode_label:<13} | {r5:.4f}   | {r10:.4f}    | {mrr_val:.4f} | {ndcg_val:.4f}  | {map_val:.4f} | {lat_val:.1f} |")


def print_compare_table(all_results: list) -> None:
    """Imprime una tabla comparativa unificada de todos los modos evaluados.

    Args:
        all_results: Lista de tuplas (label, results_dict) ordenadas.
    """
    print(f"\n## Comparativa de Modos de Retrieval\n")
    print(f"| Configuración      | Recall@5 | Recall@10 | MRR    | NDCG@10 | MAP    | Latencia (ms) |")
    print(f"|--------------------|----------|-----------|--------|---------|--------|---------------|")

    for label, res in all_results:
        agg = res["aggregate"]
        r5  = agg.get("recall@5", 0)
        r10 = agg.get("recall@10", 0)
        mrr_val  = agg.get("mrr", 0)
        ndcg_val = agg.get("ndcg@10", 0)
        map_val  = agg.get("ap", 0)
        lat_val  = agg.get("latency_median_ms", 0)
        print(f"| {label:<18} | {r5:.4f}   | {r10:.4f}    | {mrr_val:.4f} | {ndcg_val:.4f}  | {map_val:.4f} | {lat_val:.1f}        |")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación cuantitativa del retriever RAG turístico."
    )
    parser.add_argument(
        "--gold", required=True,
        help="Ruta al archivo gold_set.json",
    )
    parser.add_argument(
        "--mode", choices=["dense", "bm25", "hybrid", "compare"], default="dense",
        help="Modo de retrieval a evaluar. 'compare' ejecuta todos los modos.",
    )
    parser.add_argument(
        "--with-reranker", action="store_true",
        help="Aplica el cross-encoder reranker sobre los candidatos del retriever.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Peso del dense en fusión linear para modo hybrid (solo con fusion=linear).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Ruta de salida JSON para métricas detalladas.",
    )
    parser.add_argument(
        "--k", type=int, default=20,
        help="Número máximo de resultados a recuperar por query.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[eval_retrieval] Cargando gold set desde: {args.gold}")
    queries = load_gold_set(args.gold)
    print(f"[eval_retrieval] {len(queries)} queries cargadas.")

    if args.mode == "compare":
        # -----------------------------------------------------------------
        # Modo comparativa: ejecuta todos los modos secuencialmente
        # -----------------------------------------------------------------
        print("[eval_retrieval] Inicializando todos los componentes (puede tardar)...")
        retriever, poi_manager = _build_retriever(args.k)
        reranker = _build_reranker()

        # BM25 puro (índice sobre enriched_text)
        from app.infra.bm25_index import BM25Index
        print("[eval_retrieval] Construyendo índice BM25 puro...")
        bm25_index = BM25Index()
        all_pois = poi_manager.get_all()
        bm25_texts = [p.enriched_text for p in all_pois]
        bm25_index.build(bm25_texts)

        # BM25 para híbrido (índice sobre textos mapeados por HybridRetriever)
        from app.hybrid_retriever import HybridRetriever
        print("[eval_retrieval] Construyendo índice BM25 para híbridos...")
        hyb_texts, idx_map = HybridRetriever.build_bm25_mapping(all_pois)
        bm25_hybrid = BM25Index()
        bm25_hybrid.build(hyb_texts)

        # Híbrido RRF
        hybrid_rrf = HybridRetriever(
            dense_retriever=retriever,
            bm25_index=bm25_hybrid,
            poi_id_by_bm25_idx=idx_map,
            id_to_poi=poi_manager.get_by_id,
        )
        hybrid_rrf.fusion = "rrf"

        # Híbrido linear
        hybrid_linear = HybridRetriever(
            dense_retriever=retriever,
            bm25_index=bm25_hybrid,
            poi_id_by_bm25_idx=idx_map,
            id_to_poi=poi_manager.get_by_id,
        )
        hybrid_linear.fusion = "linear"
        if args.alpha is not None:
            hybrid_linear.linear_alpha = args.alpha

        print("[eval_retrieval] Construyendo índices BM25 e híbridos...")
        modes = [
            ("Dense (bge-m3)",      lambda q, k: retrieve_dense(q, k, retriever)),
            ("Dense + Reranker",     lambda q, k: retrieve_dense_reranked(q, k, retriever, reranker)),
            ("BM25",                 lambda q, k: retrieve_bm25(q, k, bm25_index, poi_manager)),
            ("Hybrid (rrf)",         lambda q, k: retrieve_hybrid(q, k, hybrid_rrf, poi_manager)),
            ("Hybrid (linear)",      lambda q, k: retrieve_hybrid(q, k, hybrid_linear, poi_manager)),
        ]

        all_results = []
        for label, fn in modes:
            print(f"\n[eval_retrieval] Evaluando modo '{label}' ({len(queries)} queries)...")
            res = evaluate(queries, fn, k=args.k)
            print_markdown_table(res, label)
            all_results.append((label, res))

        print_compare_table(all_results)

        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            out_data = {
                "mode": "compare",
                "k": args.k,
                "gold_set": args.gold,
                "n_queries": len(queries),
                "results": [{"label": label, **res} for label, res in all_results],
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out_data, f, ensure_ascii=False, indent=2)
            print(f"\n[eval_retrieval] Métricas guardadas en: {args.output}")

    elif args.mode == "bm25":
        # Modo BM25: construir índice desde POIs
        print("[eval_retrieval] Construyendo índice BM25...")
        retriever, poi_manager = _build_retriever(args.k)
        from app.infra.bm25_index import BM25Index
        bm25_index = BM25Index()
        all_pois = poi_manager.get_all()
        texts = [p.enriched_text for p in all_pois]
        bm25_index.build(texts)

        mode_label = "BM25"
        retrieve_fn = lambda q, k: retrieve_bm25(q, k, bm25_index, poi_manager)

        print(f"[eval_retrieval] Evaluando {len(queries)} queries en modo '{mode_label}'...")
        results = evaluate(queries, retrieve_fn, k=args.k)
        print_markdown_table(results, mode_label)

        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            out_data = {
                "mode": args.mode,
                "with_reranker": args.with_reranker,
                "k": args.k,
                "gold_set": args.gold,
                "n_queries": len(queries),
                **results,
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out_data, f, ensure_ascii=False, indent=2)
            print(f"\n[eval_retrieval] Métricas guardadas en: {args.output}")

    elif args.mode == "hybrid":
        # Modo híbrido: dense + BM25 con RRF/linear
        print("[eval_retrieval] Construyendo retriever híbrido...")
        retriever, poi_manager = _build_retriever(args.k)
        from app.infra.bm25_index import BM25Index
        from app.hybrid_retriever import HybridRetriever

        bm25_index = BM25Index()
        all_pois = poi_manager.get_all()
        texts, idx_map = HybridRetriever.build_bm25_mapping(all_pois)
        bm25_index.build(texts)

        hybrid = HybridRetriever(
            dense_retriever=retriever,
            bm25_index=bm25_index,
            poi_id_by_bm25_idx=idx_map,
            id_to_poi=poi_manager.get_by_id,
        )

        fusion = hybrid.fusion
        mode_label = f"Hybrid ({fusion})"
        if args.alpha is not None and fusion == "linear":
            hybrid.linear_alpha = args.alpha
            mode_label += f" alpha={args.alpha}"

        retrieve_fn = lambda q, k: retrieve_hybrid(q, k, hybrid, poi_manager)

        print(f"[eval_retrieval] Evaluando {len(queries)} queries en modo '{mode_label}'...")
        results = evaluate(queries, retrieve_fn, k=args.k)
        print_markdown_table(results, mode_label)

        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            out_data = {
                "mode": args.mode,
                "with_reranker": args.with_reranker,
                "k": args.k,
                "gold_set": args.gold,
                "n_queries": len(queries),
                **results,
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out_data, f, ensure_ascii=False, indent=2)
            print(f"\n[eval_retrieval] Métricas guardadas en: {args.output}")

    else:
        # ---- Modo dense ----
        print("[eval_retrieval] Inicializando componentes RAG (puede tardar)...")
        retriever, poi_manager = _build_retriever(args.k)

        reranker = None
        if args.with_reranker:
            print("[eval_retrieval] Cargando cross-encoder reranker...")
            reranker = _build_reranker()

        mode_label = "Dense (bge-m3)"
        if args.with_reranker:
            mode_label += " + Reranker"

        if reranker is not None:
            retrieve_fn = lambda q, k: retrieve_dense_reranked(q, k, retriever, reranker)
        else:
            retrieve_fn = lambda q, k: retrieve_dense(q, k, retriever)

        print(f"[eval_retrieval] Evaluando {len(queries)} queries en modo '{mode_label}'...")
        results = evaluate(queries, retrieve_fn, k=args.k)
        print_markdown_table(results, mode_label)

        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            out_data = {
                "mode": args.mode,
                "with_reranker": args.with_reranker,
                "k": args.k,
                "gold_set": args.gold,
                "n_queries": len(queries),
                **results,
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out_data, f, ensure_ascii=False, indent=2)
            print(f"\n[eval_retrieval] Métricas guardadas en: {args.output}")


if __name__ == "__main__":
    main()
