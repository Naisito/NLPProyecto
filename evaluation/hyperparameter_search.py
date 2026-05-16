"""
Barrido de hiperparámetros del sistema de retrieval.

Evalúa combinaciones de BM25 (k1, b), fusión (RRF k, linear alpha),
número de candidatos, reranking y rerank_top_n sobre el gold set,
ordenando las configuraciones por NDCG@10 y MAP.

Uso:
    python -m evaluation.hyperparameter_search \
        --gold evaluation/gold_set.json \
        --output results/hyper_search.json \
        --top 15
"""

from __future__ import annotations

import argparse
import json
import itertools
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.metrics import compute_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("hyper_search")

# ---------------------------------------------------------------------------
# Carga del gold set
# ---------------------------------------------------------------------------

def load_gold_set(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "queries" not in data:
        raise ValueError("El gold set no contiene la clave 'queries'.")
    for i, q in enumerate(data["queries"]):
        for field in ("id", "query", "relevant_poi_ids", "highly_relevant_poi_ids"):
            if field not in q:
                raise ValueError(f"Query #{i} falta el campo '{field}'.")
    return data["queries"]


# ---------------------------------------------------------------------------
# Evaluación sobre una función de retrieval
# ---------------------------------------------------------------------------

def evaluate_all(
    queries: List[Dict],
    retrieve_fn: Callable[[str, int], List[str]],
    k: int = 20,
) -> Dict[str, Any]:
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
            "metrics": metrics,
            "latency_ms": round(latency_ms, 1),
        })

    # Agregación
    if not per_query:
        return {"aggregate": {}, "per_query": []}

    metric_keys = list(per_query[0]["metrics"].keys())
    agg = {k: 0.0 for k in metric_keys}
    latencies = []
    n = len(per_query)

    for pq in per_query:
        for mk, mv in pq["metrics"].items():
            agg[mk] += mv
        latencies.append(pq.get("latency_ms", 0))

    aggregate = {k: round(v / n, 4) for k, v in agg.items()}
    if latencies:
        aggregate["latency_median_ms"] = round(sorted(latencies)[len(latencies) // 2], 1)

    # Desglose por grupo
    groups: Dict[str, List[Dict]] = {}
    for pq in per_query:
        g = pq["group"]
        groups.setdefault(g, []).append(pq)

    group_agg = {}
    for gname, gqueries in groups.items():
        gagg = {k: 0.0 for k in metric_keys}
        for pq in gqueries:
            for mk, mv in pq["metrics"].items():
                gagg[mk] += mv
        group_agg[gname] = {k: round(v / len(gqueries), 4) for k, v in gagg.items()}

    return {
        "aggregate": aggregate,
        "per_group": group_agg,
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Construcción de funciones de retrieval por configuración
# ---------------------------------------------------------------------------

def make_retrieval_fn(
    retriever,
    poi_manager,
    bm25_index,
    reranker,
    fusion: str,
    bm25_k1: float,
    bm25_b: float,
    rrf_k: int,
    linear_alpha: float,
    with_reranker: bool,
    rerank_top_n: int,
) -> Callable[[str, int], List[str]]:
    """Crea una función de retrieval para un conjunto de hiperparámetros."""
    from app.infra.bm25_index import BM25Index
    from app.hybrid_retriever import HybridRetriever, _rrf_fusion, _linear_fusion
    from app.models import UserPreferences

    # Ajustar BM25
    bm25_index.k1 = bm25_k1
    bm25_index.b = bm25_b

    # Construir HybridRetriever con los parámetros indicados
    all_pois = poi_manager.get_all()
    texts, idx_map = HybridRetriever.build_bm25_mapping(all_pois)
    hybrid = HybridRetriever(
        dense_retriever=retriever,
        bm25_index=bm25_index,
        poi_id_by_bm25_idx=idx_map,
        id_to_poi=poi_manager.get_by_id,
    )
    hybrid.fusion = fusion
    hybrid.rrf_k = rrf_k
    hybrid.linear_alpha = linear_alpha

    if with_reranker and reranker is not None and reranker.reranker_loaded:
        def fn(query: str, k: int) -> List[str]:
            raw = hybrid.retrieve_raw(query, k=k * 2)
            if not raw:
                return []
            prefs = UserPreferences()
            ranked = reranker.rank(raw, prefs, query, top_n=rerank_top_n)
            return [poi.id for poi, _, _, _ in ranked][:k]
    else:
        def fn(query: str, k: int) -> List[str]:
            results = hybrid.retrieve_raw(query, k=k)
            return [poi.id for poi, _ in results]

    return fn


# ---------------------------------------------------------------------------
# Barrido principal
# ---------------------------------------------------------------------------

def sweep(queries, retriever, poi_manager, bm25_index, reranker, args):
    """Ejecuta todas las combinaciones y devuelve los resultados ordenados."""

    # Parrilla de parámetros
    grid = {
        "bm25_k1":       [1.2, 1.5, 1.8, 2.0],
        "bm25_b":        [0.5, 0.75, 0.9],
        "retrieval_k":   [15, 20, 30],
        "fusion":        ["rrf", "linear"],
        "rrf_k":         [30, 60, 90],
        "linear_alpha":  [0.3, 0.5, 0.7],
        "with_reranker": [True, False],
        "rerank_top_n":  [8, 12, 16],
    }

    if args.fast:
        grid = {
            "bm25_k1":       [1.2, 1.5, 2.0],
            "bm25_b":        [0.5, 0.75],
            "retrieval_k":   [20, 30],
            "fusion":        ["rrf", "linear"],
            "rrf_k":         [30, 60],
            "linear_alpha":  [0.3, 0.5, 0.7],
            "with_reranker": [True, False],
            "rerank_top_n":  [8, 12],
        }

    if args.with_reranker_only:
        grid["with_reranker"] = [True]

    if args.no_reranker:
        grid["with_reranker"] = [False]
        grid["rerank_top_n"] = grid["rerank_top_n"][:1]  # no se usa, da igual el valor

    # Generar combinaciones (filtramos params irrelevantes según fusion)
    all_configs = []
    for bm25_k1 in grid["bm25_k1"]:
        for bm25_b in grid["bm25_b"]:
            for ret_k in grid["retrieval_k"]:
                for fusion in grid["fusion"]:
                    if fusion == "rrf":
                        rrf_params = [(rk, None) for rk in grid["rrf_k"]]
                    else:
                        rrf_params = [(-1, la) for la in grid["linear_alpha"]]

                    for rrf_k, linear_alpha in rrf_params:
                        for with_rr in grid["with_reranker"]:
                            if with_rr:
                                for rtn in grid["rerank_top_n"]:
                                    all_configs.append({
                                        "bm25_k1": bm25_k1,
                                        "bm25_b": bm25_b,
                                        "retrieval_k": ret_k,
                                        "fusion": fusion,
                                        "rrf_k": rrf_k if fusion == "rrf" else None,
                                        "linear_alpha": linear_alpha if fusion == "linear" else None,
                                        "with_reranker": True,
                                        "rerank_top_n": rtn,
                                    })
                            else:
                                all_configs.append({
                                    "bm25_k1": bm25_k1,
                                    "bm25_b": bm25_b,
                                    "retrieval_k": ret_k,
                                    "fusion": fusion,
                                    "rrf_k": rrf_k if fusion == "rrf" else None,
                                    "linear_alpha": linear_alpha if fusion == "linear" else None,
                                    "with_reranker": False,
                                    "rerank_top_n": None,
                                })

    total = len(all_configs)
    print(f"\n[hyper_search] {total} combinaciones a evaluar.\n")

    results: List[Dict] = []
    for i, cfg in enumerate(all_configs):
        # Label descriptivo
        if cfg["with_reranker"]:
            label = (
                f"k1={cfg['bm25_k1']:.1f} b={cfg['bm25_b']:.2f} "
                f"k={cfg['retrieval_k']} {cfg['fusion']}"
            )
            if cfg["fusion"] == "rrf":
                label += f"(rrf_k={cfg['rrf_k']})"
            else:
                label += f"(α={cfg['linear_alpha']})"
            label += f" +re {cfg['rerank_top_n']}"
        else:
            label = (
                f"k1={cfg['bm25_k1']:.1f} b={cfg['bm25_b']:.2f} "
                f"k={cfg['retrieval_k']} {cfg['fusion']}"
            )
            if cfg["fusion"] == "rrf":
                label += f"(rrf_k={cfg['rrf_k']})"
            else:
                label += f"(α={cfg['linear_alpha']})"

        print(f"[{i+1}/{total}] {label}...", end=" ", flush=True)

        try:
            fn = make_retrieval_fn(
                retriever=retriever,
                poi_manager=poi_manager,
                bm25_index=bm25_index,
                reranker=reranker,
                fusion=cfg["fusion"],
                bm25_k1=cfg["bm25_k1"],
                bm25_b=cfg["bm25_b"],
                rrf_k=cfg["rrf_k"] if cfg["rrf_k"] is not None else 60,
                linear_alpha=cfg["linear_alpha"] if cfg["linear_alpha"] is not None else 0.5,
                with_reranker=cfg["with_reranker"],
                rerank_top_n=cfg["rerank_top_n"] if cfg["rerank_top_n"] is not None else 12,
            )
            res = evaluate_all(queries, fn, k=cfg["retrieval_k"])
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        agg = res["aggregate"]
        print(
            f"R@10={agg.get('recall@10',0):.4f} "
            f"NDCG@10={agg.get('ndcg@10',0):.4f} "
            f"MAP={agg.get('ap',0):.4f}"
        )

        results.append({
            "config": cfg,
            "label": label,
            **res,
        })

    # Ordenar por NDCG@10 descendente (desempate: MAP)
    results.sort(
        key=lambda r: (
            r["aggregate"].get("ndcg@10", 0),
            r["aggregate"].get("ap", 0),
        ),
        reverse=True,
    )

    return results


# ---------------------------------------------------------------------------
# Salida
# ---------------------------------------------------------------------------

def print_top(results: List[Dict], top_n: int):
    """Imprime tabla markdown con las N mejores configuraciones."""
    print(f"\n## Top {top_n} configuraciones (ordenadas por NDCG@10 + MAP)\n")
    header = (
        "| # | Configuración | R@5 | R@10 | MRR | NDCG@10 | MAP | Lat(ms) |"
    )
    sep = (
        "|---|--------------|-----|------|-----|---------|-----|---------|"
    )
    print(header)
    print(sep)

    for rank, r in enumerate(results[:top_n], 1):
        a = r["aggregate"]
        r5  = a.get("recall@5", 0)
        r10 = a.get("recall@10", 0)
        mrr = a.get("mrr", 0)
        ndcg = a.get("ndcg@10", 0)
        ap  = a.get("ap", 0)
        lat = a.get("latency_median_ms", 0)
        print(
            f"| {rank} | {r['label'][:60]} | {r5:.4f} | {r10:.4f} | {mrr:.4f} | {ndcg:.4f} | {ap:.4f} | {lat:.0f} |"
        )

    # Detalle de la mejor
    best = results[0]
    print(f"\n---\n## Mejor configuración\n")
    print(f"```json")
    print(json.dumps(best["config"], indent=2, ensure_ascii=False))
    print(f"```\n")

    # Métricas por grupo de la mejor
    if "per_group" in best:
        pg = best["per_group"]
        print("### Métricas por grupo (mejor configuración)\n")
        print("| Grupo | R@5 | R@10 | MRR | NDCG@10 | MAP |")
        print("|-------|-----|------|-----|---------|-----|")
        for gname in sorted(pg.keys()):
            g = pg[gname]
            print(
                f"| {gname:<20} | {g.get('recall@5',0):.4f} | {g.get('recall@10',0):.4f} | "
                f"{g.get('mrr',0):.4f} | {g.get('ndcg@10',0):.4f} | {g.get('ap',0):.4f} |"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def apply_best_config(best_config: dict, config_path: str) -> None:
    """Parchea config.json con los mejores hiperparámetros encontrados."""
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    cfg.setdefault("bm25", {})
    cfg["bm25"]["k1"] = best_config["bm25_k1"]
    cfg["bm25"]["b"] = best_config["bm25_b"]

    cfg.setdefault("rag", {})
    cfg["rag"]["retrieval_k"] = best_config["retrieval_k"]
    if best_config.get("rerank_top_n") is not None:
        cfg["rag"]["rerank_top_n"] = best_config["rerank_top_n"]

    cfg.setdefault("retrieval", {})
    cfg["retrieval"]["fusion"] = best_config["fusion"]
    cfg["retrieval"]["rrf_k"] = best_config["rrf_k"]
    cfg["retrieval"]["linear_alpha"] = best_config["linear_alpha"]

    cfg.setdefault("reranker", {})
    cfg["reranker"]["enabled"] = best_config.get("with_reranker", False)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"\n[hyper_search] Mejor configuración aplicada a: {config_path}")
    print(json.dumps({
        "bm25": cfg["bm25"],
        "rag": cfg["rag"],
        "retrieval": cfg["retrieval"],
        "reranker": {"enabled": cfg["reranker"]["enabled"]},
    }, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Barrido de hiperparámetros del retriever turístico."
    )
    p.add_argument("--gold", required=True, help="Ruta al gold_set.json")
    p.add_argument("--output", default=None, help="Ruta de salida JSON")
    p.add_argument("--top", type=int, default=15, help="Cuántas mostrar en tabla")
    p.add_argument("--fast", action="store_true", help="Barrido reducido (más rápido)")
    p.add_argument("--no-reranker", dest="no_reranker", action="store_true",
                   help="Omitir reranker en todas las combinaciones (más rápido)")
    p.add_argument("--with-reranker-only", action="store_true",
                   help="Solo evaluar combinaciones con reranker")
    p.add_argument("--apply-best", action="store_true",
                   help="Parchea config.json con los mejores hiperparámetros encontrados")
    p.add_argument("--config-path", default=None,
                   help="Ruta al config.json (por defecto: ./config.json)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[hyper_search] Cargando gold set: {args.gold}")
    queries = load_gold_set(args.gold)
    print(f"[hyper_search] {len(queries)} queries cargadas.")

    # Cargar componentes una sola vez
    from app.config import settings
    from app.infra.embeddings_local import LocalHuggingFaceEmbeddings
    from app.infra.vector_chroma import LocalChromaIndex
    from app.infra.bm25_index import BM25Index
    from app.poi_manager import POIManager
    from app.retriever import SemanticRetriever
    from app.ranker import POIRanker

    print("[hyper_search] [1/5] Cargando modelo de embeddings...")
    embedder = LocalHuggingFaceEmbeddings(
        model_name=settings.embeddings["model_name"],
        cache_dir=settings.embeddings.get("cache_dir", "models_cache"),
    )

    print("[hyper_search] [2/5] Conectando a ChromaDB...")
    vector_store = LocalChromaIndex(
        db_path=settings.vector_db["path"],
        collection_name=settings.vector_db["collection_name"],
    )

    print("[hyper_search] [3/5] Cargando POIs...")
    poi_manager = POIManager(embedder=embedder, vector_store=vector_store)
    poi_manager.load_pois()

    retriever = SemanticRetriever(
        embedder=embedder,
        vector_store=vector_store,
        poi_manager=poi_manager,
        retrieval_k=30,  # usamos el máximo que necesitaremos
    )

    print("[hyper_search] [4/5] Construyendo índice BM25...")
    all_pois = poi_manager.get_all()
    bm25_texts = [p.enriched_text for p in all_pois]
    bm25_index = BM25Index()
    bm25_index.build(bm25_texts)

    print("[hyper_search] [5/5] Cargando reranker...")
    reranker = POIRanker()

    start = time.perf_counter()
    results = sweep(
        queries=queries,
        retriever=retriever,
        poi_manager=poi_manager,
        bm25_index=bm25_index,
        reranker=reranker,
        args=args,
    )
    elapsed = time.perf_counter() - start
    print(f"\n[hyper_search] Barrido completado en {elapsed:.1f} s ({len(results)} configs evaluadas).")

    print_top(results, args.top)

    if args.apply_best and results:
        config_path = args.config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.json",
        )
        apply_best_config(results[0]["config"], config_path)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out_data = {
            "gold_set": args.gold,
            "n_queries": len(queries),
            "total_configs": len(results),
            "top_n_shown": args.top,
            "best_config": results[0]["config"] if results else None,
            "results": [
                {"rank": i + 1, "config": r["config"], "aggregate": r["aggregate"]}
                for i, r in enumerate(results)
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        print(f"\n[hyper_search] Resultados guardados en: {args.output}")


if __name__ == "__main__":
    main()
