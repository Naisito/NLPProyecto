import logging
from typing import Dict, List, Optional, Tuple

from app.config import settings
from app.infra.bm25_index import BM25Index
from app.models import POI, UserPreferences
from app.retriever import SCOPE_FILTER, SemanticRetriever, _build_query

logger = logging.getLogger("turismo_rag")


def _rrf_fusion(
    dense_ranked: List[Tuple[POI, float]],
    bm25_ranked: List[Tuple[POI, float]],
    rrf_k: int = 60,
) -> List[Tuple[POI, float]]:
    """Reciprocal Rank Fusion: score(poi) = sum(1 / (k + rank))."""
    poi_scores: Dict[str, float] = {}
    poi_map: Dict[str, POI] = {}

    for rank, (poi, _) in enumerate(dense_ranked, start=1):
        rrf = 1.0 / (rrf_k + rank)
        poi_scores[poi.id] = poi_scores.get(poi.id, 0.0) + rrf
        poi_map[poi.id] = poi

    for rank, (poi, _) in enumerate(bm25_ranked, start=1):
        rrf = 1.0 / (rrf_k + rank)
        poi_scores[poi.id] = poi_scores.get(poi.id, 0.0) + rrf
        poi_map[poi.id] = poi

    sorted_ids = sorted(poi_scores.items(), key=lambda x: x[1], reverse=True)
    return [(poi_map[poi_id], score) for poi_id, score in sorted_ids]


def _linear_fusion(
    dense_ranked: List[Tuple[POI, float]],
    bm25_ranked: List[Tuple[POI, float]],
    alpha: float = 0.5,
) -> List[Tuple[POI, float]]:
    """Fusión lineal: alpha * dense_norm + (1-alpha) * bm25_norm."""

    def _norm(scores: List[float]) -> List[float]:
        if not scores:
            return []
        max_s = max(scores)
        if max_s <= 0:
            return [0.0] * len(scores)
        return [s / max_s for s in scores]

    dense_pois = [p for p, _ in dense_ranked]
    bm25_pois = [p for p, _ in bm25_ranked]
    dense_raw = [s for _, s in dense_ranked]
    bm25_raw = [s for _, s in bm25_ranked]

    d_norm = _norm(dense_raw)
    b_norm = _norm(bm25_raw)

    poi_scores: Dict[str, float] = {}
    poi_map: Dict[str, POI] = {}

    for poi, ns in zip(dense_pois, d_norm):
        poi_scores[poi.id] = poi_scores.get(poi.id, 0.0) + alpha * ns
        poi_map[poi.id] = poi

    for poi, ns in zip(bm25_pois, b_norm):
        poi_scores[poi.id] = poi_scores.get(poi.id, 0.0) + (1 - alpha) * ns
        poi_map[poi.id] = poi

    sorted_ids = sorted(poi_scores.items(), key=lambda x: x[1], reverse=True)
    return [(poi_map[poi_id], score) for poi_id, score in sorted_ids]


class HybridRetriever:
    """Fusión dense (SemanticRetriever) + sparse (BM25Index) con RRF o combinación lineal."""

    def __init__(
        self,
        dense_retriever: SemanticRetriever,
        bm25_index: BM25Index,
        poi_id_by_bm25_idx: Dict[int, str],
        id_to_poi,
    ):
        self.dense = dense_retriever
        self.bm25 = bm25_index
        self._poi_id_by_bm25_idx = poi_id_by_bm25_idx
        self._id_to_poi = id_to_poi

        retrieval_cfg = settings.retrieval
        self.fusion = retrieval_cfg.get("fusion", "rrf")
        self.rrf_k = retrieval_cfg.get("rrf_k", 60)
        self.linear_alpha = retrieval_cfg.get("linear_alpha", 0.5)

    @staticmethod
    def build_bm25_mapping(pois: List[POI]) -> Tuple[List[str], Dict[int, str]]:
        """Construye la lista de textos y el mapeo doc_idx -> poi_id."""
        texts = [p.enriched_text for p in pois]
        mapping = {i: p.id for i, p in enumerate(pois)}
        return texts, mapping

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def retrieve(
        self,
        preferences: UserPreferences,
        k: int = 20,
    ) -> List[Tuple[POI, float]]:
        """
        Recupera candidatos combinando dense + BM25 con los mismos filtros
        duros que SemanticRetriever.

        Returns:
            Lista de (POI, score) ordenada por score descendente.
        """
        # 1. Dense retrieval (sin límite estricto para dejar margen al RRF)
        dense_raw = self.dense.retrieve(preferences, k=k * 2)

        # 2. BM25 retrieval — usar la misma query expandida que dense
        self.dense._ensure_category_embeddings()
        query = _build_query(
            preferences,
            embedder=self.dense.embedder,
            category_embeddings=self.dense._category_embeddings,
        )
        bm25_raw_idx = self.bm25.search(query, k=k * 2)

        bm25_raw: List[Tuple[POI, float]] = []
        for doc_idx, score in bm25_raw_idx:
            poi_id = self._poi_id_by_bm25_idx.get(doc_idx)
            if poi_id is None:
                continue
            poi = self._id_to_poi(poi_id)
            if poi is None:
                continue
            bm25_raw.append((poi, score))

        # 3. Fusión
        if self.fusion == "linear":
            fused = _linear_fusion(dense_raw, bm25_raw, self.linear_alpha)
        else:
            fused = _rrf_fusion(dense_raw, bm25_raw, self.rrf_k)

        # 4. Filtros duros post-fusión (idénticos a SemanticRetriever)
        candidates: List[Tuple[POI, float]] = []
        allowed = SCOPE_FILTER.get(preferences.city_scope)
        allowed_set = set(m.lower() for m in allowed) if allowed else None

        for poi, score in fused:
            if allowed_set and poi.municipality.lower() not in allowed_set:
                continue
            if preferences.mobility == "reducida" and not poi.accessibility:
                continue

            # Descuento de score para POIs caros (>80% del presupuesto diario)
            daily_budget = preferences.budget_per_day
            if poi.price_numeric > daily_budget * 0.8 and poi.price_numeric > 0:
                score *= 0.7

            # Bonificación para POIs gratuitos cuando el presupuesto es bajo
            if daily_budget < 30 and poi.price_numeric == 0.0:
                score *= 1.1

            candidates.append((poi, score))
            if len(candidates) >= k:
                break

        logger.info(f"Hybrid candidates after fusion + hard filters: {len(candidates)}")
        return candidates

    def retrieve_raw(self, query: str, k: int = 20) -> List[Tuple[POI, float]]:
        """Búsqueda híbrida directa por texto, sin UserPreferences."""
        dense_raw = self.dense.search_by_text(query, k=k * 2)
        bm25_raw_idx = self.bm25.search(query, k=k * 2)

        bm25_raw: List[Tuple[POI, float]] = []
        for doc_idx, score in bm25_raw_idx:
            poi_id = self._poi_id_by_bm25_idx.get(doc_idx)
            if poi_id is None:
                continue
            poi = self._id_to_poi(poi_id)
            if poi is None:
                continue
            bm25_raw.append((poi, score))

        if self.fusion == "linear":
            return _linear_fusion(dense_raw, bm25_raw, self.linear_alpha)[:k]
        return _rrf_fusion(dense_raw, bm25_raw, self.rrf_k)[:k]

    def search_by_text(self, query: str, k: int = 10) -> List[Tuple[POI, float]]:
        return self.retrieve_raw(query, k=k)
