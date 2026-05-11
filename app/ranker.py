import logging
import math
from typing import List, Tuple, Dict, Optional

from app.config import settings
from app.models import POI, UserPreferences

logger = logging.getLogger("turismo_rag")

INTEREST_TO_CATEGORIES: Dict[str, List[str]] = {
    "museos":           ["museo"],
    "arte":             ["museo", "arte", "galería"],
    "arquitectura":     ["arquitectura", "puente", "edificio", "teatro", "castillo"],
    "gastronomía":      ["gastronomía", "café", "mercado", "puerto", "pueblo"],
    "pintxos":          ["gastronomía", "café", "mercado", "barrio"],
    "naturaleza":       ["naturaleza", "parque", "playa", "reserva"],
    "senderismo":       ["naturaleza", "parque"],
    "playa":            ["playa"],
    "surf":             ["playa", "naturaleza"],
    "historia":         ["historia", "religioso", "arquitectura", "pueblo", "museo"],
    "cultura vasca":    ["museo", "historia", "barrio", "pueblo"],
    "deporte":          ["deporte", "naturaleza", "playa"],
    "vida nocturna":    ["barrio", "gastronomía", "arte"],
    "compras":          ["barrio", "gastronomía", "mercado"],
    "fotografía":       ["naturaleza", "arquitectura", "barrio"],
    "familia":          ["parque", "museo", "playa", "naturaleza"],
    "rural":            ["pueblo", "naturaleza"],
    "pueblos costeros": ["pueblo", "playa", "gastronomía"],
}


def _preference_score(poi: POI, preferences: UserPreferences) -> float:
    """Cuánto encaja el POI con los intereses del usuario [0-1]."""
    if not preferences.interests:
        return 0.5

    matches = 0
    for interest in preferences.interests:
        target_cats = INTEREST_TO_CATEGORIES.get(interest, [interest])
        cat_lower = poi.category.lower()
        sub_lower = poi.subcategory.lower()
        tags_lower = " ".join(poi.tags).lower()
        for tc in target_cats:
            if tc in cat_lower or tc in sub_lower or tc in tags_lower:
                matches += 1
                break

    score = matches / len(preferences.interests)

    if preferences.mobility == "reducida" and poi.accessibility:
        score = min(1.0, score + 0.1)
    if preferences.budget_per_day < 30 and poi.price_numeric == 0.0:
        score = min(1.0, score + 0.1)

    return round(score, 4)


def _diversity_penalty(poi: POI, already_selected: List[POI], penalty: float = 0.3) -> float:
    """Penaliza repetición de categoría. 0.0 si es nueva, hasta 0.7 si se repite mucho."""
    selected_cats = [p.category for p in already_selected]
    count = selected_cats.count(poi.category)
    if count == 0:
        return 0.0
    return min(penalty * count, 0.7)


class POIRanker:
    """Ranker compuesto: cross-encoder rerank + preference score + diversity penalty."""

    def __init__(self, reranker_model_name: str = None, cache_dir: str = None):
        self.cross_encoder = None
        model_name = reranker_model_name or settings.reranker.get("model_name", "BAAI/bge-reranker-v2-m3")
        cache = cache_dir or settings.reranker.get("cache_dir", "models_cache")

        if settings.reranker.get("enabled", True):
            self._load_cross_encoder(model_name, cache)

        self.weights = {
            "semantic":   settings.scoring_weights.get("semantic", 0.30),
            "rerank":     settings.scoring_weights.get("rerank", 0.35),
            "preference": settings.scoring_weights.get("preference_match", 0.25),
            "diversity":  settings.scoring_weights.get("spatial_diversity", 0.10),
        }

    def _load_cross_encoder(self, model_name: str, cache_dir: str):
        try:
            import os
            from huggingface_hub import snapshot_download
            from sentence_transformers.cross_encoder import CrossEncoder

            os.makedirs(cache_dir, exist_ok=True)
            model_path = snapshot_download(repo_id=model_name, cache_dir=cache_dir, local_files_only=True)
            self.cross_encoder = CrossEncoder(model_path, max_length=512, local_files_only=True)
            logger.info(f"Cross-encoder '{model_name}' cargado.")
        except Exception as e:
            logger.warning(f"No se pudo cargar el cross-encoder: {e}")
            self.cross_encoder = None

    @property
    def reranker_loaded(self) -> bool:
        return self.cross_encoder is not None

    def rank(
        self,
        candidates: List[Tuple[POI, float]],
        preferences: UserPreferences,
        query: str,
        already_selected: Optional[List[POI]] = None,
        top_n: int = None,
    ) -> List[Tuple[POI, float, float, float]]:
        if not candidates:
            return []

        already_selected = already_selected or []
        top_n = top_n or settings.rag.get("rerank_top_n", 12)

        pois = [c[0] for c in candidates]
        s_scores = [c[1] for c in candidates]

        re_scores = self._cross_encode(query, pois)
        pref_scores = [_preference_score(p, preferences) for p in pois]
        div_penalties = [_diversity_penalty(p, already_selected) for p in pois]

        scored = []
        for i, poi in enumerate(pois):
            final = (
                self.weights["semantic"]   * s_scores[i]
                + self.weights["rerank"]   * re_scores[i]
                + self.weights["preference"] * pref_scores[i]
                - self.weights["diversity"]  * div_penalties[i]
            )
            final = max(0.0, min(1.0, final))
            scored.append((poi, s_scores[i], re_scores[i], final))

        scored.sort(key=lambda x: x[3], reverse=True)
        return scored[:top_n]

    def _cross_encode(self, query: str, pois: List[POI]) -> List[float]:
        if self.cross_encoder is None:
            return [0.5] * len(pois)

        pairs = [(query, poi.enriched_text) for poi in pois]
        try:
            raw_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
            min_s = float(min(raw_scores))
            max_s = float(max(raw_scores))
            if max_s - min_s < 1e-6:
                return [0.5] * len(pois)
            return [(float(s) - min_s) / (max_s - min_s) for s in raw_scores]
        except Exception as e:
            logger.warning(f"Error en cross-encoder predict: {e}")
            return [0.5] * len(pois)
