"""
Módulo de Reranking y Scoring de POIs.

Pipeline de ranking en tres etapas:

  Etapa 1 — Semantic score (vectorial):
    Score procedente de ChromaDB (BAAI/bge-m3). Captura similitud
    semántica global entre la consulta y el texto del POI.

  Etapa 2 — Cross-encoder reranking:
    Modelo cross-encoder/ms-marco-multilingual-MiniLM-L12-v2.
    A diferencia del bi-encoder (BAAI/bge-m3), el cross-encoder analiza
    conjuntamente la consulta y el texto del POI, ofreciendo una
    relevancia más precisa. Referencia: Reimers & Gurevych (2019)
    «Sentence-BERT». EMNLP 2019. arXiv:1908.10084.

  Etapa 3 — Scoring compuesto:
    Combina puntuaciones mediante pesos configurables:
    • semantic      (0.30): similitud vectorial
    • rerank        (0.35): cross-encoder
    • preference    (0.25): coincidencia explícita con intereses del usuario
    • diversity     (0.10): penalización por repetición de categoría

Los pesos son configurables en config.json → scoring_weights.
"""

import logging
import math
from typing import List, Tuple, Dict, Optional

from app.config import settings
from app.models import POI, UserPreferences

logger = logging.getLogger("turismo_rag")

# Mapeo interés → palabras clave que deben aparecer en la categoría/tags
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
    """
    Calcula qué bien encaja un POI con los intereses explícitos del usuario.
    Devuelve un valor en [0, 1].
    """
    if not preferences.interests:
        return 0.5  # neutral si no hay intereses

    matches = 0
    for interest in preferences.interests:
        target_cats = INTEREST_TO_CATEGORIES.get(interest, [interest])
        cat_lower   = poi.category.lower()
        sub_lower   = poi.subcategory.lower()
        tags_lower  = " ".join(poi.tags).lower()
        for tc in target_cats:
            if tc in cat_lower or tc in sub_lower or tc in tags_lower:
                matches += 1
                break

    score = matches / len(preferences.interests)

    # Bonificación por accesibilidad cuando es necesaria
    if preferences.mobility == "reducida" and poi.accessibility:
        score = min(1.0, score + 0.1)

    # Bonificación por gratuidad cuando el presupuesto es bajo
    if preferences.budget_per_day < 30 and poi.price_numeric == 0.0:
        score = min(1.0, score + 0.1)

    return round(score, 4)


def _diversity_penalty(
    poi: POI,
    already_selected: List[POI],
    penalty: float = 0.3,
) -> float:
    """
    Penaliza categorías ya presentes en el itinerario actual para favorecer
    la diversidad. Devuelve 0.0 si la categoría es nueva, >0 si ya existe.
    """
    selected_cats = [p.category for p in already_selected]
    count = selected_cats.count(poi.category)
    if count == 0:
        return 0.0
    return min(penalty * count, 0.7)   # máximo 70 % de penalización


class POIRanker:
    """
    Ranker híbrido en dos pasos:
      1. Cross-encoder reranking (ms-marco-multilingual-MiniLM-L12-v2).
      2. Score compuesto: semantic + rerank + preference + diversity.
    """

    def __init__(self, reranker_model_name: str = None, cache_dir: str = None):
        self.cross_encoder = None
        model_name = reranker_model_name or settings.reranker.get(
            "model_name", "cross-encoder/ms-marco-multilingual-MiniLM-L12-v2"
        )
        cache = cache_dir or settings.reranker.get("cache_dir", "models_cache")

        if settings.reranker.get("enabled", True):
            self._load_cross_encoder(model_name, cache)

        self.weights = {
            "semantic":    settings.scoring_weights.get("semantic", 0.30),
            "rerank":      settings.scoring_weights.get("rerank", 0.35),
            "preference":  settings.scoring_weights.get("preference_match", 0.25),
            "diversity":   settings.scoring_weights.get("spatial_diversity", 0.10),
        }

    def _load_cross_encoder(self, model_name: str, cache_dir: str):
        try:
            import os
            os.makedirs(cache_dir, exist_ok=True)
            from sentence_transformers.cross_encoder import CrossEncoder
            self.cross_encoder = CrossEncoder(model_name, max_length=512)
            logger.info(f"Cross-encoder '{model_name}' cargado correctamente.")
        except Exception as e:
            logger.warning(f"No se pudo cargar el cross-encoder: {e}. Se usará sólo el score semántico.")
            self.cross_encoder = None

    @property
    def reranker_loaded(self) -> bool:
        return self.cross_encoder is not None

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def rank(
        self,
        candidates: List[Tuple[POI, float]],
        preferences: UserPreferences,
        query: str,
        already_selected: Optional[List[POI]] = None,
        top_n: int = None,
    ) -> List[Tuple[POI, float, float, float]]:
        """
        Reordena los candidatos combinando todas las señales de scoring.

        Args:
            candidates:        Lista de (POI, semantic_score) del retriever.
            preferences:       Preferencias del usuario.
            query:             Consulta textual original o generada.
            already_selected:  POIs ya elegidos (para penalización de diversidad).
            top_n:             Número máximo de resultados a devolver.

        Returns:
            Lista de (POI, semantic_score, rerank_score, final_score)
            ordenada por final_score descendente.
        """
        if not candidates:
            return []

        already_selected = already_selected or []
        top_n = top_n or settings.rag.get("rerank_top_n", 12)

        pois   = [c[0] for c in candidates]
        s_scores = [c[1] for c in candidates]

        # ---- Etapa 1: Cross-encoder reranking -------------------------
        re_scores = self._cross_encode(query, pois)

        # ---- Etapa 2: Preference scores --------------------------------
        pref_scores = [_preference_score(p, preferences) for p in pois]

        # ---- Etapa 3: Diversity penalties ------------------------------
        # Las penalizaciones se aplican dinámicamente conforme se seleccionan
        # (aquí calculamos la penalización respecto a already_selected)
        div_penalties = [_diversity_penalty(p, already_selected) for p in pois]

        # ---- Composición final -----------------------------------------
        scored = []
        for i, poi in enumerate(pois):
            s  = s_scores[i]
            r  = re_scores[i]
            p  = pref_scores[i]
            dp = div_penalties[i]

            final = (
                self.weights["semantic"]   * s
                + self.weights["rerank"]   * r
                + self.weights["preference"] * p
                - self.weights["diversity"]  * dp
            )
            final = max(0.0, min(1.0, final))
            scored.append((poi, s, r, final))

        scored.sort(key=lambda x: x[3], reverse=True)
        return scored[:top_n]

    # ------------------------------------------------------------------
    # Cross-encoder interno
    # ------------------------------------------------------------------

    def _cross_encode(self, query: str, pois: List[POI]) -> List[float]:
        """
        Puntúa cada POI con el cross-encoder usando su enriched_text.
        Si el cross-encoder no está disponible, devuelve scores uniformes.
        Normaliza a [0, 1] mediante sigmoide.
        """
        if self.cross_encoder is None:
            return [0.5] * len(pois)

        pairs = [(query, poi.enriched_text[:512]) for poi in pois]
        try:
            raw_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
            # Sigmoide para normalizar a [0, 1]
            normalized = [float(1 / (1 + math.exp(-s))) for s in raw_scores]
            return normalized
        except Exception as e:
            logger.warning(f"Error en cross-encoder predict: {e}")
            return [0.5] * len(pois)
