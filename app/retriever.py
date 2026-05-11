import logging
from typing import List, Dict, Optional, Tuple

from app.config import settings
from app.interfaces import EmbeddingClient, VectorIndex
from app.models import POI, UserPreferences
from app.poi_manager import POIManager

logger = logging.getLogger("turismo_rag")


def _semantic_expand_interest(interest: str, embedder, category_embeddings: dict) -> str:
    """Expande un interés sin mapeo buscando las 3 categorías del corpus más cercanas en embeddings."""
    import numpy as np

    try:
        interest_vec = embedder.encode([interest])[0]
    except Exception:
        return interest

    i_vec = np.array(interest_vec)
    scores = {}
    for cat, cat_vec in category_embeddings.items():
        c_vec = np.array(cat_vec)
        sim = float(np.dot(i_vec, c_vec) / (np.linalg.norm(i_vec) * np.linalg.norm(c_vec) + 1e-10))
        if cat.lower() != interest.lower():
            scores[cat] = sim

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    expanded = interest
    for cat, sim in top:
        if sim > 0.5:
            expanded += f" {cat}"
    return expanded


INTEREST_QUERY_MAP: Dict[str, str] = {
    "museos":           "museo arte exposición colección cultural",
    "arte":             "arte pintura escultura galería",
    "arquitectura":     "arquitectura edificio puente diseño construcción",
    "gastronomía":      "gastronomía comida pintxos cocina vasca restaurante mercado",
    "pintxos":          "pintxos bares gastronomía vasca tapas bocadillos",
    "naturaleza":       "naturaleza parque senderismo bosque monte paisaje",
    "senderismo":       "senderismo montaña rutas naturaleza caminos",
    "playa":            "playa arena mar Cantábrico baño verano",
    "surf":             "surf olas deporte acuático playa",
    "historia":         "historia medieval patrimonio histórico monumento",
    "cultura vasca":    "cultura vasca tradiciones Euskal Herria identidad",
    "deporte":          "deporte actividad física estadio atletismo",
    "vida nocturna":    "bares ocio nocturno vida nocturna ambiente",
    "compras":          "compras tiendas mercado artesanía",
    "fotografía":       "fotografía paisaje vistas panorámicas icónico",
    "familia":          "familia niños parque actividades para familias",
    "rural":            "rural pueblo caserío campo naturaleza",
    "pueblos costeros": "pueblo costero pesquero mar marinero",
}

SCOPE_FILTER: Dict[str, Optional[List[str]]] = {
    "Bilbao":   ["Bilbao"],
    "Bizkaia":  None,
    "Ambos":    None,
}


def _build_query(preferences: UserPreferences, embedder=None, category_embeddings=None) -> str:
    """Construye la query enriquecida a partir de las preferencias del usuario."""
    parts = []

    for interest in preferences.interests:
        if embedder is not None and category_embeddings is not None and interest not in INTEREST_QUERY_MAP:
            expanded = _semantic_expand_interest(interest, embedder, category_embeddings)
        else:
            expanded = INTEREST_QUERY_MAP.get(interest, interest)
        parts.append(expanded)

    group_terms = {
        "familia": "actividades familiares niños",
        "pareja":  "romántico íntimo especial pareja",
        "amigos":  "animado social ocio amigos",
        "solo":    "individual contemplativo interesante",
    }
    parts.append(group_terms.get(preferences.group_type, ""))

    if preferences.pace == "tranquilo":
        parts.append("relajado tranquilo pausado sin prisas")
    elif preferences.pace == "intenso":
        parts.append("completo intenso variado activo")

    if preferences.city_scope == "Bilbao":
        parts.append("Bilbao ciudad urbano")
    elif preferences.city_scope == "Bizkaia":
        parts.append("Bizkaia rural costero natural pueblos")

    query = " ".join(p for p in parts if p).strip()
    if not query:
        query = "turismo cultural naturaleza gastronomía Bilbao Bizkaia"

    logger.debug(f"Query de recuperación: '{query}'")
    return query


class SemanticRetriever:
    """Recuperación semántica con BAAI/bge-m3 + ChromaDB."""

    def __init__(
        self,
        embedder: EmbeddingClient,
        vector_store: VectorIndex,
        poi_manager: POIManager,
        retrieval_k: int = 20,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.poi_manager = poi_manager
        self.retrieval_k = retrieval_k
        self._category_embeddings = None

    def _ensure_category_embeddings(self):
        if self._category_embeddings is not None:
            return
        categories = sorted(self.poi_manager.get_categories())
        if not categories:
            return
        logger.info(f"Calculando embeddings para {len(categories)} categorías...")
        cat_vectors = self.embedder.encode(categories)
        self._category_embeddings = dict(zip(categories, cat_vectors))

    def retrieve(self, preferences: UserPreferences, k: int = None) -> List[Tuple[POI, float]]:
        k = k or self.retrieval_k
        self._ensure_category_embeddings()
        query = _build_query(preferences, self.embedder, self._category_embeddings)

        query_vector = self.embedder.encode([query])[0]

        chroma_filter = None
        allowed_municipalities = SCOPE_FILTER.get(preferences.city_scope)
        if allowed_municipalities and len(allowed_municipalities) == 1:
            chroma_filter = {"municipality": {"$eq": allowed_municipalities[0]}}

        fetch_k = min(k * 2, self.poi_manager.total)
        raw_results = self.vector_store.search_with_scores(
            query_vector=query_vector, n_results=fetch_k, filters=chroma_filter,
        )

        logger.info(f"Recuperados {len(raw_results)} candidatos brutos del índice vectorial.")

        candidates: List[Tuple[POI, float]] = []
        for item in raw_results:
            poi_id = item["id"]
            score = float(item["score"])
            poi = self.poi_manager.get_by_id(poi_id)
            if poi is None:
                continue

            if preferences.mobility == "reducida" and not poi.accessibility:
                continue

            daily_budget = preferences.budget_per_day
            if poi.price_numeric > daily_budget * 0.8 and poi.price_numeric > 0:
                score *= 0.7
            if daily_budget < 30 and poi.price_numeric == 0.0:
                score *= 1.1

            if (
                preferences.city_scope == "Bilbao"
                and poi.municipality.lower() != "bilbao"
                and chroma_filter is None
            ):
                continue

            candidates.append((poi, score))
            if len(candidates) >= k:
                break

        logger.info(f"Candidatos tras filtros duros: {len(candidates)}")
        return candidates

    def search_by_text(self, query: str, k: int = 10) -> List[Tuple[POI, float]]:
        query_vector = self.embedder.encode([query])[0]
        raw = self.vector_store.search_with_scores(query_vector=query_vector, n_results=k)
        results = []
        for item in raw:
            poi = self.poi_manager.get_by_id(item["id"])
            if poi:
                results.append((poi, float(item["score"])))
        return results
