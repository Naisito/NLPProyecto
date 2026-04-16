"""
Módulo de Recuperación (RAG Retriever).

Implementa la fase de recuperación semántica del pipeline:
  1. Construye una consulta enriquecida a partir de las preferencias.
  2. Genera su embedding con BAAI/bge-m3.
  3. Busca en ChromaDB los k POIs más similares semánticamente.
  4. Aplica filtros duros: municipio/ámbito, accesibilidad.
  5. Devuelve los candidatos con sus scores de similitud vectorial.

Referencia del modelo: Chen et al. (2024) «BGE M3-Embedding».
arXiv:2402.03216.
"""

import logging
from typing import List, Dict, Optional, Tuple

from app.config import settings
from app.interfaces import EmbeddingClient, VectorIndex
from app.models import POI, UserPreferences
from app.poi_manager import POIManager

logger = logging.getLogger("turismo_rag")

# Mapeo de intereses del usuario a términos de búsqueda aumentados
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

# Mapeo de ámbito geográfico a municipios incluidos
SCOPE_FILTER: Dict[str, Optional[List[str]]] = {
    "Bilbao":   ["Bilbao"],
    "Bizkaia":  None,   # todos
    "Ambos":    None,   # todos
}


def _build_query(preferences: UserPreferences) -> str:
    """
    Construye un texto de consulta semántica enriquecido a partir
    de las preferencias del usuario.
    """
    parts = []

    # Intereses → términos de búsqueda aumentados
    for interest in preferences.interests:
        expanded = INTEREST_QUERY_MAP.get(interest, interest)
        parts.append(expanded)

    # Tipo de grupo
    group_terms = {
        "familia": "actividades familiares niños",
        "pareja":  "romántico íntimo especial pareja",
        "amigos":  "animado social ocio amigos",
        "solo":    "individual contemplativo interesante",
    }
    parts.append(group_terms.get(preferences.group_type, ""))

    # Ritmo
    if preferences.pace == "tranquilo":
        parts.append("relajado tranquilo pausado sin prisas")
    elif preferences.pace == "intenso":
        parts.append("completo intenso variado activo")

    # Ámbito geográfico
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
    """
    Recupera candidatos de POIs mediante búsqueda vectorial en ChromaDB.

    El texto indexado para cada POI es su 'enriched_text', que contiene
    categoría, subcategoría, tags y descripción ampliada, diseñado para
    maximizar la recuperación semántica.
    """

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

    def retrieve(
        self,
        preferences: UserPreferences,
        k: int = None,
    ) -> List[Tuple[POI, float]]:
        """
        Recupera los k POIs más relevantes para las preferencias dadas.

        Returns:
            Lista de (POI, semantic_score) ordenada por score descendente.
        """
        k = k or self.retrieval_k
        query = _build_query(preferences)

        # 1. Embedding de la consulta
        query_vector = self.embedder.encode([query])[0]

        # 2. Filtro duro por municipio (si ámbito = "Bilbao")
        chroma_filter = None
        allowed_municipalities = SCOPE_FILTER.get(preferences.city_scope)
        if allowed_municipalities and len(allowed_municipalities) == 1:
            chroma_filter = {
                "municipality": {"$eq": allowed_municipalities[0]}
            }

        # 3. Búsqueda vectorial con scores
        # Pedimos más del doble para tener margen tras el filtrado duro
        fetch_k = min(k * 2, self.poi_manager.total)
        raw_results = self.vector_store.search_with_scores(
            query_vector=query_vector,
            n_results=fetch_k,
            filters=chroma_filter,
        )

        logger.info(f"Recuperados {len(raw_results)} candidatos brutos del índice vectorial.")

        # 4. Filtros duros post-recuperación
        candidates: List[Tuple[POI, float]] = []
        for item in raw_results:
            poi_id = item["id"]
            score  = float(item["score"])
            poi = self.poi_manager.get_by_id(poi_id)
            if poi is None:
                continue

            # Filtro de accesibilidad
            if preferences.mobility == "reducida" and not poi.accessibility:
                continue

            # Filtro de presupuesto (excluir POIs claramente fuera de presupuesto)
            daily_budget = preferences.budget_per_day
            # No excluimos POIs de precio 0 (gratis) ni aplicamos filtro muy estricto
            # — el ranker penalizará los caros
            if poi.price_numeric > daily_budget * 0.8 and poi.price_numeric > 0:
                pass  # se mantienen pero serán penalizados en el ranking

            # Filtro geográfico más permisivo si se filtra sólo por Bilbao
            if (
                preferences.city_scope == "Bilbao"
                and poi.municipality.lower() != "bilbao"
                and chroma_filter is None  # el filtro de chroma ya lo haría
            ):
                continue

            candidates.append((poi, score))
            if len(candidates) >= k:
                break

        logger.info(f"Candidatos tras filtros duros: {len(candidates)}")
        return candidates

    def search_by_text(self, query: str, k: int = 10) -> List[Tuple[POI, float]]:
        """
        Búsqueda semántica libre sobre el corpus de POIs.
        Útil para el endpoint de búsqueda directa.
        """
        query_vector = self.embedder.encode([query])[0]
        raw = self.vector_store.search_with_scores(
            query_vector=query_vector,
            n_results=k,
        )
        results = []
        for item in raw:
            poi = self.poi_manager.get_by_id(item["id"])
            if poi:
                results.append((poi, float(item["score"])))
        return results
