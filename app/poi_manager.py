"""
POI Manager — carga, almacena e indexa los Puntos de Interés.

Flujo al arrancar:
  1. Carga el JSON de POIs desde disco.
  2. Si el índice ChromaDB ya tiene datos, lo reutiliza (los embeddings
     son costosos; sólo se regeneran cuando el JSON cambia).
  3. Expone métodos para consultar POIs por ID, categoría o municipio.
"""

import json
import logging
import os
from typing import Dict, List, Optional

from app.config import settings
from app.interfaces import EmbeddingClient, VectorIndex
from app.models import POI, Coordinates, Schedule

logger = logging.getLogger("turismo_rag")


def _parse_poi(raw: dict) -> POI:
    """Convierte un diccionario del JSON en un objeto POI validado."""
    # Coordenadas
    coord = raw["coordinates"]
    coordinates = Coordinates(lat=coord["lat"], lon=coord["lon"])

    # Horario: cada día puede ser None o un dict {open, close}
    schedule: Dict[str, Optional[Schedule]] = {}
    for day, val in raw.get("schedule", {}).items():
        if val is None:
            schedule[day] = None
        else:
            schedule[day] = Schedule(open=val["open"], close=val["close"])

    return POI(
        id=raw["id"],
        name=raw["name"],
        municipality=raw["municipality"],
        category=raw["category"],
        subcategory=raw["subcategory"],
        description=raw["description"],
        coordinates=coordinates,
        address=raw["address"],
        price=raw["price"],
        price_numeric=float(raw["price_numeric"]),
        schedule=schedule,
        source=raw["source"],
        url=raw["url"],
        tags=raw.get("tags", []),
        enriched_text=raw.get("enriched_text", raw["description"]),
        visit_duration_minutes=int(raw.get("visit_duration_minutes", 60)),
        accessibility=bool(raw.get("accessibility", True)),
    )


class POIManager:
    """
    Gestiona la colección de POIs: carga, indexación y consulta.

    El índice vectorial almacena el 'enriched_text' de cada POI, que es
    un texto enriquecido con categoría, etiquetas y contexto semántico
    diseñado explícitamente para la recuperación.
    """

    def __init__(self, embedder: EmbeddingClient, vector_store: VectorIndex):
        self.embedder = embedder
        self.vector_store = vector_store
        self._pois: Dict[str, POI] = {}          # poi_id → POI
        self._loaded = False

    # ------------------------------------------------------------------
    # Carga desde JSON
    # ------------------------------------------------------------------

    def load_pois(self, json_path: str = None) -> int:
        """Carga los POIs desde el JSON y los indexa si el índice está vacío."""
        if json_path is None:
            base = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            json_path = os.path.join(base, settings.poi_data.get("path", "data/pois_bilbao_bizkaia.json"))

        logger.info(f"Cargando POIs desde: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_pois = data.get("pois", [])
        for raw in raw_pois:
            poi = _parse_poi(raw)
            self._pois[poi.id] = poi

        logger.info(f"{len(self._pois)} POIs cargados en memoria.")

        # Indexar sólo si el índice está vacío
        if self.vector_store.count() == 0:
            self._index_all()
        else:
            logger.info("Índice ChromaDB ya poblado — omitiendo re-indexación.")

        self._loaded = True
        return len(self._pois)

    # ------------------------------------------------------------------
    # Indexación
    # ------------------------------------------------------------------

    def _index_all(self):
        """Genera embeddings del enriched_text de cada POI y los almacena."""
        pois = list(self._pois.values())
        texts = [p.enriched_text for p in pois]
        ids   = [p.id for p in pois]
        metadatas = [
            {
                "poi_id":       p.id,
                "name":         p.name,
                "category":     p.category,
                "municipality": p.municipality,
                "price_numeric": p.price_numeric,
                "accessibility": int(p.accessibility),
            }
            for p in pois
        ]

        logger.info(f"Generando embeddings para {len(pois)} POIs…")
        vectors = self.embedder.encode(texts)

        self.vector_store.add_vectors(
            ids=ids,
            vectors=vectors,
            metadatas=metadatas,
            documents=texts,
        )
        logger.info("POIs indexados correctamente en ChromaDB.")

    def reindex(self):
        """Fuerza la re-indexación completa (vacía el índice primero)."""
        self.vector_store.clear()
        self._index_all()

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------

    def get_by_id(self, poi_id: str) -> Optional[POI]:
        return self._pois.get(poi_id)

    def get_all(self) -> List[POI]:
        return list(self._pois.values())

    def filter_by_category(self, category: str) -> List[POI]:
        cat_lower = category.lower()
        return [p for p in self._pois.values() if cat_lower in p.category.lower()]

    def filter_by_municipality(self, municipality: str) -> List[POI]:
        mun_lower = municipality.lower()
        return [p for p in self._pois.values() if mun_lower in p.municipality.lower()]

    def get_categories(self) -> List[str]:
        return sorted({p.category for p in self._pois.values()})

    def get_municipalities(self) -> List[str]:
        return sorted({p.municipality for p in self._pois.values()})

    @property
    def total(self) -> int:
        return len(self._pois)

    @property
    def is_loaded(self) -> bool:
        return self._loaded
