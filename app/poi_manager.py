"""
POI Manager: carga, indexa y consulta los puntos de interés.

Flujo:
1. Carga el corpus completo desde un único JSON local.
2. Reindexa automáticamente ChromaDB si cambia el contenido del corpus.
3. Expone consultas por ID, categoría y municipio.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from app.config import settings
from app.interfaces import EmbeddingClient, VectorIndex
from app.models import Coordinates, POI, Schedule

logger = logging.getLogger("turismo_rag")


def _parse_poi(raw: dict) -> POI:
    """Convierte un diccionario del JSON en un objeto POI validado."""
    coordinates = Coordinates(
        lat=raw["coordinates"]["lat"],
        lon=raw["coordinates"]["lon"],
    )

    schedule: Dict[str, Optional[Schedule]] = {}
    for day, value in raw.get("schedule", {}).items():
        schedule[day] = None if value is None else Schedule(open=value["open"], close=value["close"])

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

    El índice vectorial almacena el `enriched_text` de cada POI.
    """

    def __init__(self, embedder: EmbeddingClient, vector_store: VectorIndex):
        self.embedder = embedder
        self.vector_store = vector_store
        self._pois: Dict[str, POI] = {}
        self._loaded = False
        self._load_summary: Dict[str, object] = {}
        self._corpus_signature: str = ""
        self._corpus_state_path = self._build_corpus_state_path()

    def _build_corpus_state_path(self) -> Path:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        vector_db_path = settings.vector_db.get("path", "db/chroma_db")
        absolute_db_path = vector_db_path if os.path.isabs(vector_db_path) else os.path.join(base, vector_db_path)
        collection_name = settings.vector_db.get("collection_name", "pois_turisticos")
        os.makedirs(absolute_db_path, exist_ok=True)
        return Path(absolute_db_path) / f"{collection_name}_corpus.sha256"

    def _compute_signature(self, raw_pois: List[dict]) -> str:
        payload = json.dumps(raw_pois, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _read_saved_signature(self) -> str:
        if not self._corpus_state_path.exists():
            return ""
        return self._corpus_state_path.read_text(encoding="utf-8").strip()

    def _write_saved_signature(self):
        self._corpus_state_path.write_text(self._corpus_signature, encoding="utf-8")

    def _should_reindex(self) -> bool:
        current_vectors = self.vector_store.count()
        saved_signature = self._read_saved_signature()

        if current_vectors == 0:
            return True
        if current_vectors != len(self._pois):
            return True
        if saved_signature != self._corpus_signature:
            return True
        return False

    # ------------------------------------------------------------------
    # Carga desde JSON
    # ------------------------------------------------------------------

    def load_pois(self, json_path: str = None) -> int:
        """Carga los POIs desde un único JSON local y reindexa si hace falta."""
        self._pois = {}

        if json_path is None:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            json_path = os.path.join(base, settings.poi_data.get("path", "data/pois_bilbao_bizkaia.json"))

        logger.info(f"Cargando POIs desde: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_pois = data.get("pois", [])
        self._corpus_signature = self._compute_signature(raw_pois)

        for raw in raw_pois:
            poi = _parse_poi(raw)
            self._pois[poi.id] = poi

        self._load_summary = {
            "mode": "direct_file",
            "path": json_path,
            "total_pois": len(self._pois),
            "bilbao_pois": sum(1 for poi in self._pois.values() if poi.municipality == "Bilbao"),
        }

        logger.info(f"{len(self._pois)} POIs cargados en memoria.")

        if self._should_reindex():
            logger.info("El corpus cargado difiere del índice vectorial actual. Reindexando…")
            self.vector_store.clear()
            self._index_all()
            self._write_saved_signature()
        else:
            logger.info("Índice ChromaDB reutilizado: coincide con el corpus cargado.")

        self._loaded = True
        return len(self._pois)

    # ------------------------------------------------------------------
    # Indexación
    # ------------------------------------------------------------------

    def _index_all(self):
        """Genera embeddings del enriched_text de cada POI y los almacena."""
        pois = list(self._pois.values())
        total = len(pois)
        batch_size = 64

        logger.info(f"Generando embeddings para {total} POIs…")

        for start in range(0, total, batch_size):
            batch = pois[start : start + batch_size]
            texts = [poi.enriched_text for poi in batch]
            ids = [poi.id for poi in batch]
            metadatas = [
                {
                    "poi_id": poi.id,
                    "name": poi.name,
                    "category": poi.category,
                    "municipality": poi.municipality,
                    "price_numeric": poi.price_numeric,
                    "accessibility": int(poi.accessibility),
                }
                for poi in batch
            ]

            vectors = self.embedder.encode(texts)
            self.vector_store.add_vectors(
                ids=ids,
                vectors=vectors,
                metadatas=metadatas,
                documents=texts,
            )
            logger.info(f"POIs indexados: {min(start + len(batch), total)}/{total}")

        logger.info("POIs indexados correctamente en ChromaDB.")

    def reindex(self):
        """Fuerza la reindexación completa del corpus ya cargado."""
        self.vector_store.clear()
        self._index_all()
        self._write_saved_signature()

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------

    def get_by_id(self, poi_id: str) -> Optional[POI]:
        return self._pois.get(poi_id)

    def get_all(self) -> List[POI]:
        return list(self._pois.values())

    def filter_by_category(self, category: str) -> List[POI]:
        category_lower = category.lower()
        return [poi for poi in self._pois.values() if category_lower in poi.category.lower()]

    def filter_by_municipality(self, municipality: str) -> List[POI]:
        municipality_lower = municipality.lower()
        return [poi for poi in self._pois.values() if municipality_lower in poi.municipality.lower()]

    def get_categories(self) -> List[str]:
        return sorted({poi.category for poi in self._pois.values()})

    def get_municipalities(self) -> List[str]:
        return sorted({poi.municipality for poi in self._pois.values()})

    @property
    def total(self) -> int:
        return len(self._pois)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_summary(self) -> Dict[str, object]:
        return dict(self._load_summary)
