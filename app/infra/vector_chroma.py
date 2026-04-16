import os
import logging
from typing import List, Dict, Optional, Any

import chromadb
from app.interfaces import VectorIndex

logger = logging.getLogger("turismo_rag")


class LocalChromaIndex(VectorIndex):
    """
    Índice vectorial persistente basado en ChromaDB.

    Almacena embeddings de POIs junto con metadatos (poi_id, category,
    municipality) y permite búsquedas semánticas filtradas.
    Los scores se normalizan mediante min-max para obtener valores en [0, 1]
    donde 1 representa la mayor similitud.
    """

    def __init__(self, db_path: str = None, collection_name: str = "pois_turisticos"):
        if not db_path:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            db_path = os.path.join(base, "db", "chroma_db")

        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info(f"ChromaDB conectado en: {db_path} — colección: '{collection_name}'")

    # ------------------------------------------------------------------
    # Escritura
    # ------------------------------------------------------------------

    def add_vectors(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: List[Dict],
        documents: List[str],
    ):
        if not ids:
            return
        self.collection.add(
            embeddings=vectors,
            metadatas=metadatas,
            documents=documents,
            ids=ids,
        )

    # ------------------------------------------------------------------
    # Lectura
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: List[float],
        n_results: int,
        filters: Optional[Dict] = None,
    ) -> List[str]:
        if self.collection.count() == 0:
            return []

        params = {"query_embeddings": [query_vector], "n_results": n_results}
        if filters:
            params["where"] = filters

        results = self.collection.query(**params)
        ids_list = results.get("ids", [])
        return ids_list[0] if ids_list else []

    def search_with_scores(
        self,
        query_vector: List[float],
        n_results: int,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Devuelve lista de {id, score} con scores normalizados en [0, 1]."""
        if self.collection.count() == 0:
            return []

        params = {"query_embeddings": [query_vector], "n_results": n_results}
        if filters:
            params["where"] = filters

        results = self.collection.query(**params)
        ids_list  = results.get("ids", [])
        dists_list = results.get("distances", [])

        if not ids_list:
            return []

        ids   = ids_list[0]  if ids_list   else []
        dists = dists_list[0] if dists_list else []

        if not ids:
            return []

        # Fallback por posición si no hay distancias
        if not dists or len(dists) != len(ids):
            total = max(len(ids), 1)
            return [{"id": i, "score": 1.0 - (idx / total)} for idx, i in enumerate(ids)]

        # Normalización min-max: distancia menor → score mayor
        min_d = min(dists)
        max_d = max(dists)
        denom = (max_d - min_d) if (max_d - min_d) > 1e-9 else 1.0

        return [
            {"id": id_, "score": float((max_d - d) / denom)}
            for id_, d in zip(ids, dists)
        ]

    def get_documents_by_ids(self, ids: List[str]) -> Dict[str, str]:
        if not ids:
            return {}
        try:
            results = self.collection.get(ids=ids)
            if results and results.get("documents"):
                return {
                    doc_id: doc
                    for doc_id, doc in zip(results["ids"], results["documents"])
                }
        except Exception as e:
            logger.error(f"Error recuperando documentos por ID: {e}")
        return {}

    # ------------------------------------------------------------------
    # Eliminación
    # ------------------------------------------------------------------

    def delete(self, poi_id: str):
        try:
            self.collection.delete(where={"poi_id": {"$eq": poi_id}})
            logger.info(f"Vectores eliminados para poi_id='{poi_id}'")
        except Exception as e:
            logger.error(f"Error eliminando en ChromaDB: {e}")

    def clear(self):
        try:
            name = self.collection.name
            self.client.delete_collection(name)
            self.collection = self.client.get_or_create_collection(name=name)
            logger.info("Índice ChromaDB vaciado completamente.")
        except Exception as e:
            logger.error(f"Error vaciando ChromaDB: {e}")

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def count(self) -> int:
        return self.collection.count()
