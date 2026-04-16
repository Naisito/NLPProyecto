import os
import chromadb
from typing import List, Dict, Optional
from app.interfaces import VectorIndex
import logging

logger = logging.getLogger("doc_service")

class LocalChromaIndex(VectorIndex):
    def __init__(self, db_path: str = None, collection_name: str = "documents_rag"):
        if not db_path:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            db_path = os.path.join(base_dir, "db", "chroma_db")
            
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info(f"ChromaDB conectado en: {db_path}")

    def add_vectors(self, ids: List[str], vectors: List[List[float]], metadatas: List[Dict], documents: List[str]):
        if not ids:
            return
        self.collection.add(
            embeddings=vectors,
            metadatas=metadatas,
            documents=documents,
            ids=ids
        )

    def search(self, query_vector: List[float], n_results: int, filters: Optional[Dict] = None) -> List[str]:
        if self.collection.count() == 0:
            return []

        search_params = {
            "query_embeddings": [query_vector],
            "n_results": n_results
        }
        
        if filters:
            search_params["where"] = filters

        results = self.collection.query(**search_params)
        
        # Devolver IDs en lugar de documentos para poder identificar qué resultado es cuál
        if results and results.get("ids"):
            return results["ids"][0] if results["ids"] else []
        return []

    def search_with_scores(self, query_vector: List[float], n_results: int, filters: Optional[Dict] = None) -> List[Dict[str, float]]:
        """
        Busca y devuelve pares (id, score) usando distancias de Chroma.
        Normaliza distancias por min-max para obtener scores en [0,1], donde 1 es el más similar.
        """
        if self.collection.count() == 0:
            return []

        search_params = {
            "query_embeddings": [query_vector],
            "n_results": n_results
        }

        if filters:
            search_params["where"] = filters

        results = self.collection.query(**search_params)

        ids_list = results.get("ids", []) if results else []
        dists_list = results.get("distances", []) if results else []
        if not ids_list:
            return []

        ids = ids_list[0] if ids_list else []
        dists = dists_list[0] if dists_list else []

        if not ids:
            return []

        # Si no hay distancias, devolver orden como scores descendentes por posición
        if not dists or len(dists) != len(ids):
            scored = []
            total = max(len(ids), 1)
            for i, id_ in enumerate(ids):
                score = 1.0 - (i / total)  # peor caso: fallback por posición
                scored.append({"id": id_, "score": score})
            return scored

        # Min-Max normalización: menor distancia => mayor score
        min_d = min(dists)
        max_d = max(dists)
        denom = (max_d - min_d) if (max_d - min_d) > 1e-9 else 1.0

        scored = []
        for id_, d in zip(ids, dists):
            norm = (max_d - d) / denom  # 1.0 para el más cercano, 0.0 para el más lejano
            scored.append({"id": id_, "score": float(norm)})

        return scored

    def get_documents_by_ids(self, ids: List[str]) -> Dict[str, str]:
        """Obtiene documentos (texto) por sus IDs."""
        if not ids:
            return {}
        
        try:
            results = self.collection.get(ids=ids)
            if results and results.get("documents"):
                return {doc_id: doc for doc_id, doc in zip(results["ids"], results["documents"])}
            return {}
        except Exception as e:
            logger.error(f"Error obteniendo documentos por IDs: {e}")
            return {}

    def delete(self, document_id: str):
        try:
            self.collection.delete(where={"document_id": document_id})
            logger.info(f"Vectores borrados para: {document_id}")
        except Exception as e:
            logger.error(f"Error borrando en Chroma: {e}")

    def clear(self):
        try:
            # Chroma no tiene un 'clear' simple, borramos y recreamos la coleccion
            name = self.collection.name
            self.client.delete_collection(name)
            self.collection = self.client.get_or_create_collection(name=name)
        except Exception as e:
            logger.error(f"Error limpiando DB: {e}")