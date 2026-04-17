from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class EmbeddingClient(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Convierte una lista de textos en vectores de embeddings."""
        pass


class VectorIndex(ABC):
    @abstractmethod
    def add_vectors(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: List[Dict],
        documents: List[str],
    ):
        """Almacena vectores junto con metadatos y texto original."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        n_results: int,
        filters: Optional[Dict] = None,
    ) -> List[str]:
        """Busca los fragmentos mas cercanos. Devuelve IDs."""
        pass

    @abstractmethod
    def search_with_scores(
        self,
        query_vector: List[float],
        n_results: int,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Busca y devuelve lista de {id, score} normalizados en [0, 1]."""
        pass

    @abstractmethod
    def get_documents_by_ids(self, ids: List[str]) -> Dict[str, str]:
        """Devuelve {id: texto} para los IDs indicados."""
        pass

    @abstractmethod
    def delete(self, poi_id: str):
        """Borra los vectores asociados a un POI."""
        pass

    @abstractmethod
    def clear(self):
        """Vacia completamente el indice."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Devuelve el numero total de vectores almacenados."""
        pass
