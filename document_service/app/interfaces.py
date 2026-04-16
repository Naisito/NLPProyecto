from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class EmbeddingClient(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Convierte una lista de textos en una lista de vectores."""
        pass

class VectorIndex(ABC):
    @abstractmethod
    def add_vectors(self, ids: List[str], vectors: List[List[float]], metadatas: List[Dict], documents: List[str]):
        """Guarda vectores y metadatos."""
        pass

    @abstractmethod
    def search(self, query_vector: List[float], n_results: int, filters: Optional[Dict] = None) -> List[str]:
        """Busca los fragmentos más cercanos al vector de consulta. Devuelve IDs."""
        pass

    @abstractmethod
    def get_documents_by_ids(self, ids: List[str]) -> Dict[str, str]:
        """Obtiene documentos (texto) por sus IDs. Devuelve {id: documento}."""
        pass

    @abstractmethod
    def delete(self, document_id: str):
        """Borra vectores por ID de documento."""
        pass

    @abstractmethod
    def clear(self):
        """Limpia todo el índice."""
        pass