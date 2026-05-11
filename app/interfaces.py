from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class EmbeddingClient(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
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
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        n_results: int,
        filters: Optional[Dict] = None,
    ) -> List[str]:
        pass

    @abstractmethod
    def search_with_scores(
        self,
        query_vector: List[float],
        n_results: int,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_documents_by_ids(self, ids: List[str]) -> Dict[str, str]:
        pass

    @abstractmethod
    def delete(self, poi_id: str):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def count(self) -> int:
        pass
