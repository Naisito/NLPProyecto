import os
from sentence_transformers import SentenceTransformer
from typing import List
from app.interfaces import EmbeddingClient
import logging

logger = logging.getLogger("doc_service")

class LocalHuggingFaceEmbeddings(EmbeddingClient):
    def __init__(self, model_name: str = 'BAAI/bge-m3', cache_dir: str = None):
        if not cache_dir:
            # Definir ruta por defecto relativa al proyecto si no se pasa
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            cache_dir = os.path.join(base_dir, "models_cache")
        
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Cargando modelo local desde: {cache_dir}")
        try:
            self.model = SentenceTransformer(model_name, cache_folder=cache_dir,trust_remote_code=True)
            logger.info("Modelo local cargado exitosamente.")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise e

    def encode(self, texts: List[str]) -> List[List[float]]:
        # SentenceTransformer devuelve ndarray, lo convertimos a lista
        embeddings = self.model.encode(texts,normalize_embeddings=True)
        return embeddings.tolist()