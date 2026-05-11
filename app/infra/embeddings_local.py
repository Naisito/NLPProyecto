import os
import logging
from typing import List

from sentence_transformers import SentenceTransformer
from app.interfaces import EmbeddingClient

logger = logging.getLogger("turismo_rag")


class LocalHuggingFaceEmbeddings(EmbeddingClient):
    """Embeddings locales con BAAI/bge-m3 via SentenceTransformers."""

    def __init__(self, model_name: str = "BAAI/bge-m3", cache_dir: str = None):
        if not cache_dir:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            cache_dir = os.path.join(base, "models_cache")

        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Cargando modelo de embeddings '{model_name}' desde: {cache_dir}")
        try:
            self.model = SentenceTransformer(
                model_name,
                cache_folder=cache_dir,
                trust_remote_code=True,
            )
            self.model_name = model_name
            logger.info("Modelo de embeddings cargado correctamente.")
        except Exception as e:
            logger.error(f"Error cargando modelo de embeddings: {e}")
            raise

    def encode(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts, batch_size=16, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()
