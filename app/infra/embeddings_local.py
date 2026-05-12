import os
import logging
import sys
import time
import threading
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

        device = os.environ.get("SENTENCE_TRANSFORMERS_DEVICE", "cpu")
        logger.info(
            f"Cargando modelo de embeddings '{model_name}' en {device}... "
            f"(puede tardar varios minutos en CPU, NO se ha congelado)"
        )

        # Evita deadlocks en Windows con tokenizers multiproceso
        if sys.platform == "win32":
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Hilo auxiliar que imprime un punto cada 10 s para dar feedback visual
        stop_spinner = threading.Event()

        def _dot_feedback():
            while not stop_spinner.is_set():
                stop_spinner.wait(10)
                if not stop_spinner.is_set():
                    print(".", end="", flush=True)

        spinner = threading.Thread(target=_dot_feedback, daemon=True)
        try:
            spinner.start()
            t0 = time.perf_counter()
            self.model = SentenceTransformer(
                model_name,
                cache_folder=cache_dir,
                trust_remote_code=True,
            )
            elapsed = time.perf_counter() - t0
            stop_spinner.set()
            self.model_name = model_name
            logger.info(
                "Modelo de embeddings cargado correctamente (%.1f s).", elapsed
            )
        except Exception as e:
            stop_spinner.set()
            logger.error(f"Error cargando modelo de embeddings: {e}")
            raise

    def encode(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts, batch_size=16, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()
