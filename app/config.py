import json
import os
import logging

logger = logging.getLogger("turismo_rag")

class Settings:
    def __init__(self, config_path: str = None):
        if config_path is None:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            config_path = os.path.join(base, "config.json")

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.server: dict = data.get("server", {})
        self.embeddings: dict = data.get("embeddings", {})
        self.reranker: dict = data.get("reranker", {})
        self.vector_db: dict = data.get("vector_db", {})
        self.rag: dict = data.get("rag", {})
        self.planner: dict = data.get("planner", {})
        self.llm: dict = data.get("llm", {})
        self.poi_data: dict = data.get("poi_data", {})
        self.scoring_weights: dict = data.get("scoring_weights", {})

        # La variable de entorno OLLAMA_BASE_URL tiene prioridad sobre config.json.
        # Útil en Docker para apuntar al Ollama del host sin tocar el fichero.
        env_ollama_url = os.environ.get("OLLAMA_BASE_URL")
        if env_ollama_url:
            self.llm["ollama_base_url"] = env_ollama_url
            logger.info(f"Ollama URL sobreescrita por variable de entorno: {env_ollama_url}")

        logger.info(f"Configuración cargada desde: {config_path}")

settings = Settings()
