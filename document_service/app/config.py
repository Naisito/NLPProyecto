import json
import os
import logging

logger = logging.getLogger("doc_service")

class AppConfig:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self):
        # Buscamos el archivo en la raíz del proyecto
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path = os.path.join(base_dir, self.config_path)
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
                logger.info(f"Configuración cargada desde {path}")
                return config
        except FileNotFoundError:
            logger.warning(f"No se encontró {path}, usando valores por defecto o fallará.")
            return {}
        except Exception as e:
            logger.error(f"Error leyendo config: {e}")
            raise e

    # Getters para acceder fácilmente a las secciones
    @property
    def rag(self):
        return self._config.get("rag", {})

    @property
    def embeddings(self):
        return self._config.get("embeddings", {})

    @property
    def vector_db(self):
        return self._config.get("vector_db", {})

    @property
    def storage(self):
        return self._config.get("storage", {})
    
    @property
    def llm(self):
        return self._config.get("llm", {})

# Instancia global (Singleton)
settings = AppConfig()