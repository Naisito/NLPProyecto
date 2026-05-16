import json
import os
import logging
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("turismo_rag")


class _ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1, le=65535)
    log_level: str = "INFO"


class _EmbeddingsConfig(BaseModel):
    provider: str = "huggingface"
    model_name: str = "BAAI/bge-m3"
    cache_dir: str = "models_cache"


class _RerankerConfig(BaseModel):
    model_name: str = "BAAI/bge-reranker-v2-m3"
    cache_dir: str = "models_cache"
    enabled: bool = True


class _VectorDBConfig(BaseModel):
    provider: str = "chroma"
    path: str = "db/chroma_db"
    collection_name: str = "pois_turisticos"


class _RAGConfig(BaseModel):
    retrieval_k: int = Field(20, ge=1)
    rerank_top_n: int = Field(12, ge=1)
    min_score_threshold: float = Field(0.0, ge=0.0, le=1.0)


class _SlotConfig(BaseModel):
    start: str = "09:30"
    end: str = "14:00"


class _SlotsConfig(BaseModel):
    manana: _SlotConfig = Field(default_factory=lambda: _SlotConfig(start="09:30", end="14:00"))
    tarde: _SlotConfig = Field(default_factory=lambda: _SlotConfig(start="16:00", end="20:00"))


class _PoisPerDayConfig(BaseModel):
    tranquilo: int = Field(3, ge=1, le=10)
    moderado: int = Field(4, ge=1, le=10)
    intenso: int = Field(6, ge=1, le=10)


class _PlannerConfig(BaseModel):
    slots: _SlotsConfig = Field(default_factory=_SlotsConfig)
    pois_per_day: _PoisPerDayConfig = Field(default_factory=_PoisPerDayConfig)
    walking_speed_kmh: float = Field(4.5, gt=0.0)
    avg_travel_minutes_intra_day: int = Field(15, ge=0)


class _LLMConfig(BaseModel):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model_name: str = "qwen3:8b"
    request_timeout_seconds: int = Field(1800, ge=1)
    temperature_generation: float = Field(0.5, ge=0.0, le=2.0)
    temperature_interpretation: float = Field(0.1, ge=0.0, le=2.0)
    max_tokens_generation: int = Field(800, ge=1)


class _POIDataConfig(BaseModel):
    path: str = "data/pois_bilbao_bizkaia.json"


class _ScoringWeightsConfig(BaseModel):
    semantic: float = Field(0.30, ge=0.0, le=1.0)
    rerank: float = Field(0.35, ge=0.0, le=1.0)
    preference_match: float = Field(0.25, ge=0.0, le=1.0)
    spatial_diversity: float = Field(0.10, ge=0.0, le=1.0)


class _BM25Config(BaseModel):
    k1: float = Field(1.5, gt=0.0)
    b: float = Field(0.75, ge=0.0, le=1.0)


class _RetrievalConfig(BaseModel):
    mode: str = "dense"
    fusion: str = "rrf"
    rrf_k: int = Field(60, ge=1)
    linear_alpha: float = Field(0.5, ge=0.0, le=1.0)

    @field_validator("mode")
    @classmethod
    def _check_mode(cls, v: str) -> str:
        if v not in ("dense", "bm25", "hybrid"):
            raise ValueError(f"retrieval.mode debe ser 'dense', 'bm25' o 'hybrid', no '{v}'")
        return v

    @field_validator("fusion")
    @classmethod
    def _check_fusion(cls, v: str) -> str:
        if v not in ("rrf", "linear"):
            raise ValueError(f"retrieval.fusion debe ser 'rrf' o 'linear', no '{v}'")
        return v


class _AppConfig(BaseModel):
    server: _ServerConfig = Field(default_factory=_ServerConfig)
    embeddings: _EmbeddingsConfig = Field(default_factory=_EmbeddingsConfig)
    reranker: _RerankerConfig = Field(default_factory=_RerankerConfig)
    vector_db: _VectorDBConfig = Field(default_factory=_VectorDBConfig)
    rag: _RAGConfig = Field(default_factory=_RAGConfig)
    planner: _PlannerConfig = Field(default_factory=_PlannerConfig)
    llm: _LLMConfig = Field(default_factory=_LLMConfig)
    poi_data: _POIDataConfig = Field(default_factory=_POIDataConfig)
    scoring_weights: _ScoringWeightsConfig = Field(default_factory=_ScoringWeightsConfig)
    retrieval: _RetrievalConfig = Field(default_factory=_RetrievalConfig)
    bm25: _BM25Config = Field(default_factory=_BM25Config)


# ---------------------------------------------------------------------------
# Settings — wrapper compatible con el código existente
# ---------------------------------------------------------------------------

class Settings:
    """Carga config.json y valida con Pydantic. Los typos fallan al arrancar."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            config_path = os.path.join(base, "config.json")

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = _AppConfig.model_validate(data)

        self.server = config.server.model_dump()
        self.embeddings = config.embeddings.model_dump()
        self.reranker = config.reranker.model_dump()
        self.vector_db = config.vector_db.model_dump()
        self.rag = config.rag.model_dump()
        self.planner = config.planner.model_dump()
        self.llm = config.llm.model_dump()
        self.poi_data = config.poi_data.model_dump()
        self.scoring_weights = config.scoring_weights.model_dump()
        self.retrieval = config.retrieval.model_dump()
        self.bm25 = config.bm25.model_dump()

        env_ollama_url = os.environ.get("OLLAMA_BASE_URL")
        if env_ollama_url:
            self.llm["ollama_base_url"] = env_ollama_url
            logger.info(f"Ollama URL sobreescrita por variable de entorno: {env_ollama_url}")

        logger.info(f"Configuración validada cargada desde: {config_path}")


settings = Settings()
