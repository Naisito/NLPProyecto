from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Schedule(BaseModel):
    open: str
    close: str


class Coordinates(BaseModel):
    lat: float
    lon: float


class POI(BaseModel):
    id: str
    name: str
    municipality: str
    category: str
    subcategory: str
    description: str
    coordinates: Coordinates
    address: str
    price: str
    price_numeric: float
    schedule: Dict[str, Optional[Schedule]]
    source: str
    url: str
    tags: List[str]
    enriched_text: str
    visit_duration_minutes: int
    accessibility: bool


VALID_INTERESTS = [
    "museos", "arte", "arquitectura", "gastronomía", "pintxos",
    "naturaleza", "senderismo", "playa", "surf", "historia",
    "cultura vasca", "deporte", "vida nocturna", "compras",
    "fotografía", "familia", "rural", "pueblos costeros"
]

VALID_BUDGETS   = ["bajo", "medio", "alto"]
VALID_PACES     = ["tranquilo", "moderado", "intenso"]
VALID_MOBILITY  = ["normal", "reducida"]
VALID_GROUPS    = ["solo", "pareja", "familia", "amigos"]
VALID_SCOPES    = ["Bilbao", "Bizkaia", "Ambos"]


class UserPreferences(BaseModel):
    city_scope: str = Field("Bilbao", description="Bilbao / Bizkaia / Ambos")
    duration_days: int = Field(1, ge=1, le=7)
    interests: List[str] = Field(default_factory=list, description="museos, gastronomía, naturaleza...")
    budget_per_day: float = Field(50.0, ge=0.0)
    pace: str = Field("moderado", description="tranquilo / moderado / intenso")
    mobility: str = Field("normal", description="normal / reducida")
    group_type: str = Field("pareja", description="solo / pareja / familia / amigos")
    start_hour: str = Field("09:30")
    end_hour: str = Field("20:00")
    include_meals: bool = Field(True, description="Incluir recomendaciones gastronómicas")
    extra_notes: Optional[str] = Field(None)


class RouteRequest(BaseModel):
    query: Optional[str] = Field(None, description="Texto libre. Si no hay preferences, el LLM las infiere.")
    preferences: Optional[UserPreferences] = Field(None, description="Preferencias estructuradas")


class PlannedPOI(BaseModel):
    poi: POI
    day: int
    slot: str
    start_time: str
    end_time: str
    semantic_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    travel_minutes_from_previous: int = 0


class DayItinerary(BaseModel):
    day: int
    pois: List[PlannedPOI]
    total_cost_eur: float
    total_visit_minutes: int
    total_travel_minutes: int
    day_summary: str = ""


class TouristRoute(BaseModel):
    title: str
    preferences_used: UserPreferences
    days: List[DayItinerary]
    narrative: str
    total_pois: int
    total_cost_eur: float
    generated_at: str


class EvaluationMetrics(BaseModel):
    constraint_satisfaction: float = Field(description="Cumplimiento de restricciones explícitas [0-1]")
    preference_coverage: float = Field(description="Proporción de POIs que cubren algún interés [0-1]")
    temporal_coherence: float = Field(description="Proporción de POIs abiertos en su franja [0-1]")
    geographic_consistency: float = Field(description="Compacidad geográfica media diaria [0-1]")
    budget_adherence: float = Field(description="Cumplimiento del presupuesto diario [0-1]")
    category_diversity: float = Field(description="Diversidad de categorías [0-1]")
    accessibility_compliance: float = Field(description="Proporción de POIs accesibles si aplica [0-1]")
    overall_score: float = Field(description="Puntuación global ponderada [0-1]")
    details: Dict[str, Any] = Field(default_factory=dict)


class RouteResponse(BaseModel):
    route: TouristRoute
    evaluation: EvaluationMetrics
    retrieval_info: Dict[str, Any] = Field(default_factory=dict)
    execution_time_seconds: float


class POISearchRequest(BaseModel):
    query: str
    k: int = Field(10, ge=1, le=40)
    category_filter: Optional[str] = None
    municipality_filter: Optional[str] = None


class POIListResponse(BaseModel):
    total: int
    pois: List[POI]


class HealthResponse(BaseModel):
    status: str
    index_size: int
    model_loaded: bool
    reranker_loaded: bool
