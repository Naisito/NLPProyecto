from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# POI (Punto de Interés) – unidad básica del sistema
# ---------------------------------------------------------------------------

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
    price: str                        # "gratis", "€", "€€", "€€€"
    price_numeric: float              # 0.0, 5.0, 18.0 …
    schedule: Dict[str, Optional[Schedule]]
    source: str
    url: str
    tags: List[str]
    enriched_text: str                # texto para indexación semántica
    visit_duration_minutes: int
    accessibility: bool


# ---------------------------------------------------------------------------
# Preferencias del usuario
# ---------------------------------------------------------------------------

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
    city_scope: str = Field("Bilbao", description="Ámbito geográfico: 'Bilbao', 'Bizkaia' o 'Ambos'")
    duration_days: int = Field(1, ge=1, le=7, description="Número de días del viaje")
    interests: List[str] = Field(
        default_factory=list,
        description="Lista de intereses del usuario (museos, gastronomía, naturaleza…)"
    )
    budget_per_day: float = Field(50.0, ge=0.0, description="Presupuesto diario en euros")
    pace: str = Field("moderado", description="Ritmo del viaje: tranquilo, moderado o intenso")
    mobility: str = Field("normal", description="Movilidad: normal o reducida")
    group_type: str = Field("pareja", description="Tipo de grupo: solo, pareja, familia, amigos")
    start_hour: str = Field("09:30", description="Hora de inicio de actividades (HH:MM)")
    end_hour: str = Field("20:00", description="Hora de fin de actividades (HH:MM)")
    include_meals: bool = Field(True, description="Incluir recomendaciones gastronómicas")
    extra_notes: Optional[str] = Field(None, description="Notas adicionales del usuario")


# ---------------------------------------------------------------------------
# Solicitud de ruta (entrada API)
# ---------------------------------------------------------------------------

class RouteRequest(BaseModel):
    query: Optional[str] = Field(
        None,
        description="Consulta en texto libre. Si se indica, el sistema infiere las preferencias con el LLM."
    )
    preferences: Optional[UserPreferences] = Field(
        None,
        description="Preferencias estructuradas. Si se omite junto a 'query', se usarán los valores por defecto."
    )


# ---------------------------------------------------------------------------
# POI planificado (con slot de tiempo asignado)
# ---------------------------------------------------------------------------

class PlannedPOI(BaseModel):
    poi: POI
    day: int
    slot: str                         # "mañana" | "tarde"
    start_time: str                   # "HH:MM"
    end_time: str                     # "HH:MM"
    semantic_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    travel_minutes_from_previous: int = 0


# ---------------------------------------------------------------------------
# Itinerario de un día
# ---------------------------------------------------------------------------

class DayItinerary(BaseModel):
    day: int
    pois: List[PlannedPOI]
    total_cost_eur: float
    total_visit_minutes: int
    total_travel_minutes: int
    day_summary: str = ""             # resumen LLM del día (opcional)


# ---------------------------------------------------------------------------
# Ruta turística completa
# ---------------------------------------------------------------------------

class TouristRoute(BaseModel):
    title: str
    preferences_used: UserPreferences
    days: List[DayItinerary]
    narrative: str                    # texto narrativo generado por LLM
    total_pois: int
    total_cost_eur: float
    generated_at: str


# ---------------------------------------------------------------------------
# Métricas de evaluación automática
# ---------------------------------------------------------------------------

class EvaluationMetrics(BaseModel):
    preference_coverage: float = Field(
        description="Proporción de POIs que cubren al menos un interés del usuario [0–1]"
    )
    temporal_coherence: float = Field(
        description="Proporción de POIs abiertos en su franja horaria asignada [0–1]"
    )
    geographic_consistency: float = Field(
        description="Compacidad geográfica media diaria (1 = todos los POIs en el mismo punto) [0–1]"
    )
    budget_adherence: float = Field(
        description="Cumplimiento del presupuesto diario (1 = dentro del presupuesto) [0–1]"
    )
    category_diversity: float = Field(
        description="Diversidad de categorías (categorías únicas / total POIs) [0–1]"
    )
    accessibility_compliance: float = Field(
        description="Proporción de POIs accesibles cuando se solicitó movilidad reducida [0–1]"
    )
    overall_score: float = Field(
        description="Puntuación global ponderada [0–1]"
    )
    details: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Respuesta completa de la API
# ---------------------------------------------------------------------------

class RouteResponse(BaseModel):
    route: TouristRoute
    evaluation: EvaluationMetrics
    retrieval_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Información de trazabilidad del proceso RAG y ranking"
    )
    execution_time_seconds: float


# ---------------------------------------------------------------------------
# Modelos auxiliares para endpoints de POIs
# ---------------------------------------------------------------------------

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
