"""
Módulo de Evaluación Automática de Rutas Turísticas.

Implementa las métricas definidas en el Hito 1, con definición operativa
precisa para cada una:

  1. preference_coverage   — cobertura de intereses
  2. temporal_coherence    — coherencia temporal (POIs abiertos en su slot)
  3. geographic_consistency — compacidad geográfica diaria
  4. budget_adherence       — cumplimiento del presupuesto
  5. category_diversity     — diversidad de categorías
  6. accessibility_compliance — accesibilidad cuando se requiere

Cada métrica devuelve un valor en [0, 1] (1 = perfecto).
La puntuación global es una media ponderada.
"""

import math
import logging
from typing import List, Dict, Any

from app.models import (
    DayItinerary,
    EvaluationMetrics,
    PlannedPOI,
    TouristRoute,
    UserPreferences,
)
from app.planner import (
    WEEKDAYS,
    _time_to_minutes,
    _is_open_at,
    haversine_km,
)
from app.ranker import INTEREST_TO_CATEGORIES

logger = logging.getLogger("turismo_rag")

# Pesos de la puntuación global
METRIC_WEIGHTS = {
    "preference_coverage":      0.25,
    "temporal_coherence":       0.25,
    "geographic_consistency":   0.20,
    "budget_adherence":         0.15,
    "category_diversity":       0.10,
    "accessibility_compliance": 0.05,
}


# ---------------------------------------------------------------------------
# Métricas individuales
# ---------------------------------------------------------------------------

def _preference_coverage(
    planned: List[PlannedPOI],
    preferences: UserPreferences,
) -> float:
    """
    Definición operativa:
        coverage = |POIs que cubren ≥1 interés del usuario| / |total POIs|

    Un POI 'cubre' un interés si su categoría, subcategoría o tags
    contienen alguna de las palabras clave mapeadas a ese interés.
    """
    if not planned:
        return 0.0
    if not preferences.interests:
        return 1.0   # sin intereses especificados, cualquier ruta es válida

    covered = 0
    for pp in planned:
        poi = pp.poi
        cat_text = f"{poi.category} {poi.subcategory} {' '.join(poi.tags)}".lower()
        for interest in preferences.interests:
            targets = INTEREST_TO_CATEGORIES.get(interest, [interest])
            if any(t in cat_text for t in targets):
                covered += 1
                break

    return round(covered / len(planned), 4)


def _temporal_coherence(
    planned: List[PlannedPOI],
    start_weekday: int = 1,  # 0=lunes
) -> float:
    """
    Definición operativa:
        coherence = |POIs abiertos en su franja asignada| / |total POIs|

    Un POI es temporalmente coherente si su horario de apertura contiene
    la franja [start_time, end_time] asignada por el planificador.
    """
    if not planned:
        return 0.0

    coherent = 0
    for pp in planned:
        day_offset = pp.day - 1
        weekday_idx = (start_weekday + day_offset) % 7
        weekday_key = WEEKDAYS[weekday_idx]

        start_min = _time_to_minutes(pp.start_time)
        end_min   = _time_to_minutes(pp.end_time)

        if _is_open_at(pp.poi, weekday_key, start_min, end_min):
            coherent += 1

    return round(coherent / len(planned), 4)


def _geographic_consistency(days: List[DayItinerary]) -> float:
    """
    Definición operativa:
        Para cada día, calcula la distancia media entre todos los pares de POIs.
        Normaliza: si dist_media < 1 km → 1.0; si dist_media > 20 km → 0.0.
        El score global es la media de los días.

    Un score alto indica que los POIs de cada día están geográficamente
    agrupados, reduciendo tiempos de desplazamiento.
    """
    if not days:
        return 0.0

    day_scores = []
    for day in days:
        pois = [pp.poi for pp in day.pois]
        if len(pois) < 2:
            day_scores.append(1.0)
            continue

        # Distancia media entre todos los pares
        dists = []
        for i in range(len(pois)):
            for j in range(i + 1, len(pois)):
                d = haversine_km(
                    pois[i].coordinates.lat, pois[i].coordinates.lon,
                    pois[j].coordinates.lat, pois[j].coordinates.lon,
                )
                dists.append(d)

        mean_dist = sum(dists) / len(dists)

        # Normalización: 0 km → 1.0 | 20 km → 0.0
        score = max(0.0, 1.0 - mean_dist / 20.0)
        day_scores.append(round(score, 4))

    return round(sum(day_scores) / len(day_scores), 4)


def _budget_adherence(
    days: List[DayItinerary],
    preferences: UserPreferences,
) -> float:
    """
    Definición operativa:
        Para cada día: score = max(0, 1 - max(0, coste_día - presupuesto) / presupuesto)
        Score global = media de los días.

    Penaliza proporcionalmente el exceso de presupuesto.
    Si el presupuesto es 0, cualquier gasto da 0.
    """
    if not days or preferences.budget_per_day <= 0:
        return 1.0

    daily_scores = []
    for day in days:
        excess = max(0.0, day.total_cost_eur - preferences.budget_per_day)
        score  = max(0.0, 1.0 - excess / preferences.budget_per_day)
        daily_scores.append(score)

    return round(sum(daily_scores) / len(daily_scores), 4)


def _category_diversity(planned: List[PlannedPOI]) -> float:
    """
    Definición operativa:
        diversity = |categorías únicas| / |total POIs|

    Máximo 1.0 cuando todos los POIs son de categorías distintas.
    """
    if not planned:
        return 0.0
    cats = {pp.poi.category for pp in planned}
    return round(len(cats) / len(planned), 4)


def _accessibility_compliance(
    planned: List[PlannedPOI],
    preferences: UserPreferences,
) -> float:
    """
    Definición operativa:
        Si mobility == 'reducida':
            compliance = |POIs accesibles| / |total POIs|
        Si mobility == 'normal':
            compliance = 1.0 (no aplica restricción)
    """
    if preferences.mobility != "reducida":
        return 1.0
    if not planned:
        return 0.0

    accessible = sum(1 for pp in planned if pp.poi.accessibility)
    return round(accessible / len(planned), 4)


# ---------------------------------------------------------------------------
# Evaluador principal
# ---------------------------------------------------------------------------

def evaluate_route(
    route: TouristRoute,
    preferences: UserPreferences,
    start_weekday: int = 1,
) -> EvaluationMetrics:
    """
    Calcula todas las métricas de evaluación para una ruta generada.

    Args:
        route:          La ruta turística generada.
        preferences:    Las preferencias del usuario empleadas.
        start_weekday:  Día de la semana del primer día (0=lunes, …, 6=domingo).

    Returns:
        EvaluationMetrics con todas las puntuaciones y detalles.
    """
    all_planned: List[PlannedPOI] = [pp for day in route.days for pp in day.pois]

    m_pref   = _preference_coverage(all_planned, preferences)
    m_temp   = _temporal_coherence(all_planned, start_weekday)
    m_geo    = _geographic_consistency(route.days)
    m_budget = _budget_adherence(route.days, preferences)
    m_div    = _category_diversity(all_planned)
    m_access = _accessibility_compliance(all_planned, preferences)

    overall = (
        METRIC_WEIGHTS["preference_coverage"]      * m_pref
        + METRIC_WEIGHTS["temporal_coherence"]     * m_temp
        + METRIC_WEIGHTS["geographic_consistency"] * m_geo
        + METRIC_WEIGHTS["budget_adherence"]       * m_budget
        + METRIC_WEIGHTS["category_diversity"]     * m_div
        + METRIC_WEIGHTS["accessibility_compliance"] * m_access
    )

    details: Dict[str, Any] = {
        "total_pois":             len(all_planned),
        "days":                   len(route.days),
        "unique_categories":      len({pp.poi.category for pp in all_planned}),
        "unique_municipalities":  len({pp.poi.municipality for pp in all_planned}),
        "total_cost_eur":         route.total_cost_eur,
        "avg_daily_cost_eur":     round(route.total_cost_eur / max(len(route.days), 1), 2),
        "budget_per_day_eur":     preferences.budget_per_day,
        "poi_categories":         [pp.poi.category for pp in all_planned],
        "metric_weights":         METRIC_WEIGHTS,
    }

    metrics = EvaluationMetrics(
        preference_coverage=m_pref,
        temporal_coherence=m_temp,
        geographic_consistency=m_geo,
        budget_adherence=m_budget,
        category_diversity=m_div,
        accessibility_compliance=m_access,
        overall_score=round(overall, 4),
        details=details,
    )

    logger.info(
        f"Evaluación completada: pref={m_pref:.2f} temp={m_temp:.2f} "
        f"geo={m_geo:.2f} budget={m_budget:.2f} div={m_div:.2f} "
        f"overall={overall:.2f}"
    )
    return metrics
