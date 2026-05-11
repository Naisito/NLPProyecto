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
    "constraint_satisfaction":  0.30,
    "preference_coverage":      0.20,
    "temporal_coherence":       0.20,
    "geographic_consistency":   0.15,
    "budget_adherence":         0.10,
    "category_diversity":       0.03,
    "accessibility_compliance": 0.02,
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


def _interest_satisfaction(
    planned: List[PlannedPOI],
    preferences: UserPreferences,
) -> float:
    """
    Definición operativa:
        satisfaction = |intereses cubiertos al menos una vez| / |intereses pedidos|

    A diferencia de `preference_coverage`, esta métrica mide si la ruta
    satisface el conjunto de intereses solicitados, no sólo cuántos POIs
    encajan con alguno de ellos.
    """
    if not preferences.interests:
        return 1.0
    if not planned:
        return 0.0

    covered_interests = 0
    for interest in preferences.interests:
        targets = INTEREST_TO_CATEGORIES.get(interest, [interest])
        matched = False
        for pp in planned:
            poi = pp.poi
            cat_text = f"{poi.category} {poi.subcategory} {' '.join(poi.tags)}".lower()
            if any(target in cat_text for target in targets):
                matched = True
                break
        if matched:
            covered_interests += 1

    return round(covered_interests / len(preferences.interests), 4)


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


def _scope_satisfaction(
    planned: List[PlannedPOI],
    preferences: UserPreferences,
) -> float:
    """
    Mide si el municipio de los POIs respeta el ámbito geográfico pedido.

    Nota:
      - `Bilbao`: todos los POIs deben estar en Bilbao.
      - `Ambos`: no restringe.
      - `Bizkaia`: el corpus ya pertenece a Bizkaia, por lo que no penaliza.
    """
    if not planned:
        return 0.0

    scope = preferences.city_scope.strip().lower()
    if scope in {"ambos", "bizkaia"}:
        return 1.0

    if scope == "bilbao":
        in_scope = sum(1 for pp in planned if pp.poi.municipality.strip().lower() == "bilbao")
        return round(in_scope / len(planned), 4)

    return 1.0


def _duration_satisfaction(route: TouristRoute, preferences: UserPreferences) -> float:
    """1.0 si el número de días generados coincide con el pedido; 0.0 si no."""
    return 1.0 if len(route.days) == preferences.duration_days else 0.0


def _pace_satisfaction(route: TouristRoute, preferences: UserPreferences) -> float:
    """
    Compara el número medio de POIs por día con el esperado según el ritmo.
    Penaliza desviaciones grandes de forma suave.
    """
    if not route.days:
        return 0.0

    expected_per_day = {
        "tranquilo": 3,
        "moderado": 4,
        "intenso": 6,
    }.get(preferences.pace, 4)

    total_pois = sum(len(day.pois) for day in route.days)
    avg_per_day = total_pois / max(len(route.days), 1)
    deviation = abs(avg_per_day - expected_per_day) / max(expected_per_day, 1)
    return round(max(0.0, 1.0 - deviation), 4)


def _time_window_satisfaction(
    planned: List[PlannedPOI],
    preferences: UserPreferences,
) -> float:
    """
    Proporción de POIs cuya visita queda dentro de la ventana horaria pedida
    por el usuario.
    """
    if not planned:
        return 0.0

    pref_start = _time_to_minutes(preferences.start_hour)
    pref_end = _time_to_minutes(preferences.end_hour)

    compliant = 0
    for pp in planned:
        start_min = _time_to_minutes(pp.start_time)
        end_min = _time_to_minutes(pp.end_time)
        if start_min >= pref_start and end_min <= pref_end:
            compliant += 1

    return round(compliant / len(planned), 4)


def _meals_satisfaction(
    planned: List[PlannedPOI],
    preferences: UserPreferences,
) -> float:
    """
    Si el usuario quiere incluir comidas, comprueba si la ruta contiene al
    menos una recomendación gastronómica.
    """
    if not preferences.include_meals:
        return 1.0
    if not planned:
        return 0.0

    meal_keywords = INTEREST_TO_CATEGORIES.get("gastronomía", []) + INTEREST_TO_CATEGORIES.get("pintxos", [])
    for pp in planned:
        poi = pp.poi
        cat_text = f"{poi.category} {poi.subcategory} {' '.join(poi.tags)}".lower()
        if any(keyword in cat_text for keyword in meal_keywords):
            return 1.0
    return 0.0


def _constraint_satisfaction(
    route: TouristRoute,
    planned: List[PlannedPOI],
    preferences: UserPreferences,
) -> tuple[float, Dict[str, float]]:
    """
    Evalúa el cumplimiento agregado de restricciones explícitas del usuario.

    Todas las submétricas se calculan en [0, 1] y la métrica final es la media.
    """
    breakdown = {
        "duration_match": _duration_satisfaction(route, preferences),
        "scope_match": _scope_satisfaction(planned, preferences),
        "interest_match": _interest_satisfaction(planned, preferences),
        "budget_match": _budget_adherence(route.days, preferences),
        "mobility_match": _accessibility_compliance(planned, preferences),
        "pace_match": _pace_satisfaction(route, preferences),
        "time_window_match": _time_window_satisfaction(planned, preferences),
        "meals_match": _meals_satisfaction(planned, preferences),
    }
    overall = sum(breakdown.values()) / len(breakdown)
    return round(overall, 4), breakdown


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

    m_constraints, constraint_breakdown = _constraint_satisfaction(route, all_planned, preferences)
    m_pref   = _preference_coverage(all_planned, preferences)
    m_temp   = _temporal_coherence(all_planned, start_weekday)
    m_geo    = _geographic_consistency(route.days)
    m_budget = _budget_adherence(route.days, preferences)
    m_div    = _category_diversity(all_planned)
    m_access = _accessibility_compliance(all_planned, preferences)

    overall = (
        METRIC_WEIGHTS["constraint_satisfaction"]  * m_constraints
        + METRIC_WEIGHTS["preference_coverage"]      * m_pref
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
        "constraint_breakdown":   constraint_breakdown,
        "metric_weights":         METRIC_WEIGHTS,
    }

    metrics = EvaluationMetrics(
        constraint_satisfaction=m_constraints,
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
        f"Evaluación completada: constraints={m_constraints:.2f} pref={m_pref:.2f} temp={m_temp:.2f} "
        f"geo={m_geo:.2f} budget={m_budget:.2f} div={m_div:.2f} "
        f"overall={overall:.2f}"
    )
    return metrics
