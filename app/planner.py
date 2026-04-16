"""
Módulo de Planificación de Itinerarios.

Recibe la lista de POIs rerankeados y produce un itinerario estructurado
por días y franjas horarias que sea:
  • Temporalmente coherente: los POIs se visitan cuando están abiertos.
  • Geográficamente compacto: cada día agrupa POIs cercanos entre sí
    (clustering greedy de vecino más próximo con distancia Haversine).
  • Presupuestariamente viable: se respeta el presupuesto diario.

Franjas horarias por defecto (configurables en config.json):
  • mañana : 09:30 – 14:00
  • tarde  : 16:00 – 20:00
"""

import logging
import math
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional

from app.config import settings
from app.models import POI, UserPreferences, PlannedPOI, DayItinerary
from app.ranker import _diversity_penalty

logger = logging.getLogger("turismo_rag")

# Días de la semana en inglés (como están en el JSON)
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


# ---------------------------------------------------------------------------
# Utilidades geográficas
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en km entre dos coordenadas geográficas (fórmula Haversine)."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def travel_minutes(poi_a: POI, poi_b: POI, speed_kmh: float = 4.5) -> int:
    """Tiempo de desplazamiento a pie entre dos POIs (minutos)."""
    dist = haversine_km(
        poi_a.coordinates.lat, poi_a.coordinates.lon,
        poi_b.coordinates.lat, poi_b.coordinates.lon,
    )
    return max(5, int((dist / speed_kmh) * 60))


# ---------------------------------------------------------------------------
# Horarios
# ---------------------------------------------------------------------------

def _time_to_minutes(hhmm: str) -> int:
    """'HH:MM' → minutos desde medianoche."""
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _minutes_to_time(minutes: int) -> str:
    """Minutos desde medianoche → 'HH:MM'."""
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def _is_open_at(poi: POI, weekday_key: str, start_min: int, end_min: int) -> bool:
    """
    Comprueba si un POI está abierto durante la franja [start_min, end_min].
    weekday_key: 'monday', 'tuesday', …
    """
    slot = poi.schedule.get(weekday_key)
    if slot is None:
        return False   # cerrado ese día

    open_min  = _time_to_minutes(slot.open)
    close_min = _time_to_minutes(slot.close)

    # Abierto 24h
    if open_min == 0 and close_min >= 1439:
        return True

    # Suficiente si el POI abre antes de que acabe la franja
    return open_min <= end_min and close_min >= start_min + poi.visit_duration_minutes


# ---------------------------------------------------------------------------
# Clustering geográfico greedy (vecino más próximo)
# ---------------------------------------------------------------------------

def _greedy_nearest_neighbor(pois: List[POI], start: Optional[POI] = None) -> List[POI]:
    """
    Ordena los POIs de un día minimizando el recorrido total (greedy NN).
    Si se indica un 'start' (punto de partida), se parte desde él.
    """
    if not pois:
        return []
    if len(pois) == 1:
        return pois

    unvisited = list(pois)
    ordered: List[POI] = []

    # Punto de inicio: el que tiene coordenadas más al noroeste (aprox. centro ciudad)
    if start:
        current = start
    else:
        current = min(unvisited, key=lambda p: (p.coordinates.lat, p.coordinates.lon))
        unvisited.remove(current)
        ordered.append(current)

    while unvisited:
        nearest = min(
            unvisited,
            key=lambda p: haversine_km(
                current.coordinates.lat, current.coordinates.lon,
                p.coordinates.lat, p.coordinates.lon,
            ),
        )
        unvisited.remove(nearest)
        ordered.append(nearest)
        current = nearest

    return ordered


# ---------------------------------------------------------------------------
# Planificador principal
# ---------------------------------------------------------------------------

class ItineraryPlanner:
    """
    Organiza los POIs en un itinerario estructurado por días y franjas.

    Algoritmo:
      Para cada día:
        1. Determinar el weekday de ese día (asumiendo que el día 1 es hoy).
        2. Filtrar POIs abiertos en alguna franja de ese día.
        3. Seleccionar los mejores según score, penalizando repetición de categoría.
        4. Ordenarlos geográficamente (vecino más próximo).
        5. Asignar franjas mañana/tarde y calcular tiempos.
    """

    def __init__(self):
        planner_cfg = settings.planner
        slots_cfg   = planner_cfg.get("slots", {})

        self.slot_manana = slots_cfg.get("manana", {"start": "09:30", "end": "14:00"})
        self.slot_tarde  = slots_cfg.get("tarde",  {"start": "16:00", "end": "20:00"})

        pois_per_day_cfg = planner_cfg.get("pois_per_day", {})
        self.pois_per_day = {
            "tranquilo": pois_per_day_cfg.get("tranquilo", 3),
            "moderado":  pois_per_day_cfg.get("moderado",  4),
            "intenso":   pois_per_day_cfg.get("intenso",   6),
        }
        self.speed_kmh = planner_cfg.get("walking_speed_kmh", 4.5)

    # ------------------------------------------------------------------
    # Método público
    # ------------------------------------------------------------------

    def plan(
        self,
        ranked: List[Tuple[POI, float, float, float]],
        preferences: UserPreferences,
        start_weekday: int = 1,   # 0=lunes, 1=martes …
    ) -> List[DayItinerary]:
        """
        Genera el itinerario completo.

        Args:
            ranked:         Lista de (POI, s_score, r_score, final_score) del ranker.
            preferences:    Preferencias del usuario.
            start_weekday:  Día de la semana del primer día (0=lun, …, 6=dom).

        Returns:
            Lista de DayItinerary, uno por día.
        """
        n_days   = preferences.duration_days
        n_per_day = self.pois_per_day.get(preferences.pace, 4)

        # Pool de candidatos: POI + scores
        pool: List[Tuple[POI, float, float, float]] = list(ranked)

        days: List[DayItinerary] = []
        used_pois: List[POI] = []

        for day_idx in range(n_days):
            weekday_idx = (start_weekday + day_idx) % 7
            weekday_key = WEEKDAYS[weekday_idx]

            manana_start = _time_to_minutes(self.slot_manana["start"])
            manana_end   = _time_to_minutes(self.slot_manana["end"])
            tarde_start  = _time_to_minutes(self.slot_tarde["start"])
            tarde_end    = _time_to_minutes(self.slot_tarde["end"])

            # --- Selección de POIs para este día ----------------------
            day_pois: List[Tuple[POI, float, float, float]] = []
            budget_remaining = preferences.budget_per_day
            selected_cats: List[POI] = []

            for entry in pool:
                if len(day_pois) >= n_per_day:
                    break
                poi, s, r, final = entry
                if poi in used_pois:
                    continue

                # Comprobación de apertura (al menos en una franja)
                open_manana = _is_open_at(poi, weekday_key, manana_start, manana_end)
                open_tarde  = _is_open_at(poi, weekday_key, tarde_start, tarde_end)
                if not open_manana and not open_tarde:
                    continue  # cerrado todo el día

                # Presupuesto: intentamos mantenerlo, pero no descartamos si es el único
                if poi.price_numeric > budget_remaining and budget_remaining > 0 and len(day_pois) > 0:
                    continue

                # Penalización de diversidad en la selección
                div_pen = _diversity_penalty(poi, selected_cats)
                if div_pen > 0.6 and len(day_pois) >= 2:
                    continue  # demasiada repetición de categoría

                day_pois.append(entry)
                selected_cats.append(poi)
                budget_remaining -= poi.price_numeric

            if not day_pois:
                # Si no hay POIs disponibles, tomar los mejores sin restricciones
                for entry in pool:
                    if entry[0] not in used_pois:
                        day_pois.append(entry)
                    if len(day_pois) >= min(n_per_day, 2):
                        break

            # Marcar como usados
            for entry in day_pois:
                used_pois.append(entry[0])

            # --- Ordenación geográfica (vecino más próximo) -----------
            day_pois_sorted_by_score = sorted(day_pois, key=lambda x: x[3], reverse=True)
            pois_ordered = _greedy_nearest_neighbor([e[0] for e in day_pois_sorted_by_score])

            # Reconstruir scores por el nuevo orden
            score_map = {e[0].id: (e[1], e[2], e[3]) for e in day_pois}

            # --- Asignación de franjas horarias -----------------------
            planned_pois: List[PlannedPOI] = []
            current_time = manana_start
            slot = "mañana"
            prev_poi: Optional[POI] = None
            total_cost = 0.0
            total_visit = 0
            total_travel = 0

            for poi in pois_ordered:
                s_sc, r_sc, f_sc = score_map.get(poi.id, (0.5, 0.5, 0.5))

                # Tiempo de viaje desde el POI anterior
                t_travel = 0
                if prev_poi:
                    t_travel = travel_minutes(prev_poi, poi, self.speed_kmh)
                    current_time += t_travel

                # Cambio de franja si excedemos la mañana
                if slot == "mañana" and current_time + poi.visit_duration_minutes > manana_end:
                    current_time = tarde_start
                    slot = "tarde"

                # Si excedemos la tarde, terminamos el día
                if current_time + poi.visit_duration_minutes > tarde_end + 30:
                    break

                end_time_min = current_time + poi.visit_duration_minutes

                planned_pois.append(PlannedPOI(
                    poi=poi,
                    day=day_idx + 1,
                    slot=slot,
                    start_time=_minutes_to_time(current_time),
                    end_time=_minutes_to_time(end_time_min),
                    semantic_score=round(s_sc, 4),
                    rerank_score=round(r_sc, 4),
                    final_score=round(f_sc, 4),
                    travel_minutes_from_previous=t_travel,
                ))

                total_cost   += poi.price_numeric
                total_visit  += poi.visit_duration_minutes
                total_travel += t_travel

                current_time = end_time_min + 10   # 10 min de margen entre visitas
                prev_poi = poi

                # Pausa de comida: pasar a franja tarde
                if slot == "mañana" and current_time >= manana_end - 30:
                    current_time = tarde_start
                    slot = "tarde"

            days.append(DayItinerary(
                day=day_idx + 1,
                pois=planned_pois,
                total_cost_eur=round(total_cost, 2),
                total_visit_minutes=total_visit,
                total_travel_minutes=total_travel,
            ))

        return days
