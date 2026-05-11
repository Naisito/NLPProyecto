import logging
import math
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional

from app.config import settings
from app.models import POI, UserPreferences, PlannedPOI, DayItinerary
from app.ranker import _diversity_penalty

logger = logging.getLogger("turismo_rag")

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def travel_minutes(poi_a: POI, poi_b: POI, speed_kmh: float = 4.5) -> int:
    dist = haversine_km(poi_a.coordinates.lat, poi_a.coordinates.lon,
                        poi_b.coordinates.lat, poi_b.coordinates.lon)
    return max(5, int((dist / speed_kmh) * 60))


def _time_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _minutes_to_time(minutes: int) -> str:
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def _is_open_at(poi: POI, weekday_key: str, start_min: int, end_min: int) -> bool:
    slot = poi.schedule.get(weekday_key)
    if slot is None:
        return False

    open_min = _time_to_minutes(slot.open)
    close_min = _time_to_minutes(slot.close)

    if open_min == 0 and close_min >= 1439:
        return True

    effective_start = max(open_min, start_min)
    effective_end = min(close_min, end_min)
    return (effective_end - effective_start) >= poi.visit_duration_minutes


def _greedy_nearest_neighbor(pois: List[POI], start: Optional[POI] = None) -> List[POI]:
    if not pois:
        return []
    if len(pois) == 1:
        return pois

    unvisited = list(pois)
    ordered: List[POI] = []

    if start:
        current = start
    else:
        current = min(unvisited, key=lambda p: (p.coordinates.lat, p.coordinates.lon))
        unvisited.remove(current)
        ordered.append(current)

    while unvisited:
        nearest = min(unvisited, key=lambda p: haversine_km(
            current.coordinates.lat, current.coordinates.lon,
            p.coordinates.lat, p.coordinates.lon))
        unvisited.remove(nearest)
        ordered.append(nearest)
        current = nearest

    return ordered


def _two_opt(route: List[POI]) -> List[POI]:
    """2-opt local search sobre ruta greedy. Para N<=7 tarda <10ms."""
    if len(route) < 3:
        return list(route)

    improved = True
    best = list(route)

    def _dist(i: int, j: int) -> float:
        return haversine_km(best[i].coordinates.lat, best[i].coordinates.lon,
                           best[j].coordinates.lat, best[j].coordinates.lon)

    while improved:
        improved = False
        n = len(best)
        for i in range(n - 1):
            for j in range(i + 1, n):
                j_next = (j + 1) % n
                current_cost = _dist(i, (i + 1) % n) + _dist(j, j_next)
                new_cost = _dist(i, j) + _dist((i + 1) % n, j_next)
                if new_cost < current_cost - 1e-10:
                    best[i + 1 : j + 1] = reversed(best[i + 1 : j + 1])
                    improved = True

    return best


class ItineraryPlanner:

    def __init__(self):
        planner_cfg = settings.planner
        slots_cfg = planner_cfg.get("slots", {})

        self.slot_manana = slots_cfg.get("manana", {"start": "09:30", "end": "14:00"})
        self.slot_tarde = slots_cfg.get("tarde", {"start": "16:00", "end": "20:00"})

        pois_per_day_cfg = planner_cfg.get("pois_per_day", {})
        self.pois_per_day = {
            "tranquilo": pois_per_day_cfg.get("tranquilo", 3),
            "moderado":  pois_per_day_cfg.get("moderado", 4),
            "intenso":   pois_per_day_cfg.get("intenso", 6),
        }
        self.speed_kmh = planner_cfg.get("walking_speed_kmh", 4.5)

    def plan(
        self,
        ranked: List[Tuple[POI, float, float, float]],
        preferences: UserPreferences,
        start_weekday: int = 1,
    ) -> List[DayItinerary]:
        n_days = preferences.duration_days
        n_per_day = self.pois_per_day.get(preferences.pace, 4)

        pool: List[Tuple[POI, float, float, float]] = list(ranked)
        days: List[DayItinerary] = []
        used_pois: List[POI] = []

        for day_idx in range(n_days):
            weekday_idx = (start_weekday + day_idx) % 7
            weekday_key = WEEKDAYS[weekday_idx]

            cfg_manana_start = _time_to_minutes(self.slot_manana["start"])
            cfg_manana_end = _time_to_minutes(self.slot_manana["end"])
            cfg_tarde_start = _time_to_minutes(self.slot_tarde["start"])
            cfg_tarde_end = _time_to_minutes(self.slot_tarde["end"])

            user_start = _time_to_minutes(preferences.start_hour)
            user_end = _time_to_minutes(preferences.end_hour)

            if user_end - user_start < 240:
                manana_start = user_start
                manana_end = user_end
                tarde_start = user_end
                tarde_end = user_end
            else:
                manana_start = max(cfg_manana_start, user_start)
                manana_end = min(cfg_manana_end, user_end)
                tarde_start = max(cfg_tarde_start, user_start)
                tarde_end = min(cfg_tarde_end, user_end)

            day_pois: List[Tuple[POI, float, float, float]] = []
            budget_remaining = preferences.budget_per_day
            selected_cats: List[POI] = []

            for entry in pool:
                if len(day_pois) >= n_per_day:
                    break
                poi, s, r, final = entry
                if poi in used_pois:
                    continue

                if not _is_open_at(poi, weekday_key, manana_start, manana_end) and \
                   not _is_open_at(poi, weekday_key, tarde_start, tarde_end):
                    continue

                if poi.price_numeric > budget_remaining and budget_remaining > 0 and len(day_pois) > 0:
                    continue

                div_pen = _diversity_penalty(poi, selected_cats)
                if div_pen > 0.6 and len(day_pois) >= 2:
                    continue

                day_pois.append(entry)
                selected_cats.append(poi)
                budget_remaining -= poi.price_numeric

            if not day_pois:
                for entry in pool:
                    if entry[0] not in used_pois:
                        day_pois.append(entry)
                    if len(day_pois) >= min(n_per_day, 2):
                        break

            for entry in day_pois:
                used_pois.append(entry[0])

            day_pois_sorted = sorted(day_pois, key=lambda x: x[3], reverse=True)
            pois_ordered = _greedy_nearest_neighbor([e[0] for e in day_pois_sorted])
            if len(pois_ordered) >= 3:
                pois_ordered = _two_opt(pois_ordered)

            score_map = {e[0].id: (e[1], e[2], e[3]) for e in day_pois}

            planned_pois: List[PlannedPOI] = []
            current_time = manana_start
            slot = "mañana"
            prev_poi: Optional[POI] = None
            total_cost = 0.0
            total_visit = 0
            total_travel = 0

            for poi in pois_ordered:
                s_sc, r_sc, f_sc = score_map.get(poi.id, (0.5, 0.5, 0.5))

                t_travel = 0
                if prev_poi:
                    t_travel = travel_minutes(prev_poi, poi, self.speed_kmh)
                    current_time += t_travel

                if slot == "mañana" and current_time + poi.visit_duration_minutes > manana_end:
                    current_time = tarde_start
                    slot = "tarde"

                if current_time + poi.visit_duration_minutes > tarde_end + 30:
                    break

                end_time_min = current_time + poi.visit_duration_minutes

                planned_pois.append(PlannedPOI(
                    poi=poi, day=day_idx + 1, slot=slot,
                    start_time=_minutes_to_time(current_time),
                    end_time=_minutes_to_time(end_time_min),
                    semantic_score=round(s_sc, 4),
                    rerank_score=round(r_sc, 4),
                    final_score=round(f_sc, 4),
                    travel_minutes_from_previous=t_travel,
                ))

                total_cost += poi.price_numeric
                total_visit += poi.visit_duration_minutes
                total_travel += t_travel

                current_time = end_time_min + 10
                prev_poi = poi

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
