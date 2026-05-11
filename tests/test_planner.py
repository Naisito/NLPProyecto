"""
Tests unitarios para app/planner.py.

Cubre:
  - _is_open_at: 4 casos del bug F0.1
  - plan(): 2 casos del fix F0.3 (start_hour / end_hour del usuario)
  - _greedy_nearest_neighbor: 1 caso básico de ordenación
"""

import pytest
from unittest.mock import MagicMock

from app.planner import _is_open_at, _greedy_nearest_neighbor, ItineraryPlanner
from app.models import POI, Schedule, Coordinates, UserPreferences


# ---------------------------------------------------------------------------
# Fixtures de POIs mínimos
# ---------------------------------------------------------------------------

def _make_poi(
    poi_id: str = "p1",
    open_time: str = "09:00",
    close_time: str = "20:00",
    visit_minutes: int = 60,
    lat: float = 43.26,
    lon: float = -2.93,
) -> POI:
    return POI(
        id=poi_id,
        name=poi_id,
        municipality="Bilbao",
        category="museo",
        subcategory="arte",
        description="desc",
        coordinates=Coordinates(lat=lat, lon=lon),
        address="Calle 1",
        price="€",
        price_numeric=10.0,
        schedule={"monday": Schedule(open=open_time, close=close_time)},
        source="test",
        url="",
        tags=[],
        enriched_text="texto",
        visit_duration_minutes=visit_minutes,
        accessibility=True,
    )


# ---------------------------------------------------------------------------
# Tests F0.1 — _is_open_at
# ---------------------------------------------------------------------------

class TestIsOpenAt:
    """4 casos cubriendo el bug corregido en _is_open_at."""

    def test_poi_fully_inside_slot(self):
        """POI abre 09:00 cierra 20:00, slot 09:30-14:00, visita 60 min → True."""
        poi = _make_poi(open_time="09:00", close_time="20:00", visit_minutes=60)
        assert _is_open_at(poi, "monday", 9*60+30, 14*60) is True

    def test_poi_opens_inside_slot(self):
        """POI abre 11:00 cierra 20:00, slot 09:30-14:00, visita 60 min → True (caben 11:00-12:00)."""
        poi = _make_poi(open_time="11:00", close_time="20:00", visit_minutes=60)
        assert _is_open_at(poi, "monday", 9*60+30, 14*60) is True

    def test_poi_closes_too_early(self):
        """POI abre 09:00 cierra 10:00, slot 09:30-14:00, visita 60 min → False (solo 30 min disponibles)."""
        poi = _make_poi(open_time="09:00", close_time="10:00", visit_minutes=60)
        assert _is_open_at(poi, "monday", 9*60+30, 14*60) is False

    def test_poi_closed_all_day(self):
        """POI no tiene horario para ese día (tuesday) → False."""
        poi = _make_poi(open_time="09:00", close_time="20:00", visit_minutes=60)
        assert _is_open_at(poi, "tuesday", 9*60+30, 14*60) is False

    def test_bug_regression_short_window(self):
        """Regresión del bug: POI abre 13:00-14:00, slot 09:30-14:00, visita 60 min.
        La lógica antigua devolvería True (close≥start+visit: 840≥570+60=630 ✓ y open≤end: 780≤840 ✓).
        La lógica correcta: effective 13:00-14:00 = 60 min, justo suficiente → True.
        """
        poi = _make_poi(open_time="13:00", close_time="14:00", visit_minutes=60)
        assert _is_open_at(poi, "monday", 9*60+30, 14*60) is True

    def test_bug_regression_not_enough_time(self):
        """POI abre 13:30-14:00 (30 min), visita 60 min, slot 09:30-14:00 → False."""
        poi = _make_poi(open_time="13:30", close_time="14:00", visit_minutes=60)
        assert _is_open_at(poi, "monday", 9*60+30, 14*60) is False


# ---------------------------------------------------------------------------
# Tests F0.3 — planner respeta ventana horaria del usuario
# ---------------------------------------------------------------------------

class TestPlannerTimeWindow:
    """Verifica que el planner usa start_hour/end_hour del usuario."""

    def _prefs(self, start: str = "09:30", end: str = "20:00") -> UserPreferences:
        return UserPreferences(
            city_scope="Bilbao",
            duration_days=1,
            interests=[],
            budget_per_day=100,
            pace="moderado",
            start_hour=start,
            end_hour=end,
        )

    def _make_ranked(self, n: int = 4):
        """Devuelve n POIs con schedule monday 08:00-22:00."""
        result = []
        lats = [43.26, 43.261, 43.262, 43.263]
        for i in range(n):
            poi = _make_poi(
                poi_id=f"p{i}",
                open_time="08:00",
                close_time="22:00",
                visit_minutes=60,
                lat=lats[i % len(lats)],
                lon=-2.93,
            )
            result.append((poi, 0.9, 0.9, 0.9))
        return result

    def test_start_hour_respected(self):
        """Con start_hour=11:00 ningún POI debe empezar antes de las 11:00."""
        planner = ItineraryPlanner()
        ranked = self._make_ranked(4)
        days = planner.plan(ranked, self._prefs(start="11:00", end="20:00"), start_weekday=0)
        for day in days:
            for pp in day.pois:
                start_min = int(pp.start_time.split(":")[0]) * 60 + int(pp.start_time.split(":")[1])
                assert start_min >= 11 * 60, f"POI empieza en {pp.start_time}, esperado ≥11:00"

    def test_end_hour_respected(self):
        """Con end_hour=18:00 ningún POI debe terminar después de las 18:00."""
        planner = ItineraryPlanner()
        ranked = self._make_ranked(4)
        days = planner.plan(ranked, self._prefs(start="09:30", end="18:00"), start_weekday=0)
        for day in days:
            for pp in day.pois:
                end_min = int(pp.end_time.split(":")[0]) * 60 + int(pp.end_time.split(":")[1])
                assert end_min <= 18 * 60 + 30, f"POI termina en {pp.end_time}, esperado ≤18:00 (con margen)"


# ---------------------------------------------------------------------------
# Test básico de _greedy_nearest_neighbor
# ---------------------------------------------------------------------------

class TestGreedyNN:
    def test_orders_by_proximity(self):
        """Tres POIs alineados: el más al sur (menor lat) se toma primero y el resto se encadena."""
        p_north = _make_poi("north", lat=43.27, lon=-2.93)
        p_mid   = _make_poi("mid",   lat=43.26, lon=-2.93)
        p_south = _make_poi("south", lat=43.25, lon=-2.93)

        # Sin punto de partida: elige el de menores (lat, lon) → p_south
        ordered = _greedy_nearest_neighbor([p_north, p_mid, p_south])
        assert ordered[0].id == "south"
        # El siguiente vecino más cercano a p_south es p_mid
        assert ordered[1].id == "mid"
        assert ordered[2].id == "north"
