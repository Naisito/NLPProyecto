import pytest
from app.evaluator import (
    _preference_coverage,
    _temporal_coherence,
    _geographic_consistency,
    _budget_adherence,
    _category_diversity,
    _accessibility_compliance,
    _time_window_satisfaction,
)
from app.models import (
    POI, Schedule, Coordinates, UserPreferences,
    PlannedPOI, DayItinerary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _poi(
    poi_id: str = "p1",
    category: str = "museo",
    subcategory: str = "arte",
    tags: list = None,
    open_mon: str = "09:00",
    close_mon: str = "20:00",
    visit_minutes: int = 60,
    price_numeric: float = 10.0,
    accessible: bool = True,
    lat: float = 43.26,
    lon: float = -2.93,
) -> POI:
    return POI(
        id=poi_id,
        name=poi_id,
        municipality="Bilbao",
        category=category,
        subcategory=subcategory,
        description="desc",
        coordinates=Coordinates(lat=lat, lon=lon),
        address="Calle 1",
        price="€",
        price_numeric=price_numeric,
        schedule={"monday": Schedule(open=open_mon, close=close_mon)},
        source="test",
        url="",
        tags=tags or [],
        enriched_text="texto de prueba",
        visit_duration_minutes=visit_minutes,
        accessibility=accessible,
    )


def _planned(
    poi: POI,
    day: int = 1,
    start: str = "10:00",
    end: str = "11:00",
    slot: str = "mañana",
) -> PlannedPOI:
    return PlannedPOI(
        poi=poi,
        day=day,
        slot=slot,
        start_time=start,
        end_time=end,
    )


def _day(pois_planned: list, day: int = 1, cost: float = 20.0) -> DayItinerary:
    return DayItinerary(
        day=day,
        pois=pois_planned,
        total_cost_eur=cost,
        total_visit_minutes=sum(pp.poi.visit_duration_minutes for pp in pois_planned),
        total_travel_minutes=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPreferenceCoverage:
    def test_all_covered(self):
        """Todos los POIs cubren al menos un interés → 1.0."""
        prefs = UserPreferences(interests=["museos"])
        p1 = _planned(_poi(category="museo"))
        p2 = _planned(_poi(poi_id="p2", category="museo", subcategory="historia"))
        result = _preference_coverage([p1, p2], prefs)
        assert result == 1.0

    def test_none_covered(self):
        """Ningún POI cubre los intereses del usuario → 0.0."""
        prefs = UserPreferences(interests=["playa"])
        p1 = _planned(_poi(category="museo"))
        result = _preference_coverage([p1], prefs)
        assert result == 0.0

    def test_no_interests_returns_one(self):
        """Sin intereses especificados → 1.0 (cualquier ruta es válida)."""
        prefs = UserPreferences(interests=[])
        result = _preference_coverage([_planned(_poi())], prefs)
        assert result == 1.0


class TestTemporalCoherence:
    def test_poi_open_in_slot(self):
        """POI abre 09:00-20:00, slot asignado 10:00-11:00 → coherente."""
        pp = _planned(_poi(open_mon="09:00", close_mon="20:00", visit_minutes=60),
                      start="10:00", end="11:00")
        result = _temporal_coherence([pp], start_weekday=0)  # lunes
        assert result == 1.0

    def test_poi_closed_in_slot(self):
        """POI abre 09:00-10:00 pero slot asignado 10:30-11:30 → no coherente."""
        pp = _planned(_poi(open_mon="09:00", close_mon="10:00", visit_minutes=60),
                      start="10:30", end="11:30")
        result = _temporal_coherence([pp], start_weekday=0)
        assert result == 0.0


class TestGeographicConsistency:
    def test_same_location(self):
        """Dos POIs en el mismo punto → distancia 0 → score 1.0."""
        p1 = _planned(_poi("p1", lat=43.26, lon=-2.93))
        p2 = _planned(_poi("p2", lat=43.26, lon=-2.93))
        day = _day([p1, p2])
        result = _geographic_consistency([day])
        assert result == 1.0

    def test_far_apart(self):
        """POIs muy separados (>20 km) → score cerca de 0."""
        p1 = _planned(_poi("p1", lat=43.26, lon=-2.93))
        p2 = _planned(_poi("p2", lat=43.40, lon=-2.60))  # ~30 km aprox
        day = _day([p1, p2])
        result = _geographic_consistency([day])
        assert result < 0.3


class TestBudgetAdherence:
    def test_within_budget(self):
        """Coste del día dentro del presupuesto → 1.0."""
        prefs = UserPreferences(budget_per_day=50.0)
        day = _day([_planned(_poi(price_numeric=20.0))], cost=20.0)
        result = _budget_adherence([day], prefs)
        assert result == 1.0

    def test_over_budget(self):
        """Coste del día supera el presupuesto → score < 1.0."""
        prefs = UserPreferences(budget_per_day=30.0)
        day = _day([_planned(_poi(price_numeric=60.0))], cost=60.0)
        result = _budget_adherence([day], prefs)
        assert result < 1.0


class TestCategoryDiversity:
    def test_all_different(self):
        """Todas las categorías distintas → 1.0."""
        p1 = _planned(_poi("p1", category="museo"))
        p2 = _planned(_poi("p2", category="naturaleza"))
        result = _category_diversity([p1, p2])
        assert result == 1.0

    def test_all_same(self):
        """Todos museo → diversidad = 1 cat / 3 POIs = 0.33."""
        planned = [_planned(_poi(f"p{i}", category="museo")) for i in range(3)]
        result = _category_diversity(planned)
        assert abs(result - round(1 / 3, 4)) < 1e-4


class TestAccessibilityCompliance:
    def test_normal_mobility_always_one(self):
        """Movilidad normal → compliance siempre 1.0."""
        prefs = UserPreferences(mobility="normal")
        pp = _planned(_poi(accessible=False))
        result = _accessibility_compliance([pp], prefs)
        assert result == 1.0

    def test_reduced_mobility_all_accessible(self):
        """Movilidad reducida, todos accesibles → 1.0."""
        prefs = UserPreferences(mobility="reducida")
        planned = [_planned(_poi(f"p{i}", accessible=True)) for i in range(3)]
        result = _accessibility_compliance(planned, prefs)
        assert result == 1.0

    def test_reduced_mobility_none_accessible(self):
        """Movilidad reducida, ninguno accesible → 0.0."""
        prefs = UserPreferences(mobility="reducida")
        pp = _planned(_poi(accessible=False))
        result = _accessibility_compliance([pp], prefs)
        assert result == 0.0


class TestTimeWindowSatisfaction:
    def test_all_within_window(self):
        """Todos los POIs dentro de la ventana del usuario → 1.0."""
        prefs = UserPreferences(start_hour="09:00", end_hour="20:00")
        pp = _planned(_poi(), start="10:00", end="11:00")
        result = _time_window_satisfaction([pp], prefs)
        assert result == 1.0

    def test_poi_starts_before_window(self):
        """POI empieza antes de start_hour → 0.0."""
        prefs = UserPreferences(start_hour="11:00", end_hour="20:00")
        pp = _planned(_poi(), start="09:00", end="10:00")
        result = _time_window_satisfaction([pp], prefs)
        assert result == 0.0
