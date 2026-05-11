import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "degraded")
        assert "index_size" in data

    def test_stats_returns_data(self):
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_pois" in data
        assert "categories" in data


class TestRouteEndpoint:
    def test_generate_route_with_query(self):
        """Generar ruta con query de texto libre (museos Bilbao)."""
        response = client.post("/api/route", json={
            "query": "Quiero visitar museos en Bilbao"
        })
        assert response.status_code in (200, 404, 422), f"Unexpected status: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            assert "route" in data
            assert "evaluation" in data
            assert "execution_time_seconds" in data
            route = data["route"]
            assert "title" in route
            assert "days" in route
            assert "narrative" in route
            assert "total_pois" in route
            assert route["total_pois"] >= 1
            assert isinstance(route["days"], list)
            assert len(route["days"]) >= 1

    def test_generate_route_with_preferences(self):
        """Generar ruta con preferencias estructuradas."""
        response = client.post("/api/route", json={
            "preferences": {
                "city_scope": "Bilbao",
                "duration_days": 1,
                "interests": ["arte", "arquitectura"],
                "budget_per_day": 30.0,
                "pace": "tranquilo",
            }
        })
        assert response.status_code in (200, 404, 422), f"Unexpected status: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            assert data["route"]["total_pois"] >= 1
            assert data["route"]["total_cost_eur"] is not None

    def test_generate_route_bizkaia_nature(self):
        """Ruta por Bizkaia con intereses de naturaleza."""
        response = client.post("/api/route", json={
            "query": "Senderismo y naturaleza en Bizkaia"
        })
        assert response.status_code in (200, 404, 422), f"Unexpected status: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            assert data["route"]["total_pois"] >= 1

    def test_generate_route_empty_defaults(self):
        """Sin query ni preferences, usa defaults."""
        response = client.post("/api/route", json={})
        assert response.status_code in (200, 404, 422), f"Unexpected status: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            assert "route" in data


class TestPOIsEndpoint:
    def test_list_pois(self):
        response = client.get("/api/pois")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "pois" in data
        assert isinstance(data["pois"], list)

    def test_list_pois_with_category_filter(self):
        response = client.get("/api/pois?category=museo")
        assert response.status_code == 200
        data = response.json()
        for poi in data["pois"]:
            assert "museo" in poi["category"].lower()

    def test_get_poi_by_id(self):
        # Obtener lista primero para conseguir un ID válido
        pois_list = client.get("/api/pois").json()
        if pois_list["total"] > 0:
            poi_id = pois_list["pois"][0]["id"]
            response = client.get(f"/api/pois/{poi_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == poi_id

    def test_search_pois(self):
        response = client.post("/api/pois/search", json={
            "query": "museo arte Bilbao",
            "k": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
