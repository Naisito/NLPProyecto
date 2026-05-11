"""
Tests unitarios para _extract_json (app/generator.py).

Cubre 5 casos: JSON limpio, con <think>, con markdown, con texto extra, malformado.
"""

import pytest
from app.generator import _extract_json


class TestExtractJson:

    def test_clean_json(self):
        """JSON limpio sin decoradores."""
        text = '{"city_scope": "Bilbao", "duration_days": 2}'
        result = _extract_json(text)
        assert result["city_scope"] == "Bilbao"
        assert result["duration_days"] == 2

    def test_json_inside_think_block(self):
        """JSON precedido de un bloque <think>...</think>."""
        text = '<think>Voy a analizar la consulta del usuario...</think>\n{"city_scope": "Bizkaia", "duration_days": 1}'
        result = _extract_json(text)
        assert result["city_scope"] == "Bizkaia"

    def test_json_in_markdown_block(self):
        """JSON dentro de un bloque ```json ... ```."""
        text = 'Aquí está el resultado:\n```json\n{"city_scope": "Ambos", "pace": "intenso"}\n```'
        result = _extract_json(text)
        assert result["city_scope"] == "Ambos"
        assert result["pace"] == "intenso"

    def test_json_with_leading_text(self):
        """JSON precedido de texto introductorio (sin bloque think ni markdown)."""
        text = 'Claro, aquí tienes las preferencias extraídas: {"city_scope": "Bilbao", "duration_days": 3}'
        result = _extract_json(text)
        assert result["duration_days"] == 3

    def test_malformed_json_raises(self):
        """Texto sin JSON válido lanza ValueError."""
        text = "No hay JSON aquí, solo texto plano sin llaves."
        with pytest.raises(ValueError):
            _extract_json(text)
