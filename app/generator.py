"""
Módulo de Generación de Itinerarios en Español.

Responsabilidades:
  1. Interpretar la consulta libre del usuario en preferencias estructuradas
     mediante una llamada LLM (función interpret_preferences).
  2. Generar el texto narrativo final del itinerario turístico en español
     a partir del itinerario estructurado (función generate_narrative).

Modelo: Ollama (local) — inferencia 100% local, sin dependencia de APIs externas.
Por defecto usa llama3.2, configurable en config.json → llm.ollama_model_name.
"""

import json
import logging
import re
from datetime import datetime
from typing import List, Optional

from openai import OpenAI

from app.config import settings
from app.models import (
    DayItinerary,
    PlannedPOI,
    TouristRoute,
    UserPreferences,
)

logger = logging.getLogger("turismo_rag")

# ---------------------------------------------------------------------------
# Cliente Ollama (API compatible con OpenAI)
# ---------------------------------------------------------------------------

_base_url = settings.llm.get("ollama_base_url", "http://localhost:11434")
_model    = settings.llm.get("ollama_model_name", "llama3.2")
_temp_gen    = float(settings.llm.get("temperature_generation", 0.5))
_temp_interp = float(settings.llm.get("temperature_interpretation", 0.1))
_max_tokens  = int(settings.llm.get("max_tokens_generation", 2000))

client = OpenAI(
    base_url=f"{_base_url}/v1",
    api_key="ollama",          # Ollama no requiere clave; valor ignorado
)


# ---------------------------------------------------------------------------
# Utilidad: extracción robusta de JSON desde respuesta del LLM
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """
    Extrae el primer objeto JSON válido de un texto que puede contener
    texto introductorio, bloques markdown (```json ... ```) u otras cadenas.
    Los modelos locales son más propensos a incluir texto extra alrededor.
    """
    # 1) Intentar parsear directamente
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2) Buscar bloque ```json ... ``` o ``` ... ```
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except json.JSONDecodeError:
            pass

    # 3) Buscar el primer objeto JSON con llaves balanceadas
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"No se encontró JSON válido en la respuesta: {text[:200]}")


# ---------------------------------------------------------------------------
# Interpretación de consulta libre → UserPreferences
# ---------------------------------------------------------------------------

_INTERP_SYSTEM = """Eres un asistente especializado en turismo en el País Vasco (Bilbao / Bizkaia).
Tu tarea es interpretar la solicitud de viaje del usuario y extraer las preferencias de forma estructurada.

Devuelve ÚNICAMENTE un objeto JSON válido con estos campos (sin texto adicional ni bloques markdown):
{
  "city_scope": "Bilbao" | "Bizkaia" | "Ambos",
  "duration_days": <entero 1-7>,
  "interests": [<lista de intereses>],
  "budget_per_day": <número en euros>,
  "pace": "tranquilo" | "moderado" | "intenso",
  "mobility": "normal" | "reducida",
  "group_type": "solo" | "pareja" | "familia" | "amigos",
  "start_hour": "<HH:MM>",
  "end_hour": "<HH:MM>",
  "include_meals": <true|false>,
  "extra_notes": "<notas adicionales o null>"
}

Intereses válidos: museos, arte, arquitectura, gastronomía, pintxos, naturaleza, senderismo,
playa, surf, historia, cultura vasca, deporte, vida nocturna, compras, fotografía, familia,
rural, pueblos costeros.

Valores por defecto si no se especifican:
  city_scope=Bilbao, duration_days=1, interests=[], budget_per_day=50,
  pace=moderado, mobility=normal, group_type=pareja, start_hour=09:30,
  end_hour=20:00, include_meals=true, extra_notes=null.

IMPORTANTE: responde SOLO con el JSON, sin explicaciones ni texto adicional."""


def interpret_preferences(query: str) -> UserPreferences:
    """
    Convierte texto libre del usuario en un objeto UserPreferences usando LLM local.
    Si el LLM falla (Ollama no está corriendo, timeout, JSON inválido),
    devuelve preferencias por defecto con un aviso en el log.
    """
    try:
        response = client.chat.completions.create(
            model=_model,
            messages=[
                {"role": "system", "content": _INTERP_SYSTEM},
                {"role": "user",   "content": query},
            ],
            temperature=_temp_interp,
            max_tokens=500,
        )
        raw = response.choices[0].message.content.strip()
        parsed = _extract_json(raw)
        prefs = UserPreferences(**parsed)
        logger.info(f"Preferencias interpretadas: {prefs.model_dump()}")
        return prefs

    except Exception as e:
        logger.warning(
            f"Error interpretando preferencias con LLM ({_model}): {e}. "
            "Usando valores por defecto. ¿Está Ollama corriendo?"
        )
        return UserPreferences()


# ---------------------------------------------------------------------------
# Generación narrativa del itinerario
# ---------------------------------------------------------------------------

def _format_day_for_prompt(day: DayItinerary) -> str:
    """Serializa un día del itinerario como texto estructurado para el prompt."""
    lines = [f"=== DÍA {day.day} ==="]
    for pp in day.pois:
        poi = pp.poi
        lines.append(
            f"  [{pp.start_time}–{pp.end_time}] {poi.name} ({poi.category})\n"
            f"    Dirección: {poi.address}\n"
            f"    Descripción: {poi.description[:200]}…\n"
            f"    Precio: {poi.price} ({poi.price_numeric:.0f} €) | "
            f"Duración estimada: {poi.visit_duration_minutes} min"
        )
        if pp.travel_minutes_from_previous > 0:
            lines.append(f"    → Desplazamiento desde anterior: ~{pp.travel_minutes_from_previous} min a pie")
    lines.append(f"  Coste total día: {day.total_cost_eur:.0f} €")
    return "\n".join(lines)


_GEN_SYSTEM = """Eres un experto guía turístico del País Vasco con un estilo narrativo
cálido, entusiasta y práctico. Tu tarea es escribir el texto narrativo de un itinerario
turístico en español castellano.

INSTRUCCIONES:
1. Escribe en primera persona del plural ('os recomendamos', 'disfrutaréis').
2. Para cada POI incluye: qué es, por qué merece la pena visitarlo y un consejo práctico.
3. Sugiere la comida vasca (pintxos, bacalao, txakoli) cuando sea natural hacerlo.
4. Conecta los POIs con transiciones fluidas ('A tan sólo 10 minutos a pie…').
5. Tono: amigable, inspirador, informativo. Evita el lenguaje genérico de folleto.
6. No repitas información ya dada en el itinerario estructurado de forma idéntica.
7. Longitud objetivo: 200–300 palabras por día de itinerario.
8. Comienza con un título atractivo y un párrafo de bienvenida a Bilbao/Bizkaia."""


def generate_narrative(
    days: List[DayItinerary],
    preferences: UserPreferences,
    original_query: Optional[str] = None,
) -> str:
    """
    Genera el texto narrativo completo del itinerario usando el LLM local (Ollama).
    Devuelve el texto en español o un fallback si el LLM falla.
    """
    itinerary_text = "\n\n".join(_format_day_for_prompt(d) for d in days)

    pref_summary = (
        f"Viajero: {preferences.group_type} | "
        f"Días: {preferences.duration_days} | "
        f"Intereses: {', '.join(preferences.interests) or 'general'} | "
        f"Presupuesto: {preferences.budget_per_day} €/día | "
        f"Ritmo: {preferences.pace}"
    )

    user_prompt = (
        f"Perfil del viajero: {pref_summary}\n"
        f"{'Consulta original: ' + original_query if original_query else ''}\n\n"
        f"Itinerario estructurado:\n{itinerary_text}\n\n"
        f"Por favor, genera el texto narrativo del itinerario completo."
    )

    try:
        response = client.chat.completions.create(
            model=_model,
            messages=[
                {"role": "system", "content": _GEN_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=_temp_gen,
            max_tokens=_max_tokens,
        )
        narrative = response.choices[0].message.content.strip()
        logger.info(f"Narrativa generada: {len(narrative)} caracteres.")
        return narrative

    except Exception as e:
        logger.error(
            f"Error generando narrativa con LLM ({_model}): {e}. "
            "¿Está Ollama corriendo? Usando fallback básico."
        )
        return _fallback_narrative(days, preferences)


def _fallback_narrative(days: List[DayItinerary], preferences: UserPreferences) -> str:
    """Narrativa de emergencia sin LLM cuando Ollama no está disponible."""
    lines = [
        f"# Itinerario turístico por {preferences.city_scope} — {preferences.duration_days} día(s)\n",
        f"Hemos preparado este itinerario pensando en un viajero que disfruta de "
        f"{', '.join(preferences.interests) or 'todo tipo de experiencias'}.\n",
        f"> ⚠️ Narrativa generada sin LLM (Ollama no disponible). "
        f"Arranca Ollama con `ollama serve` para obtener texto completo.\n",
    ]
    for day in days:
        lines.append(f"\n## Día {day.day}\n")
        for pp in day.pois:
            lines.append(
                f"**{pp.start_time}** — {pp.poi.name}: {pp.poi.description[:120]}…\n"
            )
        lines.append(f"*Coste estimado del día: {day.total_cost_eur:.0f} €*\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ensamblador final: construye el objeto TouristRoute
# ---------------------------------------------------------------------------

def assemble_route(
    days: List[DayItinerary],
    preferences: UserPreferences,
    narrative: str,
) -> TouristRoute:
    """Crea el objeto TouristRoute completo."""
    total_pois = sum(len(d.pois) for d in days)
    total_cost = sum(d.total_cost_eur for d in days)

    scope = preferences.city_scope
    dur   = preferences.duration_days
    day_word = "día" if dur == 1 else "días"
    interests_str = (
        " y ".join(preferences.interests[:2]) if preferences.interests else "turismo cultural"
    )
    title = f"Ruta de {dur} {day_word} por {scope}: {interests_str.capitalize()}"

    return TouristRoute(
        title=title,
        preferences_used=preferences,
        days=days,
        narrative=narrative,
        total_pois=total_pois,
        total_cost_eur=round(total_cost, 2),
        generated_at=datetime.now().isoformat(),
    )
