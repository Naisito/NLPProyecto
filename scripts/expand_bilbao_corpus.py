#!/usr/bin/env python3
"""
Regenera el corpus completo de POIs de Bilbao / Bizkaia desde fuentes automáticas.

Objetivos:
1. Eliminar la dependencia de una base manual de 40 POIs.
2. Construir un único JSON final que la aplicación carga directamente en runtime.
3. Aumentar el corpus con fuentes estructuradas y reproducibles.

Fuentes usadas:
- OpenStreetMap Overpass + Nominatim para Bilbao ciudad.
- Open Data Euskadi / Open Data Bilbao (RDF de lugares de interés turístico).
- Wikidata vía SPARQL para POIs turísticos y patrimoniales de Bilbao y Bizkaia.

La aplicación NO ejecuta este script al arrancar. Este script se usa únicamente
para regenerar el JSON estático del repositorio.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import ssl
import textwrap
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


USER_AGENT = "NLPProyecto Bilbao Corpus Builder/2.0"
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT_DIR / "data" / "pois_bilbao_bizkaia.json"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
OVERPASS_QUERY_GROUPS = [
    ('nwr["tourism"~"^(museum|gallery|attraction|artwork|viewpoint)$"]', "tourism"),
    ('nwr["historic"~"^(monument|memorial|castle|archaeological_site|ruins|building)$"]', "historic"),
    ('nwr["amenity"~"^(theatre|arts_centre|cinema|marketplace|place_of_worship|library)$"]', "amenity"),
    ('nwr["leisure"~"^(park|garden|sports_centre|stadium)$"]', "leisure"),
    ('nwr["man_made"~"^(bridge|tower)$"]', "man_made"),
    ('nwr["natural"="peak"]', "natural"),
]
OPEN_DATA_BILBAO_RDF = "https://www.bilbao.eus/bilbaoopendata/turismo/lugares_interes_turistico.rdf"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

DAYS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]

OSM_DAY_ORDER = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
OSM_TO_DAY = {
    "Mo": "monday",
    "Tu": "tuesday",
    "We": "wednesday",
    "Th": "thursday",
    "Fr": "friday",
    "Sa": "saturday",
    "Su": "sunday",
}

STOPWORDS = {
    "a",
    "al",
    "con",
    "da",
    "de",
    "del",
    "do",
    "e",
    "el",
    "en",
    "eta",
    "la",
    "las",
    "le",
    "los",
    "o",
    "por",
    "un",
    "una",
    "y",
    "bilbao",
    "bizkaia",
}

LODGING_TAGS = {
    "hotel",
    "hostel",
    "apartment",
    "guest_house",
    "motel",
    "camp_site",
    "caravan_site",
    "chalet",
}

WIKIDATA_CLASSES = [
    {"qid": "Q33506", "category": "museo", "subcategory": "museo", "label": "museo"},
    {"qid": "Q12280", "category": "arquitectura", "subcategory": "puente", "label": "puente"},
    {"qid": "Q22698", "category": "parque", "subcategory": "parque urbano", "label": "parque"},
    {"qid": "Q24354", "category": "cultura", "subcategory": "teatro", "label": "teatro"},
    {"qid": "Q16970", "category": "religioso", "subcategory": "iglesia", "label": "iglesia"},
    {"qid": "Q570116", "category": "cultura", "subcategory": "atracción turística", "label": "atracción turística"},
    {"qid": "Q4989906", "category": "historia", "subcategory": "monumento", "label": "monumento"},
    {"qid": "Q179700", "category": "arte", "subcategory": "estatua", "label": "estatua"},
    {"qid": "Q40080", "category": "naturaleza", "subcategory": "playa", "label": "playa"},
    {"qid": "Q330284", "category": "gastronomía", "subcategory": "mercado", "label": "mercado"},
    {"qid": "Q7075", "category": "cultura", "subcategory": "biblioteca", "label": "biblioteca"},
    {"qid": "Q483110", "category": "deporte", "subcategory": "estadio", "label": "estadio"},
    {"qid": "Q860861", "category": "arte", "subcategory": "escultura", "label": "escultura"},
    {"qid": "Q1107656", "category": "parque", "subcategory": "jardín", "label": "jardín"},
    {"qid": "Q23413", "category": "historia", "subcategory": "castillo", "label": "castillo"},
    {"qid": "Q839954", "category": "historia", "subcategory": "sitio arqueológico", "label": "sitio arqueológico"},
    {"qid": "Q101659", "category": "historia", "subcategory": "dolmen", "label": "dolmen"},
    {"qid": "Q1569871", "category": "historia", "subcategory": "patrimonio industrial", "label": "patrimonio industrial"},
    {"qid": "Q39715", "category": "arquitectura", "subcategory": "faro", "label": "faro"},
    {"qid": "Q16560", "category": "arquitectura", "subcategory": "palacio", "label": "palacio"},
    {"qid": "Q56750657", "category": "religioso", "subcategory": "ermita", "label": "ermita"},
    {"qid": "Q142031", "category": "cultura", "subcategory": "funicular", "label": "funicular"},
    {"qid": "Q82117", "category": "historia", "subcategory": "puerta histórica", "label": "puerta histórica"},
    {"qid": "Q16748868", "category": "historia", "subcategory": "muralla", "label": "muralla"},
    {"qid": "Q6017969", "category": "naturaleza", "subcategory": "mirador", "label": "mirador"},
    {"qid": "Q1440300", "category": "arquitectura", "subcategory": "torre mirador", "label": "torre mirador"},
    {"qid": "Q44613", "category": "religioso", "subcategory": "monasterio", "label": "monasterio"},
]

WIKIDATA_REJECT_PATTERNS = [
    r"^q\d+$",
    r"^sin nombre$",
    r"fosa común",
    r"west portal",
    r"portal of",
    r"\boficina\b",
    r"\boffice\b",
    r"sener ingeniería",
    r"sener ingenieria",
]

RDF_NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "loc": "http://www.w3.org/2007/uwa/context/location.owl#",
    "xpb": "http://www.morelab.deusto.es/ontologies/xpb#",
    "geonames": "http://www.geonames.org/ontology#",
}


def make_request(url: str, data: Optional[bytes] = None, accept: str = "*/*") -> Request:
    return Request(
        url,
        data=data,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": accept,
        },
    )


def fetch_text(url: str, timeout: int = 120, accept: str = "*/*") -> str:
    context = ssl._create_unverified_context()
    request = make_request(url, accept=accept)
    with urlopen(request, timeout=timeout, context=context) as response:
        return response.read().decode("utf-8", errors="ignore")


def fetch_json(url: str, params: Dict[str, str], timeout: int = 120) -> object:
    query_url = f"{url}?{urlencode(params)}"
    raw = fetch_text(query_url, timeout=timeout, accept="application/json")
    return json.loads(raw)


def post_json_with_retry(url: str, payload: bytes, timeout: int = 240) -> Tuple[dict, str]:
    last_error: Optional[Exception] = None
    for round_index in range(3):
        for endpoint in OVERPASS_ENDPOINTS:
            request = Request(
                endpoint,
                data=payload,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            try:
                context = ssl._create_unverified_context()
                with urlopen(request, timeout=timeout, context=context) as response:
                    raw = response.read().decode("utf-8")
                if not raw.lstrip().startswith("{"):
                    raise ValueError(f"Respuesta no JSON recibida desde {endpoint}")
                return json.loads(raw), endpoint
            except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        time.sleep(2 * (round_index + 1))
    raise RuntimeError(f"No se pudo consultar Overpass. Último error: {last_error}")


def sparql_query_with_retry(query: str, timeout: int = 120) -> dict:
    params = urlencode({"format": "json", "query": query})
    url = f"{WIKIDATA_SPARQL_URL}?{params}"
    last_error: Optional[Exception] = None

    for attempt in range(4):
        try:
            raw = fetch_text(url, timeout=timeout, accept="application/sparql-results+json")
            return json.loads(raw)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"No se pudo consultar Wikidata. Último error: {last_error}")


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def sentence_case(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:]


def normalize_text(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    return " ".join(ascii_text.split())


def significant_tokens(text: str) -> set[str]:
    return {
        token
        for token in normalize_text(text).split()
        if token not in STOPWORDS and len(token) > 1
    }


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius * c


def overlap_ratio(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / max(1, min(len(tokens_a), len(tokens_b)))


def point_in_ring(lon: float, lat: float, ring: Sequence[Sequence[float]]) -> bool:
    inside = False
    ring_size = len(ring)
    for index in range(ring_size):
        lon1, lat1 = ring[index]
        lon2, lat2 = ring[(index + 1) % ring_size]
        intersects = (lat1 > lat) != (lat2 > lat)
        if not intersects:
            continue
        lon_intersection = (lon2 - lon1) * (lat - lat1) / (lat2 - lat1 + 1e-12) + lon1
        if lon < lon_intersection:
            inside = not inside
    return inside


def point_in_polygon(lon: float, lat: float, geojson: dict) -> bool:
    geo_type = geojson.get("type")
    coordinates = geojson.get("coordinates", [])

    if geo_type == "Polygon":
        if not coordinates:
            return False
        if not point_in_ring(lon, lat, coordinates[0]):
            return False
        for hole in coordinates[1:]:
            if point_in_ring(lon, lat, hole):
                return False
        return True

    if geo_type == "MultiPolygon":
        for polygon in coordinates:
            if point_in_polygon(lon, lat, {"type": "Polygon", "coordinates": polygon}):
                return True
        return False

    return False


def schedule_all_day() -> Dict[str, Dict[str, str]]:
    return {day: {"open": "00:00", "close": "23:59"} for day in DAYS}


def schedule_closed_monday(open_time: str = "10:00", close_time: str = "19:00") -> Dict[str, Optional[dict]]:
    schedule: Dict[str, Optional[dict]] = {"monday": None}
    for day in DAYS[1:6]:
        schedule[day] = {"open": open_time, "close": close_time}
    schedule["saturday"] = {"open": open_time, "close": close_time}
    schedule["sunday"] = {"open": open_time, "close": "14:00"}
    return schedule


def schedule_uniform(open_time: str, close_time: str) -> Dict[str, Dict[str, str]]:
    return {day: {"open": open_time, "close": close_time} for day in DAYS}


def schedule_market() -> Dict[str, Optional[dict]]:
    schedule: Dict[str, Optional[dict]] = {"sunday": None}
    for day in DAYS[:-1]:
        schedule[day] = {"open": "08:30", "close": "14:30"}
    return schedule


def expand_osm_days(day_expr: str) -> List[str]:
    expanded: List[str] = []
    for part in [chunk.strip() for chunk in day_expr.split(",") if chunk.strip()]:
        if "-" in part:
            start, end = [item.strip() for item in part.split("-", 1)]
            if start not in OSM_DAY_ORDER or end not in OSM_DAY_ORDER:
                continue
            start_idx = OSM_DAY_ORDER.index(start)
            end_idx = OSM_DAY_ORDER.index(end)
            if start_idx <= end_idx:
                expanded.extend(OSM_DAY_ORDER[start_idx : end_idx + 1])
            else:
                expanded.extend(OSM_DAY_ORDER[start_idx:] + OSM_DAY_ORDER[: end_idx + 1])
        elif part in OSM_DAY_ORDER:
            expanded.append(part)
    seen = set()
    ordered = []
    for item in expanded:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def parse_opening_hours(opening_hours: str) -> Optional[Dict[str, Optional[dict]]]:
    text = clean_text(opening_hours)
    if not text:
        return None
    if text == "24/7":
        return schedule_all_day()

    schedule: Dict[str, Optional[dict]] = {day: None for day in DAYS}
    matched_any = False

    for segment in [piece.strip() for piece in text.split(";") if piece.strip()]:
        if "PH" in segment or "SH" in segment or "off" in segment.lower():
            continue

        match = re.match(
            r"^(?P<days>[A-Za-z,\-]+)\s+(?P<times>\d{1,2}:\d{2}-\d{1,2}:\d{2}(?:,\d{1,2}:\d{2}-\d{1,2}:\d{2})*)$",
            segment,
        )
        if not match:
            continue

        day_tokens = expand_osm_days(match.group("days"))
        time_ranges = re.findall(r"(\d{1,2}:\d{2})-(\d{1,2}:\d{2})", match.group("times"))
        if not day_tokens or not time_ranges:
            continue

        open_time = time_ranges[0][0].zfill(5)
        close_time = time_ranges[-1][1].zfill(5)

        for osm_day in day_tokens:
            schedule[OSM_TO_DAY[osm_day]] = {"open": open_time, "close": close_time}
            matched_any = True

    return schedule if matched_any else None


def build_default_schedule(category: str) -> Dict[str, Optional[dict]]:
    if category in {"arte", "arquitectura", "historia", "naturaleza", "parque"}:
        return schedule_all_day()
    if category == "museo":
        return schedule_closed_monday()
    if category == "gastronomía":
        return schedule_market()
    if category == "religioso":
        return schedule_uniform("10:00", "19:00")
    if category == "deporte":
        return schedule_uniform("09:00", "21:00")
    return schedule_uniform("10:00", "20:00")


def infer_price(category: str, explicit_free: Optional[bool] = None) -> Tuple[str, float]:
    if explicit_free is True:
        return "gratis", 0.0
    if explicit_free is False:
        if category == "museo":
            return "€", 8.0
        if category in {"cultura", "deporte"}:
            return "€", 6.0
        return "€", 5.0

    if category in {"parque", "arquitectura", "historia", "naturaleza", "arte", "religioso", "gastronomía"}:
        return "gratis", 0.0
    if category == "museo":
        return "€", 8.0
    if category in {"cultura", "deporte"}:
        return "€", 6.0
    return "gratis", 0.0


def infer_visit_duration(category: str, subcategory: str) -> int:
    if category == "museo":
        return 90
    if category == "parque":
        return 45
    if category == "gastronomía":
        return 45
    if category == "cultura":
        return 60
    if category == "religioso":
        return 35
    if category == "naturaleza" and "mirador" in subcategory:
        return 30
    if category == "naturaleza":
        return 45
    if category == "arte":
        return 25
    if category == "arquitectura":
        return 25
    if category == "historia":
        return 30
    if category == "deporte":
        return 60
    return 45


def infer_accessibility(category: str, subcategory: str, explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return explicit
    if category in {"museo", "cultura", "gastronomía", "parque", "arte", "arquitectura"}:
        return True
    if category == "naturaleza" and ("cima" in subcategory or "mirador" in subcategory):
        return False
    if category in {"historia", "religioso"}:
        return False
    return True


def translate_osm_value(value: str) -> str:
    translations = {
        "artwork": "obra de arte",
        "attraction": "atracción",
        "arts_centre": "centro cultural",
        "bridge": "puente",
        "building": "edificio histórico",
        "bust": "busto",
        "castle": "fortificación histórica",
        "cinema": "cine",
        "gallery": "galería",
        "garden": "jardín",
        "installation": "instalación",
        "library": "biblioteca",
        "marketplace": "mercado",
        "memorial": "memorial",
        "monument": "monumento",
        "museum": "museo",
        "park": "parque",
        "place_of_worship": "espacio religioso",
        "ruins": "ruinas",
        "sculpture": "escultura",
        "sports_centre": "centro deportivo",
        "stadium": "estadio",
        "statue": "estatua",
        "theatre": "teatro",
        "tower": "torre",
        "viewpoint": "mirador",
    }
    cleaned = clean_text(value)
    return translations.get(cleaned, cleaned)


def kind_label(tags: dict) -> str:
    for key in ["tourism", "amenity", "leisure", "man_made", "historic", "natural"]:
        value = clean_text(tags.get(key))
        if value:
            translated = translate_osm_value(value)
            if translated:
                return translated
    return "lugar de interés"


def infer_category_and_subcategory(name: str, tags: dict) -> Tuple[str, str]:
    lower_name = name.lower()
    tourism = tags.get("tourism")
    amenity = tags.get("amenity")
    leisure = tags.get("leisure")
    historic = tags.get("historic")
    man_made = tags.get("man_made")
    natural = tags.get("natural")

    if tourism == "museum":
        return "museo", clean_text(tags.get("museum")) or "museo"
    if man_made == "bridge":
        return "arquitectura", "puente"
    if tourism == "gallery":
        return "arte", "galería de arte"
    if tourism == "artwork":
        artwork_type = clean_text(tags.get("artwork_type"))
        if artwork_type == "statue" or "estatua" in lower_name:
            return "arte", "escultura pública"
        if artwork_type == "mural":
            return "arte", "mural"
        return "arte", "arte público"
    if tourism == "viewpoint" or natural == "peak":
        return "naturaleza", "mirador"
    if leisure == "park":
        return "parque", "parque urbano"
    if leisure == "garden":
        return "parque", "jardín"
    if amenity == "marketplace":
        return "gastronomía", "mercado"
    if amenity == "place_of_worship":
        if "catedral" in lower_name:
            return "religioso", "catedral"
        if "basílica" in lower_name or "basilica" in lower_name:
            return "religioso", "basílica"
        if "capilla" in lower_name:
            return "religioso", "capilla"
        if "parroquia" in lower_name:
            return "religioso", "parroquia"
        return "religioso", "templo"
    if amenity == "theatre":
        return "cultura", "teatro"
    if amenity == "arts_centre":
        return "cultura", "centro cultural"
    if amenity == "cinema":
        return "cultura", "cine"
    if amenity == "library":
        return "cultura", "biblioteca"
    if leisure == "stadium":
        return "deporte", "estadio"
    if leisure == "sports_centre":
        return "deporte", "centro deportivo"
    if man_made == "tower":
        return "arquitectura", "torre mirador"
    if historic == "castle":
        return "historia", "fortificación"
    if historic == "archaeological_site":
        return "historia", "sitio arqueológico"
    if historic == "ruins":
        return "historia", "ruinas"
    if historic == "building":
        return "historia", "edificio histórico"
    if historic == "monument":
        return "historia", "monumento"
    if historic == "memorial":
        return "historia", "memorial"
    if tourism == "attraction":
        return "cultura", "atracción urbana"
    return "cultura", kind_label(tags)


def build_address(tags: dict, municipality: str = "Bilbao") -> str:
    street = clean_text(tags.get("addr:street"))
    number = clean_text(tags.get("addr:housenumber"))
    postcode = clean_text(tags.get("addr:postcode"))
    city = clean_text(tags.get("addr:city")) or municipality
    suburb = (
        clean_text(tags.get("addr:suburb"))
        or clean_text(tags.get("addr:neighbourhood"))
        or clean_text(tags.get("addr:quarter"))
        or clean_text(tags.get("is_in:suburb"))
    )

    line_1 = " ".join(part for part in [street, number] if part).strip()
    pieces = [line_1, suburb, " ".join(part for part in [postcode, city] if part).strip(), "Bizkaia"]
    compact = [piece for piece in pieces if piece]
    return ", ".join(compact) if compact else f"{municipality}, Bizkaia"


def build_url_from_osm(element: dict) -> str:
    tags = element.get("tags", {})
    for key in ["website", "contact:website", "url"]:
        value = clean_text(tags.get(key))
        if not value:
            continue
        if value.startswith(("http://", "https://")):
            return value
        return f"https://{value}"

    wikipedia = clean_text(tags.get("wikipedia"))
    if wikipedia and ":" in wikipedia:
        lang, title = wikipedia.split(":", 1)
        return f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"

    element_type = element.get("type", "node")
    element_id = element.get("id")
    return f"https://www.openstreetmap.org/{element_type}/{element_id}"


def compose_semantic_tags(
    source_tags: Iterable[str],
    category: str,
    subcategory: str,
    accessibility: bool,
) -> List[str]:
    values = list(source_tags) + [
        category,
        subcategory,
        "accesible" if accessibility else "accesibilidad limitada",
    ]

    cleaned: List[str] = []
    seen = set()
    for value in values:
        text = clean_text(value)
        if not text or len(text) <= 1:
            continue
        if text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def build_description(
    name: str,
    municipality: str,
    category: str,
    subcategory: str,
    address: str,
    source: str,
    extra: str = "",
) -> str:
    source_prefix = {
        "OpenStreetMap Overpass": "Se ha detectado automáticamente en OpenStreetMap como punto de interés relevante para rutas turísticas.",
        "Open Data Euskadi": "Figura en el catálogo oficial de lugares de interés turístico publicado como open data.",
        "Wikidata": "Está catalogado en Wikidata como elemento turístico o patrimonial visitable.",
    }.get(source, "Se incorpora al corpus como punto de interés turístico.")

    parts = [
        f"{name} es un punto de interés de tipo {subcategory} en {municipality}.",
        source_prefix,
        f"Categoría principal: {category}.",
        f"Se localiza en {address}.",
    ]
    if extra:
        parts.append(sentence_case(extra) + ".")
    return clean_text(" ".join(parts))


def build_enriched_text(
    name: str,
    municipality: str,
    category: str,
    subcategory: str,
    description: str,
    address: str,
    semantic_tags: Sequence[str],
) -> str:
    chunks = [
        name,
        municipality,
        f"Categoría {category}",
        f"Subcategoría {subcategory}",
        description,
        address,
        ", ".join(semantic_tags),
    ]
    return ". ".join(chunk for chunk in chunks if chunk)


def make_record(
    *,
    name: str,
    municipality: str,
    category: str,
    subcategory: str,
    lat: float,
    lon: float,
    address: str,
    source: str,
    url: str,
    tags: Sequence[str],
    explicit_free: Optional[bool] = None,
    schedule: Optional[Dict[str, Optional[dict]]] = None,
    accessibility: Optional[bool] = None,
    extra_description: str = "",
) -> dict:
    price, price_numeric = infer_price(category, explicit_free=explicit_free)
    accessibility_value = infer_accessibility(category, subcategory, explicit=accessibility)
    semantic_tags = compose_semantic_tags(tags, category, subcategory, accessibility_value)
    description = build_description(
        name=name,
        municipality=municipality,
        category=category,
        subcategory=subcategory,
        address=address,
        source=source,
        extra=extra_description,
    )
    enriched_text = build_enriched_text(
        name=name,
        municipality=municipality,
        category=category,
        subcategory=subcategory,
        description=description,
        address=address,
        semantic_tags=semantic_tags,
    )

    return {
        "id": "",
        "name": name,
        "municipality": municipality,
        "category": category,
        "subcategory": subcategory,
        "description": description,
        "coordinates": {"lat": round(lat, 6), "lon": round(lon, 6)},
        "address": address,
        "price": price,
        "price_numeric": price_numeric,
        "schedule": schedule or build_default_schedule(category),
        "source": source,
        "url": url,
        "tags": semantic_tags,
        "enriched_text": enriched_text,
        "visit_duration_minutes": infer_visit_duration(category, subcategory),
        "accessibility": accessibility_value,
    }


def existing_index_from_pois(pois: Sequence[dict]) -> List[dict]:
    index: List[dict] = []
    for poi in pois:
        coordinates = poi["coordinates"]
        index.append(
            {
                "normalized_name": normalize_text(poi["name"]),
                "tokens": significant_tokens(poi["name"]),
                "lat": float(coordinates["lat"]),
                "lon": float(coordinates["lon"]),
                "category": poi["category"],
            }
        )
    return index


def is_duplicate(name: str, category: str, lat: float, lon: float, existing_index: Sequence[dict]) -> bool:
    normalized = normalize_text(name)
    tokens = significant_tokens(name)

    for item in existing_index:
        if normalized == item["normalized_name"]:
            return True

        distance = haversine_m(lat, lon, item["lat"], item["lon"])
        if distance <= 120 and overlap_ratio(tokens, item["tokens"]) >= 0.5:
            return True

        if distance <= 200 and (
            normalized in item["normalized_name"] or item["normalized_name"] in normalized
        ):
            return True

        if distance <= 300 and category == item["category"] and overlap_ratio(tokens, item["tokens"]) >= 0.75:
            return True

    return False


def register_record(record: dict, target: List[dict], dedupe_index: List[dict]) -> bool:
    lat = float(record["coordinates"]["lat"])
    lon = float(record["coordinates"]["lon"])
    if is_duplicate(record["name"], record["category"], lat, lon, dedupe_index):
        return False

    target.append(record)
    dedupe_index.append(
        {
            "normalized_name": normalize_text(record["name"]),
            "tokens": significant_tokens(record["name"]),
            "lat": lat,
            "lon": lon,
            "category": record["category"],
        }
    )
    return True


def candidate_coordinates(element: dict) -> Optional[Tuple[float, float]]:
    lat = element.get("lat")
    lon = element.get("lon")
    if lat is not None and lon is not None:
        return float(lat), float(lon)

    center = element.get("center", {})
    if "lat" in center and "lon" in center:
        return float(center["lat"]), float(center["lon"])
    return None


def should_keep_osm_candidate(tags: dict) -> bool:
    if not clean_text(tags.get("name")):
        return False
    if clean_text(tags.get("tourism")).lower() in LODGING_TAGS:
        return False
    if clean_text(tags.get("access")).lower() == "private":
        return False
    if any(key in tags for key in ["disused", "abandoned"]):
        return False
    return True


def build_overpass_query(bbox: Sequence[float], selector: str) -> str:
    south, north, west, east = bbox
    return textwrap.dedent(
        f"""
        [out:json][timeout:180];
        (
          {selector}({south},{west},{north},{east});
        );
        out center tags;
        """
    ).strip()


def fetch_bilbao_boundary() -> dict:
    response = fetch_json(
        NOMINATIM_URL,
        {
            "q": "Bilbao, Bizkaia, Euskadi, Spain",
            "format": "jsonv2",
            "limit": "1",
            "polygon_geojson": "1",
        },
    )
    if not response:
        raise RuntimeError("Nominatim no ha devuelto resultados para Bilbao.")
    return response[0]


def build_osm_record(element: dict) -> dict:
    tags = element["tags"]
    name = clean_text(tags["name"])
    coordinates = candidate_coordinates(element)
    if coordinates is None:
        raise ValueError(f"Elemento OSM sin coordenadas válidas: {name}")

    lat, lon = coordinates
    category, subcategory = infer_category_and_subcategory(name, tags)
    address = build_address(tags, municipality="Bilbao")
    schedule = parse_opening_hours(tags.get("opening_hours", "")) or build_default_schedule(category)
    explicit_free: Optional[bool] = None
    fee = clean_text(tags.get("fee")).lower()
    if fee in {"no", "free"}:
        explicit_free = True
    elif fee in {"yes", "paid"}:
        explicit_free = False

    accessibility: Optional[bool] = None
    wheelchair = clean_text(tags.get("wheelchair")).lower()
    if wheelchair in {"yes", "designated", "limited"}:
        accessibility = True
    elif wheelchair == "no":
        accessibility = False

    semantic_tags = [
        kind_label(tags),
        translate_osm_value(tags.get("tourism", "")),
        translate_osm_value(tags.get("amenity", "")),
        translate_osm_value(tags.get("historic", "")),
        translate_osm_value(tags.get("leisure", "")),
        translate_osm_value(tags.get("man_made", "")),
        translate_osm_value(tags.get("artwork_type", "")),
        clean_text(tags.get("artist_name")),
        clean_text(tags.get("architect")),
        clean_text(tags.get("operator")),
        clean_text(tags.get("addr:suburb")),
        clean_text(tags.get("addr:neighbourhood")),
    ]
    extra_bits = [
        clean_text(tags.get("description")),
        clean_text(tags.get("artist_name")),
        clean_text(tags.get("architect")),
    ]
    extra_text = ". ".join(bit for bit in extra_bits if bit)

    return make_record(
        name=name,
        municipality="Bilbao",
        category=category,
        subcategory=subcategory,
        lat=lat,
        lon=lon,
        address=address,
        source="OpenStreetMap Overpass",
        url=build_url_from_osm(element),
        tags=semantic_tags,
        explicit_free=explicit_free,
        schedule=schedule,
        accessibility=accessibility,
        extra_description=extra_text,
    )


def collect_osm_bilbao(dedupe_index: List[dict]) -> Tuple[List[dict], dict]:
    if DEFAULT_OUTPUT.exists():
        try:
            data = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
            if any(poi.get("source") == "OpenStreetMap Overpass" for poi in data.get("pois", [])):
                return _collect_osm_bilbao_fallback(dedupe_index, reason="reuse_local_materialized_osm")
        except Exception:
            pass

    try:
        return _collect_osm_bilbao_live(dedupe_index)
    except RuntimeError as exc:
        return _collect_osm_bilbao_fallback(dedupe_index, reason=str(exc))


def _collect_osm_bilbao_live(dedupe_index: List[dict]) -> Tuple[List[dict], dict]:
    boundary = fetch_bilbao_boundary()
    bbox = [float(value) for value in boundary["boundingbox"]]
    geojson = boundary["geojson"]
    records: List[dict] = []
    discarded_outside = 0
    discarded_filtered = 0
    discarded_duplicates = 0
    downloaded_candidates = 0
    endpoints_used: List[str] = []

    for selector, _label in OVERPASS_QUERY_GROUPS:
        overpass_query = build_overpass_query(bbox, selector)
        payload = urlencode({"data": overpass_query}).encode("utf-8")
        overpass_data, endpoint = post_json_with_retry(OVERPASS_ENDPOINTS[0], payload)
        endpoints_used.append(endpoint)
        downloaded_candidates += len(overpass_data.get("elements", []))

        for element in overpass_data.get("elements", []):
            tags = element.get("tags", {})
            if not should_keep_osm_candidate(tags):
                discarded_filtered += 1
                continue

            coords = candidate_coordinates(element)
            if coords is None:
                discarded_filtered += 1
                continue

            lat, lon = coords
            if not point_in_polygon(lon, lat, geojson):
                discarded_outside += 1
                continue

            record = build_osm_record(element)
            if not register_record(record, records, dedupe_index):
                discarded_duplicates += 1

    records.sort(key=lambda poi: (poi["municipality"], poi["category"], normalize_text(poi["name"])))
    summary = {
        "source": "OpenStreetMap Overpass",
        "downloaded_candidates": downloaded_candidates,
        "added": len(records),
        "discarded_outside_polygon": discarded_outside,
        "discarded_filtered": discarded_filtered,
        "discarded_duplicates": discarded_duplicates,
        "overpass_endpoints": sorted(set(endpoints_used)),
        "mode": "live",
    }
    return records, summary


def _collect_osm_bilbao_fallback(dedupe_index: List[dict], reason: str) -> Tuple[List[dict], dict]:
    if not DEFAULT_OUTPUT.exists():
        raise RuntimeError(f"No hay fallback OSM local disponible. Error original: {reason}")

    data = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    existing = [
        poi
        for poi in data.get("pois", [])
        if poi.get("source") == "OpenStreetMap Overpass"
    ]

    records: List[dict] = []
    discarded_duplicates = 0
    for poi in existing:
        cloned = json.loads(json.dumps(poi, ensure_ascii=False))
        cloned["id"] = ""
        if not register_record(cloned, records, dedupe_index):
            discarded_duplicates += 1

    records.sort(key=lambda poi: (poi["municipality"], poi["category"], normalize_text(poi["name"])))
    summary = {
        "source": "OpenStreetMap Overpass",
        "downloaded_candidates": len(existing),
        "added": len(records),
        "discarded_outside_polygon": 0,
        "discarded_filtered": 0,
        "discarded_duplicates": discarded_duplicates,
        "overpass_endpoints": [],
        "mode": "fallback_local_json",
        "fallback_reason": reason,
    }
    return records, summary


def utm30_to_wgs84(easting: float, northing: float) -> Tuple[float, float]:
    a = 6378137.0
    e = 0.081819191
    e_sq = e ** 2
    e1sq = e_sq / (1 - e_sq)
    k0 = 0.9996
    x = easting - 500000.0
    y = northing
    lon0 = math.radians(-3.0)

    m = y / k0
    mu = m / (a * (1 - e_sq / 4 - 3 * e_sq ** 2 / 64 - 5 * e_sq ** 3 / 256))

    e1 = (1 - math.sqrt(1 - e_sq)) / (1 + math.sqrt(1 - e_sq))
    j1 = 3 * e1 / 2 - 27 * e1 ** 3 / 32
    j2 = 21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32
    j3 = 151 * e1 ** 3 / 96
    j4 = 1097 * e1 ** 4 / 512
    fp = mu + j1 * math.sin(2 * mu) + j2 * math.sin(4 * mu) + j3 * math.sin(6 * mu) + j4 * math.sin(8 * mu)

    sin_fp = math.sin(fp)
    cos_fp = math.cos(fp)
    tan_fp = math.tan(fp)

    c1 = e1sq * cos_fp ** 2
    t1 = tan_fp ** 2
    r1 = a * (1 - e_sq) / ((1 - e_sq * sin_fp ** 2) ** 1.5)
    n1 = a / math.sqrt(1 - e_sq * sin_fp ** 2)
    d = x / (n1 * k0)

    q1 = n1 * tan_fp / r1
    q2 = d ** 2 / 2
    q3 = (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * e1sq) * d ** 4 / 24
    q4 = (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * e1sq - 3 * c1 ** 2) * d ** 6 / 720
    lat = fp - q1 * (q2 - q3 + q4)

    q5 = d
    q6 = (1 + 2 * t1 + c1) * d ** 3 / 6
    q7 = (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * e1sq + 24 * t1 ** 2) * d ** 5 / 120
    lon = lon0 + (q5 - q6 + q7) / cos_fp

    return math.degrees(lat), math.degrees(lon)


def map_open_data_type(type_uri: str) -> Tuple[str, str, Optional[bool]]:
    type_name = type_uri.rsplit("#", 1)[-1]
    mapping = {
        "Museum": ("museo", "museo", False),
        "Exhibition": ("arte", "exposición", True),
        "EmblematicBuilding": ("arquitectura", "edificio emblemático", True),
        "Building": ("arquitectura", "edificio singular", True),
        "Religious_Monument": ("religioso", "monumento religioso", True),
        "Classical": ("historia", "patrimonio clásico", True),
    }
    return mapping.get(type_name, ("cultura", "lugar de interés", True))


def collect_open_data_bilbao(dedupe_index: List[dict]) -> Tuple[List[dict], dict]:
    raw_xml = fetch_text(OPEN_DATA_BILBAO_RDF, timeout=120, accept="application/rdf+xml,text/xml")
    root = ET.fromstring(raw_xml)

    coords_by_ref: Dict[str, Tuple[float, float]] = {}
    place_entries: List[dict] = []
    rdf_resource = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
    rdf_about = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"
    xml_lang = "{http://www.w3.org/XML/1998/namespace}lang"

    for desc in root.findall("rdf:Description", RDF_NS):
        about = clean_text(desc.attrib.get(rdf_about))
        type_element = desc.find("rdf:type", RDF_NS)
        type_uri = type_element.attrib.get(rdf_resource, "") if type_element is not None else ""

        if about.startswith("coordinates/"):
            northing = clean_text(desc.findtext("loc:northing", default="", namespaces=RDF_NS))
            easting = clean_text(desc.findtext("loc:easting", default="", namespaces=RDF_NS))
            if northing and easting:
                # El RDF usa las etiquetas loc:northing/easting al revés respecto a UTM.
                utm_easting = float(northing)
                utm_northing = float(easting)
                coords_by_ref[about] = utm30_to_wgs84(utm_easting, utm_northing)
            continue

        if not about.startswith("place/"):
            continue

        name = ""
        for name_node in desc.findall("geonames:name", RDF_NS):
            if name_node.attrib.get(xml_lang) == "es":
                name = clean_text(name_node.text)
                break
            if not name:
                name = clean_text(name_node.text)

        coord_node = desc.find("loc:utmCoordinates", RDF_NS)
        coord_ref = coord_node.attrib.get(rdf_resource, "") if coord_node is not None else ""
        place_entries.append(
            {
                "name": name,
                "address": clean_text(desc.findtext("xpb:address", default="", namespaces=RDF_NS)),
                "type_uri": type_uri,
                "coord_ref": coord_ref,
            }
        )

    records: List[dict] = []
    discarded_missing_coords = 0
    discarded_duplicates = 0

    for entry in place_entries:
        if not entry["name"]:
            continue
        coords = coords_by_ref.get(entry["coord_ref"])
        if not coords:
            discarded_missing_coords += 1
            continue

        lat, lon = coords
        category, subcategory, explicit_free = map_open_data_type(entry["type_uri"])
        address = entry["address"] or "Bilbao, Bizkaia"
        record = make_record(
            name=entry["name"],
            municipality="Bilbao",
            category=category,
            subcategory=subcategory,
            lat=lat,
            lon=lon,
            address=address,
            source="Open Data Euskadi",
            url=OPEN_DATA_BILBAO_RDF,
            tags=["Open Data Bilbao", subcategory, "Bilbao", category],
            explicit_free=explicit_free,
            extra_description=f"Tipo oficial en el dataset: {subcategory}",
        )
        if not register_record(record, records, dedupe_index):
            discarded_duplicates += 1

    records.sort(key=lambda poi: (poi["municipality"], poi["category"], normalize_text(poi["name"])))
    summary = {
        "source": "Open Data Euskadi",
        "downloaded_candidates": len(place_entries),
        "added": len(records),
        "discarded_missing_coords": discarded_missing_coords,
        "discarded_duplicates": discarded_duplicates,
    }
    return records, summary


def parse_wikidata_point(wkt: str) -> Optional[Tuple[float, float]]:
    match = re.match(r"^Point\(([-0-9.]+)\s+([-0-9.]+)\)$", clean_text(wkt))
    if not match:
        return None
    lon = float(match.group(1))
    lat = float(match.group(2))
    return lat, lon


def wikidata_scope_query(qid: str, scope: str) -> str:
    return wikidata_scope_query_batch([qid], scope)


def wikidata_scope_query_batch(qids: Sequence[str], scope: str) -> str:
    values_clause = " ".join(f"wd:{qid}" for qid in qids)
    if scope == "bilbao":
        location_clause = "?item wdt:P131* wd:Q8692 ."
        municipality_clause = ""
        municipality_bind = 'BIND("Bilbao"@es AS ?municipalityLabel)'
    elif scope == "bizkaia":
        location_clause = "?item wdt:P131 ?municipality . ?municipality wdt:P131* wd:Q93366 ."
        municipality_clause = "OPTIONAL { ?item wdt:P131 ?municipality . }"
        municipality_bind = ""
    else:
        raise ValueError(f"Scope no soportado: {scope}")

    return textwrap.dedent(
        f"""
        SELECT ?item ?itemLabel ?itemDescription ?coord ?typeLabel ?municipalityLabel WHERE {{
          VALUES ?wantedClass {{ {values_clause} }}
          ?item wdt:P31/wdt:P279* ?wantedClass ;
                wdt:P625 ?coord ;
                wdt:P31 ?type .
          FILTER(?item != wd:Q8692)
          FILTER(?item != wd:Q93366)
          {location_clause}
          {municipality_clause}
          {municipality_bind}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "es". }}
        }}
        """
    ).strip()


def should_keep_wikidata_name(name: str) -> bool:
    normalized = normalize_text(name)
    if not normalized:
        return False
    for pattern in WIKIDATA_REJECT_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return False
    return True


def should_keep_wikidata_candidate(name: str, description: str, type_label: str) -> bool:
    if not should_keep_wikidata_name(name):
        return False
    description_norm = normalize_text(description)
    type_norm = normalize_text(type_label)
    noisy_fragments = [
        "tumba destacada",
        "panteon",
        "sepultura",
        "fosa comun",
        "oficina",
        "cementerio",
    ]
    noisy_types = [
        "biblioteca",
        "mediateca",
        "centro de documentacion",
        "vertice geodesico",
        "fosa comun",
        "adoquin conmemorativo",
        "negocio",
        "institucion educativa",
    ]
    if any(fragment in description_norm for fragment in noisy_fragments):
        return False
    if any(fragment in type_norm for fragment in noisy_types):
        return False
    return True


def normalize_municipality_name(name: str) -> str:
    mapping = {
        "Abanto y Ciérvana": "Abanto-Zierbena",
        "Arbácegui y Guerricaiz": "Munitibar-Arbatzegi Gerrikaitz",
        "Baracaldo": "Barakaldo",
        "Ceberio": "Zeberio",
        "Guecho": "Getxo",
        "Guernica y Luno": "Gernika-Lumo",
        "Lequeitio": "Lekeitio",
        "Marquina-Jeméin": "Markina-Xemein",
        "Musques": "Muskiz",
        "Santurce": "Santurtzi",
        "Valle de Carranza": "Karrantza Harana",
        "Valmaseda": "Balmaseda",
        "Vizcaya": "Bizkaia",
    }
    return mapping.get(name, name)


def categorize_wikidata_type(type_label: str, fallback_class: dict) -> Tuple[str, str]:
    normalized = normalize_text(type_label)

    keyword_mapping = [
        (["museo"], ("museo", "museo")),
        (["puente"], ("arquitectura", "puente")),
        (["parque"], ("parque", "parque urbano")),
        (["jardin"], ("parque", "jardín")),
        (["teatro"], ("cultura", "teatro")),
        (["biblioteca"], ("cultura", "biblioteca")),
        (["mercado"], ("gastronomía", "mercado")),
        (["playa"], ("naturaleza", "playa")),
        (["mirador"], ("naturaleza", "mirador")),
        (["estadio"], ("deporte", "estadio")),
        (["escultura"], ("arte", "escultura")),
        (["estatua"], ("arte", "estatua")),
        (["iglesia", "capilla", "ermita", "basilica", "basílica", "catedral", "monasterio"], ("religioso", "templo")),
        (["convento"], ("religioso", "convento")),
        (["dolmen"], ("historia", "dolmen")),
        (["tumulo", "túmulo"], ("historia", "túmulo")),
        (["castillo"], ("historia", "castillo")),
        (["muralla"], ("historia", "muralla")),
        (["sitio arqueologico", "arqueologico"], ("historia", "sitio arqueológico")),
        (["monumento", "memorial"], ("historia", "monumento")),
        (["patrimonio industrial"], ("historia", "patrimonio industrial")),
        (["casa torre"], ("historia", "casa torre")),
        (["faro"], ("arquitectura", "faro")),
        (["palacio"], ("arquitectura", "palacio")),
        (["torre"], ("arquitectura", "torre mirador")),
        (["viaducto"], ("arquitectura", "viaducto")),
        (["fuente"], ("arquitectura", "fuente monumental")),
        (["casa consistorial"], ("arquitectura", "casa consistorial")),
        (["plaza de toros"], ("cultura", "plaza de toros")),
        (["centro de interpretacion"], ("cultura", "centro de interpretación")),
        (["funicular"], ("cultura", "funicular")),
        (["puerta historica", "puerta"], ("historia", "puerta histórica")),
    ]

    for keywords, result in keyword_mapping:
        if any(keyword in normalized for keyword in keywords):
            return result

    return fallback_class["category"], fallback_class["subcategory"]


def batched(items: Sequence[dict], size: int) -> Iterable[Sequence[dict]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def collect_wikidata(dedupe_index: List[dict]) -> Tuple[List[dict], dict]:
    records: List[dict] = []
    downloaded_candidates = 0
    discarded_invalid = 0
    discarded_duplicates = 0
    seen_entities: set[str] = set()

    for class_batch in batched(WIKIDATA_CLASSES, 5):
        for scope in ["bilbao", "bizkaia"]:
            query = wikidata_scope_query_batch([item["qid"] for item in class_batch], scope)
            response = sparql_query_with_retry(query)
            bindings = response.get("results", {}).get("bindings", [])
            downloaded_candidates += len(bindings)

            for binding in bindings:
                entity_url = binding.get("item", {}).get("value", "")
                if not entity_url or entity_url in seen_entities:
                    continue

                name = clean_text(binding.get("itemLabel", {}).get("value"))
                type_label = clean_text(binding.get("typeLabel", {}).get("value"))
                description_hint = clean_text(binding.get("itemDescription", {}).get("value"))
                if not should_keep_wikidata_candidate(name, description_hint, type_label):
                    discarded_invalid += 1
                    continue

                coords = parse_wikidata_point(binding.get("coord", {}).get("value", ""))
                if not coords:
                    discarded_invalid += 1
                    continue

                lat, lon = coords
                municipality = clean_text(binding.get("municipalityLabel", {}).get("value")) or "Bilbao"
                if scope == "bilbao":
                    municipality = "Bilbao"
                if municipality.lower() in {"vizcaya", "bizkaia", "biscay"}:
                    municipality = "Bizkaia"
                municipality = normalize_municipality_name(municipality)

                matching_class = next(
                    (
                        cfg
                        for cfg in class_batch
                        if normalize_text(cfg["label"]) in normalize_text(type_label)
                    ),
                    class_batch[0],
                )
                category, subcategory = categorize_wikidata_type(type_label, matching_class)

                record = make_record(
                    name=name,
                    municipality=municipality,
                    category=category,
                    subcategory=(type_label or subcategory).lower(),
                    lat=lat,
                    lon=lon,
                    address=f"{municipality}, Bizkaia",
                    source="Wikidata",
                    url=entity_url,
                    tags=["Wikidata", matching_class["label"], type_label, municipality],
                    explicit_free=None,
                    extra_description=description_hint,
                )
                if not register_record(record, records, dedupe_index):
                    discarded_duplicates += 1
                    continue

                seen_entities.add(entity_url)

            time.sleep(2.0)

    records.sort(key=lambda poi: (poi["municipality"], poi["category"], normalize_text(poi["name"])))
    summary = {
        "source": "Wikidata",
        "downloaded_candidates": downloaded_candidates,
        "added": len(records),
        "discarded_invalid": discarded_invalid,
        "discarded_duplicates": discarded_duplicates,
    }
    return records, summary


def next_poi_id(start_index: int) -> str:
    width = max(3, len(str(start_index)))
    return f"poi_{start_index:0{width}d}"


def build_corpus() -> Tuple[dict, dict]:
    all_records: List[dict] = []
    dedupe_index = existing_index_from_pois([])

    osm_records, osm_summary = collect_osm_bilbao(dedupe_index)
    all_records.extend(osm_records)

    open_data_records, open_data_summary = collect_open_data_bilbao(dedupe_index)
    all_records.extend(open_data_records)

    wikidata_records, wikidata_summary = collect_wikidata(dedupe_index)
    all_records.extend(wikidata_records)

    all_records.sort(key=lambda poi: (poi["municipality"], poi["category"], normalize_text(poi["name"])))
    for index, record in enumerate(all_records, start=1):
        record["id"] = next_poi_id(index)

    corpus = {
        "version": "2.0",
        "region": "Bilbao / Bizkaia",
        "sources": [
            "OpenStreetMap Overpass API",
            "OpenStreetMap Nominatim",
            "Open Data Euskadi",
            "Wikidata SPARQL",
        ],
        "pois": all_records,
    }

    per_source: Dict[str, int] = {}
    per_municipality: Dict[str, int] = {}
    for poi in all_records:
        per_source[poi["source"]] = per_source.get(poi["source"], 0) + 1
        per_municipality[poi["municipality"]] = per_municipality.get(poi["municipality"], 0) + 1

    summary = {
        "total_pois": len(all_records),
        "bilbao_pois": sum(1 for poi in all_records if poi["municipality"] == "Bilbao"),
        "per_source": dict(sorted(per_source.items())),
        "top_municipalities": dict(sorted(per_municipality.items(), key=lambda item: (-item[1], item[0]))[:20]),
        "collectors": {
            "osm": osm_summary,
            "open_data_euskadi": open_data_summary,
            "wikidata": wikidata_summary,
        },
    }
    return corpus, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenera el corpus completo de POIs de Bilbao / Bizkaia.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Ruta del JSON de salida.")
    parser.add_argument("--dry-run", action="store_true", help="No escribe el fichero; solo muestra el resumen.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output.resolve()
    corpus, summary = build_corpus()

    if not args.dry_run:
        output_path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
