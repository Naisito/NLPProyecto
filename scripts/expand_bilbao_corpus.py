#!/usr/bin/env python3
"""
Amplia el corpus de POIs con nuevos puntos de interés del municipio de Bilbao.

Estrategia:
1. Cargar el JSON base existente.
2. Obtener el polígono administrativo de Bilbao desde Nominatim.
3. Descargar candidatos desde OpenStreetMap vía Overpass dentro del bbox del
   municipio y filtrar únicamente los que caen dentro del polígono real.
4. Normalizar cada elemento al esquema del proyecto.
5. Eliminar duplicados frente al corpus existente y entre candidatos nuevos.
6. Escribir el JSON ampliado en disco.

No usa dependencias externas: sólo librerías estándar de Python.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
import time
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


USER_AGENT = "NLPProyecto Bilbao Corpus Builder/1.0"
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "pois_bilbao_bizkaia.json"
DEFAULT_OUTPUT = DEFAULT_INPUT

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

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

OUTDOOR_CATEGORIES = {"arte", "arquitectura", "historia", "naturaleza", "parque"}

TOP_LEVEL_EXTRA_SOURCES = [
    "OpenStreetMap Overpass API",
    "OpenStreetMap Nominatim",
]


def http_get_json(url: str, params: Dict[str, str], timeout: int = 120) -> object:
    query_url = f"{url}?{urlencode(params)}"
    request = Request(
        query_url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def overpass_query_json(query: str, timeout: int = 240) -> Tuple[dict, str]:
    payload = urlencode({"data": query}).encode("utf-8")
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
                with urlopen(request, timeout=timeout) as response:
                    raw = response.read().decode("utf-8")
                if not raw.lstrip().startswith("{"):
                    raise ValueError(f"Respuesta no JSON recibida desde {endpoint}")
                return json.loads(raw), endpoint
            except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        time.sleep(2 * (round_index + 1))

    raise RuntimeError(f"No se pudo consultar Overpass. Último error: {last_error}")


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def sentence_case(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:]


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
        "marketplace": "mercado",
        "memorial": "memorial",
        "monument": "monumento",
        "museum": "museo",
        "park": "parque",
        "place_of_worship": "espacio religioso",
        "ruins": "ruinas",
        "sculpture": "escultura",
        "statue": "estatua",
        "theatre": "teatro",
        "viewpoint": "mirador",
    }
    cleaned = clean_text(value)
    return translations.get(cleaned, cleaned)


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


def kind_label(tags: dict) -> str:
    tourism = tags.get("tourism")
    amenity = tags.get("amenity")
    leisure = tags.get("leisure")
    historic = tags.get("historic")
    man_made = tags.get("man_made")
    natural = tags.get("natural")

    translations = {
        "museum": "museo",
        "gallery": "galería",
        "artwork": "obra de arte",
        "viewpoint": "mirador",
        "attraction": "atracción",
        "theatre": "teatro",
        "arts_centre": "centro cultural",
        "cinema": "cine",
        "marketplace": "mercado",
        "place_of_worship": "espacio religioso",
        "library": "biblioteca",
        "park": "parque",
        "garden": "jardín",
        "sports_centre": "centro deportivo",
        "bridge": "puente",
        "monument": "monumento",
        "memorial": "memorial",
        "castle": "fortificación histórica",
        "archaeological_site": "yacimiento arqueológico",
        "ruins": "ruinas",
        "building": "edificio histórico",
        "peak": "cima",
    }

    for value in [tourism, amenity, leisure, man_made, historic, natural]:
        if value and value in translations:
            return translations[value]
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

    if tourism == "viewpoint":
        return "naturaleza", "mirador urbano"

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
        if "mezquita" in lower_name or tags.get("religion") == "muslim":
            return "religioso", "mezquita"
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

    if natural == "peak":
        return "naturaleza", "cima"

    if tourism == "attraction":
        return "cultura", "atracción urbana"

    return "cultura", kind_label(tags)


def infer_schedule(tags: dict, category: str) -> Dict[str, Optional[dict]]:
    parsed = parse_opening_hours(tags.get("opening_hours", ""))
    if parsed:
        return parsed

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


def infer_price(tags: dict, category: str) -> Tuple[str, float]:
    fee = clean_text(tags.get("fee")).lower()
    if fee in {"no", "free"}:
        return "gratis", 0.0
    if fee in {"yes", "paid"}:
        if category == "museo":
            return "€", 8.0
        if category == "cultura":
            return "€", 6.0
        return "€", 5.0

    if category in {"parque", "arquitectura", "historia", "naturaleza", "arte", "religioso", "gastronomía"}:
        return "gratis", 0.0
    if category == "museo":
        return "€", 8.0
    if category == "cultura":
        return "€", 6.0
    if category == "deporte":
        return "€", 5.0
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


def infer_accessibility(tags: dict, category: str, subcategory: str) -> bool:
    wheelchair = clean_text(tags.get("wheelchair")).lower()
    if wheelchair in {"yes", "designated", "limited"}:
        return True
    if wheelchair == "no":
        return False

    if category in {"museo", "cultura", "gastronomía", "parque", "arte", "arquitectura"}:
        return True
    if category == "naturaleza" and ("cima" in subcategory or "mirador" in subcategory):
        return False
    if category in {"historia", "religioso"}:
        return False
    return True


def build_address(tags: dict) -> str:
    street = clean_text(tags.get("addr:street"))
    number = clean_text(tags.get("addr:housenumber"))
    postcode = clean_text(tags.get("addr:postcode"))
    city = clean_text(tags.get("addr:city")) or "Bilbao"
    suburb = (
        clean_text(tags.get("addr:suburb"))
        or clean_text(tags.get("addr:neighbourhood"))
        or clean_text(tags.get("addr:quarter"))
        or clean_text(tags.get("is_in:suburb"))
    )

    line_1 = " ".join(part for part in [street, number] if part).strip()
    pieces = [line_1, suburb, " ".join(part for part in [postcode, city] if part).strip(), "Bizkaia"]
    compact = [piece for piece in pieces if piece]
    return ", ".join(compact) if compact else "Bilbao, Bizkaia"


def build_url(element: dict) -> str:
    tags = element.get("tags", {})
    for key in ["website", "contact:website", "url"]:
        value = clean_text(tags.get(key))
        if not value:
            continue
        if value.startswith("http://") or value.startswith("https://"):
            return value
        return f"https://{value}"

    wikipedia = clean_text(tags.get("wikipedia"))
    if wikipedia and ":" in wikipedia:
        lang, title = wikipedia.split(":", 1)
        return f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"

    element_type = element.get("type", "node")
    element_id = element.get("id")
    return f"https://www.openstreetmap.org/{element_type}/{element_id}"


def translated_detail(tags: dict) -> List[str]:
    details: List[str] = []

    artist = clean_text(tags.get("artist_name"))
    if artist:
        details.append(f"vinculada a {artist}")

    architect = clean_text(tags.get("architect"))
    if architect:
        details.append(f"con referencia al arquitecto {architect}")

    start_date = clean_text(tags.get("start_date"))
    if start_date:
        details.append(f"documentada en OpenStreetMap con fecha {start_date}")

    artwork_type = clean_text(tags.get("artwork_type"))
    if artwork_type:
        artwork_map = {
            "statue": "escultura o estatua",
            "bust": "busto",
            "mural": "mural",
            "installation": "instalación",
            "sculpture": "escultura",
        }
        details.append(f"tipificada como {artwork_map.get(artwork_type, translate_osm_value(artwork_type))}")

    description = clean_text(tags.get("description"))
    if description:
        details.append(description[:180])

    return details


def compose_semantic_tags(tags: dict, category: str, subcategory: str, accessibility: bool) -> List[str]:
    values = [
        category,
        subcategory,
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
        "accesible" if accessibility else "accesibilidad limitada",
        "gratuito" if clean_text(tags.get("fee")).lower() in {"", "no", "free"} else "de pago",
    ]

    cleaned = []
    seen = set()
    for value in values:
        if not value:
            continue
        if len(value) <= 1:
            continue
        if value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned


def build_description(name: str, category: str, subcategory: str, address: str, tags: dict) -> str:
    context_by_category = {
        "museo": "Refuerza la cobertura museística del corpus de Bilbao para consultas de arte, historia y patrimonio.",
        "arte": "Aporta cobertura de arte público y escultura urbana dentro del tejido cultural bilbaíno.",
        "parque": "Se incorpora como espacio verde y de paseo para rutas urbanas de descanso y naturaleza.",
        "naturaleza": "Añade opciones de miradores y recorridos abiertos dentro del municipio.",
        "gastronomía": "Amplía la cobertura de mercados y puntos de vida local vinculados a la experiencia gastronómica.",
        "religioso": "Se incluye por su interés patrimonial y religioso dentro de Bilbao.",
        "cultura": "Amplía la cobertura de equipamientos culturales y de ocio del municipio.",
        "historia": "Se añade como punto de interés histórico o memorial dentro de Bilbao.",
        "arquitectura": "Refuerza la cobertura de hitos urbanos y arquitectura reconocible del municipio.",
        "deporte": "Amplía la cobertura de espacios deportivos visitables en Bilbao.",
    }

    detail_fragments = translated_detail(tags)
    parts = [
        f"{name} es un punto de interés de tipo {subcategory} en Bilbao.",
        context_by_category.get(category, "Se incorpora al corpus como punto de interés adicional de Bilbao."),
    ]
    if detail_fragments:
        parts.append(sentence_case(" ".join(detail_fragments)) + ".")
    parts.append(f"Se localiza en {address}.")
    return clean_text(" ".join(parts))


def build_enriched_text(
    name: str,
    category: str,
    subcategory: str,
    description: str,
    address: str,
    semantic_tags: Sequence[str],
) -> str:
    tags_block = ", ".join(semantic_tags)
    chunks = [
        name,
        "Bilbao",
        f"Categoría {category}",
        f"Subcategoría {subcategory}",
        description,
        address,
        tags_block,
    ]
    return ". ".join(chunk for chunk in chunks if chunk)


def build_overpass_query(bbox: Sequence[float]) -> str:
    south, north, west, east = bbox
    return textwrap.dedent(
        f"""
        [out:json][timeout:180];
        (
          nwr["tourism"~"^(museum|gallery|attraction|artwork|viewpoint)$"]({south},{west},{north},{east});
          nwr["historic"~"^(monument|memorial|castle|archaeological_site|ruins|building)$"]({south},{west},{north},{east});
          nwr["amenity"~"^(theatre|arts_centre|cinema|marketplace|place_of_worship)$"]({south},{west},{north},{east});
          nwr["leisure"~"^(park|garden)$"]({south},{west},{north},{east});
          nwr["man_made"="bridge"]({south},{west},{north},{east});
        );
        out center tags;
        """
    ).strip()


def fetch_bilbao_boundary() -> dict:
    response = http_get_json(
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


def candidate_coordinates(element: dict) -> Optional[Tuple[float, float]]:
    lat = element.get("lat")
    lon = element.get("lon")
    if lat is not None and lon is not None:
        return float(lat), float(lon)

    center = element.get("center", {})
    if "lat" in center and "lon" in center:
        return float(center["lat"]), float(center["lon"])
    return None


def should_keep_candidate(tags: dict) -> bool:
    if not clean_text(tags.get("name")):
        return False
    if clean_text(tags.get("tourism")).lower() in LODGING_TAGS:
        return False
    if clean_text(tags.get("access")).lower() == "private":
        return False
    if any(key in tags for key in ["disused", "abandoned"]):
        return False
    return True


def overlap_ratio(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / max(1, min(len(tokens_a), len(tokens_b)))


def is_duplicate(
    name: str,
    category: str,
    lat: float,
    lon: float,
    existing_index: Sequence[dict],
) -> bool:
    normalized = normalize_text(name)
    tokens = significant_tokens(name)

    for item in existing_index:
        if normalized == item["normalized_name"]:
            return True

        distance = haversine_m(lat, lon, item["lat"], item["lon"])
        if distance > 300:
            continue

        if category != item["category"]:
            continue

        if normalized in item["normalized_name"] or item["normalized_name"] in normalized:
            return True

        if overlap_ratio(tokens, item["tokens"]) >= 0.75:
            return True

    return False


def existing_index_from_pois(pois: Sequence[dict]) -> List[dict]:
    index = []
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


def next_poi_id(start_index: int) -> str:
    width = max(3, len(str(start_index)))
    return f"poi_{start_index:0{width}d}"


def build_poi_record(element: dict) -> dict:
    tags = element["tags"]
    name = clean_text(tags["name"])
    coordinates = candidate_coordinates(element)
    if coordinates is None:
        raise ValueError(f"Elemento sin coordenadas válidas: {name}")

    lat, lon = coordinates
    category, subcategory = infer_category_and_subcategory(name, tags)
    address = build_address(tags)
    schedule = infer_schedule(tags, category)
    price, price_numeric = infer_price(tags, category)
    accessibility = infer_accessibility(tags, category, subcategory)
    semantic_tags = compose_semantic_tags(tags, category, subcategory, accessibility)
    description = build_description(name, category, subcategory, address, tags)
    enriched_text = build_enriched_text(name, category, subcategory, description, address, semantic_tags)

    return {
        "id": "",
        "name": name,
        "municipality": "Bilbao",
        "category": category,
        "subcategory": subcategory,
        "description": description,
        "coordinates": {"lat": round(lat, 6), "lon": round(lon, 6)},
        "address": address,
        "price": price,
        "price_numeric": price_numeric,
        "schedule": schedule,
        "source": "OpenStreetMap Overpass",
        "url": build_url(element),
        "tags": semantic_tags,
        "enriched_text": enriched_text,
        "visit_duration_minutes": infer_visit_duration(category, subcategory),
        "accessibility": accessibility,
    }


def expand_corpus(data: dict) -> Tuple[dict, dict]:
    base_pois = list(data.get("pois", []))
    bilbao_boundary = fetch_bilbao_boundary()
    bbox = [float(value) for value in bilbao_boundary["boundingbox"]]
    overpass_query = build_overpass_query(bbox)
    overpass_data, endpoint = overpass_query_json(overpass_query)

    geojson = bilbao_boundary["geojson"]
    existing_bilbao_pois = [poi for poi in base_pois if poi.get("municipality") == "Bilbao"]
    dedupe_index = existing_index_from_pois(existing_bilbao_pois)

    new_records: List[dict] = []
    discarded_outside = 0
    discarded_duplicates = 0
    discarded_filtered = 0

    for element in overpass_data.get("elements", []):
        tags = element.get("tags", {})
        if not should_keep_candidate(tags):
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

        record = build_poi_record(element)
        if is_duplicate(
            record["name"],
            record["category"],
            record["coordinates"]["lat"],
            record["coordinates"]["lon"],
            dedupe_index,
        ):
            discarded_duplicates += 1
            continue

        new_records.append(record)
        dedupe_index.append(
            {
                "normalized_name": normalize_text(record["name"]),
                "tokens": significant_tokens(record["name"]),
                "lat": record["coordinates"]["lat"],
                "lon": record["coordinates"]["lon"],
                "category": record["category"],
            }
        )

    new_records.sort(key=lambda poi: (poi["category"], normalize_text(poi["name"])))
    next_index = len(base_pois) + 1
    for record in new_records:
        record["id"] = next_poi_id(next_index)
        next_index += 1

    expanded = deepcopy(data)
    expanded["version"] = "1.1"
    expanded["sources"] = list(dict.fromkeys(list(data.get("sources", [])) + TOP_LEVEL_EXTRA_SOURCES))
    expanded["pois"] = base_pois + new_records

    summary = {
        "base_total": len(base_pois),
        "base_bilbao": len(existing_bilbao_pois),
        "downloaded_candidates": len(overpass_data.get("elements", [])),
        "added_new_pois": len(new_records),
        "discarded_outside_polygon": discarded_outside,
        "discarded_duplicates": discarded_duplicates,
        "discarded_filtered": discarded_filtered,
        "final_total": len(expanded["pois"]),
        "final_bilbao": len([poi for poi in expanded["pois"] if poi.get("municipality") == "Bilbao"]),
        "overpass_endpoint": endpoint,
    }
    return expanded, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Amplía el corpus con más POIs de Bilbao.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Ruta al JSON de entrada.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Ruta al JSON de salida.")
    parser.add_argument("--dry-run", action="store_true", help="No escribe el fichero; solo muestra el resumen.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    data = json.loads(input_path.read_text(encoding="utf-8"))
    expanded, summary = expand_corpus(data)

    if not args.dry_run:
        output_path.write_text(
            json.dumps(expanded, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
