"""
Interfaz Streamlit — Generador de Rutas Turísticas Bilbao / Bizkaia.

Tres páginas:
  1. Generador  — crear rutas turísticas personalizadas
  2. Explorar   — navegar la colección de POIs con búsqueda semántica
  3. Cómo funciona — descripción del pipeline técnico

Conexión: HTTP a la API FastAPI (por defecto http://localhost:8000).
Variable de entorno API_BASE_URL sobreescribe el valor por defecto (útil en Docker).
"""

import os
import time
import json
from datetime import datetime
from typing import Optional

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
def _resolve_api_base() -> str:
    """
    Resuelve la URL base de la API.

    Prioridad:
    1. `API_BASE_URL` si viene por entorno.
    2. Auto-detección local, útil cuando Streamlit se lanza fuera de Docker
       pero la API corre en Docker mapeada al puerto 9000.
    3. Fallback final a localhost:8000.
    """
    env_value = os.environ.get("API_BASE_URL")
    if env_value:
        return env_value.rstrip("/")

    candidates = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:9000",
        "http://127.0.0.1:9000",
    ]

    for candidate in candidates:
        try:
            response = httpx.get(f"{candidate}/api/health", timeout=2.0)
            if response.status_code < 500:
                return candidate
        except Exception:
            continue

    return "http://localhost:8000"


API_BASE = _resolve_api_base()

try:
    API_TIMEOUT_SECONDS = int(os.environ.get("API_TIMEOUT_SECONDS", "1800"))
except Exception:
    API_TIMEOUT_SECONDS = 1800

st.set_page_config(
    page_title="Rutas Bilbao / Bizkaia",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
<style>
:root {
    --bg: #0b0f19;
    --panel: #111827;
    --panel-2: #172033;
    --panel-3: #1d2940;
    --border: #24324d;
    --text: #ecf3ff;
    --muted: #9fb0c9;
    --accent: #4da3ff;
    --accent-2: #7cc4ff;
    --success: #3ddc97;
    --warn: #ffb84d;
    --danger: #ff6b6b;
}
html, body, [class*="css"]  {
    color: var(--text);
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #0a0f18 0%, #0d1320 100%);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101722 0%, #0d1420 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--muted);
}
[data-testid="stSidebar"] [data-testid="stHeading"] {
    color: var(--text);
}
[data-testid="stSidebar"] [data-baseweb="radio"] label,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .st-b7,
[data-testid="stSidebar"] .st-b8 {
    color: var(--text) !important;
}
h1, h2, h3 {
    color: var(--text);
    letter-spacing: -0.02em;
}
p, li, div, span {
    color: inherit;
}
.stButton > button {
    border-radius: 12px;
    border: 1px solid var(--border);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    border: none;
    color: white;
    font-weight: 600;
}
.stButton > button[kind="secondary"] {
    background: var(--panel-2);
    color: var(--text);
}
[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--panel);
}
[data-testid="stExpander"] details summary p {
    color: var(--text);
}
[data-baseweb="select"] > div,
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {
    background: var(--panel) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
.stSlider [data-baseweb="slider"] * {
    color: var(--text) !important;
}
.stAlert {
    background: var(--panel-2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}
[data-baseweb="tag"] {
    background: #1f3b63 !important;
    color: #eaf4ff !important;
    border: 1px solid #31527d !important;
}
.route-hero,
.metric-card {
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.28);
}
.route-hero {
    background: linear-gradient(135deg, #142033 0%, #1f3353 100%);
    color: white;
    padding: 1.4rem 1.6rem;
    border-radius: 18px;
    margin-bottom: 1rem;
    border: 1px solid #243754;
}
.route-hero-title {
    font-size: 1.75rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}
.route-hero-subtitle {
    font-size: 0.96rem;
    opacity: 0.9;
    line-height: 1.5;
}
.summary-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem;
    margin-top: 0.9rem;
}
.summary-chip {
    background: rgba(77, 163, 255, 0.12);
    border: 1px solid rgba(124, 196, 255, 0.18);
    padding: 0.38rem 0.75rem;
    border-radius: 999px;
    font-size: 0.88rem;
    color: #eef6ff;
}
.empty-state {
    background: linear-gradient(135deg, #121b2b 0%, #172233 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 2rem 2.2rem;
    margin-top: 0.5rem;
}
.empty-title {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.4rem;
}
.empty-subtitle {
    font-size: 1rem;
    color: var(--muted);
    max-width: 760px;
    line-height: 1.6;
    margin-bottom: 1.2rem;
}
.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-bottom: 1.4rem;
}
.pill {
    background: var(--panel);
    border: 1px solid var(--border);
    color: #dce9ff;
    border-radius: 999px;
    padding: 0.45rem 0.85rem;
    font-size: 0.92rem;
    font-weight: 500;
}
.empty-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.9rem;
}
.empty-card {
    background: rgba(17, 24, 39, 0.88);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem;
}
.empty-card-title {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--accent-2);
    margin-bottom: 0.35rem;
}
.empty-card-body {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.4;
}
.metric-card {
    background: linear-gradient(135deg, #142033, #1d3353);
    padding: 1rem 1.5rem;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin: 0.3rem 0;
    border: 1px solid #27415f;
}
.metric-value { font-size: 2rem; font-weight: 700; }
.metric-label { font-size: 0.85rem; opacity: 0.85; }
.slot-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.manana { background: #1a2a3f; color: #dce9ff; border: 1px solid #29405f; }
.tarde  { background: #162235; color: #dce9ff; border: 1px solid #29405f; }
.poi-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.2);
    margin-bottom: 0.9rem;
}
.poi-time {
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.9rem 0.75rem;
    text-align: center;
    color: var(--text);
    font-weight: 700;
}
.poi-title {
    font-size: 1.08rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.35rem;
}
.poi-meta {
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.5;
    margin-bottom: 0.45rem;
}
.poi-description {
    color: #d2def1;
    line-height: 1.6;
    margin-bottom: 0.45rem;
}
.mini-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.45rem;
}
.mini-chip {
    background: #17253b;
    border: 1px solid #29405f;
    color: #d6e6ff;
    border-radius: 999px;
    padding: 0.28rem 0.65rem;
    font-size: 0.8rem;
}
.section-card {
    background: linear-gradient(135deg, #111927 0%, #162235 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.2rem 1.3rem;
    margin-bottom: 1rem;
    color: var(--muted);
}
.poi-browser-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.2);
}
.poi-browser-title {
    color: var(--text);
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.poi-browser-desc {
    color: var(--muted);
    line-height: 1.6;
    margin-bottom: 0.5rem;
}
.history-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.2);
    margin-bottom: 0.9rem;
}
.history-title {
    color: var(--text);
    font-weight: 700;
    font-size: 1.02rem;
    margin-bottom: 0.25rem;
}
.history-meta {
    color: var(--muted);
    font-size: 0.92rem;
    margin-bottom: 0.65rem;
}
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--muted) !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state — historial de llamadas
# ---------------------------------------------------------------------------

if "call_history" not in st.session_state:
    st.session_state.call_history = []

if "selected_history_idx" not in st.session_state:
    st.session_state.selected_history_idx = None

if "selected_route_id" not in st.session_state:
    st.session_state.selected_route_id = None


def _add_to_history(query_or_prefs: str, data: dict, exec_time: float):
    """Guarda una llamada exitosa en el historial de la sesión."""
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "query": query_or_prefs[:80] + ("…" if len(query_or_prefs) > 80 else ""),
        "title": data.get("route", {}).get("title", "Ruta generada"),
        "score": data.get("evaluation", {}).get("overall_score", 0),
        "exec_time": exec_time,
        "data": data,
    }
    st.session_state.call_history.insert(0, entry)   # más reciente primero
    st.session_state.selected_history_idx = None


# ---------------------------------------------------------------------------
# Helpers de API
# ---------------------------------------------------------------------------

def _api_get(path: str, params: dict = None) -> Optional[dict]:
    try:
        r = httpx.get(f"{API_BASE}{path}", params=params, timeout=API_TIMEOUT_SECONDS)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error de conexión con la API: {e}")
        return None


def _api_post(path: str, payload: dict) -> Optional[dict]:
    try:
        r = httpx.post(f"{API_BASE}{path}", json=payload, timeout=API_TIMEOUT_SECONDS)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = e.response.text[:300] or str(e)
        st.error(f"Error de la API ({e.response.status_code}): {detail}")
        return None
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return None


# ---------------------------------------------------------------------------
# Componentes de presentación
# ---------------------------------------------------------------------------

def _render_poi_card(pp: dict, show_scores: bool = False):
    try:
        poi = pp["poi"]
        slot = pp.get("slot", "mañana")

        with st.container():
            st.markdown('<div class="poi-card">', unsafe_allow_html=True)
            col_time, col_body = st.columns([1, 5])
            with col_time:
                st.markdown(
                    f'<div class="poi-time">{pp.get("start_time","—")}<br><span style="opacity:.5">↓</span><br>{pp.get("end_time","—")}</div>',
                    unsafe_allow_html=True,
                )
            with col_body:
                st.markdown(
                    f'<div class="poi-title">{poi["name"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="poi-meta">Ubicación: {poi["municipality"]} | Duración: {poi["visit_duration_minutes"]} min | Precio: {poi["price"]} ({poi["price_numeric"]:.0f} €)</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="poi-description">{poi["description"][:220]}…</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="mini-chip-row"><span class="mini-chip">{poi["category"]}</span><span class="mini-chip">{slot.title()}</span></div>',
                    unsafe_allow_html=True,
                )
                if poi.get("address"):
                    st.caption(f"Dirección: {poi['address']}")
                if show_scores:
                    s = pp.get("semantic_score", 0)
                    r = pp.get("rerank_score", 0)
                    f_ = pp.get("final_score", 0)
                    st.caption(f"Semantic: {s:.2f}  |  Rerank: {r:.2f}  |  Score final: {f_:.2f}")
                if pp.get("travel_minutes_from_previous", 0) > 0:
                    st.caption(f"{pp['travel_minutes_from_previous']} min desde el anterior")
            st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Error renderizando POI: {e}")


def _render_browser_poi_card(poi: dict, score: float | None = None):
    try:
        st.markdown('<div class="poi-browser-card">', unsafe_allow_html=True)
        col_main, col_side = st.columns([5, 1])
        with col_main:
            st.markdown(
                f'<div class="poi-browser-title">{poi["name"]} — {poi["category"]} — {poi["municipality"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="poi-browser-desc">{poi["description"][:180]}…</div>',
                unsafe_allow_html=True,
            )
            chip_row = (
                f'<div class="mini-chip-row">'
                f'<span class="mini-chip">Precio: {poi["price"]}</span>'
                f'<span class="mini-chip">Duración: {poi["visit_duration_minutes"]} min</span>'
                f'<span class="mini-chip">{"Accesible" if poi["accessibility"] else "No accesible"}</span>'
                f'</div>'
            )
            st.markdown(chip_row, unsafe_allow_html=True)
        with col_side:
            if score is not None:
                st.metric("Score", f"{score:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Error renderizando POI: {e}")


def _render_evaluation(eval_data: dict):
    try:
        st.markdown("### Evaluación automática de la ruta")
        overall = eval_data["overall_score"]
        overall_color = "#28a745" if overall >= 0.7 else "#ffc107" if overall >= 0.4 else "#dc3545"
        st.markdown(
            f'<div class="metric-card"><div class="metric-value" style="color:{overall_color}">'
            f'{overall:.0%}</div><div class="metric-label">Puntuación global</div></div>',
            unsafe_allow_html=True,
        )

        metrics = [
            ("Cumplimiento de la petición", eval_data["constraint_satisfaction"], "Grado en que la ruta respeta lo pedido"),
            ("Cobertura de intereses",  eval_data["preference_coverage"],    "POIs que encajan con tus intereses"),
            ("Coherencia temporal",     eval_data["temporal_coherence"],     "POIs abiertos en su franja horaria"),
            ("Consistencia geográfica", eval_data["geographic_consistency"], "Compacidad geográfica diaria"),
            ("Ajuste al presupuesto",   eval_data["budget_adherence"],       "Cumplimiento del presupuesto"),
            ("Diversidad",              eval_data["category_diversity"],     "Variedad de categorías"),
            ("Accesibilidad",           eval_data["accessibility_compliance"],"Cumplimiento de accesibilidad"),
        ]

        cols = st.columns(3)
        for i, (label, value, tooltip) in enumerate(metrics):
            with cols[i % 3]:
                bar_color = "#28a745" if value >= 0.7 else "#ffc107" if value >= 0.4 else "#dc3545"
                st.markdown(
                    f"**{label}**  \n<small style='color:#666'>{tooltip}</small>\n\n"
                    f"<span style='font-size:1.4rem;font-weight:700;color:{bar_color}'>{value:.0%}</span>",
                    unsafe_allow_html=True,
                )
                st.progress(value)

        with st.expander("Ver detalles de la evaluación"):
            det = eval_data.get("details", {})
            d_cols = st.columns(4)
            d_cols[0].metric("POIs totales",      det.get("total_pois", "—"))
            d_cols[1].metric("Días planificados",  det.get("days", "—"))
            d_cols[2].metric("Coste total (€)",    f"{det.get('total_cost_eur', 0):.0f} €")
            d_cols[3].metric("Coste medio/día",    f"{det.get('avg_daily_cost_eur', 0):.0f} €")
            if det.get("poi_categories"):
                st.write("**Categorías en la ruta:**", ", ".join(det["poi_categories"]))
            if det.get("constraint_breakdown"):
                st.write("**Desglose de cumplimiento:**")
                st.json(det["constraint_breakdown"])
    except Exception as e:
        st.warning(f"Error renderizando evaluación: {e}")


def _render_retrieval_info(info: dict):
    try:
        # Mostrar trazabilidad mínima (ocultando nombres de modelos y detalles técnicos)
        with st.expander("Trazabilidad del proceso"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Candidatos recuperados", info.get("candidates_retrieved", "—"))
            c2.metric("Tras reranking",         info.get("candidates_after_rerank", "—"))
            c3.metric("Reranker activo",        "Sí" if info.get("reranker_used") else "No")
            st.markdown(
                "Se muestra información agregada del proceso de recuperación y reranking. "
                "Los detalles técnicos y nombres de modelos están ocultos en la interfaz pública."
            )
    except Exception as e:
        st.warning(f"Error renderizando trazabilidad: {e}")


def _try_folium_map(day_pois: list):
    try:
        import folium
        from streamlit_folium import st_folium

        if not day_pois:
            return

        lats = [pp["poi"]["coordinates"]["lat"] for pp in day_pois]
        lons = [pp["poi"]["coordinates"]["lon"] for pp in day_pois]
        center = [sum(lats) / len(lats), sum(lons) / len(lons)]

        m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")
        colors = ["red", "blue", "green", "purple", "orange", "darkred", "lightblue"]

        for i, pp in enumerate(day_pois):
            poi = pp["poi"]
            lat = poi["coordinates"]["lat"]
            lon = poi["coordinates"]["lon"]
            color = colors[i % len(colors)]

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(
                    f"<b>{poi['name']}</b><br>{poi['category']}<br>"
                    f"{pp.get('start_time','')}-{pp.get('end_time','')}",
                    max_width=200,
                ),
                tooltip=f"{i+1}. {poi['name']}",
                icon=folium.Icon(color=color, icon="info-sign"),
            ).add_to(m)

            if i > 0:
                prev = day_pois[i - 1]["poi"]
                folium.PolyLine(
                    locations=[
                        [prev["coordinates"]["lat"], prev["coordinates"]["lon"]],
                        [lat, lon],
                    ],
                    color="#6c757d",
                    weight=2,
                    dash_array="5",
                ).add_to(m)

        st_folium(m, width="100%", height=400)

    except ImportError:
        st.info("Instala `folium` y `streamlit-folium` para ver el mapa interactivo.")
    except Exception as e:
        st.warning(f"Error renderizando mapa: {e}")


def _fetch_all_pois() -> list:
    if "cached_all_pois" not in st.session_state:
        data = _api_get("/api/pois")
        st.session_state.cached_all_pois = data.get("pois", []) if data else []
    return st.session_state.cached_all_pois


_FOLIUM_COLORS = [
    "red", "blue", "green", "purple", "orange",
    "darkred", "darkblue", "cadetblue", "darkgreen", "lightred",
]
_FOLIUM_CSS_COLORS = {
    "red": "#d63e2a", "blue": "#2a81cb", "green": "#2aad27",
    "purple": "#9c2bcb", "orange": "#cb8427", "darkred": "#a23336",
    "darkblue": "#00649f", "cadetblue": "#436978", "darkgreen": "#728224",
    "lightred": "#ff8e7f",
}


def _render_pois_map(pois: list, height: int = 480):
    try:
        import folium
        from streamlit_folium import st_folium

        BILBAO_CENTER = [43.263, -2.935]

        categories = sorted({p.get("category", "Otros") for p in pois if p.get("category")})
        cat_color = {
            cat: _FOLIUM_COLORS[i % len(_FOLIUM_COLORS)]
            for i, cat in enumerate(categories)
        }

        if not pois:
            m = folium.Map(location=BILBAO_CENTER, zoom_start=12, tiles="CartoDB positron")
            st_folium(m, width="100%", height=height, returned_objects=[])
            return

        lats = [p["coordinates"]["lat"] for p in pois if p.get("coordinates")]
        lons = [p["coordinates"]["lon"] for p in pois if p.get("coordinates")]
        center = [sum(lats) / len(lats), sum(lons) / len(lons)] if lats else BILBAO_CENTER

        m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

        for poi in pois:
            coords = poi.get("coordinates")
            if not coords:
                continue
            cat = poi.get("category", "Otros")
            color = cat_color.get(cat, "gray")
            desc = poi.get("description", "")
            popup_html = (
                f"<b>{poi['name']}</b><br>"
                f"<i>{cat} — {poi.get('municipality', '')}</i><br>"
                f"{desc[:120]}{'…' if len(desc) > 120 else ''}<br>"
                f"<small>Precio: {poi.get('price', '—')} | "
                f"{poi.get('visit_duration_minutes', '?')} min</small>"
            )
            folium.Marker(
                location=[coords["lat"], coords["lon"]],
                popup=folium.Popup(popup_html, max_width=230),
                tooltip=poi["name"],
                icon=folium.Icon(color=color, icon="info-sign"),
            ).add_to(m)

        st_folium(m, width="100%", height=height, returned_objects=[])

        chips = "".join(
            f'<span class="mini-chip">'
            f'<span style="color:{_FOLIUM_CSS_COLORS.get(col, col)}">●</span> {cat}'
            f'</span>'
            for cat, col in cat_color.items()
        )
        st.markdown(f'<div class="mini-chip-row">{chips}</div>', unsafe_allow_html=True)

    except ImportError:
        st.info("Instala `folium` y `streamlit-folium` para ver el mapa interactivo.")
    except Exception as e:
        st.warning(f"Error renderizando mapa de POIs: {e}")


# ---------------------------------------------------------------------------
# Renderizado de una ruta completa (reutilizado por generador e historial)
# ---------------------------------------------------------------------------

def _render_route(data: dict, exec_time: float, show_scores: bool = False):
    """Renderiza completamente la respuesta de /api/route."""
    route         = data.get("route", {})
    evaluation    = data.get("evaluation", {})
    retrieval_info = data.get("retrieval_info", {})

    # ── Cabecera ─────────────────────────────────────────────────────────────
    st.success(f"Ruta generada en {exec_time:.1f} s")
    st.markdown(
        f"""
        <div class="route-hero">
            <div class="route-hero-title">{route.get("title", "Ruta turística")}</div>
            <div class="route-hero-subtitle">
                Un itinerario organizado para recorrer Bilbao y Bizkaia con equilibrio entre tiempo, distancia y preferencias.
            </div>
            <div class="summary-strip">
                <div class="summary-chip">{len(route.get("days", []))} días</div>
                <div class="summary-chip">{route.get("total_pois", "—")} paradas</div>
                <div class="summary-chip">{route.get('total_cost_eur', 0):.0f} € estimados</div>
                <div class="summary-chip">Score {evaluation.get('overall_score', 0):.0%}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Días planificados",    len(route.get("days", [])))
    c2.metric("Lugares a visitar",    route.get("total_pois", "—"))
    c3.metric("Coste total estimado", f"{route.get('total_cost_eur', 0):.0f} €")
    c4.metric("Puntuación global",    f"{evaluation.get('overall_score', 0):.0%}")

    # ── Narrativa ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Narrativa del itinerario")
    narrative = route.get("narrative", "")
    if narrative:
        st.markdown(narrative)
    else:
        st.warning("No se generó narrativa (el servicio de generación no respondió a tiempo).")

    # ── Itinerario detallado ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Itinerario detallado")
    for day_data in route.get("days", []):
        try:
            day_num  = day_data.get("day", "?")
            day_pois = day_data.get("pois", [])
            cost     = day_data.get("total_cost_eur", 0)
            vis_min  = day_data.get("total_visit_minutes", 0)

            with st.expander(
                f"Día {day_num}  —  {len(day_pois)} lugares  |  {vis_min} min  |  {cost:.0f} €",
                expanded=(day_num == 1),
            ):
                tab_list, tab_map = st.tabs(["Lista", "Mapa"])
                with tab_list:
                    for pp in day_pois:
                        _render_poi_card(pp, show_scores=show_scores)
                with tab_map:
                    _try_folium_map(day_pois)
        except Exception as e:
            st.warning(f"Error renderizando día {day_data.get('day','?')}: {e}")

    # ── Evaluación ────────────────────────────────────────────────────────────
    st.markdown("---")
    _render_evaluation(evaluation)

    # ── Trazabilidad ──────────────────────────────────────────────────────────
    st.markdown("---")
    _render_retrieval_info(retrieval_info)

    # Nota: la respuesta detallada (JSON) se omite en la interfaz pública.


# ---------------------------------------------------------------------------
# Página: Generador de rutas
# ---------------------------------------------------------------------------

def page_generator():
    st.title("Generador de Rutas Turísticas")
    st.caption("Bilbao y Bizkaia — Sistema RAG híbrido con reranking semántico")

    # ── Sidebar: preferencias + historial ────────────────────────────────────
    with st.sidebar:
        st.header("Preferencias del viaje")

        mode = st.radio(
            "Modo de entrada",
            ["Consulta libre", "Formulario detallado"],
            index=0,
        )

        if mode == "Consulta libre":
            query = st.text_area(
                "Describe tu viaje ideal",
                placeholder="Ej: Quiero pasar 2 días en Bilbao con mi pareja. Nos encantan los museos y la gastronomía.",
                height=120,
            )
            preferences_payload = None
        else:
            query = None
            st.subheader("Dónde")
            scope = st.selectbox("Ámbito geográfico", ["Bilbao", "Bizkaia", "Ambos"])
            days  = st.slider("Número de días", 1, 7, 2)

            st.subheader("Intereses")
            all_interests = [
                "museos", "arte", "arquitectura", "gastronomía", "pintxos",
                "naturaleza", "senderismo", "playa", "surf", "historia",
                "cultura vasca", "deporte", "fotografía", "familia",
                "pueblos costeros"
            ]
            interests = st.multiselect("Selecciona tus intereses", all_interests,
                                       default=["museos", "gastronomía"])
            st.subheader("Viaje")
            group    = st.selectbox("Tipo de grupo", ["pareja", "solo", "familia", "amigos"])
            pace     = st.selectbox("Ritmo del viaje", ["moderado", "tranquilo", "intenso"])
            budget   = st.number_input("Presupuesto por día (€)", min_value=0, max_value=500,
                                       value=50, step=10)
            mobility = st.selectbox("Movilidad", ["normal", "reducida"])

            preferences_payload = {
                "city_scope":    scope,
                "duration_days": days,
                "interests":     interests,
                "budget_per_day": float(budget),
                "pace":          pace,
                "mobility":      mobility,
                "group_type":    group,
            }

        show_scores  = st.checkbox("Mostrar scores de ranking", value=False)
        generate_btn = st.button("Generar Ruta", type="primary", use_container_width=True)

        # ── Historial de llamadas ─────────────────────────────────────────────
        if st.session_state.call_history:
            st.markdown("---")
            st.markdown("### Historial de rutas")
            for i, entry in enumerate(st.session_state.call_history):
                label = f"{entry['timestamp']} — {entry['title'][:35]}"
                if st.button(label, key=f"hist_{i}", use_container_width=True):
                    st.session_state.selected_history_idx = i

    # ── Área principal ────────────────────────────────────────────────────────

    # Si se seleccionó una ruta del historial, mostrarla
    idx = st.session_state.selected_history_idx
    if idx is not None and not generate_btn:
        entry = st.session_state.call_history[idx]
        st.info(f"Mostrando ruta del historial — {entry['timestamp']} — «{entry['query']}»")
        _render_route(entry["data"], entry["exec_time"], show_scores=show_scores)
        return

    if not generate_btn:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-title">Diseña tu próxima ruta por Bilbao y Bizkaia</div>
                <div class="empty-subtitle">
                    Elige una consulta libre para describir tu viaje ideal o usa el formulario
                    para afinar ritmo, presupuesto e intereses con más precisión.
                </div>
                <div class="pill-row">
                    <div class="pill">Museos</div>
                    <div class="pill">Pintxos</div>
                    <div class="pill">Naturaleza</div>
                    <div class="pill">Familia</div>
                    <div class="pill">Pueblos costeros</div>
                </div>
                <div class="empty-grid">
                    <div class="empty-card">
                        <div class="empty-card-title">Consulta libre</div>
                        <div class="empty-card-body">Describe el viaje con tus palabras y genera una propuesta rápida.</div>
                    </div>
                    <div class="empty-card">
                        <div class="empty-card-title">Formulario</div>
                        <div class="empty-card-body">Ajusta días, presupuesto, movilidad e intereses de forma precisa.</div>
                    </div>
                    <div class="empty-card">
                        <div class="empty-card-title">Resultado</div>
                        <div class="empty-card-body">Obtén un itinerario ordenado, mapa diario y evaluación automática.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Llamada a la API (streaming) ─────────────────────────────────────────
    payload: dict = {}
    if query:
        payload["query"] = query
    if preferences_payload:
        payload["preferences"] = preferences_payload

    query_label = query or str(preferences_payload)

    STAGE_LABELS = {
        "interpret":  "Interpretando preferencias",
        "rag":        "Recuperando puntos de interés",
        "rerank":     "Seleccionando candidatos",
        "plan":       "Construyendo itinerario",
        "narrative":  "Generando narrativa",
        "evaluate":   "Evaluando la ruta",
    }
    STAGE_ORDER = list(STAGE_LABELS.keys())

    stage_ph    = st.empty()
    narrative_ph = st.empty()
    error_ph    = st.empty()

    narrative_chunks: list[str] = []
    result_data = None
    current_stage = None
    t0 = time.time()

    try:
        with httpx.stream(
            "POST",
            f"{API_BASE}/api/route/stream",
            json=payload,
            timeout=API_TIMEOUT_SECONDS,
        ) as r:
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except Exception:
                    continue

                etype = event.get("type")

                if etype == "status":
                    current_stage = event.get("stage", "")
                    idx = STAGE_ORDER.index(current_stage) if current_stage in STAGE_ORDER else 0
                    steps_html = " &nbsp;›&nbsp; ".join(
                        f'<span style="color:#4da3ff;font-weight:600">{STAGE_LABELS[s]}</span>'
                        if s == current_stage
                        else f'<span style="color:{"#3ddc97" if STAGE_ORDER.index(s) < idx else "#4a5568"}">{STAGE_LABELS[s]}</span>'
                        for s in STAGE_ORDER
                    )
                    stage_ph.markdown(
                        f'<div style="background:#111827;border:1px solid #24324d;border-radius:12px;padding:.75rem 1rem;font-size:.85rem">'
                        f'⏳ &nbsp;{steps_html}</div>',
                        unsafe_allow_html=True,
                    )

                elif etype == "narrative_chunk":
                    narrative_chunks.append(event["text"])

                elif etype == "result":
                    result_data = event["data"]

                elif etype == "error":
                    error_ph.error(event.get("message", "Error desconocido"))
                    break

    except Exception as e:
        st.error(f"Error de conexión con la API: {e}")
        return

    exec_time = round(time.time() - t0, 2)
    stage_ph.empty()
    narrative_ph.empty()

    if not result_data:
        return

    # Inyectar narrativa streamed (puede que result lleve la misma, pero garantizamos consistencia)
    if narrative_chunks:
        result_data.setdefault("route", {})["narrative"] = "".join(narrative_chunks).strip()

    _add_to_history(query_label, result_data, exec_time)
    _render_route(result_data, exec_time, show_scores=show_scores)


# ---------------------------------------------------------------------------
# Página: Explorar POIs
# ---------------------------------------------------------------------------

def page_explore():
    st.title("Explorar Puntos de Interés")
    st.caption("Busca y filtra el corpus completo de Bilbao y Bizkaia")
    st.markdown(
        """
        <div class="section-card">
            Recorre la colección completa, encuentra ideas por categoría o lanza una búsqueda semántica
            para descubrir lugares afines a un estilo de viaje concreto.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Filtros ───────────────────────────────────────────────────────────────
    all_pois = _fetch_all_pois()
    all_categories    = sorted({p.get("category", "") for p in all_pois if p.get("category")})
    all_municipalities = sorted({p.get("municipality", "") for p in all_pois if p.get("municipality")})

    col_cat, col_mun, col_num = st.columns([3, 3, 1])
    with col_cat:
        cat_filter = st.multiselect("Categorías en el mapa", all_categories, placeholder="Selecciona para ver en el mapa…")
    with col_mun:
        mun_filter = st.multiselect("Municipio", all_municipalities, placeholder="Todos los municipios")
    with col_num:
        map_limit = st.number_input("Máx. en mapa", min_value=1, max_value=500, value=100, step=10)

    filtered_pois = [
        p for p in all_pois
        if (not cat_filter or p.get("category") in cat_filter)
        and (not mun_filter or p.get("municipality") in mun_filter)
    ]

    # Resetear paginación cuando cambian los filtros
    filter_sig = (tuple(cat_filter), tuple(mun_filter))
    if st.session_state.get("explore_filter_sig") != filter_sig:
        st.session_state["explore_filter_sig"] = filter_sig
        st.session_state["explore_page"] = 0

    # ── Mapa global ───────────────────────────────────────────────────────────
    if not cat_filter:
        st.info("Selecciona una o varias categorías para ver los POIs en el mapa.")
        _render_pois_map([])
    elif filtered_pois:
        map_pois = filtered_pois[:map_limit]
        st.markdown(f"**{len(map_pois)}** POIs en el mapa ({len(filtered_pois)} coinciden con los filtros)")
        _render_pois_map(map_pois)
    else:
        st.warning("Sin resultados con estos filtros.")
        _render_pois_map([])

    st.markdown("---")

    # ── Búsqueda semántica ────────────────────────────────────────────────────
    if "explore_search_counter" not in st.session_state:
        st.session_state.explore_search_counter = 0

    col_q, col_btn, col_clear = st.columns([5, 1, 1])
    with col_q:
        query = st.text_input(
            "Búsqueda semántica",
            placeholder="Ej: museos de arte contemporáneo, playas con surf…",
            key=f"explore_q_{st.session_state.explore_search_counter}",
        )
    with col_btn:
        search_btn = st.button("Buscar", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Limpiar", use_container_width=True):
            st.session_state.explore_search_counter += 1
            st.session_state.pop("explore_active_query", None)
            st.rerun()

    if search_btn and query:
        st.session_state.explore_active_query = query

    active_query = st.session_state.get("explore_active_query", "")

    # ── Listado ───────────────────────────────────────────────────────────────
    PAGE_SIZE = 20

    if active_query:
        backend_cat = cat_filter[0] if len(cat_filter) == 1 else None
        backend_mun = mun_filter[0] if len(mun_filter) == 1 else None

        with st.spinner("Buscando…"):
            result = _api_post("/api/pois/search", {
                "query": active_query,
                "k": 20,
                "category_filter":    backend_cat,
                "municipality_filter": backend_mun,
            })

        if result:
            hits = result.get("results", [])
            if len(cat_filter) > 1:
                hits = [h for h in hits if h["poi"].get("category") in cat_filter]
            if len(mun_filter) > 1:
                hits = [h for h in hits if h["poi"].get("municipality") in mun_filter]

            if not hits:
                st.info(f"Sin resultados semánticos para «{active_query}» con los filtros actuales.")
            else:
                st.success(f"**{len(hits)}** resultados para «{active_query}»")
                for item in hits:
                    _render_browser_poi_card(item["poi"], score=item["score"])
    else:
        if not filtered_pois:
            st.info("No hay POIs que mostrar con los filtros seleccionados.")
        else:
            page_key = "explore_page"
            if page_key not in st.session_state:
                st.session_state[page_key] = 0

            total = len(filtered_pois)
            total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
            page = st.session_state[page_key]
            page = max(0, min(page, total_pages - 1))
            st.session_state[page_key] = page

            start = page * PAGE_SIZE
            page_pois = filtered_pois[start : start + PAGE_SIZE]

            st.info(f"**{total}** POIs — página {page + 1} de {total_pages}")
            for poi in page_pois:
                _render_browser_poi_card(poi)

            col_prev, col_info, col_next = st.columns([1, 3, 1])
            with col_prev:
                if st.button("← Anterior", disabled=(page == 0), use_container_width=True):
                    st.session_state[page_key] -= 1
                    st.rerun()
            with col_info:
                st.markdown(
                    f'<div style="text-align:center;color:var(--muted);padding-top:.4rem">'
                    f'Mostrando {start + 1}–{min(start + PAGE_SIZE, total)} de {total}</div>',
                    unsafe_allow_html=True,
                )
            with col_next:
                if st.button("Siguiente →", disabled=(page >= total_pages - 1), use_container_width=True):
                    st.session_state[page_key] += 1
                    st.rerun()


# ---------------------------------------------------------------------------
# Página: Historial de llamadas
# ---------------------------------------------------------------------------

def page_history():
    st.title("Historial de rutas")
    st.caption("Rutas guardadas de forma persistente")

    # Si hay una ruta seleccionada, mostrarla en detalle
    selected_id = st.session_state.get("selected_route_id")
    if selected_id:
        if st.button("← Volver al historial"):
            st.session_state.selected_route_id = None
            st.rerun()
        data = _api_get(f"/api/routes/saved/{selected_id}")
        if data:
            exec_time = data.get("execution_time_seconds", 0)
            _render_route(data, exec_time)
        else:
            st.error("No se pudo cargar la ruta.")
        return

    routes = _api_get("/api/routes/saved") or []

    if not routes:
        st.info("Aún no se ha guardado ninguna ruta.")
        return

    st.markdown(
        f'<div class="section-card">Hay <strong>{len(routes)}</strong> rutas guardadas.</div>',
        unsafe_allow_html=True,
    )

    for entry in routes:
        st.markdown('<div class="history-card">', unsafe_allow_html=True)
        score_pct = f"{entry['score']:.0%}" if entry.get("score") is not None else "—"
        st.markdown(
            f'<div class="history-title">{entry["title"]}</div>'
            f'<div class="history-meta">{entry["created_at"]} | Score: {score_pct} | {entry["exec_time"]:.1f} s</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Consulta: {entry['query']}")

        col_ver, col_del = st.columns([3, 1])
        with col_ver:
            if st.button("Ver ruta completa", key=f"view_{entry['id']}", use_container_width=True):
                st.session_state.selected_route_id = entry["id"]
                st.rerun()
        with col_del:
            if st.button("Eliminar", key=f"del_{entry['id']}", use_container_width=True):
                r = httpx.delete(f"{API_BASE}/api/routes/saved/{entry['id']}", timeout=10)
                if r.status_code == 200:
                    st.rerun()
                else:
                    st.error("No se pudo eliminar.")
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Página: Cómo funciona
# ---------------------------------------------------------------------------

def page_how_it_works():
    st.title("Cómo funciona el sistema")
    st.markdown(
        """
        <div class="section-card">
            El sistema transforma una petición de viaje en un itinerario utilizable combinando
            búsqueda semántica, priorización de candidatos y planificación temporal.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        - Entrada: preferencias libres o formulario estructurado.
        - Recuperación: búsqueda semántica sobre el corpus de POIs.
        - Reranking: priorización de candidatos por relevancia y diversidad.
        - Planificación: asignación temporal y geográfica de POIs.
        - Salida: itinerario detallado con evaluación automática.
        """
    )


# ---------------------------------------------------------------------------
# Navegación principal
# ---------------------------------------------------------------------------

def main():
    with st.sidebar:
        st.title("Rutas Bilbao / Bizkaia")

        page = st.radio(
            "Navegación",
            ["Generador de rutas", "Explorar POIs", "Historial", "Cómo funciona"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        health = _api_get("/api/health")
        if health:
            status_text = "API conectada" if health.get("status") == "ok" else "API disponible (problemas)"
            st.caption(f"{status_text}\n{health.get('index_size',0)} POIs indexados")
            # Mostrar estado simple del reranker (activo/inactivo)
            if "reranker_loaded" in health:
                rer_status = "Activo" if health.get("reranker_loaded") else "Inactivo"
                st.caption(f"Reranker: {rer_status}")
        else:
            st.caption("API no disponible")

    if page == "Generador de rutas":
        page_generator()
    elif page == "Explorar POIs":
        page_explore()
    elif page == "Historial":
        page_history()
    else:
        page_how_it_works()


if __name__ == "__main__":
    main()
