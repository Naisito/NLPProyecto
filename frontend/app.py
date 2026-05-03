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
    API_TIMEOUT_SECONDS = int(os.environ.get("API_TIMEOUT_SECONDS", "900"))
except Exception:
    API_TIMEOUT_SECONDS = 900

st.set_page_config(
    page_title="Rutas Bilbao / Bizkaia",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
    padding: 1rem 1.5rem;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin: 0.3rem 0;
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
.manana { background: #fff3cd; color: #856404; }
.tarde  { background: #cce5ff; color: #004085; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state — historial de llamadas
# ---------------------------------------------------------------------------

if "call_history" not in st.session_state:
    st.session_state.call_history = []   # lista de dicts con metadatos + respuesta completa

if "selected_history_idx" not in st.session_state:
    st.session_state.selected_history_idx = None


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
        detail = e.response.json().get("detail", str(e))
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
        badge_cls = "manana" if "ma" in slot else "tarde"

        with st.container():
            col_time, col_body = st.columns([1, 5])
            with col_time:
                st.markdown(f"**{pp.get('start_time','—')}**\n\n↓\n\n**{pp.get('end_time','—')}**")
                st.markdown(
                    f'<span style="background:#17a2b8;color:white;padding:2px 8px;'
                    f'border-radius:10px;font-size:0.8rem;">{poi["category"]}</span> '
                    f'Ubicación: {poi["municipality"]} &nbsp;|&nbsp; '
                    f'Duración: {poi["visit_duration_minutes"]} min &nbsp;|&nbsp; '
                    f'Precio: {poi["price"]} ({poi["price_numeric"]:.0f} €)',
                    unsafe_allow_html=True,
                )
                st.markdown(f'*{poi["description"][:220]}…*')
                if poi.get("address"):
                    st.caption(f"Dirección: {poi['address']}")
                if show_scores:
                    s = pp.get("semantic_score", 0)
                    r = pp.get("rerank_score", 0)
                    f_ = pp.get("final_score", 0)
                    st.caption(f"Semantic: {s:.2f}  |  Rerank: {r:.2f}  |  Score final: {f_:.2f}")
                if pp.get("travel_minutes_from_previous", 0) > 0:
                    st.caption(f"{pp['travel_minutes_from_previous']} min desde el anterior")
            st.divider()
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
    st.title(route.get("title", "Ruta turística"))

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
        st.markdown("""
        ## Bienvenido al Generador de Rutas Turísticas

        Este sistema usa **RAG híbrido** para crear itinerarios personalizados
        para Bilbao y Bizkaia.

        ### Pipeline técnico:
        | Paso | Componente | Tecnología |
        |------|-----------|------------|
        | 1 | Interpretación de preferencias | Ollama (local) |
        | 2 | Recuperación semántica | BAAI/bge-m3 + ChromaDB |
        | 3 | Reranking de candidatos | cross-encoder/ms-marco-multilingual |
        | 4 | Planificación del itinerario | Algoritmo greedy NN + slots horarios |
        | 5 | Generación narrativa | Ollama (local) |
        | 6 | Evaluación automática | Métricas objetivas |

        ### Cómo empezar:
        - **Consulta libre**: escribe en lenguaje natural tu viaje ideal.
        - **Formulario**: ajusta los parámetros con precisión.
        """)
        return

    # ── Llamada a la API ──────────────────────────────────────────────────────
    payload: dict = {}
    if query:
        payload["query"] = query
    if preferences_payload:
        payload["preferences"] = preferences_payload

    query_label = query or str(preferences_payload)

    with st.spinner("Generando tu ruta… (puede tardar 1-2 min en CPU)"):
        t0   = time.time()
        data = _api_post("/api/route", payload)
        exec_time = round(time.time() - t0, 2)

    if not data:
        return

    # Guardar en historial
    _add_to_history(query_label, data, exec_time)

    # Renderizar resultado
    _render_route(data, exec_time, show_scores=show_scores)


# ---------------------------------------------------------------------------
# Página: Explorar POIs
# ---------------------------------------------------------------------------

def page_explore():
    st.title("Explorar Puntos de Interés")
    st.caption("Busca y filtra el corpus completo de Bilbao y Bizkaia")

    col_search, col_filters = st.columns([3, 2])
    with col_search:
        query = st.text_input(
            "Búsqueda semántica",
            placeholder="Ej: museos de arte contemporáneo, playas con surf…",
        )
    with col_filters:
        stats = _api_get("/api/stats") or {}
        categories    = ["Todos"] + stats.get("categories", [])
        municipalities = ["Todos"] + stats.get("municipalities", [])
        cat_filter = st.selectbox("Categoría", categories)
        mun_filter = st.selectbox("Municipio", municipalities)

    if query:
        with st.spinner("Buscando…"):
            result = _api_post("/api/pois/search", {
                "query": query, "k": 15,
                "category_filter":    None if cat_filter == "Todos" else cat_filter,
                "municipality_filter": None if mun_filter == "Todos" else mun_filter,
            })
        if result:
            st.success(f"**{result['total']}** resultados para «{query}»")
            for item in result.get("results", []):
                poi   = item["poi"]
                score = item["score"]
                with st.container():
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        st.markdown(f"**{poi['name']}** — *{poi['category']}* — {poi['municipality']}")
                        st.markdown(f"{poi['description'][:180]}…")
                        st.caption(
                            f"Precio: {poi['price']} | Duración: {poi['visit_duration_minutes']} min "
                            f"| {'accesible' if poi['accessibility'] else 'no accesible'}"
                        )
                    with c2:
                        st.metric("Score", f"{score:.2f}")
                    st.divider()
    else:
        params = {}
        if cat_filter != "Todos":
            params["category"] = cat_filter
        if mun_filter != "Todos":
            params["municipality"] = mun_filter
        result = _api_get("/api/pois", params=params)
        if result:
            st.info(f"**{result['total']}** POIs en la colección.")
            for poi in result.get("pois", []):
                with st.container():
                    st.markdown(f"**{poi['name']}** — *{poi['category']}* — {poi['municipality']}")
                    st.markdown(f"{poi['description'][:160]}…")
                    st.caption(
                        f"Precio: {poi['price']} | Duración: {poi['visit_duration_minutes']} min "
                        f"| {'accesible' if poi['accessibility'] else 'no accesible'}"
                    )
                    st.divider()


# ---------------------------------------------------------------------------
# Página: Historial de llamadas
# ---------------------------------------------------------------------------

def page_history():
    st.title("Historial de llamadas")
    st.caption("Rutas generadas en esta sesión")

    if not st.session_state.call_history:
        st.info("Aún no se ha generado ninguna ruta en esta sesión.")
        return

    st.markdown(f"**{len(st.session_state.call_history)}** rutas generadas en esta sesión.")

    for i, entry in enumerate(st.session_state.call_history):
        score_color = "green" if entry["score"] >= 0.7 else "orange" if entry["score"] >= 0.4 else "red"
        with st.expander(
            f"[{entry['timestamp']}] {entry['title']}  —  "
            f"Score: {entry['score']:.0%}  |  {entry['exec_time']:.1f} s",
            expanded=(i == 0),
        ):
            st.caption(f"**Consulta:** {entry['query']}")

            col_ver, col_json = st.columns([1, 1])
            with col_ver:
                if st.button("Ver ruta completa", key=f"view_{i}"):
                    st.session_state.selected_history_idx = i
                    st.rerun()
            with col_json:
                with st.expander("JSON completo"):
                    st.json(entry["data"])

    if st.button("Limpiar historial"):
        st.session_state.call_history = []
        st.session_state.selected_history_idx = None
        st.rerun()


# ---------------------------------------------------------------------------
# Página: Cómo funciona
# ---------------------------------------------------------------------------

def page_how_it_works():
    st.title("Cómo funciona el sistema")
    st.markdown("""
    El sistema combina recuperación de información, reranking y planificación
    para generar itinerarios personalizados. La interfaz pública muestra
    resultados y métricas agregadas; los detalles internos y nombres de
    modelos no se exponen.

    - Entrada: preferencias libres o formulario estructurado.
    - Recuperación: búsqueda semántica sobre el corpus de POIs.
    - Reranking: priorización de candidatos por relevancia y diversidad.
    - Planificación: asignación temporal y geográfica de POIs.
    - Salida: itinerario detallado + evaluación objetiva.
    """)


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
            st.caption(f"{status_text}  \\n  {health.get('index_size',0)} POIs indexados")
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
