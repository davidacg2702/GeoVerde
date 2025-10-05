# app.py
import json
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import ee
import plotly.graph_objects as go
import numpy as np

from gee_utils import (
    init_ee, aoi_from_geojson, get_s2_collection, get_modis_ndvi, get_imerg_precip
)
from bloom_farm_pipeline import (
    timeseries_ndvi_s2, timeseries_evi_modis, timeseries_precip,
    timeseries_landsat_ndvi_5yr, detect_peaks, detect_greening_phases,
    anomalies_against_recent_climatology, compute_phenology_metrics,
    forecast_next_months, align_index_precip
)

# ---------------------------- util -------------------------------------
def _fmt_date(s: str) -> str:
    s = (s or "").strip().replace("/", "-")
    dt = pd.to_datetime(s, errors="raise", dayfirst=False)
    return dt.strftime("%Y-%m-%d")

def _plot_with_peaks(df, value_col, smooth_col=None, peaks=None, y_title="Valor", title=None):
    fig = go.Figure()
    x_dt = df["date"]
    if hasattr(x_dt, "dt"):
        x_dt = x_dt.dt.tz_localize(None).dt.to_pydatetime()
    fig.add_trace(go.Scatter(x=x_dt, y=df[value_col], mode="lines", name=value_col, line=dict(width=2)))
    if smooth_col and smooth_col in df.columns:
        fig.add_trace(go.Scatter(x=x_dt, y=df[smooth_col], mode="lines", name=smooth_col,
                                 line=dict(width=2, dash="dot")))
    if peaks:
        pxs = [p["date"] for p in peaks]
        pys = [p["value"] for p in peaks]
        fig.add_trace(go.Scatter(x=pxs, y=pys, mode="markers+text", name="Pico",
                                 marker=dict(symbol="star", size=14),
                                 text=["🌸"] * len(pxs), textposition="top center"))
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=50), xaxis_title="Fecha", yaxis_title=y_title,
                      title=dict(text=title or "", x=0.5, xanchor="center"))
    fig.update_xaxes(type="date", tickformat="%Y-%m", tickangle=-45)
    return fig

# ----------------------- cachés de datos (15 min) ----------------------
@st.cache_data(ttl=900)
def cached_timeseries_ndvi_s2(geojson_obj, start, end, agg, index_name, worldcover_classes):
    aoi = aoi_from_geojson(geojson_obj)
    return timeseries_ndvi_s2(aoi, start, end, agg=agg, index_name=index_name, worldcover_classes=worldcover_classes)

@st.cache_data(ttl=900)
def cached_timeseries_evi_modis(geojson_obj, start, end, agg):
    aoi = aoi_from_geojson(geojson_obj)
    return timeseries_evi_modis(aoi, start, end, agg=agg)

@st.cache_data(ttl=900)
def cached_timeseries_precip(geojson_obj, start, end, agg):
    aoi = aoi_from_geojson(geojson_obj)
    return timeseries_precip(aoi, start, end, agg=agg)

@st.cache_data(ttl=900)
def cached_timeseries_landsat_5y(geojson_obj, start, end):
    aoi = aoi_from_geojson(geojson_obj)
    return timeseries_landsat_ndvi_5yr(aoi, start, end, agg="mensual")

# -----------------------------------------------------------------------
st.set_page_config(page_title="GeoVerde – Fenología de Floración", layout="wide")
st.title("GeoVerde – Explorador de la Fenología de la Floración")
st.caption("NASA Space Apps – Monitoreo y visualización de eventos de floración con datos de observación de la Tierra")

# --- Scroll horizontal para la barra de pestañas (móvil/mediano) ---
st.markdown("""
<style>
/* Hace que la barra de tabs permita desplazamiento horizontal */
.stTabs [role="tablist"]{
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    overflow-y: hidden;
    scrollbar-width: thin;
}
.stTabs [role="tab"]{
    flex: 0 0 auto !important;  /* evita que se achiquen y se amontonen */
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.header("Parámetros")
    start_in = st.text_input("Fecha inicio", "2024-01-01")
    end_in   = st.text_input("Fecha fin",   "2025-09-25")
    agg      = st.selectbox("Agregación temporal", ["mensual", "quincenal", "diario"], index=0)
    crop     = st.selectbox("Cultivo", ["mango", "banano", "arroz", "otro"], index=0)

    st.markdown("### Detección de floración")
    min_value = st.slider("Umbral NDVI/EVI para floración", 0.10, 0.80, 0.30, 0.01)
    n_peaks   = st.slider("Número de picos a marcar", 1, 3, 1)
    sep_days  = st.slider("Separación mínima entre picos (días)", 7, 90, 30, 1)

    # umbrales sugeridos por cultivo
    if crop == "mango":
        min_value = max(min_value, 0.30)
    elif crop == "banano":
        min_value = max(min_value, 0.28)
    elif crop == "arroz":
        min_value = max(min_value, 0.35)

    st.markdown("---")
    s2_index = st.selectbox("Índice (Sentinel-2)", ["NDVI", "EVI2", "NDRE", "NIRv"], index=0)

    # Filtro WorldCover
    st.markdown("### Filtro por cobertura (WorldCover 10 m)")
    worldcover_labels = {
        10: "Árboles", 20: "Arbustos", 30: "Pastizal",
        40: "Cultivos", 50: "Construido", 60: "Suelo desnudo",
        70: "Nieve/hielo", 80: "Agua", 90: "Humedales", 95: "Manglar"
    }
    wc_sel = st.multiselect(
        "Incluir solo estas clases (opcional)",
        list(worldcover_labels.values()),
        default=["Cultivos", "Pastizal"]
    )
    wc_codes = [k for k, v in worldcover_labels.items() if v in wc_sel]

    st.markdown("### Selección del área de interés (AOI)")

aoi_option = st.radio(
    "Selecciona cómo definir tu AOI:",
    ["Ejemplo (Tambogrande, Piura)", "Subir archivo GeoJSON", "Ingresar coordenadas manualmente"]
)

if aoi_option == "Ejemplo (Tambogrande, Piura)":
    # AOI de demostración (Tambogrande, Piura, Perú)
    geojson_obj = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "Tambogrande"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-80.516, -5.016],
                    [-80.316, -5.016],
                    [-80.316, -4.866],
                    [-80.516, -4.866],
                    [-80.516, -5.016]
                ]]
            }
        }]
    }

elif aoi_option == "Subir archivo GeoJSON":
    uploaded = st.file_uploader("Sube tu AOI .geojson o .json", type=["geojson", "json"])
    if uploaded:
        geojson_obj = json.load(uploaded)
    else:
        st.stop()

elif aoi_option == "Ingresar coordenadas manualmente":
    coords_text = st.text_area(
        "Pega tus coordenadas (formato JSON o lista [[lon,lat], [lon,lat], ...]):",
        "[[-80.5, -5.0], [-80.3, -5.0], [-80.3, -4.9], [-80.5, -4.9], [-80.5, -5.0]]"
    )
    try:
        coords = json.loads(coords_text)
        geojson_obj = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"name": "AOI manual"},
                "geometry": {"type": "Polygon", "coordinates": [coords]}
            }]
        }
    except Exception as e:
        st.error(f"Error al procesar coordenadas: {e}")
        st.stop()

    st.caption("Si no subes AOI, se usa un polígono de ejemplo en Tambogrande-Piura, Perú.")

# Fechas seguras
try:
    start = _fmt_date(start_in)
    end   = _fmt_date(end_in)
    if pd.to_datetime(start) > pd.to_datetime(end):
        st.warning("La fecha inicial es mayor que la final. Intercambiándolas.")
        start, end = end, start
except Exception as e:
    st.error(f"Fecha inválida: {e}. Usa 'YYYY-MM-DD'."); st.stop()

# AOI por defecto
default_geojson = {
  "type": "FeatureCollection",
  "features": [{
      "type": "Feature",
      "properties": {"name": "Tambogrande AOI demo"},
      "geometry": {"type": "Polygon", "coordinates": [[
          [-80.45, -4.98], [-80.28, -4.98], [-80.28, -4.85], [-80.45, -4.85], [-80.45, -4.98]
      ]]}
  }]
}
geojson_obj = json.load(uploaded) if uploaded else default_geojson

# GEE
EE_OK = True
try:
    mode = init_ee(debug=False)
    st.caption(f"Earth Engine inicializado (modo: {mode})")
except Exception as e:
    EE_OK = False
    st.error(f"No se pudo inicializar Google Earth Engine: {e}")


# --------------------------- Mapa --------------------------------------
st.subheader("Mapa – Contexto fenológico (NDVI mediana)")
m = folium.Map(location=[-5.0, -80.6], zoom_start=9, tiles="OpenStreetMap")
folium.GeoJson(geojson_obj, name="AOI",
               style_function=lambda x: {"fillOpacity": 0.05, "color": "red"}).add_to(m)

if EE_OK:
    try:
        # Preview rápido con NDVI S2 mediana
        aoi = aoi_from_geojson(geojson_obj)
        col = get_s2_collection(aoi, start, end)
        cnt = ee.Number(col.size()).getInfo()
        st.caption(f"Imágenes Sentinel-2 en rango: {cnt}")
        if cnt > 0:
            ndvi = col.select("NDVI").median().clip(aoi)
            url = ndvi.visualize(
                min=0, max=0.8,
                palette=["#d73027", "#f46d43", "#fdae61", "#fee08b",
                         "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850"]
            ).getThumbURL({"region": aoi, "dimensions": 640})
            st.image(url, caption="NDVI mediana (Sentinel-2)")
    except Exception as e:
        st.warning(f"No se pudo mostrar el mapa: {e}")

# -------------------------- Gráficos -----------------------------------
st.subheader("Fenología de floración – series temporales")
tabs = st.tabs([
    f"{s2_index} (Sentinel-2)", "EVI (MODIS 061)", "Precipitación (IMERG V07)",
    "Histórico NDVI (Landsat, 5 años)", f"Anomalías {s2_index} (S2) vs climatología (5 años)",
    "Índice vs Precipitación", "Métricas fenológicas", "Pronóstico (6 meses)", "Datos y métodos"
])

if EE_OK:
    try:
        # ---- Índice S2 (con cache) ----
        with tabs[0]:
            with st.spinner("Calculando series de Sentinel-2…"):
                df_idx = cached_timeseries_ndvi_s2(geojson_obj, start, end, agg, s2_index, wc_codes)
            if df_idx.empty:
                st.warning(f"Sin datos {s2_index}. Revisa fechas/AOI o prueba otras clases de WorldCover.")
            else:
                smooth_col = f"{s2_index}_smooth"
                df_idx[smooth_col] = df_idx[s2_index].rolling(7, center=True, min_periods=1).mean()
                idx_peaks = detect_peaks(df_idx, s2_index, smooth_col,
                                         n_peaks=n_peaks, min_value=min_value, min_separation=sep_days)
                title = f"{s2_index} (S2) – Floración"
                fig = _plot_with_peaks(df_idx, s2_index, smooth_col, idx_peaks, s2_index, title)
                st.plotly_chart(fig, use_container_width=True)

                # Fases fenológicas
                phases = detect_greening_phases(df_idx, s2_index, smooth_col)
                with st.expander("Fases fenológicas (estimadas)"):
                    def _fmt(x):
                        try: return pd.to_datetime(x).date().isoformat()
                        except Exception: return "NA"
                    st.markdown(
                        f"- **Inicio green-up:** {_fmt(phases.get('greenup_start'))}\n"
                        f"- **Pico:** {_fmt(phases.get('peak'))}\n"
                        f"- **Inicio senescencia:** {_fmt(phases.get('senescence_start'))}"
                    )

                # CSV
                out = df_idx.copy()
                for i, p in enumerate(idx_peaks, 1):
                    out.loc[out["date"] == p["date"], f"PEAK_{i}"] = p["value"]
                st.download_button(f"⬇️ Descargar {s2_index} CSV",
                                   out.to_csv(index=False).encode("utf-8"),
                                   f"s2_{s2_index.lower()}_timeseries.csv", "text/csv")

        # ---- EVI MODIS (con cache) ----
        with tabs[1]:
            with st.spinner("Calculando serie EVI (MODIS)…"):
                df_evi = cached_timeseries_evi_modis(geojson_obj, start, end, agg)
            if df_evi.empty:
                st.warning("Sin datos EVI (MODIS).")
            else:
                df_evi["EVI_smooth"] = df_evi["EVI"].rolling(7, center=True, min_periods=1).mean()
                evi_peaks = detect_peaks(df_evi, "EVI", "EVI_smooth",
                                         n_peaks=n_peaks, min_value=max(0.15, min_value - 0.10),
                                         min_separation=sep_days)
                fig = _plot_with_peaks(df_evi, "EVI", "EVI_smooth", evi_peaks, "EVI", "EVI (MODIS) – Floración")
                st.plotly_chart(fig, use_container_width=True)

                out = df_evi.copy()
                for i, p in enumerate(evi_peaks, 1):
                    out.loc[out["date"] == p["date"], f"PEAK_{i}"] = p["value"]
                st.download_button("⬇️ Descargar EVI CSV",
                                   out.to_csv(index=False).encode("utf-8"),
                                   "evi_timeseries.csv", "text/csv")

        # ---- IMERG (con cache) ----
        with tabs[2]:
            with st.spinner("Calculando precipitación IMERG…"):
                df_p = cached_timeseries_precip(geojson_obj, start, end, agg)
            if df_p.empty:
                st.warning("Sin datos de precipitación IMERG.")
            else:
                st.line_chart(df_p.set_index("date"))

        # ---- Hist. Landsat (con cache) ----
        with tabs[3]:
            with st.spinner("Calculando histórico Landsat (5 años)…"):
                df_ls = cached_timeseries_landsat_5y(geojson_obj, start, end)
            if df_ls.empty:
                st.warning("Sin datos Landsat en los últimos 5 años para el AOI/fechas.")
            else:
                df_ls["NDVI_smooth"] = df_ls["NDVI"].rolling(5, center=True, min_periods=1).mean()
                t = np.arange(len(df_ls))
                coef = np.polyfit(t, df_ls["NDVI"].astype(float), 1)
                df_ls["NDVI_trend"] = coef[0] * t + coef[1]
                fig = _plot_with_peaks(df_ls, "NDVI", "NDVI_smooth", None, "NDVI", "Histórico NDVI (Landsat, 5 años)")
                x_dt = df_ls["date"].dt.tz_localize(None).dt.to_pydatetime()
                fig.add_trace(go.Scatter(x=x_dt, y=df_ls["NDVI_trend"], mode="lines",
                                         name="Tendencia", line=dict(dash="dash")))
                st.plotly_chart(fig, use_container_width=True)

                st.download_button("⬇️ Descargar histórico Landsat CSV",
                                   df_ls.to_csv(index=False).encode("utf-8"),
                                   "landsat_ndvi_5y.csv", "text/csv")
                
        # ---- Anomalías índice S2 ----
        with tabs[4]:
            with st.spinner("Calculando anomalías vs climatología (5 años)…"):
                df_idx_m = cached_timeseries_ndvi_s2(geojson_obj, start, end, "mensual", s2_index, wc_codes)

            if df_idx_m.empty:
                st.warning(f"Necesitamos {s2_index} mensual (S2) para anomalías.")
            else:
                anom = anomalies_against_recent_climatology(df_idx_m, s2_index, end, years=5)
                if anom.empty:
                    st.info("No fue posible calcular climatología/anomalías con los datos disponibles.")
                else:
                    # Define colors and text labels
                    anom["color"] = np.where(anom["anomaly"] >= 0, "#1a9850", "#d73027")
                    anom["label"] = np.where(
                        anom["anomaly"] >= 0,
                        "Above climatology (Greener)",
                        "Below climatology (Drier/Stress)"
                    )

                    # Create figure
                    fig = go.Figure()
                    for lbl, df_sub in anom.groupby("label"):
                        fig.add_trace(go.Bar(
                            x=df_sub["date"], y=df_sub["anomaly"],
                            name=lbl,
                            marker_color=df_sub["color"].iloc[0],
                            text=[f"{v:+.2f}" for v in df_sub["anomaly"]],
                            textposition="outside"
                        ))

                    # Update layout
                    fig.update_layout(
                        margin=dict(l=40, r=40, t=60, b=60),
                        xaxis_title="Date",
                        yaxis_title=f"{s2_index} Anomaly (Index – 5yr climatology)",
                        title=f"{s2_index} anomalies (12 months) vs 5-year climatology",
                        legend=dict(
                            title="Legend",
                            orientation="h",
                            yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5,
                            bgcolor="rgba(255,255,255,0.5)"
                        ),
                        bargap=0.3
                    )
                    fig.update_xaxes(type="date", tickformat="%Y-%m", tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

                    # Download button
                    st.download_button(
                        "⬇️ Download anomaly CSV",
                        anom.to_csv(index=False).encode("utf-8"),
                        f"anomalies_{s2_index.lower()}.csv",
                        "text/csv"
                    )


        # ---- Índice vs Precipitación (lag) ----
        with tabs[5]:
            lag = st.slider("Retardo aplicado a precipitación (días)", -90, 90, 0, 10)
            with st.spinner("Alineando índice y precipitación…"):
                df_idx_m = cached_timeseries_ndvi_s2(geojson_obj, start, end, "mensual", s2_index, wc_codes)
                df_p_m   = cached_timeseries_precip(geojson_obj, start, end, "mensual")
            if df_idx_m.empty or df_p_m.empty:
                st.warning("Se requieren series mensuales del índice y de precipitación.")
            else:
                rel = align_index_precip(df_idx_m, s2_index, df_p_m, lag_days=lag)
                if rel.empty:
                    st.info("No hubo intersección de meses tras aplicar el retardo.")
                else:
                    r = float(rel[s2_index].corr(rel["PRECIP"]))
                    b1, b0 = np.polyfit(rel["PRECIP"].values.astype(float),
                                        rel[s2_index].values.astype(float), 1)
                    xline = np.linspace(rel["PRECIP"].min(), rel["PRECIP"].max(), 100)
                    yline = b1 * xline + b0

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=rel["PRECIP"], y=rel[s2_index],
                                             mode="markers", name="Meses"))
                    fig.add_trace(go.Scatter(x=xline, y=yline, mode="lines",
                                             name="Regresión", line=dict(dash="dash")))
                    fig.update_layout(
                        margin=dict(l=40,r=40,t=50,b=50),
                        xaxis_title="Precipitación IMERG (mm/mes)",
                        yaxis_title=s2_index,
                        title=f"Relación {s2_index} vs Precipitación · r={r:.2f} · R²={r*r:.2f}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.download_button("⬇️ Descargar tabla relación (CSV)",
                                       rel.to_csv(index=False).encode("utf-8"),
                                       f"relation_{s2_index.lower()}_precip_lag{lag}.csv", "text/csv")

        # ---- Métricas fenológicas ----
        with tabs[6]:
            with st.spinner("Calculando métricas fenológicas…"):
                df_idx_m = cached_timeseries_ndvi_s2(geojson_obj, start, end, "mensual", s2_index, wc_codes)
            if df_idx_m.empty:
                st.warning(f"Necesitamos {s2_index} mensual (S2) para métricas.")
            else:
                smooth_col = f"{s2_index}_smooth"
                df_idx_m[smooth_col] = df_idx_m[s2_index].rolling(5, center=True, min_periods=1).mean()
                metrics = compute_phenology_metrics(df_idx_m, value_col=s2_index, smooth_col=smooth_col)
                table = pd.DataFrame([
                    ["Inicio green-up", metrics["greenup_start"]],
                    ["Pico", metrics["peak"]],
                    ["Inicio senescencia", metrics["senescence_start"]],
                    ["Duración (días)", metrics["duration_days"]],
                    ["Valor pico", metrics["peak_value"]],
                    ["Valor mínimo", metrics["min_value"]],
                    ["Amplitud", metrics["amplitude"]],
                    ["AUC (área bajo curva)", metrics["auc"]],
                ], columns=["Métrica", "Valor"])
                st.dataframe(table, use_container_width=True)

                st.markdown("**Interpretaciones rápidas**")
                st.markdown(
                    "- La **ventana probable de floración** va desde el **inicio de green-up** hasta el "
                    "**inicio de senescencia**.\n"
                    "- La **intensidad** se aproxima con la **amplitud** (pico – mínimo). "
                    "Valores altos sugieren floraciones más marcadas.\n"
                    "- El **AUC** resume la actividad del dosel; útil para comparar campañas."
                )

                st.download_button("⬇️ Descargar métricas (CSV)",
                                   table.to_csv(index=False).encode("utf-8"),
                                   f"metrics_{s2_index.lower()}.csv", "text/csv")

        # ---- Pronóstico 6 meses ----
        with tabs[7]:
            with st.spinner("Calculando pronóstico (experimental)…"):
                df_idx_m = cached_timeseries_ndvi_s2(geojson_obj, start, end, "mensual", s2_index, wc_codes)
            if df_idx_m.empty:
                st.warning(f"Necesitamos {s2_index} mensual (S2) para pronóstico.")
            else:
                fc = forecast_next_months(df_idx_m, value_col=s2_index, months=6, years=5)
                if fc.empty:
                    st.info("No fue posible calcular el pronóstico con los datos disponibles.")
                else:
                    fig = go.Figure()
                    hist = df_idx_m.tail(12)
                    fig.add_trace(go.Scatter(x=hist["date"], y=hist[s2_index], mode="lines+markers",
                                             name=f"{s2_index} observado"))
                    fig.add_trace(go.Scatter(x=fc["date"], y=fc["pred"], mode="lines+markers",
                                             name="Pronóstico"))
                    fig.add_trace(go.Scatter(x=fc["date"], y=fc["high"], mode="lines",
                                             name="Alta (95%)", line=dict(dash="dot")))
                    fig.add_trace(go.Scatter(x=fc["date"], y=fc["low"], mode="lines",
                                             name="Baja (95%)", line=dict(dash="dot")))
                    fig.add_trace(go.Scatter(x=fc["date"], y=fc["clim"], mode="lines",
                                             name="Climatología", line=dict(dash="dash")))
                    fig.update_layout(margin=dict(l=40,r=40,t=60,b=40),
                                      xaxis_title="Fecha", yaxis_title=s2_index,
                                      title=f"Pronóstico {s2_index} (próximos 6 meses) – experimental")
                    fig.update_xaxes(type="date", tickformat="%Y-%m", tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    st.download_button("⬇️ Descargar pronóstico (CSV)",
                                       fc.to_csv(index=False).encode("utf-8"),
                                       f"forecast_{s2_index.lower()}.csv", "text/csv")

        # ---- Datos y métodos (nuevo tab) ----
        with tabs[8]:
            st.markdown("## Datos y métodos")
            st.markdown("""
**Datasets utilizados**  
- **NASA GPM IMERG V07** (precipitación diaria → agregación mensual).  
- **NASA MODIS 061 (MOD13Q1)**: EVI/NDVI (res. 250–500 m, compuestos 16 días → mensual).  
- **Sentinel-2 SR (Harmonized)**: NDVI, **EVI2**, **NDRE**, **NIRv** (10 m), máscara SCL para nubes/sombras.  
- **Landsat L2 C2 (L5/L7/L8/L9)**: NDVI histórico (5 años, ~30 m).

**Procesamiento y métricas**  
- **Agregación temporal**: mediana/quincenal/mensual según selección; compuestos mensuales vía mediana.  
- **Reducción zonal**: media sobre el AOI, `bestEffort=True`.  
- **Detección de picos**: máximos locales de la serie suavizada; parámetros: umbral, número de picos, separación mínima.  
- **Fases fenológicas**: green-up, pico y senescencia a partir del signo de la derivada suavizada.  
- **Anomalías**: último año vs climatología mediana de 5 años (por mes).  
- **Pronóstico (experimental)**: climatología mensual + anomalía reciente; intervalos ±1.96·σ de residuos.

**Limitaciones**  
- Nubes en S2; MODIS con resolución 250–500 m puede mezclar coberturas.  
- Enmascaramiento por **WorldCover** ayuda, pero puede excluir píxeles útiles si el AOI es heterogéneo.  
- El pronóstico es heurístico y no sustituye un modelo fenológico calibrado por cultivo/región.

**Reproducibilidad**  
- Requiere autenticación de Google Earth Engine y variables: `EARTHENGINE_PROJECT`.  
""")

    except Exception as e:
        st.error(f"Error al calcular series: {e}")

else:
    st.info("Sin sesión de GEE; no se pueden calcular series.")
