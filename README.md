# 🌿 GeoVerde – Phenology Explorer for Bloom Events

**GeoVerde** is an interactive dashboard built with **Streamlit** for monitoring and visualizing **plant bloom phenology** using satellite data from **NASA** and **ESA**.  
It detects flowering peaks (🌸), analyzes their relationship with rainfall, historical trends, and climate anomalies — helping bridge Earth observations to local agricultural and ecological action.

---

## 🌍 Earth Observation Datasets

| Source | Dataset | Resolution | Variable |
|:--------|:---------|:------------|:----------|
| NASA | **GPM IMERG V07** | 10 km | Precipitation |
| NASA | **MODIS 061 (MOD13Q1)** | 250–500 m | NDVI / EVI |
| ESA | **Sentinel-2 SR Harmonized** | 10 m | NDVI / EVI2 / NDRE / NIRv |
| USGS / NASA | **Landsat L2 C2 (L5–L9)** | 30 m | NDVI (5-year historical trend) |
| ESA | **WorldCover 10 m** | 10 m | Land cover classification |

---

## ⚙️ Requirements

```bash
# Python 3.10+ recommended
pip install -r requirements.txt

----------------------------------------------------------------------------------------------------------------------------------

# requirements.txt
streamlit==1.37.0
earthengine-api==1.4.3
folium==0.16.0
streamlit-folium==0.20.0
plotly==5.23.0
pandas==2.2.2
numpy==1.26.4
kaleido==0.2.1

-----------------------------------------------------------------------------------------------------------------

# Google Earth Engine Authentication

Run once:

python -c "import ee; ee.Authenticate()"


Then set your project ID:

Windows (CMD / PowerShell)

setx EARTHENGINE_PROJECT jovial-sunrise-393204


macOS / Linux

export EARTHENGINE_PROJECT=jovial-sunrise-393204

----------------------------------------------------------------------------------------------------------------

# Run the App

streamlit run app.py

---------------------------------------------------------------------------------------------------------------

# Quick Start

In the sidebar, choose:

Date range and temporal aggregation (monthly / biweekly / daily).

Crop type (auto-adjusts NDVI/EVI thresholds).

Bloom detection parameters: threshold, number of peaks, and min. separation (days).

Upload your AOI (.geojson) or use the built-in Piura, Peru demo area.

Explore the tabs:

NDVI (Sentinel-2) and EVI (MODIS) → time-series with detected flowering peaks 🌸.

Precipitation (IMERG) → climate context.

Landsat History → 5-year NDVI trend.

Anomalies → recent index vs. 5-year climatology.

Index vs. Rainfall → correlation and regression.

Phenological Metrics → key dates, amplitude, duration, and AUC.

Forecast (6 months) → experimental NDVI trend projection.

Data & Methods → dataset references, algorithms, and assumptions.

------------------------------------------------------------------------------------------------------------------------

# AOI (.geojson) Format

Coordinate system: WGS84 (EPSG:4326)

Coordinates ordered as [longitude, latitude]

Closed polygon (last vertex = first vertex)

-----------------------------------------------------------------------------------------------------------------------------

# Alignment with NASA Space Apps Challenge – “BloomWatch”

GeoVerde directly supports the “From EarthData to Action: BloomWatch” challenge goal by:

Detecting where and when global or regional flowering occurs.

Integrating NASA datasets (MODIS, IMERG) and ESA Sentinel-2 imagery.

Enabling localized phenological monitoring for crops and ecosystems.

Providing actionable insights for:

Sustainable agriculture and crop management.

Ecological and conservation monitoring.

Public health and allergen (pollen) forecasting.

----------------------------------------------------------------------------------------------------------------------------------

# Performance Tips

Use monthly aggregation for large AOIs.

The app uses median monthly composites and adaptive scaling per AOI size.

If Sentinel-2 NDVI takes long to load, reduce the time range or AOI area.

---------------------------------------------------------------------------------------------------------------------------------

# Troubleshooting

| Error                                    | Likely Cause / Solution                          |
| :--------------------------------------- | :----------------------------------------------- |
| `EEException / not initialized`          | Run `ee.Authenticate()` and restart the app.     |
| `Dictionary does not contain key 'NDVI'` | Cloudy months — extend date range or change AOI. |
| Slow map load                            | Use smaller AOIs or longer temporal aggregation. |

--------------------------------------------------------------------------------------------------------------------------

# Reproducibility & Methods

Compositing: Monthly medians for all indices.

Cloud masking: Sentinel-2 SCL ≠ (3, 8, 9, 10).

Zonal statistics: Mean over AOI using bestEffort=True.

Phenology: Peak detection in smoothed NDVI/EVI curve.

Anomalies: Last 12 months vs. 5-year median climatology.

Forecast: Climatology + recent anomaly ±1.96σ (simple heuristic model).

---------------------------------------------------------------------------------------------------------------------------


