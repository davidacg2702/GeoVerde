# bloom_farm_pipeline.py
import ee, pandas as pd
import numpy as np
from gee_utils import (
    get_s2_collection, get_modis_ndvi, get_imerg_precip, zonal_timeseries,
    get_landsat_merged, get_s2_collection_with_indices
)

# ----------------------------- util ------------------------------------
def _fc_to_df(fc, value_col):
    size = ee.Number(fc.size()).getInfo()
    if size == 0:
        return pd.DataFrame({"date": [], value_col: []})
    lst = fc.toList(size)
    recs = [ee.Feature(lst.get(i)).toDictionary().getInfo() for i in range(size)]
    df = pd.DataFrame(recs)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

def _resample_df(df, value_col, agg):
    if df.empty:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    rule = {"diario": "D", "quincenal": "2W", "mensual": "MS"}[agg]
    out = df.resample(rule)[value_col].median().to_frame()
    out.index = out.index.rename("date")
    return out.reset_index()

def _monthly_composites(col, start, end, expected_band, reducer=None):
    reducer = reducer or ee.Reducer.median()
    startD, endD = ee.Date(start), ee.Date(end)
    nmonths = endD.difference(startD, "month").toInt()

    def _comp(m):
        d1 = startD.advance(m, "month")
        d2 = d1.advance(1, "month")
        monthcol = col.filterDate(d1, d2)
        reduced = ee.Image(monthcol.reduce(reducer))
        img = ee.Image(ee.Algorithms.If(
            monthcol.size().gt(0),
            ee.Image(reduced).rename([expected_band]),
            ee.Image.constant(0).rename([expected_band]).updateMask(ee.Image.constant(0))
        ))
        return img.set({"system:time_start": d1.millis(), "date": d1.format("YYYY-MM-01")})

    return ee.ImageCollection(ee.List.sequence(0, nmonths.subtract(1)).map(_comp))

# ------------------------------ series ---------------------------------
def timeseries_ndvi_s2(aoi, start, end, agg="mensual", index_name="NDVI", worldcover_classes=None):
    """
    Serie S2 para índice elegido: 'NDVI' | 'EVI2' | 'NDRE' | 'NIRv'.
    Puede aplicar máscara WorldCover con worldcover_classes (p. ej., [40,30]).
    """
    col_all = get_s2_collection_with_indices(aoi, start, end, worldcover_classes).select([index_name])

    if ee.Number(col_all.size()).getInfo() == 0:
        return pd.DataFrame({"date": [], index_name: []})

    area_km2 = ee.Number(aoi.area().divide(1e6)).getInfo()
    if area_km2 > 700:
        scale = 120
    elif area_km2 > 250:
        scale = 80
    elif area_km2 > 100:
        scale = 60
    else:
        scale = 20

    if agg != "diario":
        col_all = _monthly_composites(col_all, start, end, expected_band=index_name, reducer=ee.Reducer.median())

    ts = zonal_timeseries(col_all, aoi, index_name, scale=scale)
    df = _fc_to_df(ts, index_name)
    return _resample_df(df, index_name, agg)

def timeseries_evi_modis(aoi, start, end, agg="mensual"):
    col = get_modis_ndvi(aoi, start, end).select("EVI_scaled")
    if ee.Number(col.size()).getInfo() == 0:
        return pd.DataFrame({"date": [], "EVI": []})
    area_km2 = ee.Number(aoi.area().divide(1e6)).getInfo()
    scale = 500 if area_km2 > 300 else 250

    if agg != "diario":
        col = _monthly_composites(col, start, end, expected_band="EVI_scaled", reducer=ee.Reducer.median())

    ts = zonal_timeseries(col, aoi, "EVI_scaled", scale=scale)
    df = _fc_to_df(ts, "EVI_scaled").rename(columns={"EVI_scaled": "EVI"})
    return _resample_df(df, "EVI", agg)

def timeseries_precip(aoi, start, end, agg="mensual"):
    col = get_imerg_precip(aoi, start, end).select("PRECIP")
    if ee.Number(col.size()).getInfo() == 0:
        return pd.DataFrame({"date": [], "PRECIP": []})
    area_km2 = ee.Number(aoi.area().divide(1e6)).getInfo()
    scale = 20000 if area_km2 > 300 else 10000

    if agg != "diario":
        col = _monthly_composites(col, start, end, expected_band="PRECIP", reducer=ee.Reducer.mean())

    ts = zonal_timeseries(col, aoi, "PRECIP", scale=scale)
    df = _fc_to_df(ts, "PRECIP")
    return _resample_df(df, "PRECIP", agg)

def timeseries_landsat_ndvi_5yr(aoi, start, end, agg="mensual"):
    end_ts = pd.to_datetime(end)
    start_5y = (end_ts - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    col = get_landsat_merged(aoi, start_5y, end).select("NDVI")

    area_km2 = ee.Number(aoi.area().divide(1e6)).getInfo()
    scale = 90 if area_km2 > 700 else 60

    if agg != "diario":
        col = _monthly_composites(col, start_5y, end, expected_band="NDVI", reducer=ee.Reducer.median())

    ts = zonal_timeseries(col, aoi, "NDVI", scale=scale)
    df = _fc_to_df(ts, "NDVI")
    return _resample_df(df, "NDVI", "mensual")

# --------------------- Fenología (picos y fases) -----------------------
def detect_peaks(df: pd.DataFrame,
                 value_col: str,
                 smooth_col: str | None = None,
                 n_peaks: int = 1,
                 min_value: float = 0.28,
                 min_separation: int = 20) -> list[dict]:
    if df.empty or value_col not in df.columns:
        return []
    s = df[smooth_col] if smooth_col and smooth_col in df.columns else df[value_col]
    s = s.astype(float)
    s_smooth = s.rolling(window=5, center=True, min_periods=1).mean()
    greater_prev = s_smooth > s_smooth.shift(1)
    greater_next = s_smooth > s_smooth.shift(-1)
    is_peak = greater_prev & greater_next & (s_smooth >= min_value)
    candidates = df.loc[is_peak, ['date', value_col]].copy()
    if candidates.empty:
        return []
    candidates.sort_values(value_col, ascending=False, inplace=True)
    chosen, taken_dates = [], []
    for _, row in candidates.iterrows():
        d = pd.to_datetime(row['date'])
        if all(abs((d - td).days) >= min_separation for td in taken_dates):
            chosen.append({'date': d, 'value': float(row[value_col])})
            taken_dates.append(d)
            if len(chosen) >= n_peaks:
                break
    chosen.sort(key=lambda x: x['date'])
    return chosen

def detect_greening_phases(df: pd.DataFrame,
                           value_col: str = "NDVI",
                           smooth_col: str | None = None,
                           deriv_threshold: float = 0.002) -> dict:
    if df.empty or value_col not in df.columns:
        return {}
    s = df[smooth_col] if smooth_col and smooth_col in df.columns else df[value_col]
    s = s.astype(float)
    s_s = s.rolling(5, center=True, min_periods=1).mean()
    d1 = s_s.diff()
    pos = d1.rolling(3, min_periods=1).mean() > deriv_threshold
    neg = d1.rolling(3, min_periods=1).mean() < -deriv_threshold
    try: greenup_start = df.loc[pos, "date"].iloc[0]
    except Exception: greenup_start = pd.NaT
    try:
        peak_idx = s_s.idxmax()
        peak = df.loc[peak_idx, "date"]
    except Exception:
        peak = pd.NaT
    try:
        after_peak = df["date"] > peak if pd.notna(peak) else pd.Series([False]*len(df))
        senescence_start = df.loc[after_peak & neg, "date"].iloc[0]
    except Exception:
        senescence_start = pd.NaT
    return {"greenup_start": greenup_start, "peak": peak, "senescence_start": senescence_start}

# ---------------- Anomalías vs climatología reciente -------------------
def anomalies_against_recent_climatology(df: pd.DataFrame,
                                         value_col: str = "NDVI",
                                         end_date: str | pd.Timestamp | None = None,
                                         years: int = 5) -> pd.DataFrame:
    if df.empty or value_col not in df.columns:
        return pd.DataFrame(columns=["date", value_col, "clim", "anomaly"])
    dd = df.copy()
    dd["date"] = pd.to_datetime(dd["date"])
    dd = dd.set_index("date")
    end_dt = pd.to_datetime(end_date) if end_date is not None else dd.index.max()
    start_dt = end_dt - pd.DateOffset(years=years)
    clim_base = dd.loc[(dd.index >= start_dt) & (dd.index <= end_dt)].copy()
    if clim_base.empty:
        return pd.DataFrame(columns=["date", value_col, "clim", "anomaly"])
    clim_base["month"] = clim_base.index.month
    clim = clim_base.groupby("month")[value_col].median().to_frame("clim")
    last12 = dd.loc[(dd.index > end_dt - pd.DateOffset(months=12)) & (dd.index <= end_dt)].copy()
    if last12.empty:
        return pd.DataFrame(columns=["date", value_col, "clim", "anomaly"])
    last12["month"] = last12.index.month
    out = last12.join(clim, on="month")[["month", value_col, "clim"]]
    out["anomaly"] = out[value_col] - out["clim"]
    out.index = out.index.rename("date")
    return out.reset_index()[["date", value_col, "clim", "anomaly"]]

# ---------------- Métricas fenológicas --------------------------------
def compute_phenology_metrics(df: pd.DataFrame,
                              value_col: str = "NDVI",
                              smooth_col: str | None = None) -> dict:
    out = {
        "greenup_start": None, "peak": None, "senescence_start": None,
        "duration_days": None, "peak_value": None, "min_value": None,
        "amplitude": None, "auc": None
    }
    if df.empty or value_col not in df.columns:
        return out
    dd = df.copy()
    dd["date"] = pd.to_datetime(dd["date"])
    dd = dd.dropna(subset=[value_col]).sort_values("date").reset_index(drop=True)
    if smooth_col and smooth_col in dd.columns:
        s = dd[smooth_col].astype(float)
    else:
        s = dd[value_col].astype(float).rolling(5, center=True, min_periods=1).mean()
    phases = detect_greening_phases(dd.assign(**{value_col: s}), value_col=value_col, smooth_col=None)
    for k in ["greenup_start", "peak", "senescence_start"]:
        try:
            out[k] = pd.to_datetime(phases.get(k)).date().isoformat() if phases.get(k) is not pd.NaT else None
        except Exception:
            out[k] = None
    out["peak_value"] = float(np.nanmax(s.values)) if len(s) else None
    out["min_value"]  = float(np.nanmin(s.values)) if len(s) else None
    if out["peak_value"] is not None and out["min_value"] is not None:
        out["amplitude"] = float(out["peak_value"] - out["min_value"])
    try:
        if out["greenup_start"] and out["senescence_start"]:
            d0 = pd.to_datetime(out["greenup_start"]); d1 = pd.to_datetime(out["senescence_start"])
            out["duration_days"] = int((d1 - d0).days)
    except Exception:
        out["duration_days"] = None
    try:
        x = dd["date"].map(pd.Timestamp.toordinal).astype(float).values
        x = x - x.min(); y = s.values.astype(float)
        out["auc"] = float(np.trapz(y, x))
    except Exception:
        out["auc"] = None
    return out

# ---------------- Climatología + Pronóstico ---------------------------
def monthly_climatology(df: pd.DataFrame, value_col: str, years: int = 5) -> pd.Series:
    if df.empty or value_col not in df.columns:
        return pd.Series(dtype=float)
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").sort_index()
    end_dt = d.index.max()
    if pd.isna(end_dt):
        return pd.Series(dtype=float)
    start_dt = end_dt - pd.DateOffset(years=years)
    base = d.loc[(d.index >= start_dt) & (d.index <= end_dt), [value_col]].dropna()
    if base.empty:
        return pd.Series(dtype=float)
    base["month"] = base.index.month
    return base.groupby("month")[value_col].median()

def forecast_next_months(df: pd.DataFrame, value_col: str, months: int = 6, years: int = 5) -> pd.DataFrame:
    """
    Pronóstico sencillo:
      pred = climatología(mes) + anomalía_reciente
    Donde anomalía_reciente = mediana de anomalías de los últimos 6–12 meses.
    Intervalo (muy simple): ±1.96 * std(residuos históricos).
    """
    if df.empty or value_col not in df.columns:
        return pd.DataFrame(columns=["date", "pred", "clim", "low", "high"])

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").sort_index()   # <-- aquí el fix

    # Climatología
    clim = monthly_climatology(d.reset_index(), value_col=value_col, years=years)
    if clim.empty:
        return pd.DataFrame(columns=["date", "pred", "clim", "low", "high"])

    # Residuos históricos respecto a la climatología
    hist = d.copy()
    hist["month"] = hist.index.month
    hist = hist.join(clim.rename("clim"), on="month")
    hist["resid"] = hist[value_col] - hist["clim"]
    resid_std = float(hist["resid"].std(skipna=True)) if not hist["resid"].empty else 0.05

    # Anomalía reciente: mediana últimos 12 meses (si no hay, usa 6)
    tail = hist.dropna(subset=["resid"]).tail(12)
    if tail.empty:
        tail = hist.dropna(subset=["resid"]).tail(6)
    recent_anom = float(tail["resid"].median()) if not tail.empty else 0.0

    # Fechas futuras (primer día de cada mes)
    last_month = (d.index.max() + pd.offsets.MonthBegin(1)).normalize()
    dates = [last_month + pd.DateOffset(months=i) for i in range(months)]
    months_idx = [dt.month for dt in dates]
    clim_vals = [float(clim.get(m, np.nan)) for m in months_idx]
    preds = [c + recent_anom if not np.isnan(c) else np.nan for c in clim_vals]
    low = [p - 1.96 * resid_std if not np.isnan(p) else np.nan for p in preds]
    high = [p + 1.96 * resid_std if not np.isnan(p) else np.nan for p in preds]

    out = pd.DataFrame({"date": dates, "pred": preds, "clim": clim_vals, "low": low, "high": high})
    return out.dropna(subset=["pred"]).reset_index(drop=True)


# --------------- NUEVO: alineación índice–precipitación ----------------
def align_index_precip(df_index: pd.DataFrame,
                       idx_col: str,
                       df_precip: pd.DataFrame,
                       lag_days: int = 0) -> pd.DataFrame:
    """
    Fusiona serie mensual del índice con IMERG aplicando retardo (lag) en días
    a la precipitación (positivo = la lluvia antecede al índice).
    Devuelve columnas: date, <idx_col>, PRECIP
    """
    if df_index.empty or df_precip.empty or idx_col not in df_index.columns:
        return pd.DataFrame(columns=["date", idx_col, "PRECIP"])

    a = df_index[["date", idx_col]].copy()
    b = df_precip[["date", "PRECIP"]].copy()

    a["date"] = pd.to_datetime(a["date"]).dt.to_period("M").dt.to_timestamp()
    b["date"] = pd.to_datetime(b["date"]) + pd.to_timedelta(lag_days, unit="D")
    b["date"] = b["date"].dt.to_period("M").dt.to_timestamp()

    m = pd.merge(a, b, on="date", how="inner").dropna()
    return m.reset_index(drop=True)
