# gee_utils.py
import os, json
import ee

# --- Colecciones ---
S2_COLLECTION   = "COPERNICUS/S2_SR_HARMONIZED"
MODIS_NDVI      = "MODIS/061/MOD13Q1"
IMERG_DAILY     = "NASA/GPM_L3/IMERG_V07"
WORLDCOVER_IC   = "ESA/WorldCover/v100"  # usa v200 si está disponible en tu entorno

# Landsat L2 (Collection 2): escala reflectancia = DN*0.0000275 - 0.2
L5_SR = "LANDSAT/LT05/C02/T1_L2"
L7_SR = "LANDSAT/LE07/C02/T1_L2"
L8_SR = "LANDSAT/LC08/C02/T1_L2"
L9_SR = "LANDSAT/LC09/C02/T1_L2"

# ---------------------------------------------------------------------
def init_ee(debug: bool = False) -> str:
    """
    Inicializa Google Earth Engine.
    - En Streamlit Cloud: usa Service Account de st.secrets (EE_SERVICE_ACCOUNT, EE_PRIVATE_KEY_JSON).
    - En local: intenta credenciales existentes; si fallan, permite ee.Authenticate().
    Devuelve 'service' (SA) o 'local' (OAuth/local).
    """
    project = os.getenv("EARTHENGINE_PROJECT") or "jovial-sunrise-393204"

    # Intentar leer secretos (si estamos dentro de Streamlit)
    sa = None
    key_json = None
    in_streamlit = False
    try:
        import streamlit as st  # import diferido
        in_streamlit = True
        project  = st.secrets.get("EARTHENGINE_PROJECT", project)
        sa       = st.secrets.get("EE_SERVICE_ACCOUNT", None)
        key_json = st.secrets.get("EE_PRIVATE_KEY_JSON", None)
    except Exception:
        # Modo CLI/local: también permitimos variables de entorno
        sa       = os.getenv("EE_SERVICE_ACCOUNT")
        key_json = os.getenv("EE_PRIVATE_KEY_JSON")

    # 1) Si hay Service Account + key -> usar SIEMPRE esto (modo nube)
    if sa and key_json:
        if isinstance(key_json, dict):
            key_data = json.dumps(key_json)
        else:
            key_data = key_json
        creds = ee.ServiceAccountCredentials(sa, key_data=key_data)
        ee.Initialize(credentials=creds, project=project)
        return "service"

    # 2) Sin SA: intentar credenciales locales (útil en tu PC)
    try:
        ee.Initialize(project=project)  # tokens guardados localmente
        return "local"
    except Exception:
        # En la nube NO hay gcloud → no intentes OAuth
        if in_streamlit:
            raise RuntimeError(
                "Streamlit Cloud sin Service Account. "
                "Agrega EE_SERVICE_ACCOUNT y EE_PRIVATE_KEY_JSON en Settings → Secrets."
            )
        # En local sí permitimos OAuth una vez
        ee.Authenticate()
        ee.Initialize(project=project)
        return "local"

# --------------------------- Sentinel-2 --------------------------------
def _s2_mask(img):
    scl = img.select("SCL")
    # Excluir sombra (3), nubes (8,9), cirrus (10)
    mask = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)))
    return img.updateMask(mask)

def _s2_add_indices(img):
    # Índices calculados siempre desde bandas S2 (sin getInfo dentro de map)
    nir = img.select("B8")
    red = img.select("B4")
    re5 = img.select("B5")

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    evi2 = img.expression("2.5*(NIR-RED)/(NIR+2.4*RED+1.0)", {"NIR": nir, "RED": red}).rename("EVI2")
    ndre = nir.subtract(re5).divide(nir.add(re5)).rename("NDRE")
    nirv = ndvi.multiply(nir).rename("NIRv")

    return img.addBands([ndvi, evi2, ndre, nirv], overwrite=True)

# --------- WorldCover: máscara por clases permitidas -------------------
def get_worldcover_mask(aoi, allowed_classes=None) -> ee.Image:
    """
    Devuelve una máscara booleana a partir de ESA WorldCover (10 m).
    allowed_classes: lista de códigos (10 árboles, 20 arbustos, 30 pastizal, 40 cultivos, 50 construido, 60 suelo desnudo, 70 nieve, 80 agua, 90 humedales, 95 manglar).
    Si None o lista vacía => no filtra (todo True).
    """
    img = ee.ImageCollection(WORLDCOVER_IC).first().select("Map").clip(aoi)
    if not allowed_classes:
        return img.gte(0)  # True para todos los píxeles
    m = img.eq(allowed_classes[0])
    for c in allowed_classes[1:]:
        m = m.Or(img.eq(c))
    return m

def get_s2_collection(aoi, start, end):
    """Colección S2 con NDVI (retrocompatible; sin WorldCover)."""
    return (ee.ImageCollection(S2_COLLECTION)
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
            .map(_s2_mask)
            .map(lambda im: _s2_add_indices(im).select("NDVI")))

def get_s2_collection_with_indices(aoi, start, end, worldcover_classes=None):
    """
    Sentinel-2 con bandas: NDVI, EVI2, NDRE, NIRv.
    Si worldcover_classes se provee (p. ej., [40,30]) aplica esa máscara.
    """
    col = (ee.ImageCollection(S2_COLLECTION)
           .filterBounds(aoi)
           .filterDate(start, end)
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
           .map(_s2_mask)
           .map(_s2_add_indices))

    if worldcover_classes:
        wc_mask = get_worldcover_mask(aoi, worldcover_classes)
        col = col.map(lambda i: i.updateMask(wc_mask))

    return col.select(["NDVI", "EVI2", "NDRE", "NIRv"])

# ----------------------------- MODIS ----------------------------------
def get_modis_ndvi(aoi, start, end):
    def _scale(img):
        return (img
                .addBands(img.select("NDVI").multiply(0.0001).rename("NDVI_scaled"), overwrite=True)
                .addBands(img.select("EVI").multiply(0.0001).rename("EVI_scaled"),   overwrite=True))
    col = (ee.ImageCollection(MODIS_NDVI)
           .filterBounds(aoi)
           .filterDate(start, end)
           .map(_scale))
    return col.select(["NDVI_scaled", "EVI_scaled"])

# ----------------------------- IMERG ----------------------------------
def get_imerg_precip(aoi, start, end):
    """
    Devuelve IMERG con banda 'PRECIP'.
    Hace size().getInfo() una vez (válido aquí, fuera de map()).
    """
    col = ee.ImageCollection(IMERG_DAILY).filterBounds(aoi).filterDate(start, end)
    n = ee.Number(col.size()).getInfo()
    if n == 0:
        return col

    first = ee.Image(col.first())
    bands = first.bandNames().getInfo()
    band  = "precipitationCal" if "precipitationCal" in bands else bands[0]
    return col.select([band], ["PRECIP"])

# ---------------------------- Landsat L2 -------------------------------
def _ls_scale_reflectance(img, red_name, nir_name):
    red = img.select(red_name).multiply(0.0000275).add(-0.2)
    nir = img.select(nir_name).multiply(0.0000275).add(-0.2)
    return img.addBands(red.rename("RED"), overwrite=True).addBands(nir.rename("NIR"), overwrite=True)

def _ls_cloud_mask(img):
    qa = img.select('QA_PIXEL')
    cloud = qa.bitwiseAnd(1 << 3).neq(0)
    shadow = qa.bitwiseAnd(1 << 4).neq(0)
    return img.updateMask(cloud.Or(shadow).Not())

def _ls_ndvi(img):
    ndvi = img.select("NIR").subtract(img.select("RED")).divide(img.select("NIR").add(img.select("RED"))).rename("NDVI")
    return img.addBands(ndvi, overwrite=True)

def _landsat_l5(aoi, start, end):
    return (ee.ImageCollection(L5_SR)
            .filterBounds(aoi).filterDate(start, end)
            .map(_ls_cloud_mask)
            .map(lambda im: _ls_scale_reflectance(im, "SR_B3", "SR_B4"))
            .map(_ls_ndvi))

def _landsat_l7(aoi, start, end):
    return (ee.ImageCollection(L7_SR)
            .filterBounds(aoi).filterDate(start, end)
            .map(_ls_cloud_mask)
            .map(lambda im: _ls_scale_reflectance(im, "SR_B3", "SR_B4"))
            .map(_ls_ndvi))

def _landsat_l89(aoi, start, end, collection):
    return (ee.ImageCollection(collection)
            .filterBounds(aoi).filterDate(start, end)
            .map(_ls_cloud_mask)
            .map(lambda im: _ls_scale_reflectance(im, "SR_B4", "SR_B5"))
            .map(_ls_ndvi))

def get_landsat_merged(aoi, start, end):
    l5 = _landsat_l5(aoi, start, end)
    l7 = _landsat_l7(aoi, start, end)
    l8 = _landsat_l89(aoi, start, end, L8_SR)
    l9 = _landsat_l89(aoi, start, end, L9_SR)
    return ee.ImageCollection(l5.merge(l7).merge(l8).merge(l9))

# ------------------------- AOI + Zonal stats ---------------------------
def aoi_from_geojson(obj):
    if obj.get("type") == "FeatureCollection":
        geoms = [ee.Feature(f).geometry() for f in obj["features"]]
        return ee.FeatureCollection(geoms).geometry().dissolve()
    if obj.get("type") == "Feature":
        return ee.Feature(obj).geometry()
    return ee.Geometry(obj)

def zonal_timeseries(imgcol, aoi, band, reducer=None, scale=20):
    reducer = reducer or ee.Reducer.mean()

    def reduce_img(img):
        date_prop = ee.Algorithms.If(
            img.propertyNames().contains('date'),
            img.get('date'),
            img.date().format('YYYY-MM-dd')
        )
        stat = img.reduceRegion(
            reducer=reducer, geometry=aoi, scale=scale,
            maxPixels=2e13, bestEffort=True, tileScale=8
        )
        dct = ee.Dictionary(stat)
        val = ee.Algorithms.If(dct.contains(band), dct.get(band), None)
        return ee.Feature(None, {"date": date_prop, band: val})

    fc = ee.FeatureCollection(imgcol.map(reduce_img))
    return fc.filter(ee.Filter.notNull([band]))
