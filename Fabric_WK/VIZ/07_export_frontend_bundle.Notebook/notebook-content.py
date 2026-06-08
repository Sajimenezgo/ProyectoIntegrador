# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "15b10a14-49d8-4232-afca-2659551cd285",
# META       "default_lakehouse_name": "Data_LakeHouse",
# META       "default_lakehouse_workspace_id": "981be398-56d1-4882-a799-8c7592882837",
# META       "known_lakehouses": [
# META         {
# META           "id": "15b10a14-49d8-4232-afca-2659551cd285"
# META         }
# META       ]
# META     },
# META     "warehouse": {
# META       "default_warehouse": "1998873b-7175-4a5f-83dc-293eeddc5480",
# META       "known_warehouses": [
# META         {
# META           "id": "1998873b-7175-4a5f-83dc-293eeddc5480",
# META           "type": "Lakewarehouse"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# # 07 · Export front-end bundle
# 
# Stage 7 (optional, for the React+FastAPI app). Writes the **self-contained fallback bundle** the
# backend uses when OneLake is unavailable: latest-year comuna features, feature ranges, the slim
# posterior, the scaler, the comunas GeoJSON, and a reconciled code map. Everything lands in
# `Files/frontend_bundle/` for download into `frontend_app/backend/bundle/`.

# CELL ********************

%pip install arviz

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import os, json, glob
import numpy as np
import pandas as pd

SEED  = 2024
K     = 3
RESP  = ["p_left", "p_center", "p_right"]
COMP  = ["left", "center", "right"]
IDS   = ["anio", "corporacion", "codigo_comuna"]
PREDS = [
    "ESTRATO_STD", "EDUCACION_STD", "ALFABETISMO", "DIM_RECREACION",
    "IND_DISCRIMINACION", "IND_BRECHA_PERCEPCION", "DIM_PARTICIPACION",
    "DIM_MEDIO_AMBIENTE", "DIM_MOVILIDAD", "SATISFACCION_MUNICIPIO",
    "CONFIANZA_INSTITUCIONES", "CUMPLIMIENTO_NORMAS", "DIM_SEGURIDAD_ALIMENTARIA",
    "DIM_CAPITAL_FISICO", "INDICE_CARENCIA_SERVICIOS", "IND_HACINAMIENTO",
]
RESP_ADJ     = [c + "_adj" for c in RESP]        # zero-replaced (open simplex)
ART             = "/lakehouse/default/Files/models/dirichlet_ecv"
EXPERIMENT      = "dirichlet_ecv"
REGISTERED_NAME = "dirichlet_ecv"
SOURCE_TABLE    = "join_to_gold"
os.makedirs(ART, exist_ok=True)

import shutil, json
import arviz as az
BUNDLE = "/lakehouse/default/Files/frontend_bundle"
os.makedirs(BUNDLE, exist_ok=True)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- comuna_features.csv (latest anio) + feature_meta.csv (all rows) --------
pdf = spark.sql(f"SELECT * FROM {SOURCE_TABLE}").toPandas()
latest = int(pdf["anio"].max())
cols = IDS + PREDS + RESP
feat = pdf[pdf["anio"] == latest][cols].copy()
feat.to_csv(f"{BUNDLE}/comuna_features.csv", index=False)

meta = pd.DataFrame({"name": PREDS,
                     "min": [float(pdf[p].min()) for p in PREDS],
                     "max": [float(pdf[p].max()) for p in PREDS],
                     "mean": [float(pdf[p].mean()) for p in PREDS]})
meta.to_csv(f"{BUNDLE}/feature_meta.csv", index=False)
print(f"latest anio = {latest}; comunas = {feat['codigo_comuna'].nunique()}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- ensure slim posterior + scaler exist ----------------------------------
slim = f"{ART}/posterior_slim.npz"
if not os.path.exists(slim):
    post = az.from_netcdf(f"{ART}/posterior.nc").posterior
    ex  = [d for d in post["beta"].dims  if d not in ("chain", "draw")]
    ex0 = [d for d in post["beta0"].dims if d not in ("chain", "draw")]
    beta = post["beta"].stack(s=("chain", "draw")).transpose("s", *ex).values
    b0   = post["beta0"].stack(s=("chain", "draw")).transpose("s", *ex0).values
    np.savez_compressed(slim, beta=beta, beta0=b0,
                        preds=np.array(PREDS), comp=np.array(COMP))
    print("built", slim)
shutil.copy(slim, f"{BUNDLE}/posterior_slim.npz")
shutil.copy(f"{ART}/scaler.joblib", f"{BUNDLE}/scaler.joblib")
print("copied scaler + slim posterior")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- GeoJSON + code reconciliation -----------------------------------------
# Place a public Medellin comunas GeoJSON at this path first (upload to Files).
gj_path = f"{ART}/comunas_medellin.geojson"
codes = sorted(int(c) for c in pdf["codigo_comuna"].unique())
print("distinct codigo_comuna in data:", codes)

if os.path.exists(gj_path):
    gj = json.load(open(gj_path, encoding="utf-8"))
    rows = []
    for ft in gj["features"]:
        props = ft.get("properties", {})
        rows.append({"feature_id": ft.get("id"), **props})
    gdf = pd.DataFrame(rows)
    print("GeoJSON feature properties (inspect to choose id + name columns):")
    print(gdf.head(20).to_string(index=False))
    # Heuristic: a property whose integer values match the data codes is the id.
    id_col = None
    for col in gdf.columns:
        try:
            vals = sorted(int(float(v)) for v in gdf[col].dropna().unique())
        except (ValueError, TypeError):
            continue
        if set(vals) >= set(codes):
            id_col = col; break
    name_col = next((c for c in gdf.columns
                     if gdf[c].dtype == object and c != id_col), id_col)
    if id_col:
        cmap = pd.DataFrame({"codigo_comuna": codes})
        lut = {int(float(r[id_col])): (str(r["feature_id"]), str(r[name_col]))
               for _, r in gdf.iterrows() if pd.notna(r[id_col])}
        cmap["geojson_id"] = cmap["codigo_comuna"].map(lambda k: lut.get(k, (str(k), ""))[0])
        cmap["name"] = cmap["codigo_comuna"].map(lambda k: lut.get(k, ("", f"Comuna {k}"))[1])
        cmap.to_csv(f"{BUNDLE}/comuna_code_map.csv", index=False)
        shutil.copy(gj_path, f"{BUNDLE}/comunas_medellin.geojson")
        print(f"auto-matched on '{id_col}' (name='{name_col}'); wrote comuna_code_map.csv")
        print(cmap.to_string(index=False))
    else:
        print("!! No GeoJSON property matched the data codes. Inspect the table above and")
        print("   hand-write comuna_code_map.csv with columns: codigo_comuna,geojson_id,name")
else:
    print(f"!! No GeoJSON at {gj_path}. Upload a Medellin comunas GeoJSON there and re-run")
    print("   this cell. Until then the app falls back to a comuna dropdown.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- list bundle -----------------------------------------------------------
print("frontend_bundle contents:")
for fn in sorted(os.listdir(BUNDLE)):
    print("  ", fn, os.path.getsize(os.path.join(BUNDLE, fn)), "bytes")
print("\nDownload these into frontend_app/backend/bundle/.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
