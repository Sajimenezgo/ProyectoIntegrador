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

# # 01 · Prep & validate
# 
# Stage 1 of the Dirichlet-regression pipeline. Reads `join_to_gold`, runs data-quality
# gates, applies multiplicative zero-replacement, makes the **round(sqrt(N))** train/test
# split, standardizes predictors (**fit on train only**), computes VIF, and persists the
# prepared dataset + scaler. Starts the shared MLflow run.

# CELL ********************

%pip install scikit-learn statsmodels

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import os, json, glob
import numpy as np
import pandas as pd

SEED  = 2026
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
ART          = "/lakehouse/default/Files/models/dirichlet_ecv"
EXPERIMENT   = "dirichlet_ecv"
SOURCE_TABLE = "join_to_gold"
os.makedirs(ART, exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib, mlflow

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- Read gold table -------------------------------------------------------
pdf = spark.sql(f"SELECT * FROM {SOURCE_TABLE}").toPandas()
print("rows:", len(pdf), "| cols:", len(pdf.columns))
need = IDS + RESP + PREDS + ["suma_proporciones"]
missing = [c for c in need if c not in pdf.columns]
assert not missing, f"missing columns: {missing}"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- Data-quality gates ----------------------------------------------------
dq = {"n_raw": int(len(pdf))}

dups = int(pdf.duplicated(subset=IDS).sum())
dq["duplicate_keys"] = dups
assert dups == 0, f"{dups} duplicate (anio, corporacion, codigo_comuna) keys"

nulls = pdf[RESP + PREDS].isnull().sum()
dq["null_counts"] = {k: int(v) for k, v in nulls.items() if v > 0}
n_before = len(pdf)
pdf = pdf.dropna(subset=RESP + PREDS).reset_index(drop=True)
dq["n_after_dropna"] = int(len(pdf))
print(f"dropna: {n_before} -> {len(pdf)}  | null cols: {dq['null_counts']}")

for c in RESP:
    assert pdf[c].between(0, 1).all(), f"{c} has values outside [0,1]"
sum_err = float((pdf[RESP].sum(axis=1) - pdf["suma_proporciones"]).abs().max())
dq["max_sum_error"] = sum_err
assert sum_err < 1e-3, f"sum(p_*) vs suma_proporciones mismatch: {sum_err}"
print("max |sum(p_*) - suma_proporciones|:", sum_err)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- Response matrix + multiplicative zero-replacement ---------------------
# Dirichlet support is the OPEN simplex, so any exact 0/1 must be nudged inside.
def multiplicative_replacement(Y, delta=1e-4):
    Y = Y / Y.sum(axis=1, keepdims=True)
    out = Y.copy()
    n_adj = 0
    for i in range(Y.shape[0]):
        zeros = out[i] <= 0
        if zeros.any():
            n_adj += 1
            z = int(zeros.sum())
            out[i, zeros] = delta
            out[i, ~zeros] = out[i, ~zeros] * (1.0 - z * delta)
        out[i] = out[i] / out[i].sum()
    return out, n_adj

Y_obs = pdf[RESP].to_numpy(float)
Y_obs = Y_obs / Y_obs.sum(axis=1, keepdims=True)          # renormalized observed (evaluation)
Y_adj, n_adj = multiplicative_replacement(Y_obs, 1e-4)    # open simplex (Dirichlet likelihood)
dq["rows_zero_adjusted"] = int(n_adj)
print("rows zero-adjusted:", n_adj)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- Split + standardize (fit on train only) + VIF -------------------------
N = len(pdf)
n_test = int(round(np.sqrt(N)))
train_idx, test_idx = train_test_split(np.arange(N), test_size=n_test, random_state=SEED)
print(f"N={N} | n_train={len(train_idx)} | n_test={n_test}")

X_full = pdf[PREDS].to_numpy(float)
scaler = StandardScaler().fit(X_full[train_idx])          # NO leakage
X_std  = scaler.transform(X_full)

vif = pd.DataFrame({
    "predictor": PREDS,
    "VIF": [variance_inflation_factor(X_std[train_idx], j) for j in range(len(PREDS))],
}).sort_values("VIF", ascending=False).reset_index(drop=True)
vif["flag"] = np.where(vif.VIF > 10, ">10", np.where(vif.VIF > 5, ">5", ""))
max_vif = float(vif.VIF.max())
print(vif.to_string(index=False))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- Build + persist train/test (Delta) + scaler + reports -----------------
def build_split(positions, label):
    d = pdf.loc[positions, IDS].reset_index(drop=True).copy()
    Xs = X_std[positions]
    for j, c in enumerate(PREDS):    d[c] = Xs[:, j]            # standardized predictor
    for k, c in enumerate(RESP):     d[c] = Y_obs[positions, k] # observed proportion
    for k, c in enumerate(RESP_ADJ): d[c] = Y_adj[positions, k] # zero-replaced
    d["split"] = label
    return d

train_df = build_split(train_idx, "train")
test_df  = build_split(test_idx,  "test")

(spark.createDataFrame(train_df).write.mode("overwrite")
      .option("overwriteSchema", "true").format("delta").saveAsTable("model_train"))
(spark.createDataFrame(test_df).write.mode("overwrite")
      .option("overwriteSchema", "true").format("delta").saveAsTable("model_test"))
print("wrote tables: model_train, model_test")

joblib.dump(scaler, f"{ART}/scaler.joblib")
vif.to_csv(f"{ART}/vif.csv", index=False)
with open(f"{ART}/dq_report.json", "w") as f:
    json.dump(dq, f, indent=2)
print("wrote scaler.joblib, vif.csv, dq_report.json")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- Start the shared MLflow run -------------------------------------------
mlflow.set_experiment(EXPERIMENT)
run = mlflow.start_run(run_name="dirichlet_ecv_pipeline")
run_id = run.info.run_id
mlflow.log_params({
    "K": K, "P": len(PREDS), "predictors": ",".join(PREDS),
    "n_train": int(len(train_idx)), "n_test": int(n_test),
    "split_rule": "round(sqrt(N))", "seed": SEED, "zero_replacement_delta": 1e-4,
})
mlflow.log_metrics({
    "n_raw": dq["n_raw"], "n_after_dropna": dq["n_after_dropna"],
    "rows_zero_adjusted": dq["rows_zero_adjusted"],
    "max_sum_error": dq["max_sum_error"], "max_vif": max_vif,
})
mlflow.log_artifact(f"{ART}/vif.csv")
mlflow.log_artifact(f"{ART}/dq_report.json")
mlflow.end_run()
with open(f"{ART}/run_id.txt", "w") as f:
    f.write(run_id)
print("MLflow run_id:", run_id)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
