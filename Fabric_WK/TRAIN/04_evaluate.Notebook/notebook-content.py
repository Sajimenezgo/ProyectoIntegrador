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

# # 04 · Evaluate & compare
# 
# Stage 4. Loads the posterior + baseline predictions, produces full-posterior Dirichlet
# predictions, calibration (90% coverage), and the cross-entropy / RMSE / winning-bloc
# comparison across the three models. Logs everything to the shared MLflow run.

# CELL ********************

%pip install arviz scikit-learn

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
ART          = "/lakehouse/default/Files/models/dirichlet_ecv"
EXPERIMENT   = "dirichlet_ecv"
SOURCE_TABLE = "join_to_gold"
os.makedirs(ART, exist_ok=True)

import arviz as az, mlflow

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- Test set --------------------------------------------------------------
test = spark.read.table("model_test").toPandas()
Y_test = test[RESP].to_numpy(float); Y_test = Y_test / Y_test.sum(axis=1, keepdims=True)
X_test = test[PREDS].to_numpy(float)
ids = test[IDS].reset_index(drop=True)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- Full-posterior Dirichlet predictions on test --------------------------
idata = az.from_netcdf(f"{ART}/posterior.nc")
post = idata.posterior
ex   = [d for d in post["beta"].dims  if d not in ("chain", "draw")]
ex0  = [d for d in post["beta0"].dims if d not in ("chain", "draw")]
beta = post["beta"].stack(s=("chain", "draw")).transpose("s", *ex).values    # (S, P, K)
b0   = post["beta0"].stack(s=("chain", "draw")).transpose("s", *ex0).values  # (S, K)
print("draws:", beta.shape[0], "| P,K:", beta.shape[1:])

eta   = np.einsum("ip,spk->sik", X_test, beta) + b0[:, None, :]   # (S, Nte, K)
alpha = np.exp(eta)
m     = alpha / alpha.sum(axis=2, keepdims=True)                  # per-draw mean proportions
p_dir = m.mean(axis=0)
p_dir = np.clip(p_dir, 1e-9, 1.0); p_dir = p_dir / p_dir.sum(1, keepdims=True)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- Calibration: 90% POSTERIOR-PREDICTIVE coverage ------------------------
# Use generative replicates y_rep ~ Dirichlet(alpha) so the interval includes
# the Dirichlet observation dispersion. (The interval of the mean m omits it
# and under-covers by construction.)
rng = np.random.default_rng(SEED)
y_rep = rng.standard_gamma(alpha)                 # Dirichlet via Gamma trick -> (S, Nte, K)
y_rep = y_rep / y_rep.sum(axis=2, keepdims=True)
lo = np.quantile(y_rep, 0.05, axis=0); hi = np.quantile(y_rep, 0.95, axis=0)
inside   = (Y_test >= lo) & (Y_test <= hi)
cov_comp = inside.mean(axis=0)
cov_all  = float(inside.mean())
calib = pd.DataFrame({"component": COMP, "coverage_90": cov_comp})
calib.loc[len(calib)] = ["overall", cov_all]
calib.to_csv(f"{ART}/calibration.csv", index=False)
print(calib.to_string(index=False), "\n(want ~0.90)")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- Load baseline predictions (merge on keys to keep row alignment) -------
rf_df   = pd.read_csv(f"{ART}/pred_rf.csv")
flat_df = pd.read_csv(f"{ART}/pred_flat.csv")
P_rf   = ids.merge(rf_df,   on=IDS, how="left")[RESP].to_numpy()
P_flat = ids.merge(flat_df, on=IDS, how="left")[RESP].to_numpy()
assert not np.isnan(P_rf).any() and not np.isnan(P_flat).any(), "baseline/test key mismatch"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- Metrics + comparison table --------------------------------------------
def cross_entropy(Yt, P):
    return float((-(Yt * np.log(np.clip(P, 1e-9, 1.0))).sum(axis=1)).mean())
def rmse(Yt, P):
    return float(np.sqrt(((Yt - P) ** 2).mean()))
def bloc_accuracy(Yt, P):
    return float((P.argmax(1) == Yt.argmax(1)).mean())

# posterior-predictive CE: mean over draws of per-draw CE
pp_ce = float((-(Y_test[None] * np.log(np.clip(m, 1e-9, 1.0))).sum(2)).mean(1).mean())

comparison = pd.DataFrame([
    {"model": "Dirichlet (point)",      "cross_entropy": cross_entropy(Y_test, p_dir),
     "rmse": rmse(Y_test, p_dir),       "bloc_accuracy": bloc_accuracy(Y_test, p_dir)},
    {"model": "Dirichlet (post.pred.)", "cross_entropy": pp_ce,
     "rmse": np.nan,                    "bloc_accuracy": np.nan},
    {"model": "Random forest",          "cross_entropy": cross_entropy(Y_test, P_rf),
     "rmse": rmse(Y_test, P_rf),        "bloc_accuracy": bloc_accuracy(Y_test, P_rf)},
    {"model": "Flat Dirichlet(1,1,1)",  "cross_entropy": cross_entropy(Y_test, P_flat),
     "rmse": rmse(Y_test, P_flat),      "bloc_accuracy": np.nan},
])
comparison.to_csv(f"{ART}/comparison.csv", index=False)
print(comparison.to_string(index=False))
print("log(3) reference =", round(float(np.log(3)), 4))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- Verdict ---------------------------------------------------------------
ce = comparison.set_index("model")["cross_entropy"]
print(f"Dirichlet CE = {ce['Dirichlet (point)']:.4f}")
print(f"RF CE        = {ce['Random forest']:.4f}")
print(f"Flat CE      = {ce['Flat Dirichlet(1,1,1)']:.4f}  (== log 3)")
print("Dirichlet beats RF  :", bool(ce['Dirichlet (point)'] < ce['Random forest']))
print("Dirichlet beats flat:", bool(ce['Dirichlet (point)'] < ce['Flat Dirichlet(1,1,1)']))
print("90% coverage overall:", round(cov_all, 3), "(want ~0.90; << 0.90 => under-dispersed)")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- Log evaluation to the shared MLflow run -------------------------------
mlflow.set_experiment(EXPERIMENT)
run_id = open(f"{ART}/run_id.txt").read().strip()
with mlflow.start_run(run_id=run_id):
    mlflow.log_metrics({
        "test_ce_dirichlet":      cross_entropy(Y_test, p_dir),
        "test_ce_dirichlet_pp":   pp_ce,
        "test_ce_rf":             cross_entropy(Y_test, P_rf),
        "test_ce_flat":           cross_entropy(Y_test, P_flat),
        "test_rmse_dirichlet":    rmse(Y_test, p_dir),
        "test_rmse_rf":           rmse(Y_test, P_rf),
        "test_bloc_acc_dirichlet": bloc_accuracy(Y_test, p_dir),
        "test_bloc_acc_rf":        bloc_accuracy(Y_test, P_rf),
        "coverage_90_overall":    cov_all,
    })
    for comp_name, c in zip(COMP, cov_comp):
        mlflow.log_metric(f"coverage_90_{comp_name}", float(c))
    mlflow.log_artifact(f"{ART}/comparison.csv")
    mlflow.log_artifact(f"{ART}/calibration.csv")
print("logged eval to run", run_id)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
