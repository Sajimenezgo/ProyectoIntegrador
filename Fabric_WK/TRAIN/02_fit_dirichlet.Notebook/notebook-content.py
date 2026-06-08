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

# # 02 · Fit Dirichlet (cmdstanpy)
# 
# Stage 2. Bootstraps CmdStan, compiles the QR-reparameterized Dirichlet regression,
# samples, runs diagnostics, builds the coefficient table, and persists the posterior.
# Heavy step — first run builds CmdStan (~10 min). Depends on NB1; runs in parallel with NB3.

# CELL ********************

%pip install cmdstanpy arviz

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

import cmdstanpy
from cmdstanpy import install_cmdstan, set_cmdstan_path, cmdstan_path, CmdStanModel
import arviz as az, mlflow

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import multiprocessing
from cmdstanpy import install_cmdstan
install_cmdstan(dir="/tmp/cmdstan", cores=multiprocessing.cpu_count(),
                overwrite=True, verbose=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- Load train data -> stan_data ------------------------------------------
train = spark.read.table("model_train").toPandas()
X_train = train[PREDS].to_numpy(float)
Y_train = train[RESP_ADJ].to_numpy(float)                 # zero-replaced (open simplex)
Y_train = Y_train / Y_train.sum(axis=1, keepdims=True)
stan_data = {"N": int(X_train.shape[0]), "K": K, "P": int(X_train.shape[1]),
             "y": Y_train.tolist(), "X": X_train.tolist()}
print("N:", stan_data["N"], "P:", stan_data["P"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from cmdstanpy import set_cmdstan_path, cmdstan_path
set_cmdstan_path("/tmp/cmdstan/cmdstan-2.39.0")
print(cmdstan_path())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

STAN_CODE = r"""
data {
  int<lower=1> N;
  int<lower=2> K;
  int<lower=1> P;
  array[N] simplex[K] y;
  matrix[N, P] X;
}
transformed data {
  // thin QR reparameterization; scale so Q_ast / R_ast are unit-ish
  matrix[N, P] Q_ast = qr_thin_Q(X) * sqrt(N - 1);
  matrix[P, P] R_ast = qr_thin_R(X) / sqrt(N - 1);
  matrix[P, P] R_ast_inverse = inverse(R_ast);
}
parameters {
  vector[K] beta0;        // per-component intercepts (original scale)
  matrix[P, K] theta;     // slopes in QR space, per component
}
transformed parameters {
  array[N] vector<lower=0>[K] alpha;
  for (i in 1:N)
    for (k in 1:K)
      alpha[i][k] = exp(beta0[k] + dot_product(Q_ast[i], theta[, k]));
}
model {
  beta0 ~ normal(0, 5);
  to_vector(theta) ~ normal(0, 2);   // weakly-informative in rotated space
  for (i in 1:N)
    y[i] ~ dirichlet(alpha[i]);
}
generated quantities {
  matrix[P, K] beta;                 // interpretable slopes (standardized-predictor scale)
  vector[N] log_lik;
  for (k in 1:K) beta[, k] = R_ast_inverse * theta[, k];
  for (i in 1:N) log_lik[i] = dirichlet_lpdf(y[i] | alpha[i]);
}
"""

stan_path = f"{ART}/dirichlet_reg.stan"
with open(stan_path, "w") as f:
    f.write(STAN_CODE)
print("wrote", stan_path)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- Compile + sample ------------------------------------------------------
model = CmdStanModel(stan_file=stan_path)
fit = model.sample(data=stan_data, chains=4, parallel_chains=4,
                   iter_warmup=1000, iter_sampling=1000,
                   seed=SEED, adapt_delta=0.9, max_treedepth=12, show_progress=True)
print(fit.diagnose())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- ArviZ idata + traceplots ----------------------------------------------
idata = az.from_cmdstanpy(fit, log_likelihood="log_lik")
az.plot_trace(idata, var_names=["beta0", "beta"], compact=True)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- Coefficient table (recovered beta, standardized-predictor scale) ------
beta_draws = fit.stan_variable("beta")    # (S, P, K)
b0_draws   = fit.stan_variable("beta0")   # (S, K)

rhat_b = az.rhat(idata, var_names=["beta"])["beta"].values
essb_b = az.ess(idata, var_names=["beta"], method="bulk")["beta"].values
esst_b = az.ess(idata, var_names=["beta"], method="tail")["beta"].values
rhat_0 = az.rhat(idata, var_names=["beta0"])["beta0"].values
essb_0 = az.ess(idata, var_names=["beta0"], method="bulk")["beta0"].values
esst_0 = az.ess(idata, var_names=["beta0"], method="tail")["beta0"].values

rows = []
for k, comp in enumerate(COMP):
    v = b0_draws[:, k]
    rows.append({"param": "intercept", "predictor": "(intercept)", "component": comp,
                 "mean": float(v.mean()), "sd": float(v.std()),
                 "hdi_3": float(np.quantile(v, 0.03)), "hdi_97": float(np.quantile(v, 0.97)),
                 "r_hat": float(rhat_0[k]), "ess_bulk": float(essb_0[k]), "ess_tail": float(esst_0[k])})
for j, pred in enumerate(PREDS):
    for k, comp in enumerate(COMP):
        v = beta_draws[:, j, k]
        rows.append({"param": "slope", "predictor": pred, "component": comp,
                     "mean": float(v.mean()), "sd": float(v.std()),
                     "hdi_3": float(np.quantile(v, 0.03)), "hdi_97": float(np.quantile(v, 0.97)),
                     "r_hat": float(rhat_b[j, k]), "ess_bulk": float(essb_b[j, k]), "ess_tail": float(esst_b[j, k])})
coef = pd.DataFrame(rows)
coef.to_csv(f"{ART}/coef_table.csv", index=False)
coef

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- Sanity: intercept-implied proportions vs train marginal means ---------
a0 = np.exp(b0_draws.mean(0)); softmax0 = a0 / a0.sum()
print("intercept-implied proportions:", np.round(softmax0, 3))
print("train marginal observed means:", np.round(Y_train.mean(0), 3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- Persist posterior + log to MLflow -------------------------------------
nc_path = f"{ART}/posterior.nc"
idata.to_netcdf(nc_path)

num_div      = int(idata.sample_stats["diverging"].sum())
max_rhat     = float(np.nanmax([rhat_b.max(), rhat_0.max()]))
min_ess_bulk = float(np.nanmin([essb_b.min(), essb_0.min()]))
min_ess_tail = float(np.nanmin([esst_b.min(), esst_0.min()]))
print({"num_divergent": num_div, "max_rhat": max_rhat,
       "min_ess_bulk": min_ess_bulk, "min_ess_tail": min_ess_tail})

mlflow.set_experiment(EXPERIMENT)
run_id = open(f"{ART}/run_id.txt").read().strip()
with mlflow.start_run(run_id=run_id):
    mlflow.log_params({"chains": 4, "iter_warmup": 1000, "iter_sampling": 1000,
                       "adapt_delta": 0.9, "max_treedepth": 12,
                       "prior_beta0": "normal(0,5)", "prior_theta": "normal(0,2)",
                       "reparam": "thin_QR"})
    mlflow.log_metrics({"num_divergent": num_div, "max_rhat": max_rhat,
                        "min_ess_bulk": min_ess_bulk, "min_ess_tail": min_ess_tail})
    for p in [stan_path, nc_path, f"{ART}/coef_table.csv"]:
        mlflow.log_artifact(p)
print("logged fit to run", run_id)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
