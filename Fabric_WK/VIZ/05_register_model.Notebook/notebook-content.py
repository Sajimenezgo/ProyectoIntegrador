# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# # 05 · Register model (pyfunc)
# 
# Stage 5 (optional, for serving). Builds a **slim** posterior artifact (just the `beta0`/`beta`
# draws as a compressed `.npz`, no `log_lik` bloat), wraps scaler + draws in an `mlflow.pyfunc`
# model with a `predict()` that returns point shares **and** 90% predictive bands, and registers it.
# A front end can then load one object and call `.predict(raw_features_df)`.

# CELL ********************

%pip install mlflow arviz scikit-learn

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

import arviz as az, mlflow, joblib
import mlflow.pyfunc

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- Slim artifact: keep only the draws the predictor needs ----------------
post = az.from_netcdf(f"{ART}/posterior.nc").posterior
ex   = [d for d in post["beta"].dims  if d not in ("chain", "draw")]
ex0  = [d for d in post["beta0"].dims if d not in ("chain", "draw")]
beta = post["beta"].stack(s=("chain", "draw")).transpose("s", *ex).values    # (S, P, K)
b0   = post["beta0"].stack(s=("chain", "draw")).transpose("s", *ex0).values  # (S, K)
np.savez_compressed(f"{ART}/posterior_slim.npz",
                    beta=beta, beta0=b0,
                    preds=np.array(PREDS), comp=np.array(COMP))
print("slim draws:", beta.shape, b0.shape)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- pyfunc model: scaler + draws -> shares + 90% predictive bands ----------
class DirichletECV(mlflow.pyfunc.PythonModel):
    def load_context(self, ctx):
        import joblib, numpy as np
        self.scaler = joblib.load(ctx.artifacts["scaler"])
        d = np.load(ctx.artifacts["draws"], allow_pickle=True)
        self.beta  = d["beta"]; self.b0 = d["beta0"]
        self.preds = [str(x) for x in d["preds"]]
        self.comp  = [str(x) for x in d["comp"]]
        self.rng   = np.random.default_rng(2024)

    def predict(self, ctx, model_input):
        import numpy as np, pandas as pd
        X = model_input[self.preds].to_numpy(float)              # RAW features (16 cols)
        Xs = self.scaler.transform(X)
        alpha = np.exp(np.einsum("ip,spk->sik", Xs, self.beta) + self.b0[:, None, :])
        m = alpha / alpha.sum(2, keepdims=True)
        p = m.mean(0)                                            # posterior-mean shares
        yrep = self.rng.standard_gamma(alpha); yrep /= yrep.sum(2, keepdims=True)
        lo, hi = np.quantile(yrep, 0.05, 0), np.quantile(yrep, 0.95, 0)  # 90% predictive
        out = pd.DataFrame(p, columns=[f"p_{c}" for c in self.comp])
        for j, c in enumerate(self.comp):
            out[f"p_{c}_lo90"] = lo[:, j]; out[f"p_{c}_hi90"] = hi[:, j]
        return out

# raw-feature input example (NOT the standardized model_train table)
example = spark.sql(f"SELECT * FROM {SOURCE_TABLE} LIMIT 3").toPandas()[PREDS]

mlflow.set_experiment(EXPERIMENT)
run_id = open(f"{ART}/run_id.txt").read().strip()
with mlflow.start_run(run_id=run_id):
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=DirichletECV(),
        artifacts={"scaler": f"{ART}/scaler.joblib", "draws": f"{ART}/posterior_slim.npz"},
        input_example=example,
        pip_requirements=["numpy", "pandas", "scikit-learn", "joblib"],
        registered_model_name=REGISTERED_NAME,
    )
print("logged model:", info.model_uri)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# --- Verify: reload and predict on the example -----------------------------
reloaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
print(reloaded.predict(example).round(3).to_string(index=False))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
