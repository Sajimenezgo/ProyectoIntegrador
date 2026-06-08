# Fabric notebook source


# MARKDOWN ********************

# # 06 · Batch score
# 
# Stage 6 (serving example). Loads the registered model, scores a table of **raw** ECV features,
# and writes shares + 90% bands to a Delta table (`predictions_dirichlet`) that a front end / report
# can read. Swap `SOURCE_TABLE` for whatever incoming table holds the rows you want predicted.

# CELL ********************

%pip install mlflow

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

import mlflow

# CELL ********************

# --- Load model + score raw features ---------------------------------------
new = spark.sql(f"SELECT * FROM {SOURCE_TABLE}").toPandas()      # incoming rows to predict
run_id = open(f"{ART}/run_id.txt").read().strip()
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
# registry alternative once versions exist:
#   model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_NAME}/<version>")

preds = model.predict(new[PREDS])                               # pass RAW features
result = pd.concat([new[IDS].reset_index(drop=True),
                    preds.reset_index(drop=True)], axis=1)

(spark.createDataFrame(result).write.mode("overwrite")
      .option("overwriteSchema", "true").format("delta").saveAsTable("predictions_dirichlet"))
result.to_csv(f"{ART}/predictions_dirichlet.csv", index=False)
print("wrote table predictions_dirichlet and CSV")
print(result.head().to_string(index=False))
