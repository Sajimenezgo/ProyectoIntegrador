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

# # 03 · Baselines
# 
# Stage 3 (parallel with NB2). Fits the random-forest predictive baseline and the flat
# `Dirichlet(1,1,1)` reference, and persists their test-set predictions. Depends only on NB1.

# CELL ********************

%pip install scikit-learn

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

from sklearn.ensemble import RandomForestRegressor
import joblib

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

train = spark.read.table("model_train").toPandas()
test  = spark.read.table("model_test").toPandas()
X_train = train[PREDS].to_numpy(float); Y_train = train[RESP].to_numpy(float)
X_test  = test[PREDS].to_numpy(float)

# Baseline 1: random forest (native multi-output), predictions projected to the simplex
rf = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)
rf.fit(X_train, Y_train)
pred_rf = np.clip(rf.predict(X_test), 0, None)
pred_rf = pred_rf / pred_rf.sum(axis=1, keepdims=True)

# Baseline 2: flat Dirichlet(1,1,1) -> uniform (1/3,1/3,1/3); CE == log 3
pred_flat = np.full((len(test), K), 1.0 / K)

joblib.dump(rf, f"{ART}/rf.joblib")
ids = test[IDS].reset_index(drop=True)
pd.concat([ids, pd.DataFrame(pred_rf,   columns=RESP)], axis=1).to_csv(f"{ART}/pred_rf.csv",   index=False)
pd.concat([ids, pd.DataFrame(pred_flat, columns=RESP)], axis=1).to_csv(f"{ART}/pred_flat.csv", index=False)
print("wrote rf.joblib, pred_rf.csv, pred_flat.csv")
print("RF rows sum to 1:", bool(np.allclose(pred_rf.sum(1), 1.0)))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
