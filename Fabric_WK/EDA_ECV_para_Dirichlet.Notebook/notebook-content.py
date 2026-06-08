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

# # EDA — ECV (gold) & Votaciones por comuna
# **Objetivo:** análisis exploratorio del *dataset gold* a nivel `(YEAR, COMUNA_STD)` y de la tabla de proporciones electorales `votaciones_proporciones_comunas`, con el fin de **identificar features útiles, colinealidad, drift temporal y supuestos para la regresión Dirichlet** (outcome = proporción left/center/right ∈ Δ²).
# 
# **Estructura:**
# 
# | Sección | Tema |
# |---|---|
# | 0 | Setup, helpers, carga de datos |
# | 1 | EDA de la *outcome* (votaciones) — composición simplex, transformaciones ALR |
# | 2 | EDA de los *features* (ecv_gold) — distribuciones, drift, missingness |
# | 3 | JOIN outcome ↔ features, scatter y correlaciones |
# | 4 | Multicolinealidad: heatmap, PCA, VIF |
# | 5 | Ideas de feature engineering (interacciones, ratios, transformaciones) |
# | 6 | Outliers, leverage y casos frontera |
# | 7 | Chequeos específicos para Dirichlet (boundary, zero-inflation) |
# 
# ** Exportación de gráficos:** Exportar individualmente (cada gráfico) tiene mediante el `# save_fig(fig, "nombre")` comentada justo antes de `plt.show()`. Descomentar para guardar como .png en `Files/EDA_graficos/`. 
# 
# Exportación global de las graficas a /Files/EDA_graficos, solo con `EXPORT_ALL = True` en celda 0 (Setup).


# MARKDOWN ********************

# ## 0 · Setup

# MARKDOWN ********************

# Carga de datos y normalización de nombres entre los datasets (votos y gold de ECV)

# CELL ********************

# 0.1 — Imports y configuración global
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.gridspec import GridSpec
from itertools import combinations

# Configuración visual
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Para el export global 
# False aún no exportar
# Crear carpeta
EXPORT_ALL = True   
EDA_EXPORT_DIR = "/lakehouse/default/Files/EDA_graficos"
os.makedirs(EDA_EXPORT_DIR, exist_ok=True)

def save_fig(fig, name, dpi=160):
    """Guardar figura como PNG en Files/EDA_graficos/."""
    path = os.path.join(EDA_EXPORT_DIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  ✔ guardado: {path}")

def maybe_save(fig, name):
    """Guardar sólo si EXPORT_ALL está activo"""
    if EXPORT_ALL:
        save_fig(fig, name)

print(f"Setup listo. Export dir: {EDA_EXPORT_DIR}")
print(f"EXPORT_ALL = {EXPORT_ALL}  (si True: guardar los plots")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 0.2 — Carga de datos: gold (features / pre-features por ahora) + votaciones (outcome)
# desde delta tablas a .Pandas
gold = spark.sql("SELECT * FROM Data_LakeHouse.dbo.ecv_gold_promedios_comuna").toPandas()
votos = spark.read.format("delta").load("Tables/dbo/votaciones_proporciones_comunas").toPandas()

print(f"GOLD:    {gold.shape}  ·  años: {sorted(gold['YEAR'].unique())}")
print(f"VOTOS:   {votos.shape}  ·  columnas: {votos.columns.tolist()}")

display(gold.head(3))
display(votos.head(3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 0.3 — Normalizar nombres de columnas de la tabla de votos
# Rename a p_ de proporción + redondeo a 4 decimales.

votos = votos.rename(columns={
    "izquierda": "p_left",
    "centro":    "p_center",
    "derecha":   "p_right",
})

# Reordenar y redondear proporciones a 4 decimales
votos = votos[["anio", "corporacion", "codigo_comuna", "p_left", "p_center", "p_right"]].copy()
votos[["p_left", "p_center", "p_right"]] = votos[["p_left", "p_center", "p_right"]].round(4)

print("Votos normalizado:")
display(votos.head())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Ahora debemos matchear ambas columnas referentes a comuna, además confirmando su adecuada etiqueta

# CELL ********************

from pyspark.sql import functions as F

mapeo = (
    spark.table("mmv_bronze")
         .filter(F.col("codigo_municipio") == "001")   # Codigo Mde
         .groupBy("codigo_comuna", "comuna")
         .count()
         .orderBy("codigo_comuna", F.desc("count"))
)
display(mapeo)

# 2. Sólo los códigos altos (17–21) que necesitamos validar
display(
    mapeo.filter(F.col("codigo_comuna").isin("17","18","19","20","21","00"))
         .orderBy("codigo_comuna", F.desc("count"))
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 0.4 — Corrigiendo códigos de comuna entre VOTOS (string '00'-'21') y GOLD (int 1-16, 50-90)

#   '00'           → total agregado / sin asignar → DESCARTAR
#   '01' a '16'    → comunas 1-16
#   '17' a '21'    → corregimientos 50 a 90, reasorteados cada coincidir ambas numeraciones

# Mapeo string → código oficial de comuna
votos_to_gold = {
    "00": None,   # descartar
    "01": 1,  "02": 2,  "03": 3,  "04": 4,  "05": 5,
    "06": 6,  "07": 7,  "08": 8,  "09": 9,  "10": 10,
    "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16,
    "17": 70,   # Altavista 
    "18": 80,   # San Antonio de prado
    "19": 50,   # Palmitas
    "20": 60,   # San Cristobal
    "21": 90,  # Santa Elena
}

# Asegurar string padded para el lookup (por si entra como '1' en vez de '01')
votos["codigo_comuna"] = votos["codigo_comuna"].astype(str).str.zfill(2)
votos["codigo_comuna_gold"] = votos["codigo_comuna"].map(votos_to_gold)

# Descartar el código '00' (sin contraparte en gold)
n_drop = votos["codigo_comuna_gold"].isna().sum()
print(f"Filas descartadas (código '00' o sin mapeo): {n_drop}")
votos = votos.dropna(subset=["codigo_comuna_gold"]).copy()
votos["codigo_comuna_gold"] = votos["codigo_comuna_gold"].astype(int)

# Reemplazar la columna original con el código gold
votos = votos.drop(columns=["codigo_comuna"]).rename(columns={"codigo_comuna_gold": "codigo_comuna"})
votos = votos[["anio", "corporacion", "codigo_comuna", "p_left", "p_center", "p_right"]]

# Extraer código numérico de COMUNA_STD ("01 - Popular" → 1)
gold["codigo_comuna"] = (
    gold["COMUNA_STD"].astype(str).str.extract(r"^(\d+)").astype(float).astype("Int64")
)
gold = gold.rename(columns={"YEAR": "anio"})

# Verificación
codigos_gold = sorted(gold["codigo_comuna"].dropna().unique().tolist())
codigos_votos = sorted(votos["codigo_comuna"].unique().tolist())
print(f"\nCódigos en GOLD:  {codigos_gold}")
print(f"Códigos en VOTOS: {codigos_votos}")

solo_en_gold  = set(codigos_gold) - set(codigos_votos)
solo_en_votos = set(codigos_votos) - set(codigos_gold)
en_ambos      = set(codigos_gold) & set(codigos_votos)
print(f"\n✓ En ambos:       {sorted(en_ambos)}  ({len(en_ambos)} comunas/corregimientos)")
print(f"⚠ Solo en GOLD:   {sorted(solo_en_gold)}")
print(f"⚠ Solo en VOTOS:  {sorted(solo_en_votos)}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

spark.createDataFrame(votos).write.format("delta").mode("overwrite").saveAsTable("Data_LakeHouse.dbo.votos_Silver_matching")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 0.5 — Lista canónica de features del modelo
FEATURES = [
    "ESTRATO_STD",
    "EDUCACION_STD",
    "ALFABETISMO",
    "DIM_RECREACION",
    "IND_DISCRIMINACION",
    "IND_BRECHA_PERCEPCION",
    "DIM_PARTICIPACION",
    "DIM_MEDIO_AMBIENTE",
    "DIM_MOVILIDAD",
    "SATISFACCION_MUNICIPIO",
    "CONFIANZA_INSTITUCIONES",
    "CUMPLIMIENTO_NORMAS",
    "DIM_SEGURIDAD_ALIMENTARIA",
    "DIM_CAPITAL_FISICO",
    "INDICE_CARENCIA_SERVICIOS",
    "IND_HACINAMIENTO",
]
# Filtrar a las que efectivamente están en gold
FEATURES = [c for c in FEATURES if c in gold.columns]
print(f"{len(FEATURES)} features disponibles:")
for f in FEATURES:
    print(f"  · {f}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 1 · EDA de la outcome (votaciones)
# 
# La regresión Dirichlet trabaja sobre el simplex Δ² = `{(l,c,r) : l+c+r=1, l,c,r ≥ 0}`. Antes de modelar necesitamos verificar:
# 
# 1. **Cierre del simplex:** ¿suman las 3 proporciones a 1?
# 2. **Composición típica:** ¿qué tan dispersas/balanceadas están las observaciones?
# 3. **Heterogeneidad por `corporacion`:** ¿el mismo comuna-año vota distinto según el tipo de elección?
# 4. **Casos frontera:** observaciones con alguna proporción = 0 o = 1 rompen la verosimilitud Dirichlet (log de 0). Hay que detectarlas y decidir tratamiento (ε-bumping, modelo zero-inflated, o agregar votos).
# 5. **Trayectorias temporales:** ¿cómo se mueve cada comuna entre olas?

# CELL ********************

# 1.1 — Verificación del cierre del valor de proporciones (simplx)
votos["suma_proporciones"] = votos[["p_left", "p_center", "p_right"]].sum(axis=1)
print("Estadísticas de la suma de proporciones (debería ≈ 1):")
print(votos["suma_proporciones"].describe().round(4))

# Tolerancia: cualquier cosa > 1.01 o < 0.99 es sospechosa (votos a otros candidatos, blancos, etc.)
fuera = votos[(votos["suma_proporciones"] < 0.99) | (votos["suma_proporciones"] > 1.01)]
print(f"\nFilas fuera de [0.99, 1.01]: {len(fuera)} / {len(votos)}")
if len(fuera):
    display(fuera.head(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 1.2 — Distribución univariada de cada proporción + frecuencia de casos frontera
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
colors = {"p_left": "#D62728", "p_center": "#FFD500", "p_right": "#1F77B4"}
labels = {"p_left": "Izquierda", "p_center": "Centro", "p_right": "Derecha"}

for ax, col in zip(axes, ["p_left", "p_center", "p_right"]):
    ax.hist(votos[col].dropna(), bins=30, color=colors[col], edgecolor="white", alpha=0.85)
    ax.axvline(votos[col].mean(), color="black", linestyle="--", linewidth=1, label=f"μ={votos[col].mean():.2f}")
    ax.set_title(f"Distribución · {labels[col]}")
    ax.set_xlabel("proporción")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
fig.suptitle("Distribución de proporciones de voto por comuna-año-corporación", y=1.02)
# save_fig(fig, "01_distrib_proporciones_voto")
maybe_save(fig, "01_distrib_proporciones_voto")
plt.show()

# Contar casos frontera
print("\nObservaciones con proporción exacta = 0 o = 1 (problemáticas para Dirichlet):")
for col in ["p_left","p_center","p_right"]:
    n_zero = (votos[col] == 0).sum()
    n_one = (votos[col] == 1).sum()
    print(f"  {col}: {n_zero} ceros · {n_one} unos")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 1.3 — Boxplots por corporación
# ¿Cambian las proporciones para cada tipo de elección (entre corporaciones)?
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["p_left", "p_center", "p_right"]):
    sns.boxplot(data=votos, x="corporacion", y=col, ax=ax, color=colors[col])
    ax.set_title(labels[col])
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)
fig.suptitle("Composición de voto por corporación electoral", y=1.02)
# save_fig(fig, "02_boxplot_por_corporacion")
maybe_save(fig, "02_boxplot_por_corporacion")
plt.show()

# Tabla resumen
print("\nMedia de proporciones por corporación:")
display(
    votos.groupby("corporacion")[["p_left","p_center","p_right"]]
         .mean()
         .round(3)
         .sort_values("p_right")
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 1.4 — Diagrama ternario 
# Visualización 2D del simplex Δ²
# Proyección barycéntrica: cada punto (l,c,r) ->
#a  posición en triángulo equilátero
fig, ax = plt.subplots(figsize=(7, 6.5))

# Vértices del triángulo equilátero
sqrt3_2 = np.sqrt(3) / 2
V_left   = np.array([0.0, 0.0])
V_right  = np.array([1.0, 0.0])
V_center = np.array([0.5, sqrt3_2])

def baryc(l, c, r):
    return l * V_left + c * V_center + r * V_right

# Dibujar triángulo
tri = plt.Polygon([V_left, V_right, V_center], fill=False, edgecolor="black", linewidth=1.5)
ax.add_patch(tri)

# Líneas de cuadrícula interior (cada 0.2)
for s in np.arange(0.2, 1.0, 0.2):
    # líneas paralelas a cada lado
    p1 = baryc(s, 0, 1-s);   p2 = baryc(s, 1-s, 0)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="lightgray", linewidth=0.5)
    p1 = baryc(0, s, 1-s);   p2 = baryc(1-s, s, 0)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="lightgray", linewidth=0.5)
    p1 = baryc(0, 1-s, s);   p2 = baryc(1-s, 0, s)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="lightgray", linewidth=0.5)

# Plotear cada observación, color por corporación
corp_colors = {c: plt.cm.tab10(i) for i, c in enumerate(votos["corporacion"].unique())}
for corp, sub in votos.groupby("corporacion"):
    pts = sub[["p_left","p_center","p_right"]].apply(lambda r: baryc(*r), axis=1, result_type="expand")
    ax.scatter(pts[0], pts[1], s=22, alpha=0.65, label=corp, color=corp_colors[corp], edgecolor="white", linewidth=0.4)

# Etiquetas de los vértices
ax.text(V_left[0]-0.04, V_left[1]-0.05, "Izquierda", fontsize=11, fontweight="bold", color="#D62728", ha="right")
ax.text(V_right[0]+0.04, V_right[1]-0.05, "Derecha", fontsize=11, fontweight="bold", color="#1F77B4", ha="left")
ax.text(V_center[0], V_center[1]+0.04, "Centro", fontsize=11, fontweight="bold", color="#B59500", ha="center")

ax.set_xlim(-0.15, 1.15)
ax.set_ylim(-0.12, 1.05)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Simplex Δ² · cada punto = una comuna-año-corporación", pad=10)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
# save_fig(fig, "03_ternario_simplex")
maybe_save(fig, "03_ternario_simplex")
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 1.5 — Trayectoria temporal por comuna 
# ¿Tendecias se sostienen? o ¿se mueven cada año?
# Como referencia p_left
# Una línea por (cada corporacion x comuna) a lo largo de los años 
# con 4 puntos de los años electorales (2018, 2019, 2022, 2023)
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

for ax, col in zip(axes, ["p_left", "p_center", "p_right"]):
    for (cod, corp), sub in votos.groupby(["codigo_comuna","corporacion"]):
        sub = sub.sort_values("anio")
        if len(sub) > 1:
            ax.plot(sub["anio"], sub[col], alpha=0.55, marker="o", markersize=4, linewidth=0.9)
    ax.set_ylabel(labels[col])
    ax.set_ylim(0, 1)
    ax.set_title(f"Trayectoria de {labels[col]} · cada línea = (comuna × corporación)")
axes[-1].set_xlabel("Año")
fig.tight_layout()
# save_fig(fig, "04_trayectorias_comuna_corporacion")
maybe_save(fig, "04_trayectorias_comuna_corporacion")
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 1.6 — Transformación ALR (additive log-ratio)
# Usa p_center como referencia.
# alr(l,c,r) = (log(l/c), log(r/c))
EPS = 1e-4  # ε-bumping para evitar log(0)

def alr_transform(df, ref="p_center"):
    out = df.copy()
    cols = ["p_left","p_center","p_right"]
    for c in cols:
        out[c] = out[c].clip(EPS, 1 - EPS)
    others = [c for c in cols if c != ref]
    for c in others:
        out[f"alr_{c.replace('p_','')}"] = np.log(out[c] / out[ref])
    return out

votos_alr = alr_transform(votos)
print("Transformación ALR aplicada (referencia: centro). Nuevas columnas:")
print([c for c in votos_alr.columns if c.startswith("alr_")])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(votos_alr["alr_left"].dropna(),  bins=30, ax=axes[0], color="#D62728")
axes[0].set_title("ALR(izquierda / centro)")
axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
sns.histplot(votos_alr["alr_right"].dropna(), bins=30, ax=axes[1], color="#1F77B4")
axes[1].set_title("ALR(derecha / centro)")
axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
fig.suptitle("Outcome en coordenadas ALR (log-ratio vs. centro)", y=1.02)
# save_fig(fig, "05_alr_distribuciones")
maybe_save(fig, "05_alr_distribuciones")
plt.show()

print(f"\nNOTA: se aplicó ε={EPS} para evitar log(0). Revisar cuántas observaciones fueron clipadas.")
n_clip = ((votos[["p_left","p_center","p_right"]] < EPS) | (votos[["p_left","p_center","p_right"]] > 1-EPS)).any(axis=1).sum()
print(f"Observaciones afectadas por el clip: {n_clip} de {len(votos)}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 1.7 — Heatmap: elección (anio + corporación) × comuna, por ideología
# Cada panel: una proporción; cada celda: proporción en esa elección×comuna.
# Permite ver consistencia de voto a través de elecciones.

votos["eleccion"] = votos["anio"].astype(str) + " · " + votos["corporacion"]

orden_comunas = sorted(votos["codigo_comuna"].unique())
orden_elecciones = (
    votos[["anio","corporacion","eleccion"]]
        .drop_duplicates()
        .sort_values(["anio","corporacion"])["eleccion"]
        .tolist()
)

fig, axes = plt.subplots(3, 1, figsize=(max(10, 0.55*len(orden_comunas)),
                                          0.5*len(orden_elecciones)*3 + 2))
configs = [
    ("p_left",   "Izquierda", "OrRd"),
    ("p_center", "Centro",    "Greens"),
    ("p_right",  "Derecha",   "Blues"),
]

for ax, (col, label, cmap) in zip(axes, configs):
    pivot = (
        votos.pivot_table(index="eleccion", columns="codigo_comuna",
                          values=col, aggfunc="mean")
             .reindex(index=orden_elecciones, columns=orden_comunas)
    )
    sns.heatmap(
        pivot, ax=ax, cmap=cmap, vmin=0, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": f"proporción · {label}", "shrink": 0.7},
    )
    ax.set_title(f"{label} · proporción por elección y comuna",
                 fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel("Comuna" if col == "p_right" else "")
    ax.set_ylabel("Elección")
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    ax.tick_params(axis="x", rotation=0, labelsize=8)

fig.suptitle(
    "Consistencia del voto por comuna a través de elecciones",
    y=1.005, fontsize=13, fontweight="bold"
)
fig.tight_layout()
# save_fig(fig, "06_heatmap_eleccion_x_comuna")
maybe_save(fig, "06_heatmap_eleccion_x_comuna")
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 2 · EDA de los features (ecv_gold)
# 
# A nivel `(YEAR, COMUNA_STD)`. Las preguntas guía:
# 
# 1. **¿Hay missingness sistemático?** Una comuna sin muestra ese año romperá el modelo.
# 2. **¿Qué tan estables son los features en el tiempo?** Una variable que oscila violentamente puede ser ruido; una que es constante no aporta varianza para modelar.
# 3. **¿Hay drift entre olas (2018↔2023)?** Si una pregunta cambió de escala, lo veremos.
# 4. **¿Qué variables tienen poca varianza entre comunas?** Drop candidates: si todas las comunas tienen ~5 en una escala 1–5, esa variable no discrimina.

# CELL ********************

# 2.1 — Mapa de missingness: comuna × año × feature
missing_matrix = gold.set_index(["anio","COMUNA_STD"])[FEATURES].isna().astype(int)

fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(missing_matrix.T, cmap="Reds", cbar_kws={"label":"1 = NaN"}, 
            yticklabels=True, xticklabels=False, ax=ax, linewidths=0.1, linecolor="lightgray")
ax.set_title("Patrón de missingness · cada celda = (feature × comuna-año)")
ax.set_xlabel("Comuna-año (ordenado)")
ax.set_ylabel("Feature")
# save_fig(fig, "07_missingness_heatmap")
maybe_save(fig, "07_missingness_heatmap")
plt.show()

print("\nPorcentaje de missing por feature:")
miss_pct = (gold[FEATURES].isna().mean() * 100).round(2).sort_values(ascending=False)
display(miss_pct.to_frame("% missing"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 2.2 — Distribuciones univariadas (un panel por feature)
n = len(FEATURES)
ncols = 4
nrows = int(np.ceil(n/ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.8*nrows))
axes = axes.flatten()

for i, f in enumerate(FEATURES):
    ax = axes[i]
    data = gold[f].dropna()
    ax.hist(data, bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(data.mean(), color="black", linestyle="--", linewidth=1)
    cv = (data.std() / abs(data.mean())) if data.mean() != 0 else np.nan
    ax.set_title(f"{f}\n(μ={data.mean():.2f}, CV={cv:.2f})", fontsize=9)
    ax.tick_params(labelsize=8)
for j in range(i+1, len(axes)):
    axes[j].axis("off")
fig.suptitle("Distribución de cada feature a nivel comuna-año", y=1.005, fontsize=12, fontweight="bold")
fig.tight_layout()
# save_fig(fig, "08_distrib_features")
maybe_save(fig, "08_distrib_features")
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 2.3 — Drift temporal: observar movimiento de variables a lo largo de los años
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.8*nrows))
axes = axes.flatten()

years = sorted(gold["anio"].unique())
year_palette = sns.color_palette("viridis", n_colors=len(years))

for i, f in enumerate(FEATURES):
    ax = axes[i]
    for y, color in zip(years, year_palette):
        data = gold.loc[gold["anio"] == y, f].dropna()
        if len(data) > 1:
            ax.hist(data, bins=12, alpha=0.45, label=str(y), color=color, edgecolor="none")
    ax.set_title(f, fontsize=9)
    ax.tick_params(labelsize=8)
    if i == 0:
        ax.legend(fontsize=7, title="Año")
for j in range(i+1, len(axes)):
    axes[j].axis("off")
fig.suptitle("Drift temporal por feature (2018 → 2023)", y=1.005, fontsize=12, fontweight="bold")
fig.tight_layout()
# save_fig(fig, "09_drift_temporal")
maybe_save(fig, "09_drift_temporal")
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 2.4 — Varianza entre-comuna vs. dentro-comuna
# Si var(between) >> var(within): la variable distingue comunas → buen feature
# Si var(within) >> var(between): la variable cambia más en el tiempo → quizá modelar como diff
def variance_decomp(df, feature, group_col="codigo_comuna"):
    grand_mean = df[feature].mean()
    group_means = df.groupby(group_col)[feature].mean()
    n_per = df.groupby(group_col)[feature].count()
    var_between = ((group_means - grand_mean)**2 * n_per).sum() / max(n_per.sum() - 1, 1)
    var_within  = df.groupby(group_col)[feature].apply(lambda x: ((x - x.mean())**2).sum()).sum() / max(n_per.sum() - 1, 1)
    return var_between, var_within

decomp = []
for f in FEATURES:
    vb, vw = variance_decomp(gold.dropna(subset=[f]), f)
    decomp.append({"feature": f, "var_between": vb, "var_within": vw, 
                   "ratio_between_total": vb/(vb+vw) if (vb+vw)>0 else np.nan})
decomp = pd.DataFrame(decomp).sort_values("ratio_between_total", ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(decomp))
ax.barh(y_pos, decomp["ratio_between_total"], color="#4C72B0", edgecolor="white")
ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="50% entre comunas")
ax.set_yticks(y_pos)
ax.set_yticklabels(decomp["feature"], fontsize=8)
ax.set_xlabel("Var(entre comunas) / Var(total)")
ax.set_xlim(0, 1)
ax.set_title("¿Qué tanto de la varianza de cada feature está entre comunas?\n(más alto = mejor para distinguir comunas)")
ax.legend()
# save_fig(fig, "10_descomposicion_varianza")
maybe_save(fig, "10_descomposicion_varianza")
plt.show()

display(decomp.round(3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 2.5 — Perfil por comuna
# Cada perfil en relación a z-score de cada variable
# En heatmap
# Cada fila = comuna, cada columna = feature. Z-score sobre todas las (comuna,año).
gold_features_only = gold[["codigo_comuna","COMUNA_STD","anio"] + FEATURES].copy()
mean_per_comuna = gold_features_only.groupby("COMUNA_STD")[FEATURES].mean()

# Z-score por columna
zscored = (mean_per_comuna - mean_per_comuna.mean()) / mean_per_comuna.std()

# Ordenar comunas por una proxy de "perfil socioeconómico" (estrato promedio)
order = mean_per_comuna["ESTRATO_STD"].sort_values().index if "ESTRATO_STD" in mean_per_comuna.columns else zscored.index
zscored = zscored.loc[order]

fig, ax = plt.subplots(figsize=(11, max(6, 0.3*len(zscored))))
sns.heatmap(zscored, cmap="RdBu_r", center=0, annot=True, fmt=".1f", 
            cbar_kws={"label":"z-score"}, annot_kws={"size":7},
            linewidths=0.3, linecolor="white", ax=ax)
ax.set_title("Perfil de cada comuna en cada feature (z-score, comunas ordenadas por estrato)")
ax.set_xlabel(""); ax.set_ylabel("")
plt.xticks(rotation=45, ha="right", fontsize=8)
# save_fig(fig, "11_perfil_zscore_comunas")
maybe_save(fig, "11_perfil_zscore_comunas")
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 3 · Bivariado: features ↔ outcome
# 
# Primero JOIN (votos + ECV). Con exportación de tabla completa al Gold.
# 
# Y con miras a implementar el modelo análisis bivariado. Para Dirichlet partimos de los valores de ALR, los predictores se modelan **linealmente contra `alr_left` y `alr_right`** simultáneamente.

# CELL ********************

# 3.1 — Join: gold (features) + votos (outcome) por (anio, codigo_comuna)
# LEFT JOIN desde votos para (mantiene granularidad de todas las elecciones).
# Cada fila de votos hereda los features ECV de su comuna-año del correspondiente
modeling = votos_alr.merge(
    gold[["anio","codigo_comuna"] + FEATURES],
    on=["anio","codigo_comuna"],
    how="left"
)
print(f"Modeling table: {modeling.shape}")
print(f"  · obs con outcome: {len(votos_alr)}")
print(f"  · obs con features: {modeling[FEATURES].notna().any(axis=1).sum()}")
print(f"  · obs completas (todas las features): {modeling[FEATURES].notna().all(axis=1).sum()}")

# Diagnóstico: ¿qué (anio, comuna) de votos no tienen contraparte en gold?
no_match = modeling[modeling[FEATURES].isna().all(axis=1)][["anio","codigo_comuna","corporacion"]]
print(f"\nFilas de votos sin features ECV: {len(no_match)}")
if len(no_match):
    display(no_match.head(15))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3.1b — Exportar modeling table a delta (intermedio para entrenamiento)
spark.createDataFrame(modeling).write.format("delta").mode("overwrite").saveAsTable("Data_LakeHouse.dbo.join_to_gold")
print(f"✔ Exportado: Data_LakeHouse.dbo.join_to_gold ({modeling.shape[0]} filas, {modeling.shape[1]} columnas)")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3.2 — Correlación de cada feature con ALR(left) y ALR(right)
corrs = []
for f in FEATURES:
    sub = modeling[[f,"alr_left","alr_right"]].dropna()
    if len(sub) < 10:
        continue
    corrs.append({
        "feature": f,
        "corr_alr_left":  sub[f].corr(sub["alr_left"]),
        "corr_alr_right": sub[f].corr(sub["alr_right"]),
        "n": len(sub),
    })
corrs = pd.DataFrame(corrs)
corrs["abs_max"] = corrs[["corr_alr_left","corr_alr_right"]].abs().max(axis=1)
corrs = corrs.sort_values("abs_max", ascending=False)

# Visualización lollipop
fig, ax = plt.subplots(figsize=(10, 0.4*len(corrs) + 1.5))
y_pos = np.arange(len(corrs))
ax.hlines(y_pos, 0, corrs["corr_alr_left"], color="#D62728", alpha=0.6, linewidth=2)
ax.scatter(corrs["corr_alr_left"], y_pos, color="#D62728", s=70, label="vs ALR(izq)", zorder=3)
ax.hlines(y_pos + 0.25, 0, corrs["corr_alr_right"], color="#1F77B4", alpha=0.6, linewidth=2)
ax.scatter(corrs["corr_alr_right"], y_pos + 0.25, color="#1F77B4", s=70, label="vs ALR(der)", zorder=3)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y_pos + 0.12)
ax.set_yticklabels(corrs["feature"], fontsize=9)
ax.set_xlabel("Correlación de Pearson")
ax.set_title("Correlación de cada feature con el outcome (en coordenadas ALR)")
ax.legend()
# save_fig(fig, "12_corr_feature_alr")
maybe_save(fig, "12_corr_feature_alr")
plt.show()

display(corrs.round(3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3.3 — Scatter matrix: top 6 features vs ALR(left) y ALR(right)
TOP_K = 6
top_features = corrs.head(TOP_K)["feature"].tolist()
print(f"Top {TOP_K} features por |corr| con ALR: {top_features}")

fig, axes = plt.subplots(TOP_K, 2, figsize=(11, 2.5*TOP_K))
for i, f in enumerate(top_features):
    sub = modeling[[f,"alr_left","alr_right"]].dropna()
    for j, target in enumerate(["alr_left","alr_right"]):
        ax = axes[i, j]
        ax.scatter(sub[f], sub[target], alpha=0.45, s=22, color=("#D62728" if "left" in target else "#1F77B4"))
        # Línea de regresión simple
        if len(sub) > 2:
            m, b = np.polyfit(sub[f], sub[target], 1)
            xs = np.linspace(sub[f].min(), sub[f].max(), 50)
            ax.plot(xs, m*xs + b, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel(f, fontsize=9)
        ax.set_ylabel(target, fontsize=9)
        ax.tick_params(labelsize=8)
fig.suptitle(f"Top {TOP_K} features vs. outcome ALR", y=1.001, fontsize=12, fontweight="bold")
fig.tight_layout()
# save_fig(fig, "13_scatter_top_features")
maybe_save(fig, "13_scatter_top_features")
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 4 · Multicolinealidad
# 
# Con miras a reducción de dimensionalidad, seleccionar más relevantes al modelo.
# 1. **Heatmap de correlaciones** — identificar pares con |r|>0.7.
# 2. **PCA** — ver cuántas dimensiones latentes realmente hay (regla del codo).
# 3. **VIF** — flag a variables que necesitan ser combinadas o eliminadas (VIF > 5 sospechoso, > 10 problemático).

# CELL ********************

# 4.1 — Heatmap de correlaciones entre features
corr_matrix = gold[FEATURES].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
            square=True, linewidths=0.3, linecolor="white", 
            cbar_kws={"shrink":0.7,"label":"corr"}, annot_kws={"size":7}, ax=ax)
ax.set_title("Correlaciones entre features")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
# save_fig(fig, "14_corr_features")
maybe_save(fig, "14_corr_features")
plt.show()

# Listar relevantes =>0.7 o -0.7
high_pairs = []
for i, j in combinations(FEATURES, 2):
    r = corr_matrix.loc[i,j]
    if abs(r) >= 0.7:
        high_pairs.append({"feature_1": i, "feature_2": j, "corr": r})
high_pairs = pd.DataFrame(high_pairs).sort_values("corr", key=abs, ascending=False)
print(f"\nPares con |r| ≥ 0.7: {len(high_pairs)}")
if len(high_pairs):
    display(high_pairs.round(3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 4.2 — PCA sobre los features estandarizados
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

X = gold[FEATURES].copy()
X_imp = SimpleImputer(strategy="median").fit_transform(X)
X_sc = StandardScaler().fit_transform(X_imp)

pca = PCA().fit(X_sc)
explained = pca.explained_variance_ratio_
cum = np.cumsum(explained)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(range(1, len(explained)+1), explained*100, color="#4C72B0", edgecolor="white")
axes[0].set_xlabel("Componente principal")
axes[0].set_ylabel("Varianza explicada (%)")
axes[0].set_title("Scree plot")

axes[1].plot(range(1, len(cum)+1), cum*100, "o-", color="#4C72B0")
axes[1].axhline(80, color="red", linestyle="--", linewidth=1, label="80%")
axes[1].axhline(95, color="orange", linestyle="--", linewidth=1, label="95%")
axes[1].set_xlabel("# componentes"); axes[1].set_ylabel("Varianza acumulada (%)")
axes[1].set_title("Varianza acumulada")
axes[1].legend()
# save_fig(fig, "15_pca_scree")
maybe_save(fig, "15_pca_scree")
plt.show()

n_80 = (cum < 0.80).sum() + 1
n_95 = (cum < 0.95).sum() + 1
print(f"\nComponentes necesarios para 80%: {n_80}  ·  para 95%: {n_95}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 4.3 — Cargas (loadings) de los primeros 3 componentes
loadings = pd.DataFrame(
    pca.components_[:4].T,
    index=FEATURES,
    columns=["PC1","PC2","PC3","PC4"]
)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(loadings, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
            linewidths=0.3, linecolor="white", cbar_kws={"shrink":0.7}, ax=ax)
ax.set_title("Cargas de los primeros 4 componentes principales\n(magnitudes grandes → la variable define el componente)")
# save_fig(fig, "16_pca_loadings")
maybe_save(fig, "16_pca_loadings")
plt.show()

print("\nINTERPRETACIÓN sugerida (basada en cargas):")
for k in range(4):
    top3 = loadings[f"PC{k+1}"].abs().nlargest(4).index.tolist()
    print(f"  PC{k+1} ({explained[k]*100:.1f}% var) está dominado por: {', '.join(top3)}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 4.4 — Variance Inflation Factors (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_imp_df = pd.DataFrame(X_imp, columns=FEATURES)
vif = pd.DataFrame({
    "feature": FEATURES,
    "VIF": [variance_inflation_factor(X_imp_df.values, i) for i in range(len(FEATURES))]
}).sort_values("VIF", ascending=False)

fig, ax = plt.subplots(figsize=(9, 0.35*len(vif) + 1))
colors = vif["VIF"].apply(lambda v: "#C73E1D" if v > 10 else ("#E89D45" if v > 5 else "#4C72B0"))
ax.barh(vif["feature"], vif["VIF"], color=colors, edgecolor="white")
ax.axvline(5,  color="orange", linestyle="--", linewidth=1, label="VIF=5  (sospechoso)")
ax.axvline(10, color="red",    linestyle="--", linewidth=1, label="VIF=10 (problemático)")
ax.set_xlabel("VIF")
ax.set_title("Variance Inflation Factors")
ax.invert_yaxis()
ax.legend()
# save_fig(fig, "17_vif")
maybe_save(fig, "17_vif")
plt.show()

display(vif.round(2))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 5 · Outliers y casos frontera
# 
# Para regresión Dirichlet con tan pocas observaciones (21 comunas × N elecciones), un solo outlier puede dominar el ajuste. Identifiquemos:

# CELL ********************

# 6.1 — Comunas-año con composición de voto atípica (Mahalanobis sobre ALR)
from scipy.stats import chi2

alr_data = votos_alr[["alr_left","alr_right"]].dropna()
mu = alr_data.mean().values
S  = np.cov(alr_data.T)
S_inv = np.linalg.pinv(S)
diff = alr_data.values - mu
mahal = np.sqrt(np.einsum("ij,jk,ik->i", diff, S_inv, diff))
votos_alr_complete = votos_alr.loc[alr_data.index].copy()
votos_alr_complete["mahalanobis"] = mahal

# Threshold: 97.5% percentil de chi-cuadrado con 2 df
thresh = np.sqrt(chi2.ppf(0.975, df=2))
print(f"Umbral Mahalanobis (97.5% χ²): {thresh:.2f}")
print(f"Observaciones flag: {(mahal > thresh).sum()} de {len(mahal)}")

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(alr_data["alr_left"], alr_data["alr_right"], c=mahal, 
                cmap="YlOrRd", s=40, edgecolor="black", linewidth=0.4)
plt.colorbar(sc, label="distancia Mahalanobis")

# Anotar outliers
outliers = votos_alr_complete[votos_alr_complete["mahalanobis"] > thresh].head(8)
for _, r in outliers.iterrows():
    ax.annotate(f"C{int(r['codigo_comuna'])}-{int(r['anio'])}", 
                (r["alr_left"], r["alr_right"]), fontsize=7,
                xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("ALR(izquierda / centro)")
ax.set_ylabel("ALR(derecha / centro)")
ax.set_title("Composición de voto en ALR · puntos rojos = outliers multivariados")
# save_fig(fig, "20_outliers_mahalanobis")
maybe_save(fig, "20_outliers_mahalanobis")
plt.show()

display(outliers[["anio","corporacion","codigo_comuna","p_left","p_center","p_right","mahalanobis"]].round(3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 7 · Chequeos específicos para Dirichlet regression
# 
# Estos son los **gotchas** del modelo Dirichlet que conviene tener resueltos antes de ajustar:
# 
# 1. **Cero exacto en proporciones:** la log-verosimilitud Dirichlet contiene `log(yᵢ)` — un cero genera `-∞`. Soluciones: (a) ε-bumping (lo que ya aplicamos arriba), (b) zero-inflated Dirichlet, (c) re-agregar votos a nivel mesa.
# 2. **Concentración estimada (precisión φ):** Dirichlet tiene un parámetro de precisión que es difícil de identificar con pocas observaciones por celda.
# 3. **Tamaño efectivo de muestra:** con N≈80 observaciones y ~16 features → estamos en zona de overfitting; necesitas regularización o reducción de dimensionalidad.

# CELL ********************

# 7.1 — Auditoría de boundary cases
print("=" * 70)
print("AUDITORÍA DE FRONTERA DEL SIMPLEX")
print("=" * 70)

zeros_per_col = {}
for col in ["p_left","p_center","p_right"]:
    zeros_per_col[col] = {
        "exactly_zero": (votos[col] == 0).sum(),
        "lt_0.01":      (votos[col] < 0.01).sum(),
        "lt_0.05":      (votos[col] < 0.05).sum(),
        "gt_0.95":      (votos[col] > 0.95).sum(),
    }
print("\nConteo de casos por proporción:")
display(pd.DataFrame(zeros_per_col).T)

# Recomendación
total_zero_rows = ((votos[["p_left","p_center","p_right"]] < 1e-6).any(axis=1)).sum()
print(f"\nFilas con AL MENOS UNA proporción ≈ 0: {total_zero_rows}")
if total_zero_rows / len(votos) > 0.1:
    print("⚠ Más del 10% de filas tienen ceros. CONSIDERAR modelo zero-inflated.")
elif total_zero_rows > 0:
    print("✓ Pocos ceros — el ε-bumping (ya aplicado en sección 1.6) debería ser suficiente.")
else:
    print("✓ Sin ceros exactos — Dirichlet estándar OK.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 7.2 — Tamaño efectivo de muestra por celda (anio × corporacion)
sample_per_cell = (
    votos.groupby(["anio","corporacion"])["codigo_comuna"]
         .nunique()
         .rename("n_comunas")
         .reset_index()
)
display(sample_per_cell.pivot_table(index="anio", columns="corporacion", values="n_comunas", fill_value=0))

# Total efectivo (después del join con features)
n_complete = modeling[FEATURES + ["alr_left","alr_right"]].dropna().shape[0]
print(f"\nN total para modeling (filas completas): {n_complete}")
print(f"Features candidatas: {len(FEATURES)}")
print(f"Ratio N/p: {n_complete/len(FEATURES):.1f}")
if n_complete / len(FEATURES) < 10:
    print("⚠ Ratio N/p < 10 → regularizar (L1/L2) o reducir features vía PCA/dominio.")
else:
    print("✓ Ratio N/p ≥ 10 → modelo identificable con cuidado.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
