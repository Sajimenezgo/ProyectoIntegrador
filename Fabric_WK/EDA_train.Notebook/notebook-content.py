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

# # EDA · `model_train` — contraste de hipótesis del marco teórico
# 
# Notebook de análisis exploratorio,trabaja **únicamente sobre `model_train`**. Con el propósito de ver el comportamiento de los features del modelo en su entorno de aplicación, con especial énfasis a la literatura evaluada como marco teórico (**hipótesis declaradas en el marco teórico §2.1–§2.9**). 
# 
# Exploramos como cada de calidad de vida (ECV) tiene una dirección esperada sobre el vector composicional (izquierda, centro, derecha) que aquí se contrasta empíricamente sobre Medellín.
# 
# **Notación de veredictos:**
# 
# | Símbolo | Significado |
# |---|---|
# | ✓ | La evidencia es consistente con la dirección del marco |
# | ✗ | La evidencia se invierte respecto al marco (interesante: posible "inversión" colombiana) |
# | ~ | Señal débil o ambigua (\|r\| < 0.15) |
# | · | Hipótesis no testeable con esta variable o sin variable proxy |
# 
# **Para exportar gráficos:** cada plot tiene `# save_fig(fig, "nombre")` comentado antes de `plt.show()`. Cambia `EXPORT_ALL = True` en el setup para guardar todos en `Files/EDA_graficos_train/`.


# MARKDOWN ********************

# ## 0 · Setup

# CELL ********************

# 0.1 — Imports y configuración
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
from scipy.stats import skew, kurtosis, pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})

EXPORT_ALL = True
EDA_EXPORT_DIR = "/lakehouse/default/Files/EDA_graficos_train"
ART_DIR        = "/lakehouse/default/Files/models/dirichlet_ecv"
os.makedirs(EDA_EXPORT_DIR, exist_ok=True)

def save_fig(fig, name, dpi=160):
    path = os.path.join(EDA_EXPORT_DIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  ✔ guardado: {path}")

def maybe_save(fig, name):
    if EXPORT_ALL: save_fig(fig, name)

# Acumulador de veredictos para la tabla síntesis (sección 5)
verdicts = []

print(f"Export dir: {EDA_EXPORT_DIR}")
print(f"EXPORT_ALL = {EXPORT_ALL}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 0.2 — Cargar model_train y resumir el split
train = spark.sql("SELECT * FROM model_train").toPandas()
print(f"model_train: {train.shape}")
print(f"Comunas:        {sorted(train['codigo_comuna'].unique())}")
print(f"Años:           {sorted(train['anio'].unique())}")
print(f"Corporaciones:  {sorted(train['corporacion'].unique())}")
print(f"\nDistribución corporación × año:")
display(train.groupby(["anio","corporacion"]).size().unstack(fill_value=0))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 0.3 — Constantes y metadatos teóricos por constructo
IDS   = ["anio", "corporacion", "codigo_comuna"]
PREDS = [
    "ESTRATO_STD", "EDUCACION_STD", "ALFABETISMO", "DIM_RECREACION",
    "IND_DISCRIMINACION", "IND_BRECHA_PERCEPCION", "DIM_PARTICIPACION",
    "DIM_MEDIO_AMBIENTE", "DIM_MOVILIDAD", "SATISFACCION_MUNICIPIO",
    "CONFIANZA_INSTITUCIONES", "CUMPLIMIENTO_NORMAS", "DIM_SEGURIDAD_ALIMENTARIA",
    "DIM_CAPITAL_FISICO", "INDICE_CARENCIA_SERVICIOS", "IND_HACINAMIENTO",
]
RESP     = ["p_left", "p_center", "p_right"]
RESP_ADJ = [c + "_adj" for c in RESP]
LABELS_RESP = {"p_left":"Izquierda","p_center":"Centro","p_right":"Derecha"}
COLORS_RESP = {"p_left":"#D62728","p_center":"#E89D45","p_right":"#1F77B4"}

# Mapeo constructo → familia teórica (sección del marco). Sirve para la tabla síntesis.
THEORY = {
    "EDUCACION_STD":            {"familia":"Educación (MT §2.1)",      "dir_norte":"+ izquierda",  "nota":"Diploma divide; signo puede invertir en Colombia"},
    "ALFABETISMO":              {"familia":"Educación (MT §2.1)",      "dir_norte":"+ izquierda",  "nota":"Simetría del clivaje educativo; signo puede invertir"},
    "DIM_SEGURIDAD_ALIMENTARIA":{"familia":"Privación (MT §2.2)",      "dir_norte":"+ izquierda",  "nota":"ELCSA; demanda redistributiva. Clientelismo puede desviarlo"},
    "IND_HACINAMIENTO":         {"familia":"Privación (MT §2.2)",      "dir_norte":"+ izquierda",  "nota":"Déficit habitacional"},
    "DIM_CAPITAL_FISICO":       {"familia":"Privación (MT §2.2)",      "dir_norte":"– izquierda",  "nota":"Activos altos → centro-derecha"},
    "INDICE_CARENCIA_SERVICIOS":{"familia":"Privación (MT §2.2)",      "dir_norte":"+ izquierda",  "nota":"Pro-incumbente puede dominar (clientelismo)"},
    "CONFIANZA_INSTITUCIONES":  {"familia":"Confianza (MT §2.4)",      "dir_norte":"– polos",      "nota":"Baja confianza empuja a izq O der, no al centro"},
    "SATISFACCION_MUNICIPIO":   {"familia":"Confianza (MT §2.4)",      "dir_norte":"~ contexto",   "nota":"Pro-incumbente local"},
    "DIM_PARTICIPACION":        {"familia":"Participación (MT §2.5)",  "dir_norte":"U-shape",      "nota":"Polos > centro (medido como conocimiento cívico aquí)"},
    "DIM_RECREACION":           {"familia":"Recreación (MT §2.5)",     "dir_norte":"+ izquierda",  "nota":"Postmaterialismo, efecto débil"},
    "IND_DISCRIMINACION":       {"familia":"Género (MT §2.6)",         "dir_norte":"+ izquierda",  "nota":"Percibir discriminación → progresismo"},
    "IND_BRECHA_PERCEPCION":    {"familia":"Género (MT §2.6)",         "dir_norte":"+ derecha",    "nota":"Visión tradicional → conservadurismo"},
    "DIM_MEDIO_AMBIENTE":       {"familia":"Ambiente (MT §2.7)",       "dir_norte":"~ inverso",    "nota":"Mide SATISFACCIÓN con entorno, no CONCERN. Más calidad → menos preocupación"},
    "DIM_MOVILIDAD":            {"familia":"Movilidad (MT §2.8)",      "dir_norte":"~ contexto",   "nota":"Satisfacción con infraestructura, no victimización"},
    "CUMPLIMIENTO_NORMAS":      {"familia":"Seguridad (MT §2.8)",      "dir_norte":"– derecha",    "nota":"Buena percepción de cumplimiento → menos demanda mano-dura → menos der"},
    "ESTRATO_STD":              {"familia":"Control SES",              "dir_norte":"– izquierda",  "nota":"Activos altos → centro-derecha"},
}

# Damos foco bivariado a la corporacion: presidente pues eviudencia muestra un voto mas ideologico
for c in PREDS + RESP + RESP_ADJ:
    assert c in train.columns, f"Falta columna: {c}"
print("✓ Columnas verificadas")

CORPS_DISPONIBLES = sorted(train["corporacion"].unique())
CORP_FOCO = "Presidente" if "PRESIDENTE" in CORPS_DISPONIBLES else CORPS_DISPONIBLES[0]
print(f"Corporación foco bivariado: {CORP_FOCO!r}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 0.4 — Reanudar el MLflow run del pipeline (creado en 01_prep_validate)
run_id_path = f"{ART_DIR}/run_id.txt"
RESUME_MLFLOW = os.path.exists(run_id_path)
if RESUME_MLFLOW:
    with open(run_id_path) as f: RUN_ID = f.read().strip()
    print(f"Run ID encontrado: {RUN_ID}")
else:
    RUN_ID = None
    print("Sin MLflow — solo escritura local")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 1 · Outcome en train
# 
# Validación rápida del outcome antes de trabajar con las hipótesis.

# CELL ********************

# 1.1 — Comparar p_* (observado) vs p_*_adj (zero-replaced)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, col in zip(axes, RESP):
    col_adj = col + "_adj"
    ax.hist(train[col],     bins=25, color=COLORS_RESP[col], alpha=0.55, label="observado",       edgecolor="white")
    ax.hist(train[col_adj], bins=25, color="black",          alpha=0.30, label="ajustado",         edgecolor="white", histtype="step", linewidth=1.5)
    ax.set_title(f"{LABELS_RESP[col]}")
    ax.set_xlim(-0.02, 1.02); ax.set_xlabel("proporción"); ax.legend(fontsize=8)
fig.suptitle("Outcome (train) — observado vs. zero-adjusted", y=1.02)
fig.tight_layout()
# save_fig(fig, "01_p_vs_padj")
maybe_save(fig, "01_p_vs_padj")
plt.show()

delta = (train[RESP_ADJ].values - train[RESP].values)
n_changed = (np.abs(delta).sum(axis=1) > 1e-6).sum()
print(f"\nFilas con ajuste no-trivial: {n_changed} / {len(train)} ({n_changed/len(train)*100:.1f}%)")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 1.2 — Ternario sobre train
sqrt3_2 = np.sqrt(3) / 2
V_L, V_R, V_C = np.array([0.0,0.0]), np.array([1.0,0.0]), np.array([0.5, sqrt3_2])
def baryc(l, c, r): return l*V_L + c*V_C + r*V_R

fig, ax = plt.subplots(figsize=(7.5, 7))
ax.add_patch(plt.Polygon([V_L, V_R, V_C], fill=False, edgecolor="black", linewidth=1.5))
for s in np.arange(0.2, 1.0, 0.2):
    for p1,p2 in [(baryc(s,0,1-s),baryc(s,1-s,0)),(baryc(0,s,1-s),baryc(1-s,s,0)),(baryc(0,1-s,s),baryc(1-s,0,s))]:
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]], color="lightgray", linewidth=0.5)

corp_palette = {c: plt.cm.Set2(i) for i, c in enumerate(CORPS_DISPONIBLES)}
for corp, sub in train.groupby("corporacion"):
    pts = sub[["p_left","p_center","p_right"]].apply(lambda r: baryc(*r), axis=1, result_type="expand")
    ax.scatter(pts[0], pts[1], s=35, alpha=0.7, label=corp, color=corp_palette[corp], edgecolor="white", linewidth=0.4)

ax.text(V_L[0]-0.04, -0.05, "Izquierda", fontsize=11, fontweight="bold", color="#D62728", ha="right")
ax.text(V_R[0]+0.04, -0.05, "Derecha", fontsize=11, fontweight="bold", color="#1F77B4", ha="left")
ax.text(V_C[0], V_C[1]+0.04, "Centro", fontsize=11, fontweight="bold", color="#B59500", ha="center")
ax.set_xlim(-0.15, 1.15); ax.set_ylim(-0.12, 1.05); ax.set_aspect("equal"); ax.axis("off")
ax.set_title(f"Simplex Δ² · train (n={len(train)})", pad=10)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
# save_fig(fig, "02_ternario_train")
maybe_save(fig, "02_ternario_train")
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 1.3 — Coordenadas ALR sobre _adj (necesarias para los tests de hipótesis)
train["alr_left"]  = np.log(train["p_left_adj"]  / train["p_center_adj"])
train["alr_right"] = np.log(train["p_right_adj"] / train["p_center_adj"])
# Métrica derivada: distancia angular desde el centro (útil para hipótesis de "polos")
train["dist_centro"] = np.sqrt(train["alr_left"]**2 + train["alr_right"]**2)

print("ALR + distancia al centro creados.")
display(train[["alr_left","alr_right","dist_centro"]].describe().round(3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 2 · Predictores estandarizados — chequeo rápido

# CELL ********************

# 2.1 — Descriptivos de los predictores
desc = pd.DataFrame({
    "media":     train[PREDS].mean().round(3),
    "sd":        train[PREDS].std().round(3),
    "skew":      train[PREDS].apply(skew).round(3),
    "kurtosis":  train[PREDS].apply(kurtosis).round(3),
    "n_out_3sd": (train[PREDS].abs() > 3).sum(),
})
display(desc.sort_values("kurtosis", ascending=False))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 3 · ¿Por qué comuna ≠ estrato? (motivación del efecto fijo de comuna, MT §2.9)
# 
# En el marco teórico (2.9) indicamos que la **comuna o corregimiento debe entrar como efecto fijo**: es la unidad que captura bastiones locales y autocorrelación espacial que tenemos disponible con datos electorales y calidad de vida.
# 
# Acá damos **justificación empírica** a esta decisión: si los features ECV varían entre comunas *más allá* de lo que explica estrato (que es el proxy mas difundido y genérico de capital de los diferentes sectores sociales en Colombia), entonces demuestra que los efectos van más allá que similitudes en torno al capital captado en el estrato. Los efectos sectoriales de "estrato socioeconómico" resultan menos prometedores para cualquier modelo


# CELL ********************

# 3.1 — Descomposición de varianza: R²(estrato) vs R²(estrato + comuna FE)

def r2_estrato(df, feature):
    s = df[["ESTRATO_STD", feature]].dropna().astype(np.float64)
    if len(s) < 5:
        return np.nan
    X = s[["ESTRATO_STD"]].values
    y = s[feature].values
    return LinearRegression().fit(X, y).score(X, y)

def r2_estrato_plus_comuna(df, feature):
    s = df[["ESTRATO_STD", "codigo_comuna", feature]].dropna()
    if len(s) < 5:
        return np.nan
    # Estrato como float64
    estr = s[["ESTRATO_STD"]].astype(np.float64).values
    # Dummies de comuna como float64 (no Int64)
    com = s["codigo_comuna"].astype(int).astype(str)
    dummies = pd.get_dummies(com, drop_first=True).astype(np.float64).values
    X = np.hstack([estr, dummies])
    y = s[feature].astype(np.float64).values
    return LinearRegression().fit(X, y).score(X, y)

feats_eval = [f for f in PREDS if f != "ESTRATO_STD"]
rows = []
for f in feats_eval:
    r2_e  = r2_estrato(train, f)
    r2_ec = r2_estrato_plus_comuna(train, f)
    rows.append({
        "feature": f,
        "R²(estrato)": r2_e,
        "R²(estrato + comuna)": r2_ec,
        "Δ atribuible a comuna": r2_ec - r2_e,
    })
varianza = pd.DataFrame(rows).sort_values("Δ atribuible a comuna", ascending=True)

fig, ax = plt.subplots(figsize=(11, 0.45*len(varianza) + 1))
y = np.arange(len(varianza))
ax.barh(y, varianza["R²(estrato)"], color="#efe8eeff", label="explicado por estrato")
ax.barh(y, varianza["Δ atribuible a comuna"], left=varianza["R²(estrato)"],
        color="#441b10ff", label="ganancia al añadir comuna")
ax.set_yticks(y); ax.set_yticklabels(varianza["feature"], fontsize=9)
ax.set_xlabel("R² del feature"); ax.set_xlim(0, 1)
ax.set_title("Descomposición de varianza · estrato vs estrato + comuna FE\n(barra naranja grande → la comuna añade información que estrato no captura)")
ax.legend(loc="lower right")
# save_fig(fig, "03_varianza_estrato_vs_comuna")
maybe_save(fig, "03_varianza_estrato_vs_comuna")
plt.show()
display(varianza.round(3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 3.2 — Clustering jerárquico de comunas en el espacio de 16 features
profile = train.groupby("codigo_comuna")[PREDS].mean()
Z = linkage(profile.values, method="ward")
K_CLUSTERS = 4
profile["cluster"] = fcluster(Z, t=K_CLUSTERS, criterion="maxclust")
profile["rank_estrato"] = profile["ESTRATO_STD"].rank(method="dense").astype(int)

fig, ax = plt.subplots(figsize=(11, 5))
dendrogram(Z, labels=[f"C{c}" for c in profile.index], leaf_font_size=9, ax=ax,
           color_threshold=Z[-K_CLUSTERS+1, 2])
ax.set_title(f"Clustering jerárquico de comunas (Ward, k={K_CLUSTERS})")
ax.set_ylabel("distancia")
# save_fig(fig, "04_dendrograma_comunas")
maybe_save(fig, "04_dendrograma_comunas")
plt.show()

# Cluster vs ranking de estrato
fig, ax = plt.subplots(figsize=(10, 6))
cluster_palette = sns.color_palette("Set1", n_colors=K_CLUSTERS)
for cl in sorted(profile["cluster"].unique()):
    sub = profile[profile["cluster"] == cl]
    ax.scatter(sub["rank_estrato"], sub["ESTRATO_STD"],
               s=180, color=cluster_palette[cl-1], edgecolor="black", linewidth=1,
               label=f"Cluster {cl}", alpha=0.85)
    for cod, row in sub.iterrows():
        ax.annotate(f"C{cod}", (row["rank_estrato"], row["ESTRATO_STD"]),
                    fontsize=8, ha="center", va="center", fontweight="bold")
ax.set_xlabel("Ranking por estrato promedio (1 = más bajo)")
ax.set_ylabel("ESTRATO_STD (z-score)")
ax.set_title("Cluster (espacio de 16 features) vs ranking por estrato\nDos comunas con MISMO estrato pueden caer en DISTINTO cluster — evidencia para usar comuna FE")
ax.legend(); ax.grid(True, alpha=0.3)
# save_fig(fig, "05_cluster_vs_estrato")
maybe_save(fig, "05_cluster_vs_estrato")
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 4 · Contraste hipótesis-por-hipótesis del marco teórico
# 
# Cada sub-sección 4.X corresponde a un constructo de la sección 2 del marco. Para cada uno: hipótesis declarada, test empírico sobre train (corporación foco = `CORP_FOCO`), visualización y veredicto.

# CELL ********************

# 4.0 — Helper: scatter feature → ALR(left), ALR(right) con correlaciones anotadas
# Match flexible por si el casing/espacios del valor difieren
import unicodedata
def _norm(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode().strip().lower()

mask_corp = train["corporacion"].apply(_norm) == _norm(CORP_FOCO)
sub_corp = train[mask_corp].copy()
print(f"Subconjunto para tests bivariados: {CORP_FOCO!r}, n={len(sub_corp)}")
print(f"Valores únicos de corporación en train: {sorted(train['corporacion'].unique())}")

assert len(sub_corp) > 10, f"Subconjunto vacío o muy pequeño para {CORP_FOCO!r}. Ajusta CORP_FOCO en celda 0.3."

def hipo_scatter(feature, expected_direction, hypo_text, ref_str, fig_name):
    """Plot feature vs ALR(izq) y ALR(der), retorna correlaciones y veredicto."""
    s = sub_corp[[feature, "alr_left", "alr_right"]].dropna()
    if len(s) < 2:
        print(f"⚠ {feature}: muy pocos datos (n={len(s)}) — se omite")
        return np.nan, np.nan, np.nan, np.nan
    r_l, p_l = pearsonr(s[feature], s["alr_left"])
    r_r, p_r = pearsonr(s[feature], s["alr_right"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, target, color, r_val, p_val in zip(
            axes, ["alr_left","alr_right"], ["#D62728","#1F77B4"], [r_l,r_r], [p_l,p_r]):
        ax.scatter(s[feature], s[target], s=35, alpha=0.55, color=color, edgecolor="white", linewidth=0.4)
        m, b = np.polyfit(s[feature], s[target], 1)
        xs = np.linspace(s[feature].min(), s[feature].max(), 50)
        ax.plot(xs, m*xs + b, color="black", linewidth=1.5, linestyle="--")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel(f"{feature} (z-score)")
        ax.set_ylabel(target)
        ax.set_title(f"{target} · r={r_val:+.3f} (p={p_val:.3f})", fontsize=10)
    fig.suptitle(f"HIPÓTESIS — {hypo_text}\nFuente: {ref_str}", y=1.04, fontsize=10, fontweight="bold")
    fig.tight_layout()
    # save_fig(fig, fig_name)
    maybe_save(fig, fig_name)
    plt.show()
    return r_l, r_r, p_l, p_r

def emit_verdict(constructo, hipotesis, evidencia, signo_match):
    if signo_match == "match":    sym = "✓"
    elif signo_match == "invert": sym = "✗"
    elif signo_match == "weak":   sym = "~"
    else:                         sym = "·"
    verdicts.append({
        "constructo": constructo,
        "familia": THEORY.get(constructo, {}).get("familia",""),
        "hipotesis MT": hipotesis,
        "evidencia (train)": evidencia,
        "veredicto": sym,
    })
    print(f"  → veredicto: {sym}  ·  {evidencia}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 4.1 · Educación — clivaje educativo y "diploma divide" (MT §2.1)
# 
# **Hipótesis del marco:** en el Norte global, mayor escolaridad → más izquierda (la educación actúa como proxy de apertura valorativa más que de ingreso, Zingher 2022). En el contexto colombiano el signo puede invertirse, por lo que el efecto se trata como **ambiguo a priori**.
# 
# **Test empírico:** signo y magnitud de la correlación EDUCACION_STD ↔ ALR. Si en Medellín es **negativa con `alr_left`** (mayor escolaridad → menos izquierda), confirma la inversión. Adicionalmente probamos la **interacción ALFABETISMO × ESTRATO**: el efecto de la alfabetización sobre el voto puede depender del nivel socioeconómico de la comuna.

# CELL ********************

# 4.1.a — Educación → ALR
r_l, r_r, p_l, p_r = hipo_scatter(
    "EDUCACION_STD", "Norte: + → izquierda · Colombia: posible inversión",
    "MT §2.1 — Diploma divide", "Zingher (2022)",
    "06_h_educacion"
)
# Veredicto
if r_l > 0.15:   evidencia, m = f"r(alr_izq)={r_l:+.2f} → consistente con Norte",       "match"
elif r_l < -0.15: evidencia, m = f"r(alr_izq)={r_l:+.2f} → INVERSIÓN colombiana detectada", "invert"
else:             evidencia, m = f"r(alr_izq)={r_l:+.2f} (débil)",                      "weak"
emit_verdict("EDUCACION_STD", "+ izquierda (con posible inversión)", evidencia, m)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 4.1.b — Alfabetismo → ALR (simétrico al clivaje educativo)
r_l, r_r, p_l, p_r = hipo_scatter(
    "ALFABETISMO", "Por simetría: + alfabet. → + izquierda (Norte) · ambiguo en Colombia",
    "MT §2.1", "Zingher (2022)",
    "07_h_alfabetismo"
)
if r_l > 0.15:    evidencia, m = f"r(alr_izq)={r_l:+.2f} → consistente con Norte", "match"
elif r_l < -0.15: evidencia, m = f"r(alr_izq)={r_l:+.2f} → inversión",             "invert"
else:             evidencia, m = f"r(alr_izq)={r_l:+.2f}",                         "weak"
emit_verdict("ALFABETISMO", "+ izquierda (simetría clivaje educativo)", evidencia, m)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Cómo interactua la ganacia de educación con el estrato? Este ultimo tiene algún efecto acumulativo?

# CELL ********************

# 4.1.c — Interacción ALFABETISMO × ESTRATO_STD (test del "diploma divide condicional")
sub_corp["INTER_ALF_ESTR"] = sub_corp["ALFABETISMO"] * sub_corp["ESTRATO_STD"]

def fit_r2(df, X_cols, y_col):
    s = df[X_cols + [y_col]].dropna()
    m = LinearRegression().fit(s[X_cols], s[y_col])
    return m.score(s[X_cols], s[y_col]), m.coef_, m.intercept_, len(s)

print(f"=== Test de interacción ALFABETISMO × ESTRATO_STD ({CORP_FOCO}, n={len(sub_corp)}) ===\n")
inter_results = []
for y in ["alr_left", "alr_right"]:
    r2_0, _, _, _ = fit_r2(sub_corp, ["ESTRATO_STD"], y)
    r2_1, _, _, _ = fit_r2(sub_corp, ["ESTRATO_STD","ALFABETISMO"], y)
    r2_2, coefs, _, _ = fit_r2(sub_corp, ["ESTRATO_STD","ALFABETISMO","INTER_ALF_ESTR"], y)
    inter_results.append({
        "target": y,
        "R²(solo estrato)":       r2_0,
        "R²(+ alfabet.)":         r2_1,
        "R²(+ interacción)":      r2_2,
        "Δ por interacción":      r2_2 - r2_1,
        "coef interacción":       coefs[2],
    })
inter_df = pd.DataFrame(inter_results)
display(inter_df.round(4))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, row in zip(axes, inter_results):
    bars = ["solo estrato", "+ alfabet.", "+ interacción"]
    vals = [row["R²(solo estrato)"], row["R²(+ alfabet.)"], row["R²(+ interacción)"]]
    ax.bar(bars, vals, color=["#b2b8c2ff","#b5330fff","#ed5a39ff"], edgecolor="white")
    ax.set_ylim(0, max(vals)*1.25 + 0.02)
    ax.set_title(row["target"])
    for i,v in enumerate(vals):
        ax.text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
fig.suptitle("Ganancia incremental al añadir alfabetismo y luego su interacción con estrato", y=1.02)
fig.tight_layout()
# save_fig(fig, "08_h_interaccion_alfa_estrato")
maybe_save(fig, "08_h_interaccion_alfa_estrato")
plt.show()

max_delta = max(r["Δ por interacción"] for r in inter_results)
if max_delta > 0.05:    evi, m = f"max ΔR²(interacción)={max_delta:+.3f} → interacción FUERTE",  "match"
elif max_delta > 0.02:  evi, m = f"max ΔR²(interacción)={max_delta:+.3f} → modesta",             "weak"
else:                   evi, m = f"max ΔR²(interacción)={max_delta:+.3f} → no relevante",        "invert"
emit_verdict("ALFABETISMO × ESTRATO", "Efecto de alfabetización modulado por estrato", evi, m)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 4.2 · Privación material y vivienda (MT §2.2)
# 
# **Hipótesis:** la privación impulsa demanda redistributiva → izquierda. Pero el clientelismo puede desviar el voto del pobre hacia maquinarias e incumbentes (Nichter 2018; Diamond 2021). Por tanto, *si el signo se invierte en Medellín, el clientelismo es la explicación candidata.*
# 
# Constructos: `DIM_SEGURIDAD_ALIMENTARIA`, `IND_HACINAMIENTO`, `INDICE_CARENCIA_SERVICIOS` (privación → izq); `DIM_CAPITAL_FISICO` (riqueza → centro-der).

# CELL ********************

# 4.2.a — Seguridad alimentaria (ELCSA)
r_l, r_r, _, _ = hipo_scatter("DIM_SEGURIDAD_ALIMENTARIA", 
    "+ inseguridad → + izquierda (redistribución)", "MT §2.2", 
    "Ciccolini (2025); Nichter (2018)", "09_h_elcsa")
if r_l > 0.15:    evi,m = f"r(izq)={r_l:+.2f} → privación → izq confirmada", "match"
elif r_l < -0.15: evi,m = f"r(izq)={r_l:+.2f} → INVERSIÓN (¿clientelismo?)", "invert"
else:             evi,m = f"r(izq)={r_l:+.2f} → señal débil",                "weak"
emit_verdict("DIM_SEGURIDAD_ALIMENTARIA", "+ izquierda (redistribución)", evi, m)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 4.2.b — Hacinamiento
r_l, _, _, _ = hipo_scatter("IND_HACINAMIENTO",
    "+ hacinamiento → + izquierda", "MT §2.2",
    "Ciccolini (2025)", "10_h_hacinamiento")
if r_l > 0.15:    evi,m = f"r(izq)={r_l:+.2f} → consistente",     "match"
elif r_l < -0.15: evi,m = f"r(izq)={r_l:+.2f} → INVERSIÓN",       "invert"
else:             evi,m = f"r(izq)={r_l:+.2f} → débil",            "weak"
emit_verdict("IND_HACINAMIENTO", "+ izquierda (déficit habitacional)", evi, m)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 4.2.c — Capital físico (riqueza por activos)
r_l, r_r, _, _ = hipo_scatter("DIM_CAPITAL_FISICO",
    "+ activos → centro/derecha (NO izquierda)", "MT §2.2",
    "Ciccolini (2025)", "11_h_capital_fisico")
if r_l < -0.15:    evi,m = f"r(izq)={r_l:+.2f} → riqueza aleja de izq, consistente",  "match"
elif r_l > 0.15:   evi,m = f"r(izq)={r_l:+.2f} → INVERSIÓN (raro)",                    "invert"
else:              evi,m = f"r(izq)={r_l:+.2f} → débil",                                "weak"
emit_verdict("DIM_CAPITAL_FISICO", "– izquierda (riqueza → centro-derecha)", evi, m)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 4.2.d — Carencia de servicios públicos (puede ser pro-incumbente, no ideológico)
r_l, r_r, _, _ = hipo_scatter("INDICE_CARENCIA_SERVICIOS",
    "+ carencia → + izquierda (redistribución) PERO pro-incumbente posible", "MT §2.2",
    "Nichter (2018)", "12_h_carencia_servicios")
if r_l > 0.15:    evi,m = f"r(izq)={r_l:+.2f} → redistribución prima",          "match"
elif r_l < -0.15: evi,m = f"r(izq)={r_l:+.2f} → INVERSIÓN (clientelismo?)",     "invert"
else:             evi,m = f"r(izq)={r_l:+.2f} → débil/ambiguo",                  "weak"
emit_verdict("INDICE_CARENCIA_SERVICIOS", "+ izquierda (con posible inversión clientelar)", evi, m)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 4.3 · Confianza institucional → ¿efecto en los polos? (MT §2.4)
# 
# **Hipótesis no-lineal:** la baja confianza institucional empuja hacia los polos (izq O der) y aleja del centro — sin signo direccional único (Baccini 2025). Esto **no se ve en una correlación lineal con `alr_left` o `alr_right` por separado**, sino con la **distancia al centro** `√(alr_left² + alr_right²)`.
# 
# **Test:** correlación negativa entre CONFIANZA_INSTITUCIONES y `dist_centro` → consistente con polarización antiestablishment.

# CELL ********************

# 4.3.a — Confianza institucional vs distancia al centro
s = sub_corp[["CONFIANZA_INSTITUCIONES", "dist_centro"]].dropna()
r, p = pearsonr(s["CONFIANZA_INSTITUCIONES"], s["dist_centro"])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

# Panel izq: scatter con regresión
axes[0].scatter(s["CONFIANZA_INSTITUCIONES"], s["dist_centro"], s=40, alpha=0.6,
                color="#6b7280", edgecolor="white", linewidth=0.4)
m_c = np.polyfit(s["CONFIANZA_INSTITUCIONES"], s["dist_centro"], 1)
xs = np.linspace(s["CONFIANZA_INSTITUCIONES"].min(), s["CONFIANZA_INSTITUCIONES"].max(), 50)
axes[0].plot(xs, m_c[0]*xs + m_c[1], "--", color="black", linewidth=1.5)
axes[0].set_xlabel("CONFIANZA_INSTITUCIONES (z-score)")
axes[0].set_ylabel("Distancia al centro = √(ALR_izq² + ALR_der²)")
axes[0].set_title(f"Confianza ↔ distancia al centro · r={r:+.3f} (p={p:.3f})")

# Panel der: scatter en ALR-space, color por nivel de confianza
sc = axes[1].scatter(sub_corp["alr_left"], sub_corp["alr_right"], 
                     c=sub_corp["CONFIANZA_INSTITUCIONES"], cmap="RdYlGn",
                     s=45, alpha=0.7, edgecolor="white", linewidth=0.4)
plt.colorbar(sc, ax=axes[1], label="CONFIANZA")
axes[1].axhline(0, color="gray", linewidth=0.5); axes[1].axvline(0, color="gray", linewidth=0.5)
axes[1].set_xlabel("ALR(izquierda / centro)"); axes[1].set_ylabel("ALR(derecha / centro)")
axes[1].set_title("Mapa ALR · centro de la nube = centro electoral")

fig.suptitle("HIPÓTESIS MT §2.4 — baja confianza empuja a los polos\nFuente: Baccini (2025)", y=1.04, fontsize=10, fontweight="bold")
fig.tight_layout()
# save_fig(fig, "13_h_confianza_polos")
maybe_save(fig, "13_h_confianza_polos")
plt.show()

if r < -0.15:     evi,m = f"r(confianza, dist_centro)={r:+.2f} → polarización confirmada", "match"
elif r > 0.15:    evi,m = f"r(confianza, dist_centro)={r:+.2f} → INVERSIÓN (alta confianza → polos?)", "invert"
else:             evi,m = f"r(confianza, dist_centro)={r:+.2f} → señal débil", "weak"
emit_verdict("CONFIANZA_INSTITUCIONES", "– empuja a polos (no centro)", evi, m)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 4.3.b — Satisfacción municipio (efecto contexto-dependiente: pro-incumbente)
r_l, r_r, _, _ = hipo_scatter("SATISFACCION_MUNICIPIO",
    "Pro-incumbente local (depende de quién gobierne ese año)", "MT §2.4",
    "Baccini (2025)", "14_h_satisfaccion_mun")
emit_verdict("SATISFACCION_MUNICIPIO", "Pro-incumbente, signo dependiente del contexto",
             f"r(izq)={r_l:+.2f} · r(der)={r_r:+.2f}", "weak" if max(abs(r_l),abs(r_r))<0.15 else "match")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 4.4 · Participación / conocimiento cívico — ¿forma de U? (MT §2.5)
# 
# **Hipótesis no-lineal:** la participación política exhibe **patrón en U** — los extremos ideológicos están más involucrados que el centro (Pew 2014).
# 
# **Nota importante:** la variable `DIM_PARTICIPACION` en nuestro dataset mide **conocimiento cívico** (3 preguntas sobre quién hace las leyes), no participación política activa. La interpretación esperada se ajusta: si la hipótesis-U aplica, vemos correlación cuadrática positiva con `dist_centro` (más conocimiento → ¿más extremo?).
# 
# **Test:** ajustar `alr_target = a + b·X + c·X²` y verificar si el coeficiente cuadrático es positivo y significativo.

# CELL ********************

# 4.4 — Test cuadrático: U-shape en participación
def fit_quadratic(df, x_col, y_col):
    s = df[[x_col, y_col]].dropna()
    if len(s) < 10: return None
    X = np.column_stack([s[x_col], s[x_col]**2])
    m = LinearRegression().fit(X, s[y_col])
    r2 = m.score(X, s[y_col])
    # Comparar contra lineal puro
    m_lin = LinearRegression().fit(s[[x_col]], s[y_col])
    r2_lin = m_lin.score(s[[x_col]], s[y_col])
    return {"coef_lin": m.coef_[0], "coef_quad": m.coef_[1], 
            "R²(cuad)": r2, "R²(lin)": r2_lin, "Δ por cuad": r2 - r2_lin}

results_quad = []
for target in ["alr_left", "alr_right", "dist_centro"]:
    res = fit_quadratic(sub_corp, "DIM_PARTICIPACION", target)
    if res: 
        res["target"] = target
        results_quad.append(res)
quad_df = pd.DataFrame(results_quad)
print("Test cuadrático sobre DIM_PARTICIPACION:")
display(quad_df.round(4))

# Visualización
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, target, color in zip(axes, ["alr_left","alr_right","dist_centro"], ["#D62728","#1F77B4","#6b7280"]):
    s = sub_corp[["DIM_PARTICIPACION", target]].dropna()
    ax.scatter(s["DIM_PARTICIPACION"], s[target], s=35, alpha=0.5, color=color, edgecolor="white", linewidth=0.4)
    # Curva cuadrática ajustada
    X = np.column_stack([s["DIM_PARTICIPACION"], s["DIM_PARTICIPACION"]**2])
    m = LinearRegression().fit(X, s[target])
    xs = np.linspace(s["DIM_PARTICIPACION"].min(), s["DIM_PARTICIPACION"].max(), 100)
    ys = m.intercept_ + m.coef_[0]*xs + m.coef_[1]*xs**2
    ax.plot(xs, ys, color="black", linewidth=2)
    ax.set_xlabel("DIM_PARTICIPACION (z-score)")
    ax.set_ylabel(target)
    coef_q = m.coef_[1]
    ax.set_title(f"{target}\ncoef cuadrático = {coef_q:+.3f}")
fig.suptitle("HIPÓTESIS MT §2.5 — participación con forma de U\nFuente: Pew Research (2014)",
             y=1.04, fontsize=10, fontweight="bold")
fig.tight_layout()
# save_fig(fig, "15_h_participacion_u_shape")
maybe_save(fig, "15_h_participacion_u_shape")
plt.show()

# Veredicto sobre dist_centro
coef_q_dist = quad_df.loc[quad_df["target"]=="dist_centro","coef_quad"].iloc[0]
delta_dist  = quad_df.loc[quad_df["target"]=="dist_centro","Δ por cuad"].iloc[0]
if coef_q_dist > 0 and delta_dist > 0.02:
    evi, m = f"coef_cuad(dist)={coef_q_dist:+.3f}, ΔR²={delta_dist:+.3f} → U confirmada", "match"
elif coef_q_dist < 0:
    evi, m = f"coef_cuad(dist)={coef_q_dist:+.3f} → forma de ∩ (extremos menos participativos)", "invert"
else:
    evi, m = f"coef_cuad(dist)={coef_q_dist:+.3f} → sin curvatura clara", "weak"
emit_verdict("DIM_PARTICIPACION", "U-shape: polos > centro", evi, m)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 4.5 · Identidad de género (MT §2.6)
# 
# **Hipótesis:** la "brecha de género moderna" indica que a mayor autonomía femenina y percepción de discriminación → posiciones más progresistas / izquierda (Reyes-Housholder & Schwindt-Bayer 2025; Inglehart & Norris 2019).
# 
# - `IND_DISCRIMINACION` (1 = sí existe discriminación contra la mujer): + → izquierda esperado
# - `IND_BRECHA_PERCEPCION` (visión tradicional: hombres mejores líderes / título más importante para hombres): + → derecha esperado

# CELL ********************

# 4.5.a — Percepción de discriminación
r_l, r_r, _, _ = hipo_scatter("IND_DISCRIMINACION",
    "+ percibir discriminación → + izquierda", "MT §2.6",
    "Reyes-Housholder & Schwindt-Bayer (2025)", "16_h_discrim")
if r_l > 0.15:    evi,m = f"r(izq)={r_l:+.2f} → consistente",   "match"
elif r_l < -0.15: evi,m = f"r(izq)={r_l:+.2f} → INVERSIÓN",     "invert"
else:             evi,m = f"r(izq)={r_l:+.2f} → débil",          "weak"
emit_verdict("IND_DISCRIMINACION", "+ izquierda", evi, m)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 4.5.b — Visión tradicional de género (brecha de percepción)
r_l, r_r, _, _ = hipo_scatter("IND_BRECHA_PERCEPCION",
    "+ visión tradicional → + derecha", "MT §2.6",
    "Inglehart & Norris (2019)", "17_h_brecha_genero")
# Hipótesis: relación POSITIVA con alr_right
if r_r > 0.15:    evi,m = f"r(der)={r_r:+.2f} → consistente",     "match"
elif r_r < -0.15: evi,m = f"r(der)={r_r:+.2f} → INVERSIÓN",       "invert"
else:             evi,m = f"r(der)={r_r:+.2f} → débil",            "weak"
emit_verdict("IND_BRECHA_PERCEPCION", "+ derecha (visión tradicional)", evi, m)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 4.6 · Medio ambiente (MT §2.7)
# 
# **Hipótesis del marco:** preocupación ambiental → izquierda/verde (Papp 2022).
# 
# **Importante:** nuestra variable `DIM_MEDIO_AMBIENTE` mide **satisfacción con la calidad del entorno** (aire, basuras, contaminación visual, arborización) — es decir, lo *opuesto* a "preocupación ambiental". Una comuna con alto `DIM_MEDIO_AMBIENTE` no es necesariamente más verde; es una comuna donde el entorno percibido es bueno. La predicción se invierte: si la hipótesis del marco aplica, esperaríamos correlación **negativa** con `alr_left` (alta calidad de entorno → poca demanda de programas verdes → menos izq).

# CELL ********************

# 4.6 — Satisfacción con entorno medio ambiental
r_l, r_r, _, _ = hipo_scatter("DIM_MEDIO_AMBIENTE",
    "+ buena percepción del entorno = – preocupación → menos izq esperada",
    "MT §2.7 (interpretación invertida del proxy)", "Papp (2022)",
    "18_h_medio_ambiente")
if r_l < -0.15:    evi,m = f"r(izq)={r_l:+.2f} → consistente con marco invertido",  "match"
elif r_l > 0.15:   evi,m = f"r(izq)={r_l:+.2f} → INVERSIÓN del proxy",              "invert"
else:              evi,m = f"r(izq)={r_l:+.2f} → débil",                              "weak"
emit_verdict("DIM_MEDIO_AMBIENTE", "– izquierda (proxy de baja preocupación verde)", evi, m)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 4.7 · Movilidad y seguridad (MT §2.8)
# 
# **Hipótesis del marco:** inseguridad y victimización → derecha (mano dura) (Visconti 2020).
# 
# Nuestras variables miden **satisfacción con movilidad** (pavimentación, andenes) y **percepción de cumplimiento de normas de tránsito**. Si la hipótesis del marco aplica, las áreas con peor percepción de cumplimiento (señal de desorden urbano) deberían inclinarse a la derecha. Esto se traduce en correlación **negativa** entre `CUMPLIMIENTO_NORMAS` y `alr_right`.

# CELL ********************

# 4.7.a — Movilidad (satisfacción con pavimentación/andenes)
r_l, r_r, _, _ = hipo_scatter("DIM_MOVILIDAD",
    "Satisfacción con infraestructura — no es proxy directo de victimización",
    "MT §2.8 (proxy parcial)", "Visconti (2020)", "19_h_movilidad")
emit_verdict("DIM_MOVILIDAD", "Ambiguo (no mide victimización directa)",
             f"r(izq)={r_l:+.2f} · r(der)={r_r:+.2f}",
             "weak" if max(abs(r_l),abs(r_r))<0.15 else "match")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 4.7.b — Cumplimiento de normas de tránsito
r_l, r_r, _, _ = hipo_scatter("CUMPLIMIENTO_NORMAS",
    "Baja percepción de cumplimiento (desorden) → + derecha esperado",
    "MT §2.8", "Visconti (2020)", "20_h_cumplimiento")
# Hipótesis: relación NEGATIVA con alr_right (más cumplimiento → menos demanda de mano dura)
if r_r < -0.15:    evi,m = f"r(der)={r_r:+.2f} → consistente",  "match"
elif r_r > 0.15:   evi,m = f"r(der)={r_r:+.2f} → INVERSIÓN",    "invert"
else:              evi,m = f"r(der)={r_r:+.2f} → débil",         "weak"
emit_verdict("CUMPLIMIENTO_NORMAS", "– derecha (proxy invertido de inseguridad)", evi, m)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 4.8 · Recreación (MT §2.5, postmaterialismo)
# 
# **Hipótesis:** la participación en actividades recreativas como expresión postmaterialista se asocia débilmente con posiciones progresistas (Copeland 2014; Cheng et al. 2023).

# CELL ********************

# 4.8 — Recreación
r_l, r_r, _, _ = hipo_scatter("DIM_RECREACION",
    "+ recreación → + izquierda (efecto débil postmaterialista)",
    "MT §2.5", "Copeland (2014)", "21_h_recreacion")
if r_l > 0.10:    evi,m = f"r(izq)={r_l:+.2f} → consistente (débil)",  "match"
elif r_l < -0.10: evi,m = f"r(izq)={r_l:+.2f} → INVERSIÓN",            "invert"
else:             evi,m = f"r(izq)={r_l:+.2f} → sin señal",             "weak"
emit_verdict("DIM_RECREACION", "+ izquierda (débil)", evi, m)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 5 · Síntesis: hipótesis del marco vs. evidencia en train
# 
# Tabla consolidada — cada fila es un constructo de la sección 2 del marco teórico con su veredicto empírico sobre `model_train` (corporación foco).

# CELL ********************

# 5.1 — Tabla síntesis ordenada por veredicto
synth = pd.DataFrame(verdicts)
order = {"✓": 0, "✗": 1, "~": 2, "·": 3}
synth["_o"] = synth["veredicto"].map(order)
synth = synth.sort_values(["_o","familia","constructo"]).drop(columns="_o").reset_index(drop=True)
display(synth)

# Resumen por familia
print("\nResumen por familia teórica:")
display(synth.groupby("familia")["veredicto"].value_counts().unstack(fill_value=0))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 5.2 — Visualización tipo "report card" 
fig, ax = plt.subplots(figsize=(8.5, 0.4*len(synth) + 1))
veredicto_colors = {"✓": "#2ca02c", "✗": "#d62728", "~": "#ff7f0e", "·": "#9e9e9e"}
y = np.arange(len(synth))
ax.barh(y, np.ones(len(synth)), color=[veredicto_colors[v] for v in synth["veredicto"]], alpha=0.7)
for i, row in synth.iterrows():
    ax.text(0.02, i, f"{row['veredicto']}  {row['constructo']}", 
            va="center", fontsize=9, fontweight="bold", color="white")
    ax.text(0.55, i, row["evidencia (train)"], va="center", fontsize=8, color="white")
ax.set_yticks([]); ax.set_xticks([]); ax.set_xlim(0, 1)
ax.invert_yaxis()
ax.set_title("Veredictos empíricos por constructo (train)", fontsize=12, fontweight="bold")
# Leyenda
handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.7) for c in veredicto_colors.values()]
labels  = ["✓ concuerda con MT", "✗ se invierte (interesante)", "~ débil/ambigua", "· no testeable"]
ax.legend(handles, labels, loc="lower right", fontsize=8, framealpha=0.95)
# save_fig(fig, "22_sintesis_veredictos")
maybe_save(fig, "22_sintesis_veredictos")
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 6 · Preview multivariado: OLS lineal sobre cada ALR
# 
# Antes del Dirichlet final, ajustamos OLS lineal **separado** sobre `alr_left` y `alr_right` para tener una intuición de qué predictores sobreviven en presencia de los demás. No es el modelo final — el Dirichlet ajusta ambos componentes conjuntamente y respeta la geometría composicional — pero **un predictor que es fuerte en bivariado y muere en multivariado es candidato a estar mediado por estrato u otro feature**.

# CELL ********************

# 6.1 — OLS multivariado: ALR ~ predictores estandarizados
import statsmodels.api as sm

def fit_ols_alr(df, target):
    s = df[PREDS + [target]].dropna()
    X = sm.add_constant(s[PREDS])
    return sm.OLS(s[target], X).fit()

print(f"=== OLS multivariado · {CORP_FOCO}, n={len(sub_corp)} ===\n")
fit_left  = fit_ols_alr(sub_corp, "alr_left")
fit_right = fit_ols_alr(sub_corp, "alr_right")

coef_table = pd.DataFrame({
    "β(alr_left)":   fit_left.params,
    "p(alr_left)":   fit_left.pvalues,
    "β(alr_right)":  fit_right.params,
    "p(alr_right)":  fit_right.pvalues,
}).drop("const")

print(f"R²(alr_left)  = {fit_left.rsquared:.3f}")
print(f"R²(alr_right) = {fit_right.rsquared:.3f}")
display(coef_table.round(3))

# Forest plot — coeficientes con sus IC
fig, axes = plt.subplots(1, 2, figsize=(12, 0.4*len(PREDS) + 1.5))
for ax, fit, color, title in zip(axes, [fit_left, fit_right], ["#D62728","#1F77B4"],
                                  ["ALR(izquierda / centro)", "ALR(derecha / centro)"]):
    ci = fit.conf_int().drop("const")
    ci.columns = ["lo","hi"]
    coefs = fit.params.drop("const").sort_values()
    ci = ci.loc[coefs.index]
    y = np.arange(len(coefs))
    ax.hlines(y, ci["lo"], ci["hi"], color=color, alpha=0.7, linewidth=2)
    ax.scatter(coefs, y, color=color, s=60, zorder=3)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks(y); ax.set_yticklabels(coefs.index, fontsize=8)
    ax.set_title(title); ax.set_xlabel("β estandarizado (IC 95%)")
fig.suptitle("Preview OLS multivariado por componente ALR", y=1.02, fontweight="bold")
fig.tight_layout()
# save_fig(fig, "23_ols_multivariado")
maybe_save(fig, "23_ols_multivariado")
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 7 · Logging MLflow
# 
# Adjuntar los artefactos clave de esta EDA (tabla síntesis, veredictos, coeficientes OLS preview) al run del pipeline para trazabilidad.

# CELL ********************

# 7.1 — Persistir CSVs y loguear como nested run
synth.to_csv(f"{EDA_EXPORT_DIR}/sintesis_veredictos.csv", index=False)
varianza.to_csv(f"{EDA_EXPORT_DIR}/varianza_estrato_vs_comuna.csv", index=False)
profile.to_csv(f"{EDA_EXPORT_DIR}/cluster_assignments.csv")
inter_df.to_csv(f"{EDA_EXPORT_DIR}/interaccion_alfa_estrato.csv", index=False)
coef_table.to_csv(f"{EDA_EXPORT_DIR}/ols_preview_coefs.csv")

if RESUME_MLFLOW:
    with mlflow.start_run(run_id=RUN_ID):
        with mlflow.start_run(run_name="eda_train_marco_teorico", nested=True):
            mlflow.log_param("corp_foco_bivariado", CORP_FOCO)
            mlflow.log_param("k_clusters", K_CLUSTERS)
            mlflow.log_metric("n_match",  int((synth["veredicto"]=="✓").sum()))
            mlflow.log_metric("n_invert", int((synth["veredicto"]=="✗").sum()))
            mlflow.log_metric("n_weak",   int((synth["veredicto"]=="~").sum()))
            mlflow.log_metric("ols_r2_alr_left",  float(fit_left.rsquared))
            mlflow.log_metric("ols_r2_alr_right", float(fit_right.rsquared))
            mlflow.log_metric("max_delta_comuna_vs_estrato", float(varianza["Δ atribuible a comuna"].max()))
            for fn in ["sintesis_veredictos.csv","varianza_estrato_vs_comuna.csv",
                       "cluster_assignments.csv","interaccion_alfa_estrato.csv","ols_preview_coefs.csv"]:
                mlflow.log_artifact(f"{EDA_EXPORT_DIR}/{fn}")
        print(f"✔ EDA logueada como nested run dentro de {RUN_ID}")
else:
    print("⚠ Sin MLflow — artefactos solo en disco")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## ☑ Resumen ejecutivo
# 
# Después de correr el notebook completo, los hallazgos relevantes que esta EDA deja para el modelo Dirichlet:
# 
# 1. **Decisión sobre comuna FE (MT §2.9):** la sección 3 cuantifica cuánto añade la comuna sobre estrato. Si la columna "Δ atribuible a comuna" supera 0.2 para varios features, los efectos fijos están bien justificados — el modelo Dirichlet final debe incluir `codigo_comuna` como dummies.
# 
# 2. **Inversiones colombianas:** los constructos con veredicto ✗ son los más informativos teóricamente — son donde el contexto colombiano difiere del Norte global. El reporte final del proyecto debería discutirlos explícitamente.
# 
# 3. **Interacción ALFABETISMO × ESTRATO:** si la sección 4.1.c muestra ΔR² > 0.02, añadirla a `PREDS` en `01_prep_validate` y re-correr el split.
# 
# 4. **Forma de U en participación (MT §2.5):** si la sección 4.4 confirma curvatura positiva sobre `dist_centro`, el Dirichlet se beneficiará de añadir `DIM_PARTICIPACION²` como feature.
# 
# 5. **Polarización por desconfianza (MT §2.4):** la correlación CONFIANZA ↔ `dist_centro` en 4.3.a es el test directo de la hipótesis Baccini. Un valor < -0.2 es señal fuerte.
# 
# 6. **Preview OLS (sección 6):** los predictores con β significativo en presencia de los demás son los candidatos a entrar al Dirichlet sin penalización. Los que mueren al controlar por el resto probablemente están mediados — útil para discusión, no para inclusión naive.

