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

# # Votos Medellín — Pipeline consolidación + diccionario político
# **Años:** 2018, 2019, 2022, 2023  
# **Fuente data electoral:** `Files/bronze/mmv/ANTIOQUIA/<subcarpeta>/part-*.csv`  
# **Tabla de mapeo zona/puesto → comuna:** `dbo.mmv_bronze`  
# **Diccionario curaduría de espectro por partido:** `Files/bronze/mmv/MDE_MERGED_18_19_22_23/diccionario_agrupaciones_politicas.csv`
# 
# **Salidas:**
# - Merged CSV → `Files/bronze/mmv/MDE_MERGED_18_19_22_23/`
# - Proporciones (Delta tabla) → `Files/Silver/Votaciones_proporciones_comunas/`
# 
# ---
# ## Pasos
# 0. Setup Spark y rutas
# 1. Cargar los 18 archivos crudos
# 2. Verificar esquema entre subcarpetas
# 3. Filtrar a Medellín (codigo_municipio = '001')
# 4. Enriquecer codigo_comuna usando dbo.mmv_bronze (para elecciones nacionales: relaciona comuna a cada puesto de votación)
# 5. Unir con diccionario de agrupaciones políticas
# 6. Exportar merged a Bronze
# 7. Agregar proporciones por cada combinación a unidad de análisis (anio × corporacion × codigo_comuna)
# 8. Exportar tabla final como Delta a Silver


# MARKDOWN ********************

# ## 0. Setup

# CELL ********************

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.window import Window
import re

# La sesión Spark ya existe en Fabric como variable global `spark`
print(f"Spark version: {spark.version}")
print(f"App: {spark.sparkContext.appName}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Definimos rutas / input / output de .csv consolidado y delta a Silver 

BRONZE_ANTIOQUIA   = "Files/bronze/mmv/ANTIOQUIA"
BRONZE_MDE_MERGED  = "Files/bronze/mmv/MDE_MERGED_18_19_22_23"
DICT_FILE          = f"{BRONZE_MDE_MERGED}/diccionario_agrupaciones_politicas.csv"
MERGED_OUTPUT      = f"{BRONZE_MDE_MERGED}/MDE_MERGED_18_19_22_23" 
SILVER_PROPORCIONES = "Files/Silver/Votaciones_proporciones_comunas"
MMV_BRONZE_TABLE   = "mmv_bronze"   # tabla lakehouse (dbo)

# Definir origen: lugar del mapeo de las 18 carpetas: una a cada elección de los 4 años

SUBCARPETAS = [
    "MMV_2018_CAMARA", "MMV_2018_PRESIDENTE_1V", "MMV_2018_PRESIDENTE_2V", "MMV_2018_SENADO",
    "MMV_2019_ALCALDE", "MMV_2019_ASAMBLEA", "MMV_2019_CONCEJO", "MMV_2019_GOBERNADOR", "MMV_2019_JAL",
    "MMV_2022_CAMARA", "MMV_2022_PRESIDENTE_1V", "MMV_2022_PRESIDENTE_2V", "MMV_2022_SENADO",
    "MMV_2023_ALCALDE", "MMV_2023_ASAMBLEA", "MMV_2023_CONCEJO", "MMV_2023_GOBERNADOR", "MMV_2023_JAL",
]

EXPECTED_COLS = [
    'codigo_municipio', 'municipio', 'codigo_zona', 'codigo_puesto', 'puesto',
    'mesa', 'codigo_comuna', 'comuna', 'codigo_corporacion', 'corporacion',
    'codigo_circunscripcion', 'codigo_partido', 'agrupacion_politica',
    'codigo_candidato', 'candidato'
]

CODIGO_MEDELLIN = "001"   # código municipal de Medellín en Antioquia

print(f"Subcarpetas: {len(SUBCARPETAS)}")
print(f"Salida merged: {MERGED_OUTPUT}")
print(f"Salida proporciones: {SILVER_PROPORCIONES}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# ## 1. Cargar los 18 archivos crudos


# CELL ********************

def load_subcarpeta(subcarpeta: str):
 
    folder_path = f"{BRONZE_ANTIOQUIA}/{subcarpeta}"
    parts    = subcarpeta.split('_')
    anio     = int(parts[1])
    eleccion = '_'.join(parts[2:])

    df = (
        spark.read
        .option('header', 'true')
        .option('inferSchema', 'false')   # todo como string, igual que antes
        .option('encoding', 'UTF-8')
        .option('multiline', 'true')
        .option('escape', '"')
        .csv(folder_path)
    )

    # Normalizar nombres de columnas
    df = df.toDF(*[c.strip().lower() for c in df.columns])
    df = df.withColumn('anio', F.lit(anio))
    df = df.withColumn('eleccion', F.lit(eleccion))
    return df, anio, eleccion


# Cargar las 18 subcarpetas
dfs = {}
print(f"{'SUBCARPETA':<37} {'COLUMNAS':>10}")
print("-" * 55)
for sc in SUBCARPETAS:
    try:
        df, anio, eleccion = load_subcarpeta(sc)
        dfs[sc] = df
        print(f"  ✅ {sc:<35} {len(df.columns):>10}")
    except Exception as e:
        print(f"  ❌ {sc}: {e}")

print(f"\nCargadas: {len(dfs)} / {len(SUBCARPETAS)}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# ## 2. Verificar esquema entre subcarpetas

# CELL ********************

from collections import Counter

schema_map = {sc: tuple(sorted(df.columns)) for sc, df in dfs.items()}
unique_schemas = Counter(schema_map.values())

if len(unique_schemas) == 1:
    cols = list(list(unique_schemas.keys())[0])
    print(f"✅ Todas las subcarpetas tienen el mismo esquema ({len(cols)} columnas)")
else:
    print(f"⚠️  {len(unique_schemas)} esquemas distintos:")
    for i, (schema, count) in enumerate(unique_schemas.most_common(), 1):
        owners = [sc for sc, s in schema_map.items() if s == schema]
        print(f"\n  Esquema {i} ({count} subcarpeta(s)): {owners}")
        print(f"  Columnas: {list(schema)}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# ## 3. Unir todos los años y filtrar a Medellín
# 
# Filtramos por `codigo_municipio = '001'` (más robusto que el nombre).

# CELL ********************

# Gran DataFrame 
df_all = None
for sc, df in dfs.items():
    if df_all is None:
        df_all = df
    else:
        # unionByName - permitir orden diferente
        df_all = df_all.unionByName(df, allowMissingColumns=True)

# Filtrar Medellín por su codigo
df_medellin = (
    df_all
    .withColumn('codigo_municipio', F.trim(F.col('codigo_municipio')))
    .filter(F.col('codigo_municipio') == CODIGO_MEDELLIN)
)

# Cache para reusar
df_medellin.cache()

# Acción
total = df_medellin.count()
print(f"Filas Medellín (todos los años): {total:,}")

print("\nDistribución por año × elección:")
df_medellin.groupBy('anio', 'eleccion').count().orderBy('anio', 'eleccion').show(30, truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# ## 4. Enriquecer `codigo_comuna` con `mmv_bronze`
# 
# Aunque esquema es conservado entre elecciones, elecciones de escala Nacional (2018, 2022 — Presidente, Cámara, Senado) no contiene 'comuna' ni 'codigo_comuna'.
# La tabla `mmv_bronze` tiene granularidad mesa×candidato pero contiene el mapeo  
# `(codigo_municipio, codigo_zona, codigo_puesto) → codigo_comuna`.  
# Extraemos ese mapeo único y hacemos left join.
# Usamos deltatabla del bronze como diccionario.

# CELL ********************

# Leer tabla del Lakehouse
df_bronze_table = spark.read.table(MMV_BRONZE_TABLE)

# Normalizamos nombre en columnas
df_bronze_table = df_bronze_table.toDF(*[c.strip().lower() for c in df_bronze_table.columns])

print(f"Columnas en mmv_bronze: {df_bronze_table.columns}")
print(f"Total filas: {df_bronze_table.count():,}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Matcheamos cada combinación de zonax puesto a comuna
# Se reducen valores sin asignar comuna

df_mapeo_comuna = (
    df_bronze_table
    .filter(F.col('codigo_municipio') == CODIGO_MEDELLIN)
    .filter(F.col('codigo_comuna').isNotNull())
    .filter(F.trim(F.col('codigo_comuna')) != '')   # descartar string vacío
    .select(
        F.lpad(F.trim(F.col('codigo_municipio')), 3, '0').alias('codigo_municipio'),
        F.lpad(F.trim(F.col('codigo_zona')),      3, '0').alias('codigo_zona'),
        F.lpad(F.trim(F.col('codigo_puesto')),    2, '0').alias('codigo_puesto'),
        F.trim(F.col('codigo_comuna')).alias('codigo_comuna_lookup'),
        F.trim(F.col('comuna')).alias('comuna_lookup') if 'comuna' in df_bronze_table.columns
            else F.lit(None).alias('comuna_lookup')
    )
    .dropDuplicates(['codigo_municipio', 'codigo_zona', 'codigo_puesto'])
)

df_mapeo_comuna.cache()
print(f"Mapeos únicos zona×puesto → comuna en Medellín: {df_mapeo_comuna.count():,}")
df_mapeo_comuna.show(10, truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Upper cell correcting non matching values:::::

# CELL ********************

# Zonas x puestos que continúan son comuna, con Join al delta como diccionario 
# codigo_comuna final: si no venía en la data incopora el del matcheo

df_enriched = (
    df_medellin
    .withColumn('codigo_municipio', F.lpad(F.trim(F.col('codigo_municipio')), 3, '0'))
    .withColumn('codigo_zona',      F.lpad(F.trim(F.col('codigo_zona')),      3, '0'))
    .withColumn('codigo_puesto',    F.lpad(F.trim(F.col('codigo_puesto')),    2, '0'))
    .join(df_mapeo_comuna, on=['codigo_municipio','codigo_zona','codigo_puesto'], how='left')
    .withColumn('codigo_comuna_final', F.coalesce(F.col('codigo_comuna'), F.col('codigo_comuna_lookup')))
    .withColumn('comuna_final',        F.coalesce(F.col('comuna'),        F.col('comuna_lookup')))
    .drop('codigo_comuna','comuna','codigo_comuna_lookup','comuna_lookup')
    .withColumnRenamed('codigo_comuna_final','codigo_comuna')
    .withColumnRenamed('comuna_final','comuna')
)
df_enriched.cache()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Diagnóstico: cobertura de codigo_comuna por año × elección DESPUÉS del enriquecimiento
print("Cobertura de codigo_comuna tras enriquecimiento:")
(df_enriched
    .groupBy('anio', 'eleccion')
    .agg(
        F.count('*').alias('total'),
        F.sum(F.when(F.col('codigo_comuna').isNotNull(), 1).otherwise(0)).alias('con_comuna')
    )
    .withColumn('pct', F.round(F.col('con_comuna') / F.col('total') * 100, 1))
    .orderBy('anio', 'eleccion')
    .show(30, truncate=False)
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# En la mayoria de elecciones se conservan 100% de las zonas x puestos donde se produjeron las elecciones (incluidas las Nacionales) pudieron ser matcheadas a una comuna.

# MARKDOWN ********************

# ---
# ## 5. Unir con diccionario de espectro político
# 
# Carga el CSV producido por un experto: asigna a cada movimiento/partido un valor.
# 
# Normalizamos primero los nombres (eliminar `\`, `'`, `"`),  
# y luego mediante left join incorporamos.

# CELL ********************

# Leer el diccionario
df_dict = (
    spark.read
    .option('header', 'true')
    .option('sep', ';')
    .option('encoding', 'UTF-8')
    .option('multiline', 'true')
    .option('escape', '"')
    .csv(DICT_FILE)
)
df_dict = df_dict.toDF(*[c.strip().lower() for c in df_dict.columns])

# Eliminar columnas Unnamed
cols_keep = [c for c in df_dict.columns if not c.startswith('unnamed')and c != '']
df_dict = df_dict.select(*cols_keep)

print(f"Columnas diccionario: {df_dict.columns}")
df_dict.show(10, truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Función de normalización: quita \, ', " y espacios extra (partidos no uniformemente descritos)
def normalize_agrupacion(col):
    cleaned = F.regexp_replace(col, r'[\\\\\'"]', '')
    return F.trim(F.upper(cleaned))


df_dict_clean = (
    df_dict
    .select(
        normalize_agrupacion(F.col('agrupacion_politica')).alias('agrupacion_politica_norm'),
        F.trim(F.upper(F.col('espectro_politico'))).alias('espectro_politico')
    )
    .filter(F.col('espectro_politico').isNotNull())
    .filter(F.col('espectro_politico') != '')
    .dropDuplicates(['agrupacion_politica_norm'])
)

print(f"Agrupaciones con espectro asignado: {df_dict_clean.count()}")
df_dict_clean.groupBy('espectro_politico').count().show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Normalizar la columna en el dataset principal y unir
df_with_espectro = (
    df_enriched
    .withColumn('agrupacion_politica_norm', normalize_agrupacion(F.col('agrupacion_politica')))
    .join(df_dict_clean, on='agrupacion_politica_norm', how='left')
    .drop('agrupacion_politica_norm')
)

# Varificar los no asignados
sin_espectro = df_with_espectro.filter(F.col('espectro_politico').isNull())
n_sin = sin_espectro.count()
print(f"Filas sin espectro tras el join: {n_sin:,}")

print("\nValores únicos de agrupacion_politica sin clasificar:")
sin_espectro.select('agrupacion_politica').distinct().show(50, truncate=False)

# Descartar (corresponden a votos nulos, blancos, casillas totalizadoras)
df_final = df_with_espectro.filter(F.col('espectro_politico').isNotNull())
df_final.cache()

print(f"\nFilas restantes: {df_final.count():,}")
print("\nDistribución por año:")
df_final.groupBy('anio').count().orderBy('anio').show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# ## 6. Exportar merged a Bronze
# 
# Archivo unificado, limpio, asignado a espectro político.
# 
# Ahora a seleccionar columnas relevantes y guardar como CSV

# CELL ********************

COLS_FINALES = [
    'anio',
    'codigo_comuna', 'comuna',
    'codigo_corporacion', 'corporacion',
    'espectro_politico'
]

df_export = df_final.select(*COLS_FINALES)

# Coalesce (llevar a un solo CSV)
(df_export
    .coalesce(1)
    .write
    .mode('overwrite')
    .option('header', 'true')
    .option('encoding', 'UTF-8')
    .csv(MERGED_OUTPUT)
)

print(f"✅ Merged exportado a: {MERGED_OUTPUT}")
print(f"   Filas: {df_export.count():,}")
print(f"   Columnas: {df_export.columns}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# ## 7. Agregar a proporciones por combinación para la unidad de predicción (anio × corporacion × codigo_comuna)
# 
# Para cada combinación, su proporción de votos en cada espectro (tri-componente; suma = 1).

# CELL ********************

# Solo filas con codigo_comuna
df_for_agg = df_final.filter(F.col('codigo_comuna').isNotNull())

print(f"Filas con codigo_comuna: {df_for_agg.count():,} / {df_final.count():,}")
print("\nDistribución de observaciones completas de cada año:")
df_for_agg.groupBy('anio').count().orderBy('anio').show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Primero conteo de votos por combinación y espectro
counts = (
    df_for_agg
    .groupBy('anio', 'corporacion', 'codigo_comuna', 'espectro_politico')
    .count()
    .withColumnRenamed('count', 'votos')
)

# Total en cada combinación con Window
w = Window.partitionBy('anio', 'corporacion', 'codigo_comuna')
counts = counts.withColumn('total_combo', F.sum('votos').over(w))
counts = counts.withColumn('proporcion', F.col('votos') / F.col('total_combo'))

# Una combinación por fila (anio × corporacion × codigo_comuna)
# 3 columnas (izquierda/centro/derecha) es decir por 3 componentes
df_proporciones = (
    counts
    .groupBy('anio', 'corporacion', 'codigo_comuna')
    .pivot('espectro_politico', ['IZQUIERDA', 'CENTRO', 'DERECHA'])
    .agg(F.first('proporcion'))
)

# Renombrar a minúsculas, llenar con 0 restantes
df_proporciones = (
    df_proporciones
    .withColumnRenamed('IZQUIERDA', 'izquierda')
    .withColumnRenamed('CENTRO',    'centro')
    .withColumnRenamed('DERECHA',   'derecha')
    .na.fill({'izquierda': 0.0, 'centro': 0.0, 'derecha': 0.0})
)

df_proporciones.cache()
n_combos = df_proporciones.count()
print(f"✅ {n_combos:,} combinaciones (anio × corporacion × codigo_comuna)")
df_proporciones.orderBy('anio', 'corporacion', 'codigo_comuna').show(20, truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Validación: las proporciones de cada fila deben sumar 1
df_check = df_proporciones.withColumn(
    'suma', F.col('izquierda') + F.col('centro') + F.col('derecha')
)

fuera_rango = df_check.filter(F.abs(F.col('suma') - 1.0) > 1e-6).count()
print(f"Filas donde la suma ≠ 1: {fuera_rango}")
assert fuera_rango == 0, "Hay combinaciones donde las proporciones no suman 1"
print("Proporciones completas (total = 1)")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ---
# ## 8. Exportar tabla final como Delta a Silver

# CELL ********************

(df_proporciones
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save(SILVER_PROPORCIONES)
)

print(f"✅ Delta table escrita en: {SILVER_PROPORCIONES}")

# Verificación: leer de vuelta
df_verify = spark.read.format('delta').load(SILVER_PROPORCIONES)
print(f"   Filas:    {df_verify.count():,}")
print(f"   Columnas: {df_verify.columns}")
df_verify.show(10, truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# delta tabla a /Tables
(df_proporciones
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .saveAsTable('votaciones_proporciones_comunas')
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

spark.sql("SHOW TABLES").show(truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cleanup — liberar memoria del cache
df_medellin.unpersist()
df_mapeo_comuna.unpersist()
df_enriched.unpersist()
df_final.unpersist()
df_proporciones.unpersist()
print("Cache liberado.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
