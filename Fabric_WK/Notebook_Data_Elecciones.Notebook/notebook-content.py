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
# META     }
# META   }
# META }

# CELL ********************

# =========================================================
# NOTEBOOK 1: Cargar Elecciones Antioquia a Delta Table
# =========================================================
# 1. Leer todos los archivos CSV de la carpeta ANTIOQUIA a la vez.
# Se agrega "Files/" al inicio porque en Fabric los archivos del Lakehouse viven ahí.
df_elecciones = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("Files/Bronze/DATA/ANTIOQUIA/MMV_*.csv")
# 2. Definir la ruta donde se guardará la Delta Table.
# Se usa "Tables/" para que Fabric la reconozca automáticamente como una tabla manejada.
ruta_delta_elecciones = "Tables/elecciones_antioquia"


# 2. Guardar y registrar como Tabla Delta particionada
print("Escribiendo y registrando tabla Delta...")
df_elecciones.write.format("delta") \
    .mode("overwrite") \
    .partitionBy("year", "corporation") \
    .saveAsTable("elecciones_antioquia") # <--- AQUI ESTÁ EL CAMBIO

print("¡Tabla de elecciones creada y registrada con éxito!")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =========================================================
# NOTEBOOK 2: Crear Diccionario de Comunas en Delta Table
# =========================================================
from pyspark.sql.types import StructType, StructField, StringType

data_comunas = [
    ("1", "Popular"),
    ("2", "Santa Cruz"),
    ("3", "Manrique"),
    ("4", "Aranjuez"),
    ("5", "Castilla"),
    ("6", "Doce de Octubre"),
    ("7", "Robledo"),
    ("8", "Villa Hermosa"),
    ("9", "Buenos Aires"),
    ("10", "La Candelaria"),
    ("11", "Laureles - Estadio"),
    ("12", "La América"),
    ("13", "San Javier"),
    ("14", "El Poblado"),
    ("15", "Guayabal"),
    ("16", "Belén"),
    ("50", "Palmitas"),
    ("60", "San Cristóbal"),
    ("70", "Altavista"),
    ("80", "San Antonio de Prado"),
    ("90", "Santa Elena")
]

schema_comunas = StructType([
    StructField("codigo_comuna", StringType(), True),
    StructField("nombre_comuna", StringType(), True)
])

df_comunas = spark.createDataFrame(data=data_comunas, schema=schema_comunas)

print("Creando y registrando tabla del diccionario de comunas...")

# Guardar y registrar como tabla directamente
df_comunas.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("diccionario_comunas") # <--- AQUI ESTÁ EL CAMBIO

print("¡Diccionario de comunas guardado y registrado con éxito!")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
