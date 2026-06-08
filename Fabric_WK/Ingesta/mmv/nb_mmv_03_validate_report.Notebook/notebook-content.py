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

# # nb_mmv_03_validate_report  —  MMV Phase 3: validate + report
# 
# Reads the bronze tables and prints the audit the post-mortem describes (§3.12):
# rows per department×year and year×corporation, the availability flags for year/election
# combinations with no MMV, and the **conservation-law** check (kept + quarantined accounted
# for). Read-only — produces evidence, writes nothing new.

# PARAMETERS CELL ********************

departments = "ANTIOQUIA"
years = "2023,2022,2019,2018"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

%run nb_00_shared_utils

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql import functions as F

bronze = spark.table(T_MMV_BRONZE)
manifest = spark.table(T_MMV_MANIFEST)
try:
    quar = spark.table(T_MMV_QUARANTINE)
    n_quar = quar.count()
except Exception:
    n_quar = 0

n_rows = bronze.count()
log("=" * 70)
log(f"MMV bronze summary — {n_rows:,} rows, {n_quar:,} quarantined")
log("=" * 70)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Rows per department x year
log("\nRows per department x year:")
piv = (bronze.groupBy("departamento").pivot("year").count().orderBy("departamento"))
piv.show(50, truncate=False)

# Rows per year x corporation
log("\nRows per year x corporation:")
(bronze.groupBy("year").pivot("corporacion").count().orderBy("year")).show(50, truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Availability flags: year/election combos recorded with no MMV data.
log("\nAvailability flags (sources with 0 rows / no MMV):")
flagged = manifest.filter((F.col("n_rows") == 0)).select("year", "election", "department", "note")
if flagged.count() == 0:
    log("  (none — every requested source produced MMV data)")
else:
    flagged.show(50, truncate=False)

# Conservation law: manifest kept-count must equal the bronze row count.
man_kept = manifest.agg(F.sum("n_rows")).collect()[0][0] or 0
log("\nConservation-law check:")
log(f"  manifest n_rows total = {man_kept:,}")
log(f"  bronze table rows     = {n_rows:,}")
log(f"  quarantined rows      = {n_quar:,}")
log("  STATUS: " + ("OK ✅ (kept rows reconcile)" if man_kept == n_rows
                    else "MISMATCH ❌ — investigate manifest vs bronze"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
