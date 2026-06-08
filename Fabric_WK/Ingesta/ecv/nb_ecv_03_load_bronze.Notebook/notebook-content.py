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

# # nb_ecv_03_load_bronze  —  ECV Phase 3: load per-year bronze (Spark)
# 
# Reads the converted `microdata.csv` for one `year` (all columns as `StringType`, faithful at
# bronze) and writes Delta **`ecv_bronze_<year>`** as-is plus lineage columns
# (`ecv_year`, `source_file`, `ingested_ts`), a CSV copy under `Files/bronze/ecv/<year>/`, and the
# **`ecv_dictionary_<year>`** reference table. One table per year — no cross-year schema
# reconciliation at bronze (that's silver's job). Updates the shared **`ecv_manifest`** idempotently.

# PARAMETERS CELL ********************

year = 2018

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

import re
import json
from pathlib import Path
from pyspark.sql import functions as F

year = int(year)
norm_dir = Path(ECV_RAW_NORMALIZED) / str(year)
meta = json.loads((norm_dir / "_meta.json").read_text(encoding="utf-8")) \
    if (norm_dir / "_meta.json").exists() else {"year": year, "base_file": None, "note": "no _meta.json"}
log(f"ECV load — year={year}  base_file={meta.get('base_file')}")


def sanitize_columns(df):
    """Make column names Delta-safe (no space , ; {} () \\n \\t =). Keep them recognisable."""
    new = []
    seen = {}
    for c in df.columns:
        s = re.sub(r"[ ,;{}()\n\t=]+", "_", c).strip("_") or "col"
        if s in seen:
            seen[s] += 1; s = f"{s}_{seen[s]}"
        else:
            seen[s] = 0
        new.append(s)
    return df.toDF(*new)


def read_csv(path_local):
    return (spark.read.option("header", True).option("inferSchema", False)
            .option("multiLine", True).option("quote", '"').option("escape", '"')
            .option("encoding", "UTF-8")
            .csv(path_local.replace(FILES_ROOT, FILES_ABFSS)))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

micro = norm_dir / "microdata.csv"
if not micro.exists():
    log(f"  microdata.csv missing for {year}; run nb_ecv_02_convert first. ({meta.get('note','')})")
else:
    df = sanitize_columns(read_csv(str(micro)))

    # Optional wave filter: if this year's source genuinely stacks multiple waves,
    # ECV_SOURCES[year]["year_column"] names the column to filter to `year`. None for
    # all in-scope years (2018 is a single wave). `ecv_year` lineage is added regardless.
    ycol = (ECV_SOURCES.get(year) or {}).get("year_column")
    if ycol:
        ysan = re.sub(r"[ ,;{}()\n\t=]+", "_", ycol).strip("_")
        if ysan in df.columns:
            before = df.count()
            df = df.filter(F.col(ysan).cast("string") == str(year))
            log(f"  year filter on '{ysan}': kept {df.count():,} of {before:,} rows for {year}")
        else:
            log(f"  WARNING: year_column '{ycol}' not found in microdata columns; no filter applied")

    df = (df.withColumn("ecv_year", F.lit(str(year)))
            .withColumn("source_file", F.lit(meta.get("base_file") or ""))
            .withColumn("ingested_ts", F.current_timestamp()))
    table = f"ecv_bronze_{year}"
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
    n = spark.table(table).count()
    log(f"  wrote {table}: {n:,} rows x {len(df.columns)} cols")

    # "Both": CSV copy under bronze/ecv/<year>/
    out = f"{FILES_ABFSS}/bronze/ecv/{year}/microdata"
    df.coalesce(1).write.option("header", True).mode("overwrite").csv(out)
    log(f"  wrote CSV copy -> {out}")

    # dictionary reference table (if present)
    dct = norm_dir / "dictionary.csv"
    if dct.exists():
        ddf = sanitize_columns(read_csv(str(dct))).withColumn("ecv_year", F.lit(str(year)))
        dtable = f"ecv_dictionary_{year}"
        ddf.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(dtable)
        log(f"  wrote {dtable}: {ddf.count():,} rows")
    else:
        log("  (no dictionary.csv to load)")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =============================================================================
# Idempotent manifest upsert: replace this year's row(s) in ecv_manifest.
# =============================================================================
row = [(str(year), meta.get("base_file") or "", meta.get("dictionary_file") or "",
        int(meta.get("microdata_rows") or 0), int(meta.get("microdata_cols") or 0),
        meta.get("note") or "")]
cols = ["ecv_year", "base_file", "dictionary_file", "microdata_rows", "microdata_cols", "note"]
new_row = spark.createDataFrame(row, cols)

try:
    existing = spark.table(T_ECV_MANIFEST).filter(F.col("ecv_year") != str(year))
    combined = existing.unionByName(new_row)
except Exception:
    combined = new_row
combined.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(T_ECV_MANIFEST)
log(f"  updated {T_ECV_MANIFEST} for year {year}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
