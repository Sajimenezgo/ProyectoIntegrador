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

# # nb_mmv_02_parse_load  —  MMV Phase 2: parse + load (Spark rebuild)
# 
# The parsing/transform core, **rebuilt as Spark DataFrame operations** (the reference is
# `_ingest_records`/`_emit_row` in `mmv_antioquia.py:850-961`):
# 
# * **CSV** staged members → `spark.read.csv(mode="PERMISSIVE", columnNameOfCorruptRecord=...)`;
#   corrupt rows → quarantine. Header cells mapped onto the canonical `SCHEMA` **by name** via
#   `_HEADER_ALIASES`.
# * **XLSX** (2018 presidential) → read via **pandas/openpyxl** (base runtime; no external library)
#   into Spark DataFrames for the FINAL sheet + `DIVIPOL` + `PARTIDOS`; geography/party names
#   recovered via **Spark joins**; the 2v runoff reuses 1v's lookups (cached per year across the loop).
# * Rows filtered to the requested departments by **name** (never code); rows with no department
#   name → quarantine (`unroutable`), never dropped. Adds `year/election/scope/corporation`;
#   national rows split by department.
# * Writes Delta **`mmv_bronze`** (partitioned `departamento,year,corporacion`) + the per-department
#   CSV copy + **`mmv_quarantine`** + **`mmv_manifest`** (the "both" storage, idempotent via dynamic
#   partition overwrite).
# 
# XLSX reading uses **pandas/openpyxl** from the Fabric base runtime — no external Maven library
# (e.g. `spark-excel`) or custom Environment is required.


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

import json
from pathlib import Path
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

departments_request, keep = parse_departments_param(departments)
keep_set = keep  # None => keep all
log(f"MMV parse_load — departments={departments_request!r}  keep={keep_set}")

acquire = json.loads((Path(MMV_CONTROL) / "mmv_acquire.json").read_text(encoding="utf-8"))
log(f"  loaded acquire manifest: {len(acquire)} sources")

# Year-level XLSX lookup caches (1v harvests DIVIPOL/PARTIDOS; 2v reuses them).
geo_ref = {}    # year -> Spark DataFrame [codigo_departamento,codigo_municipio,codigo_zona,codigo_puesto, dep,mun,pue names]
party_ref = {}  # year -> Spark DataFrame [codigo_partido, agrupacion_politica]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =============================================================================
# Canonicalisation: rename a raw-header DataFrame onto SCHEMA by name, then select
# the 18 canonical columns (missing ones filled with ''), all as trimmed strings.
# =============================================================================
def to_canonical(df, corrupt_col="_corrupt_record"):
    present = {}
    for c in df.columns:
        if c == corrupt_col:
            continue
        canon = _HEADER_ALIASES.get(norm_key(c))
        if canon and canon not in present:     # first column wins for a canonical name
            present[canon] = c
    cols = []
    for name in SCHEMA:
        if name in present:
            cols.append(F.trim(F.coalesce(F.col("`%s`" % present[name]).cast(StringType()),
                                          F.lit(""))).alias(name))
        else:
            cols.append(F.lit("").alias(name))
    keep_extra = [corrupt_col] if corrupt_col in df.columns else []
    return df.select(*cols, *[F.col(c) for c in keep_extra])


def strip_dot0(df, cols):
    """spark-excel and pandas both render integer cells as '1.0'; restore '1' (mirrors the
    reference _xlsx_cell_str integer rendering) so codes, vote counts and join keys stay faithful."""
    for c in cols:
        if c in df.columns:
            df = df.withColumn(c, F.regexp_replace(F.col(c), r"^(\d+)\.0$", "$1"))
    return df


def add_analysis_cols(df, year, election, scope, corporation_default):
    if scope == "national":
        df = df.withColumn("corporacion",
                           F.when(F.trim(F.col("corporacion")) == "", F.lit("PRESIDENTE"))
                            .otherwise(F.col("corporacion")))
        corp_col = F.lit("PRESIDENTE")
    else:
        corp_col = F.when(F.trim(F.col("corporacion")) == "", F.lit(corporation_default)) \
                    .otherwise(F.col("corporacion"))
    return (df.withColumn("year", F.lit(str(year)))
              .withColumn("election", F.lit(election))
              .withColumn("scope", F.lit(scope))
              .withColumn("corporation", corp_col))


def split_keep_quarantine(df, src):
    """Return (valid_df, quarantine_df) applying the conservation law:
    corrupt rows and rows with no department name are quarantined, never dropped."""
    base_cols = SCHEMA + EXTRA_COLS
    corrupt = None
    if "_corrupt_record" in df.columns:
        corrupt = (df.filter(F.col("_corrupt_record").isNotNull())
                     .withColumn("reason", F.lit("parse error (PERMISSIVE corrupt record)"))
                     .withColumn("raw", F.col("_corrupt_record")))
        df = df.filter(F.col("_corrupt_record").isNull()).drop("_corrupt_record")
    unroutable = (df.filter(F.trim(F.col("departamento")) == "")
                    .withColumn("reason", F.lit("unroutable: no department name"))
                    .withColumn("raw", F.concat_ws("|", *[F.col(c) for c in SCHEMA])))
    valid = df.filter(F.trim(F.col("departamento")) != "")
    if keep_set is not None:
        # route by NAME: keep only requested departments (national rows get split here)
        valid = valid.filter(_norm_name_udf(F.col("departamento")).isin(list(keep_set)))
    quar_parts = [q for q in (corrupt, unroutable) if q is not None]
    quar = None
    for q in quar_parts:
        q = (q.withColumn("year", F.lit(str(src["year"]))).withColumn("election", F.lit(src["election"]))
              .withColumn("scope", F.lit(src["scope"]))
              .withColumn("department", F.lit(src.get("department") or ""))
              .withColumn("source_key", F.lit(src["key"]))
              .select("year", "election", "scope", "department", "source_key", "reason", "raw"))
        quar = q if quar is None else quar.unionByName(q)
    return valid.select(*base_cols), quar


# norm_name as a Spark UDF so department-name routing runs distributed (matches nb_00 semantics).
import re as _re, unicodedata as _ud
def _norm_name_py(s):
    if s is None:
        return ""
    s = "".join(c for c in _ud.normalize("NFKD", s) if not _ud.combining(c)).upper()
    return _re.sub(r"[\s_]+", " ", s).strip()
_norm_name_udf = F.udf(_norm_name_py, StringType())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =============================================================================
# CSV reader (Spark, PERMISSIVE). Header present is the real shape for every MMV CSV
# on the wire (Territoriales, Congreso, national Presidencial CSV).
# =============================================================================
def read_csv_member(staged_path, delimiter):
    return (spark.read
            .option("header", True).option("sep", delimiter or ",")
            .option("quote", '"').option("escape", '"').option("multiLine", True)
            .option("mode", "PERMISSIVE").option("columnNameOfCorruptRecord", "_corrupt_record")
            .option("encoding", "UTF-8")
            .csv(staged_path.replace(FILES_ROOT, FILES_ABFSS)))


# =============================================================================
# XLSX reader (pandas/openpyxl — base runtime, no external library). Enumerate sheet
# names with openpyxl (cheap metadata), pick the FINAL data sheet + lookups, recover
# names via Spark joins.
# =============================================================================
def _xlsx_sheet_names(local_path):
    import openpyxl
    wb = openpyxl.load_workbook(local_path, read_only=True, data_only=True)
    names = list(wb.sheetnames); wb.close(); return names


def _pick_data_sheet(names):
    norm = [(n, norm_key(n)) for n in names]
    for n, k in norm:
        if "mmv" in k and "final" in k: return n
    for n, k in norm:
        if "mmv" in k and "antes" not in k and "modif" not in k: return n
    for n, k in norm:
        if "mmv" in k: return n
    return names[0]


def _read_sheet(local_path, sheet, header=True):
    """Read ONE worksheet to a string-typed Spark DataFrame.

    Uses pandas/openpyxl (both in the Fabric base runtime) instead of the
    `com.crealytics.spark.excel` data source, so NO external Maven library /
    Environment is required and there is no Spark-version coupling. Cells are read
    as strings (dtype=str); integer-rendered cells like '1.0' are repaired later by
    strip_dot0, exactly as with spark-excel. Empty cells become '' (the downstream
    trim==''/coalesce logic treats '' and null identically).
    """
    import pandas as pd
    from pyspark.sql.types import StructType, StructField, StringType
    pdf = pd.read_excel(local_path, sheet_name=sheet,
                        header=(0 if header else None),
                        dtype=str, engine="openpyxl").fillna("")
    # Stable, unique, string column names (header=False -> positional '0','1',...).
    if header:
        cols, seen = [], {}
        for c in pdf.columns:
            s = str(c)
            if s in seen:
                seen[s] += 1; s = f"{s}_{seen[s]}"
            else:
                seen[s] = 0
            cols.append(s)
        pdf.columns = cols
    else:
        pdf.columns = [str(i) for i in range(pdf.shape[1])]
    schema = StructType([StructField(c, StringType(), True) for c in pdf.columns])
    if pdf.shape[0] == 0:                       # empty sheet -> typed empty DF
        return spark.createDataFrame([], schema)
    return spark.createDataFrame(pdf.astype(str), schema=schema)


def read_xlsx_source(local_path, year):
    names = _xlsx_sheet_names(local_path)
    data_sheet = _pick_data_sheet(names)
    df = strip_dot0(to_canonical(_read_sheet(local_path, data_sheet, header=True), corrupt_col=""), SCHEMA)

    # Harvest this workbook's lookups if present; else reuse the year's cached ones.
    divipol = next((n for n in names if "divipol" in norm_key(n)), None)
    partidos = next((n for n in names if "partido" in norm_key(n)), None)
    if divipol:
        g = _read_sheet(local_path, divipol, header=True)
        gsel = {norm_key(c): c for c in g.columns}
        if all(k in gsel for k in ("dep", "mun", "zona", "psto", "departamento", "municipio")):
            puesto_name = (F.trim(F.col("`%s`" % gsel["puesto"]).cast("string")).alias("g_puesto")
                           if "puesto" in gsel else F.lit("").alias("g_puesto"))
            geo = g.select(
                F.trim(F.col("`%s`" % gsel["dep"]).cast("string")).alias("codigo_departamento"),
                F.trim(F.col("`%s`" % gsel["mun"]).cast("string")).alias("codigo_municipio"),
                F.trim(F.col("`%s`" % gsel["zona"]).cast("string")).alias("codigo_zona"),
                F.trim(F.col("`%s`" % gsel["psto"]).cast("string")).alias("codigo_puesto"),
                F.trim(F.col("`%s`" % gsel["departamento"]).cast("string")).alias("g_departamento"),
                F.trim(F.col("`%s`" % gsel["municipio"]).cast("string")).alias("g_municipio"),
                puesto_name,
            )
            geo = strip_dot0(geo, ["codigo_departamento", "codigo_municipio", "codigo_zona", "codigo_puesto"])
            geo = geo.dropDuplicates(["codigo_departamento", "codigo_municipio", "codigo_zona", "codigo_puesto"])
            geo_ref[year] = geo.cache()
    if partidos:
        p = _read_sheet(local_path, partidos, header=False)
        pc = p.columns
        if len(pc) >= 2:
            party = p.select(
                F.trim(F.col("`%s`" % pc[0]).cast("string")).alias("codigo_partido"),
                F.trim(F.col("`%s`" % pc[1]).cast("string")).alias("p_agrupacion"))
            party = strip_dot0(party, ["codigo_partido"])
            party_ref[year] = party.where(F.col("codigo_partido") != "").dropDuplicates(["codigo_partido"]).cache()

    geo = geo_ref.get(year); party = party_ref.get(year)
    if geo is not None:
        df = (df.join(geo, ["codigo_departamento", "codigo_municipio", "codigo_zona", "codigo_puesto"], "left")
                .withColumn("departamento", F.when(F.trim(F.col("departamento")) == "",
                                                   F.coalesce(F.col("g_departamento"), F.lit(""))).otherwise(F.col("departamento")))
                .withColumn("municipio", F.when(F.trim(F.col("municipio")) == "",
                                                F.coalesce(F.col("g_municipio"), F.lit(""))).otherwise(F.col("municipio")))
                .withColumn("puesto", F.when(F.trim(F.col("puesto")) == "",
                                             F.coalesce(F.col("g_puesto"), F.lit(""))).otherwise(F.col("puesto")))
                .drop("g_departamento", "g_municipio", "g_puesto"))
    if party is not None:
        df = (df.join(party, ["codigo_partido"], "left")
                .withColumn("agrupacion_politica", F.when(F.trim(F.col("agrupacion_politica")) == "",
                            F.coalesce(F.col("p_agrupacion"), F.lit(""))).otherwise(F.col("agrupacion_politica")))
                .drop("p_agrupacion"))
    return df.select(*SCHEMA)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =============================================================================
# Drive every staged source through the Spark core; union into one bronze frame.
# =============================================================================
def default_corp(election):
    return "PRESIDENTE" if "presiden" in election.lower() else slugify(election)


valid_frames = []; quar_frames = []; manifest_rows = []
for src in acquire:
    if not src.get("members"):
        manifest_rows.append((src["year"], src.get("department") or "", "", src["scope"], src.get("url") or "",
                              src.get("http_status", ""), int(src.get("bytes") or 0), src.get("sha256", ""),
                              0, 0, src["election"], src.get("note") or "no data member"))
        continue
    log(f"\n=== parse {src['key']} ===")
    dcorp = default_corp(src["election"])
    for m in src["members"]:
        try:
            if m["format"] == "xlsx":
                df = read_xlsx_source(m["staged"], src["year"])
            else:
                raw = read_csv_member(m["staged"], m["delimiter"])
                df = to_canonical(raw, corrupt_col="_corrupt_record")
            df = add_analysis_cols(df, src["year"], src["election"], src["scope"], dcorp)
            valid, quar = split_keep_quarantine(df, src)
            nkept = valid.count()
            nquar = quar.count() if quar is not None else 0
            valid_frames.append(valid)
            if quar is not None:
                quar_frames.append(quar)
            manifest_rows.append((src["year"], src.get("department") or "",
                                  "", src["scope"], src.get("url") or "", src.get("http_status", ""),
                                  int(src.get("bytes") or 0), src.get("sha256", ""), nkept, nquar,
                                  src["election"], f"member={m['member']}; enc={m['encoding']}"))
            log(f"    {m['member']}: kept={nkept:,}  quarantined={nquar:,}")
        except Exception as exc:                       # never let one member sink the run
            manifest_rows.append((src["year"], src.get("department") or "", "", src["scope"],
                                  src.get("url") or "", "parse_error", int(src.get("bytes") or 0),
                                  src.get("sha256", ""), 0, 0, src["election"],
                                  f"parse error in {m['member']}: {exc!r}"[:500]))
            log(f"    ERROR parsing {m['member']}: {exc!r}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =============================================================================
# Write the bronze outputs (idempotent: dynamic partition overwrite).
# =============================================================================
from functools import reduce

if valid_frames:
    # Cache the union so the per-department CSV-copy loop below doesn't re-run the
    # (driver-side) xlsx reads + joins once per group. Materialize once.
    bronze = reduce(lambda a, b: a.unionByName(b), valid_frames).cache()
    bronze.count()
    # NOTE: 'overwriteSchema' is intentionally NOT set here — Delta forbids combining it
    # with dynamic partition overwrite ([DELTA_OVERWRITE_SCHEMA_WITH_DYNAMIC_PARTITION_OVERWRITE]).
    # The bronze schema is fixed (SCHEMA + EXTRA_COLS, all strings), so no schema evolution is
    # needed. If you ever DO change the schema, drop the table once (or run a one-off static
    # overwrite) before re-running. Dynamic partition overwrite keeps reruns idempotent per dept.
    (bronze.write.format("delta").mode("overwrite")
           .partitionBy("departamento", "year", "corporacion")
           .saveAsTable(T_MMV_BRONZE))
    total = spark.table(T_MMV_BRONZE).count()
    log(f"\nWrote {T_MMV_BRONZE}: {total:,} rows (current table total).")

    # "Both": per-department CSV copy at bronze/mmv/<DEPT>/MMV_<stem>.csv.
    # IMPORTANT: read the copy source from the persisted Delta TABLE, not the in-memory
    # `bronze` frame. The xlsx path builds frames via spark.createDataFrame(pandas), which
    # inlines data into the logical plan; with coalesce(1) that becomes one oversized task
    # (> spark.rpc.message.maxSize). Reading from Delta streams from files instead.
    bronze_tbl = spark.table(T_MMV_BRONZE)
    groups = (bronze_tbl.select("departamento", "year", "scope", "election", "corporation")
                        .dropDuplicates().collect())
    for g in groups:
        dept_slug = slugify(g["departamento"])
        rl = "1V" if "1v" in (g["election"] or "").lower() else ("2V" if "2v" in (g["election"] or "").lower() else "")
        stem = file_stem(int(g["year"]), g["scope"], rl, g["corporation"])
        out = f"{FILES_ABFSS}/bronze/mmv/{dept_slug}/MMV_{stem}"
        sub = bronze_tbl.filter((F.col("departamento") == g["departamento"]) & (F.col("year") == g["year"]) &
                                (F.col("corporation") == g["corporation"]) & (F.col("election") == g["election"]))
        sub.coalesce(1).write.option("header", True).mode("overwrite").csv(out)
    log(f"  wrote {len(groups)} per-department CSV folders under bronze/mmv/")
else:
    log("\nNo valid MMV frames produced — check the acquire manifest.")

if quar_frames:
    quarantine = reduce(lambda a, b: a.unionByName(b), quar_frames)
    quarantine.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(T_MMV_QUARANTINE)
    log(f"Wrote {T_MMV_QUARANTINE}: {spark.table(T_MMV_QUARANTINE).count():,} rows.")
else:
    log(f"{T_MMV_QUARANTINE}: 0 rows (conservation law: nothing quarantined).")

# manifest (one row per processed source/member)
man_cols = ["year", "department", "corporation", "scope", "source_url", "http_status",
            "bytes", "sha256", "n_rows", "n_quarantined", "election", "note"]
spark.createDataFrame(manifest_rows, man_cols) \
     .write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(T_MMV_MANIFEST)
log(f"Wrote {T_MMV_MANIFEST}: {len(manifest_rows)} rows.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
