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

# # nb_90_selftest  —  in-Fabric test harness for the MMV + ECV bronze framework
# 
# Runs **inside Fabric** (it needs the platform `spark` and `%run`). Import it like any other
# notebook (attach the same default Lakehouse), then run top-to-bottom. It prints a PASS/FAIL
# line per check and **raises at the end if anything failed**, so it can also be dropped into a
# pipeline as a gate activity.
# 
# Four layers, matching how we want to test the pipeline:
# 
# 1. **Unit** — the pure helpers in `nb_00_shared_utils` (encoding cascade, header aliasing,
#    `norm_name` §7 fix, slugify, param parsing, `file_stem`). No datasource, no Spark.
# 2. **I/O contracts** — structure *and* values of inputs/outputs with the datasources **mocked**
#    (in-memory bytes/strings; the `ECV_SOURCES` + `SCHEMA` contracts). No network, no Lakehouse.
# 3. **Transformations** — the real Fabric `spark`: header→canonical mapping, the trim/coalesce
#    contract, department-name routing (incl. the §7 `SAN ANDRÉS`/`SAN_ANDRES` case) and the
#    quarantine **conservation law** (kept + quarantined == total), on tiny in-memory DataFrames.
# 4. **Orchestration / outputs** — after a real pipeline run, assert the bronze tables exist with
#    the right shape and that the conservation law holds end-to-end. Guarded by `run_integration`
#    so layers 1–3 are runnable before any pipeline has executed.


# PARAMETERS CELL ********************

run_integration = True           # set True only AFTER pl_mmv_bronze / pl_ecv_bronze have run
mmv_departments = "ANTIOQUIA"        # expected department(s) in mmv_bronze (layer 4)
ecv_years = "2018,2019,2022,2023"    # expected ecv_bronze_<year> tables (layer 4)

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

# =============================================================================
# Tiny assertion harness: record every check, print a per-check line, summarise,
# and raise at the end if any failed (so a pipeline gate activity goes red).
# =============================================================================
_RESULTS = []   # (layer, name, passed, detail)


def check(layer, name, condition, detail=""):
    passed = bool(condition)
    _RESULTS.append((layer, name, passed, detail))
    mark = "PASS" if passed else "FAIL"
    log(f"  [{mark}] L{layer} · {name}" + (f"  — {detail}" if detail and not passed else ""))
    return passed


def check_eq(layer, name, got, expected):
    return check(layer, name, got == expected, f"got={got!r} expected={expected!r}")


def section(title):
    log("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =============================================================================
# LAYER 1 — unit tests of the shared helpers (pure; no datasource, no Spark)
# =============================================================================
section("LAYER 1 — shared helpers (nb_00_shared_utils)")

# text helpers
check_eq(1, "strip_accents", strip_accents("Cámara"), "Camara")
check_eq(1, "norm_key", norm_key("Código Departamento"), "codigodepartamento")
check_eq(1, "slugify corporation", slugify("Cámara de Representantes"), "CAMARA_DE_REPRESENTANTES")
check_eq(1, "slugify empty -> sentinel", slugify(""), "SIN_CORPORACION")

# norm_name §7 fix: space and underscore equivalent
check(1, "norm_name §7 SAN ANDRES", norm_name("SAN ANDRÉS") == norm_name("SAN_ANDRES"),
      "space/underscore must compare equal")
check_eq(1, "norm_name collapse+upper", norm_name("  la__guajira "), "LA GUAJIRA")

# param parsing
_req, _keep = parse_departments_param("ALL")
check(1, "parse_departments ALL", _req == "ALL" and _keep is None)
_req, _keep = parse_departments_param("Antioquia, Atlántico")
check(1, "parse_departments list+keepset",
      _req == ["Antioquia", "Atlántico"] and _keep == {"ANTIOQUIA", "ATLANTICO"})
check_eq(1, "parse_years string", parse_years_param("2023,2022,2019,2018"), [2023, 2022, 2019, 2018])
check_eq(1, "parse_years list", parse_years_param([2018, 2019]), [2018, 2019])

# file stem (output naming contract)
check_eq(1, "file_stem national 1v", file_stem(2018, "national", "1v", "PRESIDENTE"), "2018_PRESIDENTE_1V")
check_eq(1, "file_stem department", file_stem(2023, "department", "", "Asamblea de Antioquia"),
         "2023_ASAMBLEA_DE_ANTIOQUIA")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =============================================================================
# LAYER 2 — I/O contracts with datasources MOCKED (in-memory bytes/strings only)
# =============================================================================
section("LAYER 2 — input/output contracts (mocked datasources)")

# --- encoding cascade: the four real shapes we ingest, as raw bytes ----------
_t, _enc = decode_bytes("ANTIOQUIA;NIÑO".encode("utf-8"))
check(2, "decode utf-8", _t == "ANTIOQUIA;NIÑO" and _enc.startswith("utf-8"))
_t, _enc = decode_bytes("HOLA".encode("utf-16"))                       # has a BOM
check(2, "decode utf-16 BOM", _t == "HOLA" and "utf-16" in _enc)
_t, _enc = decode_bytes(b"MUNICIPIO DE NI\xd1O")                       # lone 0xD1 'Ñ'
check(2, "decode lone-0xD1 Ñ fixup", _t == "MUNICIPIO DE NIÑO" and "0xD1" in _enc)
_t, _enc = decode_bytes("Bogotá".encode("latin-1"))                    # latin-1 fallback
check(2, "decode latin-1 fallback", _t == "Bogotá" and _enc == "latin-1")
check(2, "decode never yields U+FFFD", "�" not in decode_bytes("Ñoño".encode("utf-8"))[0])

# --- delimiter detection -----------------------------------------------------
check_eq(2, "detect ';' delimiter", detect_delimiter("a;b;c\n1;2;3"), ";")
check_eq(2, "detect ',' delimiter", detect_delimiter("a,b,c\n1,2,3"), ",")

# --- header map: structure AND values, for the 3 real header vocabularies ----
_present, _w, _names = build_header_map(["Codigo Departamento", "Nombre Departamento", "Total Votos"])
check(2, "header territoriales mapped",
      _present and _names == ["codigo_departamento", "departamento", "total_votos"])
_present, _w, _names = build_header_map(["DEP", "MunNombre", "VOTOS"])   # abbreviated presidential
check(2, "header presidential aliased",
      _present and _names == ["codigo_departamento", "municipio", "total_votos"])
_present, _w, _names = build_header_map(["01", "ANTIOQUIA", "005"])      # headerless -> SCHEMA order
check(2, "headerless falls back to SCHEMA order", (not _present) and _names == SCHEMA[:3])

# --- schema + registry contracts (the output/landing contracts) --------------
check(2, "SCHEMA width == 18", N_FIELDS == 18 and len(SCHEMA) == 18)
check_eq(2, "EXTRA_COLS", EXTRA_COLS, ["year", "election", "scope", "corporation"])
check(2, "ECV_SOURCES has 4 in-scope years", set(ECV_SOURCES) == {2018, 2019, 2022, 2023})
check(2, "ECV_SOURCES entries well-formed",
      all("microdata" in c and "year_column" in c for c in ECV_SOURCES.values()))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =============================================================================
# LAYER 3 — transformations on the real Fabric Spark (tiny in-memory DataFrames)
# These mirror the contract that nb_mmv_02_parse_load.to_canonical / split_keep_
# quarantine must satisfy. (To bind them to the *exact* functions, see the README
# step that exposes those defs from nb_mmv_02 — this validates the same behaviour.)
# =============================================================================
section("LAYER 3 — transformations (Spark)")

from pyspark.sql import functions as F
from pyspark.sql.types import StringType


def _canonicalize(df):
    """Header→SCHEMA by name, trimmed strings, missing cols filled '' — the
    nb_mmv_02 to_canonical contract."""
    present = {}
    for c in df.columns:
        canon = _HEADER_ALIASES.get(norm_key(c))
        if canon and canon not in present:
            present[canon] = c
    cols = []
    for name in SCHEMA:
        if name in present:
            cols.append(F.trim(F.coalesce(F.col("`%s`" % present[name]).cast(StringType()),
                                          F.lit(""))).alias(name))
        else:
            cols.append(F.lit("").alias(name))
    return df.select(*cols)


# raw frame with messy headers + whitespace + a missing canonical column (mesa)
raw = spark.createDataFrame(
    [(" 05 ", "ANTIOQUIA", "001", "MEDELLIN", " 100 ")],
    ["Codigo Departamento", "Nombre Departamento", "Codigo Municipio", "Nombre Municipio", "Total Votos"],
)
canon = _canonicalize(raw)
check_eq(3, "canonical columns == SCHEMA", canon.columns, SCHEMA)
_r = canon.first()
check_eq(3, "value trimmed (codigo_departamento)", _r["codigo_departamento"], "05")
check_eq(3, "value mapped (total_votos)", _r["total_votos"], "100")
check_eq(3, "missing canonical col filled ''", _r["mesa"], "")

# --- conservation law + department-name routing (incl. §7) -------------------
routed = spark.createDataFrame(
    [("ANTIOQUIA",), ("SAN ANDRÉS",), ("SAN_ANDRES",), ("",)],
    ["departamento"],
)
valid = routed.filter(F.trim(F.col("departamento")) != "")
quar = routed.filter(F.trim(F.col("departamento")) == "")
check(3, "conservation law (kept+quar==total)",
      valid.count() + quar.count() == routed.count())
check_eq(3, "empty department quarantined (not dropped)", quar.count(), 1)

# route by NAME with the §7-normalised key, exactly like split_keep_quarantine
_norm_udf = F.udf(lambda s: norm_name(s) if s is not None else "", StringType())
keep_set = {norm_name("SAN ANDRÉS")}
kept_sa = valid.filter(_norm_udf(F.col("departamento")).isin(list(keep_set)))
check_eq(3, "§7 routing matches both spellings of SAN ANDRES", kept_sa.count(), 2)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =============================================================================
# LAYER 4 — orchestration / output assertions (run AFTER a real pipeline run)
# Guarded by run_integration so layers 1–3 work before anything has executed.
# =============================================================================
section("LAYER 4 — orchestration & bronze outputs (run_integration=%s)" % run_integration)

if not run_integration:
    log("  (skipped — set run_integration=True after pl_mmv_bronze / pl_ecv_bronze have run)")
else:
    def table_exists(name):
        try:
            spark.table(name).limit(1).count()
            return True
        except Exception:
            return False

    # --- MMV outputs ---------------------------------------------------------
    if check(4, "mmv_bronze exists", table_exists(T_MMV_BRONZE)):
        bronze = spark.table(T_MMV_BRONZE)
        n_bronze = bronze.count()
        check(4, "mmv_bronze non-empty", n_bronze > 0, f"rows={n_bronze}")
        check(4, "mmv_bronze has full schema",
              set(SCHEMA + EXTRA_COLS).issubset(set(bronze.columns)))
        want = {norm_name(d) for d in parse_departments_param(mmv_departments)[0]} \
            if parse_departments_param(mmv_departments)[1] else None
        if want:
            got = {norm_name(r["departamento"]) for r in bronze.select("departamento").distinct().collect()}
            check(4, "mmv_bronze only requested departments", got.issubset(want), f"got={got}")

        if check(4, "mmv_manifest exists", table_exists(T_MMV_MANIFEST)):
            man = spark.table(T_MMV_MANIFEST)
            man_kept = man.agg(F.sum("n_rows")).collect()[0][0] or 0
            check(4, "MMV conservation law (manifest n_rows == bronze count)",
                  man_kept == n_bronze, f"manifest={man_kept} bronze={n_bronze}")

        n_quar = spark.table(T_MMV_QUARANTINE).count() if table_exists(T_MMV_QUARANTINE) else 0
        check(4, "mmv_quarantine clean (==0 for clean sources)", n_quar == 0, f"quarantined={n_quar}")

    # --- ECV outputs ---------------------------------------------------------
    for y in parse_years_param(ecv_years):
        t = f"ecv_bronze_{y}"
        if check(4, f"{t} exists", table_exists(t)):
            n = spark.table(t).count()
            cols = set(spark.table(t).columns)
            check(4, f"{t} non-empty", n > 0, f"rows={n}")
            check(4, f"{t} has lineage cols",
                  {"ecv_year", "source_file", "ingested_ts"}.issubset(cols))

    if check(4, "ecv_manifest exists", table_exists(T_ECV_MANIFEST)):
        years_in = {r["ecv_year"] for r in spark.table(T_ECV_MANIFEST).select("ecv_year").collect()}
        check(4, "ecv_manifest covers requested years",
              {str(y) for y in parse_years_param(ecv_years)}.issubset(years_in), f"have={years_in}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =============================================================================
# SUMMARY — print totals and FAIL LOUDLY so a pipeline gate activity goes red.
# =============================================================================
section("SUMMARY")
_total = len(_RESULTS)
_failed = [r for r in _RESULTS if not r[2]]
by_layer = {}
for layer, name, passed, _detail in _RESULTS:
    d = by_layer.setdefault(layer, [0, 0])
    d[0 if passed else 1] += 1
for layer in sorted(by_layer):
    p, f = by_layer[layer]
    log(f"  Layer {layer}: {p} passed, {f} failed")
log(f"\n  TOTAL: {_total - len(_failed)}/{_total} passed")

if _failed:
    for layer, name, _p, detail in _failed:
        log(f"   FAILED  L{layer} · {name}  — {detail}")
    raise AssertionError(f"{len(_failed)} of {_total} self-test checks FAILED (see log above).")
log("\n  ALL CHECKS PASSED ✅")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
