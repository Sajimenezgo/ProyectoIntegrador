# Fabric notebook source


# MARKDOWN ********************

# # nb_00_shared_utils
# 
# Shared configuration + helpers for the **MMV** and **ECV** bronze ingestion notebooks.
# This notebook is meant to be pulled into the phase notebooks with `%run nb_00_shared_utils`
# (Fabric runs it in the caller's namespace, so every name defined here becomes available).
# 
# It contains **no side effects** — only constants and pure functions. Most helpers are ported
# verbatim from the proven reference script `mmv_antioquia/mmv_antioquia.py` (the row-for-row spec),
# with one deliberate fix folded in: `norm_name` now treats spaces and underscores as equivalent so
# `LA GUAJIRA`/`LA_GUAJIRA` and `SAN ANDRÉS`/`SAN_ANDRES` match across the three election sites
# (post-mortem §7, the top open item).

# CELL ********************

import csv
import re
import unicodedata
from typing import Dict, List, Tuple, Optional

# =============================================================================
# Lakehouse layout — everything is relative to the attached default Lakehouse.
# In Fabric, the attached Lakehouse is mounted at /lakehouse/default; Files live
# under .../Files and managed Delta tables under .../Tables. The phase notebooks
# use these roots so the framework is re-pointable from one place.
# =============================================================================
LH_ROOT = "/lakehouse/default"          # mounted path of the attached Lakehouse
FILES_ROOT = f"{LH_ROOT}/Files"         # local (driver) filesystem view of OneLake Files
FILES_ABFSS = "Files"                   # Spark-relative path (spark.read/write) for Files

# --- MMV file areas ----------------------------------------------------------
MMV_RAW_ZIPS = f"{FILES_ROOT}/raw/mmv/zips"
MMV_RAW_NORMALIZED = f"{FILES_ROOT}/raw/mmv/normalized"
MMV_BRONZE_CSV = f"{FILES_ROOT}/bronze/mmv"
MMV_CONTROL = f"{FILES_ROOT}/bronze/_control"

# --- ECV file areas ----------------------------------------------------------
ECV_RAW = f"{FILES_ROOT}/raw/ecv"
ECV_RAW_NORMALIZED = f"{FILES_ROOT}/raw/ecv/normalized"
ECV_BRONZE_CSV = f"{FILES_ROOT}/bronze/ecv"

# --- Managed (Delta) table names ---------------------------------------------
T_MMV_BRONZE = "mmv_bronze"
T_MMV_MANIFEST = "mmv_manifest"
T_MMV_QUARANTINE = "mmv_quarantine"
# ECV bronze tables are per-year: ecv_bronze_<year> / ecv_dictionary_<year>
T_ECV_MANIFEST = "ecv_manifest"

# =============================================================================
# Default workload — narrow by design, widened only via pipeline parameters.
# =============================================================================
DEFAULT_DEPARTMENTS = "ANTIOQUIA"          # MMV: one department for now
PRIORITY_DEPARTMENT = "ANTIOQUIA"
DEFAULT_MMV_YEARS = [2023, 2022, 2019, 2018]
DEFAULT_ECV_YEARS = [2018, 2019, 2022, 2023]

ORIGIN = "https://observatorio.registraduria.gov.co"
PAGE_URL = f"{ORIGIN}/views/electoral/historicos-resultados.php"

# Engineering knobs (mirrors the reference script) ----------------------------
HTTP_TIMEOUT = 180
DISCOVERY_TIMEOUT = 60
MAX_RETRIES = 3
BACKOFF_BASE = 3.0
POLITE_DELAY = 2.0
CHUNK = 1 << 16
USER_AGENT = (
    "Mozilla/5.0 (compatible; MMV-consolidator/1.0; academic research; +contact: researcher)"
)

# CELL ********************

# =============================================================================
# Canonical 18-field MMV schema + the 4 analysis columns the pipeline adds.
# All kept as TEXT so significant leading zeros ("01", "001", "00000") survive.
# =============================================================================
SCHEMA: List[str] = [
    "codigo_departamento", "departamento", "codigo_municipio", "municipio",
    "codigo_zona", "codigo_puesto", "puesto", "mesa", "codigo_comuna", "comuna",
    "codigo_corporacion", "corporacion", "codigo_circunscripcion",
    "codigo_partido", "agrupacion_politica", "codigo_candidato", "candidato",
    "total_votos",
]
EXTRA_COLS = ["year", "election", "scope", "corporation"]
N_FIELDS = len(SCHEMA)  # 18

# Header cell -> canonical schema, BY NAME (keys are accent-stripped/lower/alnum-only).
# Covers the three real header vocabularies on the wire (Territoriales, Congreso, and the
# abbreviated national Presidencial CSV/XLSX). Ported from mmv_antioquia.py:142-193.
_HEADER_ALIASES: Dict[str, str] = {
    "codigodepartamento": "codigo_departamento",
    "nombredepartamento": "departamento",
    "codigomunicipio": "codigo_municipio",
    "nombremunicipio": "municipio",
    "codigozona": "codigo_zona",
    "codigopuesto": "codigo_puesto",
    "nombrepuesto": "puesto",
    "codigocomuna": "codigo_comuna",
    "nombrecomuna": "comuna",
    "codigocorporacion": "codigo_corporacion",
    "nombrecorporacion": "corporacion",
    "codigocircunscripcion": "codigo_circunscripcion",
    "codigopartido": "codigo_partido",
    "nombrepartido": "agrupacion_politica",
    "codigocandidato": "codigo_candidato",
    "nombrecandidato": "candidato",
    "totalvotos": "total_votos",
    # abbreviated national Presidencial (CSV)
    "dep": "codigo_departamento",
    "depnombre": "departamento",
    "mun": "codigo_municipio",
    "munnombre": "municipio",
    "zona": "codigo_zona",
    "puesto": "codigo_puesto",
    "puesnombre": "puesto",
    "corcodigo": "codigo_corporacion",
    "cornombre": "corporacion",
    "cir": "codigo_circunscripcion",
    "par": "codigo_partido",
    "parnombre": "agrupacion_politica",
    "can": "codigo_candidato",
    "cannombre": "candidato",
    "votos": "total_votos",
    # national Presidencial 2018 .xlsx (code-only sheet)
    "psto": "codigo_puesto",
    "cor": "codigo_corporacion",
    "partido": "agrupacion_politica",
    "codcan": "codigo_candidato",
    # canonical self-aliases (idempotent re-reads)
    "departamento": "departamento",
    "municipio": "municipio",
    "mesa": "mesa",
    "comuna": "comuna",
    "corporacion": "corporacion",
    "candidato": "candidato",
    "agrupacionpolitica": "agrupacion_politica",
}

# =============================================================================
# ECV online source registry — Medellín's open-data publications. URLs are
# irregular per year (different host/path), so they live here as the single
# re-pointable place (override via a pipeline parameter if a link moves).
# Provenance: medata.gov.co (MEData) + medellin.gov.co document center.
#   * 2018 -> medellin.gov.co document-center per-year file (ECV_2018.xlsx, 30,942 rows,
#            has Form/Hogar/Orden identifiers — a clean single wave). NOTE: do NOT use the
#            MEData `encuesta_calidad_vida.csv` for 2018; that one is the stacked 2011-2019
#            panel with no Year/ID columns (361k rows) and cannot be split per year.
#   * 2022/2023 -> medellin.gov.co per-year "Base de datos" + "Diccionario".
#   * 2019 -> SAP portal: microdata is a .rar (no Fabric unrar) -> stage the extracted
#            BD_ECV_2019.xlsx locally; the .xlsm dictionary loads directly online.
# `year_column`: set ONLY if a file genuinely stacks multiple waves — the loader
# then filters rows to the requested year. None = ingest the file as-is (the case
# for all four in-scope years). Every bronze table also carries an `ecv_year`
# lineage column, which is the canonical filter key for downstream silver.
# =============================================================================
ECV_SOURCES: Dict[int, Dict[str, Optional[str]]] = {
    2018: {
        "microdata": "https://www.medellin.gov.co/es/wp-content/uploads/2024/08/ECV_2018.xlsx",
        "dictionary": "https://www.medellin.gov.co/es/wp-content/uploads/2024/08/Diccionario-de-Datos-2018.xlsx",
        "year_column": None,  # proper per-year file (single wave, ~30,942 rows)
        "note": "medellin.gov.co ECV 2018 (per-year; NOT the MEData stacked 2011-2019 csv)",
    },
    2019: {
        # SAP portal serves the microdata as a .rar (contains BD_ECV_2019.xlsx). Fabric has no
        # unrar binary, so the .rar can't be auto-extracted: stage the EXTRACTED xlsx locally
        # (acquire skips download when it's present; convert reads it). The .rar URL is kept as
        # provenance / pilot. The .xlsm dictionary loads directly.
        "microdata": "https://www.medellin.gov.co/irj/go/km/docs/pccdesign/medellin/Temas/PlaneacionMunicipal/Programas/Shared%20Content/Documentos/2021/BD_ECV_2019.rar",
        "dictionary": "https://www.medellin.gov.co/irj/go/km/docs/pccdesign/medellin/Temas/PlaneacionMunicipal/IndicadoresEstadisticas/Shared%20Content/Documento/2022/Diccionario%20de%20datos%20ECV%202019.xlsm",
        "archive": "rar",
        "year_column": None,
        "note": "medellin.gov.co SAP portal; microdata is .rar (no Fabric unrar) -> stage extracted "
                "BD_ECV_2019.xlsx in Files/raw/ecv/2019/; dictionary .xlsm loads directly",
    },
    2022: {
        "microdata": "https://www.medellin.gov.co/es/wp-content/uploads/2023/01/Base-de-Datos-ECV2022.csv",
        "dictionary": "https://www.medellin.gov.co/es/wp-content/uploads/2023/01/Diccionario-de-Datos-ECV2022-1.xlsx",
        "year_column": None,
        "note": "medellin.gov.co ECV 2022",
    },
    2023: {
        "microdata": "https://www.medellin.gov.co/es/wp-content/uploads/2022/08/Viviendas-Hogares-Personas_ECV-2023.csv",
        "dictionary": "https://www.medellin.gov.co/es/wp-content/uploads/2022/08/Diccionario-de-Datos-Encuesta-Calidad-de-Vida-2023.xlsx",
        "year_column": None,
        "note": "medellin.gov.co ECV 2023",
    },
}

# Confirmed-good fallback URLs (Antioquia per-department + national presidential zips).
# Discovery is the extensibility path; for the narrow Antioquia run these cover everything.
FALLBACK_URLS: Dict[str, str] = {
    "2023_territoriales": f"{ORIGIN}/comprimidos/MMV_TERRITORIALES2023_{PRIORITY_DEPARTMENT}.zip",
    "2019_territoriales": f"{ORIGIN}/comprimidosTwo/MMV_TERRITORIALES2019_{PRIORITY_DEPARTMENT}.zip",
    "2022_congreso": f"{ORIGIN}/congreso22/MMV_CONGRESO_2022_{PRIORITY_DEPARTMENT}.zip",
    "2022_presidente_1v": f"{ORIGIN}/anexos/MMV_NACIONAL_PRESIDENTE_2022_1v.zip",
    "2022_presidente_2v": f"{ORIGIN}/anexos/MMV_NACIONAL_PRESIDENTE_2022_2v.zip",
    "2018_presidente_1v": f"{ORIGIN}/anexos/MMV_NACIONAL_PRESIDENTE_2018_1v.zip",
    "2018_presidente_2v": f"{ORIGIN}/anexos/MMV_NACIONAL_PRESIDENTE_2018_2v.zip",
}

# CELL ********************

# =============================================================================
# Small text helpers (ported from mmv_antioquia.py:212-232, with the §7 fix).
# =============================================================================
def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def norm_key(s: str) -> str:
    """Normalise a header cell for alias lookup: accent-strip, lower, drop non-alnum."""
    return re.sub(r"[^a-z0-9]", "", strip_accents(s or "").lower())


def norm_name(s: str) -> str:
    """Normalise a department/value for equality comparison.

    FIX (post-mortem §7): spaces and underscores are treated as equivalent, so
    'LA GUAJIRA' == 'LA_GUAJIRA' and 'SAN ANDRÉS' == 'SAN_ANDRES' when matching
    names discovered across the three election sites.
    """
    s = strip_accents(s or "").upper()
    s = re.sub(r"[\s_]+", " ", s)
    return s.strip()


def slugify(s: str) -> str:
    """Filesystem-safe UPPER slug, e.g. 'Cámara de Representantes' -> 'CAMARA_DE_REPRESENTANTES'."""
    s = strip_accents(s or "").upper()
    s = re.sub(r"[^A-Z0-9]+", "_", s).strip("_")
    return s or "SIN_CORPORACION"


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_departments_param(raw) -> Tuple[object, Optional[set]]:
    """Turn the pipeline `departments` parameter into (request, keep_set).

    'ALL' -> ("ALL", None) keep everything; a comma list -> (list, {norm names}).
    """
    raw = str(raw).strip()
    if raw.upper() == "ALL":
        return "ALL", None
    names = [d.strip() for d in raw.split(",") if d.strip()]
    return names, {norm_name(n) for n in names}


def parse_years_param(raw) -> List[int]:
    if isinstance(raw, (list, tuple)):
        return [int(y) for y in raw]
    return [int(y) for y in str(raw).split(",") if str(y).strip()]


def download_file(url: str, dest, *, min_bytes: int = 64) -> Tuple[str, int]:
    """Generic idempotent file download (used by ECV acquire). Returns (status, bytes).

    Skips when a non-trivial file is already cached; retries transient errors with
    exponential backoff; raises RuntimeError on 404 / empty payload. Driver-side only.
    """
    import time as _time
    import requests as _req
    from pathlib import Path as _Path
    dest = _Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        log(f"    cached: {dest.name} ({dest.stat().st_size:,} bytes) -> skip download")
        return "cached", dest.stat().st_size
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log(f"    GET {url}  (attempt {attempt}/{MAX_RETRIES})")
            with _req.get(url, stream=True, timeout=HTTP_TIMEOUT, headers=headers) as r:
                if r.status_code == 404:
                    raise RuntimeError(f"404 Not Found: {url}")
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                got = 0
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        if chunk:
                            fh.write(chunk); got += len(chunk)
                if got < min_bytes:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f"payload too small ({got} bytes) from {url}")
                tmp.replace(dest)
                log(f"    OK: {dest.name} ({got:,} bytes)")
                return str(r.status_code), got
        except RuntimeError:
            raise
        except Exception as exc:  # transient network/OS error
            last = exc
            if attempt == MAX_RETRIES:
                break
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            log(f"    transient error ({exc}); retrying in {wait:.0f}s"); _time.sleep(wait)
    raise RuntimeError(f"failed after {MAX_RETRIES} attempts: {url} ({last})")

# CELL ********************

# =============================================================================
# Encoding cascade (ported from mmv_antioquia.py:593-651). Pure bytes->text; used
# driver-side AND inside the Spark transcoding UDF so executors decode the same way.
# =============================================================================
def _strip_bom(text: str) -> str:
    return text[1:] if text[:1] == "﻿" else text


def decode_bytes(data: bytes) -> Tuple[str, str]:
    """Return (text, encoding_label) using the documented robust cascade:
    BOM UTF-16/UTF-8-SIG -> BOM-less UTF-16 heuristic -> UTF-8 -> lone-0xD1 'Ñ' fixup -> latin-1.
    """
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            text = _strip_bom(data.decode("utf-16"))
            assert "�" not in text
            return text, "utf-16 (BOM)"
        except (UnicodeDecodeError, AssertionError):
            pass
    if data[:3] == b"\xef\xbb\xbf":
        try:
            text = data.decode("utf-8-sig")
            assert "�" not in text
            return text, "utf-8-sig (BOM)"
        except (UnicodeDecodeError, AssertionError):
            pass

    sample = data[:4096]
    if sample and sample.count(0) > len(sample) * 0.25:
        codec = "utf-16-le" if sample[1::2].count(0) >= sample[0::2].count(0) else "utf-16-be"
        try:
            text = _strip_bom(data.decode(codec))
            assert "�" not in text
            return text, f"{codec} (no BOM, heuristic)"
        except (UnicodeDecodeError, AssertionError):
            pass

    try:
        text = data.decode("utf-8")
        label = "utf-8"
    except UnicodeDecodeError:
        fixed = data.replace(b"\xd1", "Ñ".encode("utf-8"))  # lone 0xD1 -> 'Ñ'
        try:
            text = fixed.decode("utf-8")
            label = "utf-8 (0xD1->N tilde fixup)"
        except UnicodeDecodeError:
            text = data.decode("latin-1")
            label = "latin-1"
    assert "�" not in text, "decoded text contains U+FFFD replacement chars"
    return _strip_bom(text), label


def detect_delimiter(text: str) -> str:
    """Comma vs semicolon, per file (ported from mmv_antioquia.py:657-667)."""
    sample = text[:65536]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        if dialect.delimiter in (",", ";"):
            return dialect.delimiter
    except csv.Error:
        pass
    first = next((ln for ln in sample.splitlines() if ln.strip()), "")
    return ";" if first.count(";") > first.count(",") else ","


def build_header_map(first_row: List[str]) -> Tuple[bool, int, List[str]]:
    """Inspect the first record -> (header_present, width, idx_to_name).
    Ported from mmv_antioquia.py:673-692. Used to map a source header onto SCHEMA by name.
    """
    keys = [norm_key(c) for c in first_row]
    matches = sum(1 for k in keys if k in _HEADER_ALIASES)
    width = len(first_row)
    if matches >= max(3, width // 2):
        idx_to_name = [_HEADER_ALIASES.get(k, (k or f"col{i}")) for i, k in enumerate(keys)]
        return True, width, idx_to_name
    idx_to_name = [SCHEMA[i] if i < N_FIELDS else f"col{i}" for i in range(width)]
    return False, width, idx_to_name

# CELL ********************

# =============================================================================
# Spark schema for the MMV bronze table: 18 canonical + 4 analysis columns, all
# StringType (preserve leading zeros; cast in silver). Imported lazily so this
# notebook is also importable in a plain-Python context for the self-checks.
# =============================================================================
def mmv_bronze_struct():
    from pyspark.sql.types import StructType, StructField, StringType
    cols = SCHEMA + EXTRA_COLS
    return StructType([StructField(c, StringType(), True) for c in cols])


def file_stem(year: int, scope: str, round_label: str, corporation: str) -> str:
    """Output file stem (without 'MMV_' / '.csv'). Mirrors mmv_antioquia.py:838-847."""
    if scope == "national":
        rl = (round_label or "").upper()
        return f"{year}_PRESIDENTE_{rl}".rstrip("_")
    return f"{year}_{slugify(corporation)}"


log("nb_00_shared_utils loaded: schema, aliases, helpers, paths ready.")
