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

# # nb_mmv_01_acquire  —  MMV Phase 1: acquire + stage
# 
# **Driver-side only** (the three things that genuinely aren't DataFrame operations):
# 1. **Download** the per-department + national presidential zips idempotently into
#    `Files/raw/mmv/zips/` (cache hit → skip; retries + backoff; zip validation).
# 2. **Extract** the data member(s) from each zip.
# 3. **Byte-transcode** messy CSV members (UTF-16 BOM / latin-1 / lone-0xD1 'Ñ') to clean
#    UTF-8 staging under `Files/raw/mmv/normalized/<source>/`. `.xlsx` workbooks are copied
#    through verbatim for `spark-excel` to read in Phase 2.
# 
# All *structural* parsing (header mapping, validation, joins, routing, quarantine) happens
# in Phase 2 on Spark. This notebook writes an **acquire manifest** (`_control/mmv_acquire.json`)
# describing every staged source so Phase 2 can read it without re-deriving anything.
# 
# Default workload: **ANTIOQUIA**. Other departments are the extensibility seam (parameter).

# PARAMETERS CELL ********************

# Pipeline parameters (overridden by the Data Pipeline; defaults make the notebook runnable alone)
departments = "ANTIOQUIA"          # "ALL" or comma list, e.g. "ANTIOQUIA,ATLANTICO"
years = "2023,2022,2019,2018"      # comma list

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

import os
import json
import time
import hashlib
import zipfile
from pathlib import Path
import requests

departments_request, _keep = parse_departments_param(departments)
year_set = set(parse_years_param(years))
log(f"MMV acquire — departments={departments_request!r}  years={sorted(year_set)}")


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*",
                      "Accept-Language": "es-CO,es;q=0.9,en;q=0.8"})
    return s


class DownloadError(RuntimeError):
    pass


def _sha256_and_size(path: Path):
    h = hashlib.sha256(); n = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk); n += len(chunk)
    return h.hexdigest(), n


def download_zip(session, url, dest_dir: Path):
    """Idempotent download with retries/backoff + zip validation. Ported from
    mmv_antioquia.py:488-561, retargeted to the Lakehouse Files path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / url.rstrip("/").split("/")[-1]
    if dest.exists() and zipfile.is_zipfile(dest):
        sha, size = _sha256_and_size(dest)
        log(f"    cached: {dest.name} ({size:,} bytes) -> skip download")
        return dest, "cached", size, sha

    tmp = dest.with_suffix(dest.suffix + ".part")
    last_exc = None; status_code = "ERR"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log(f"    GET {url}  (attempt {attempt}/{MAX_RETRIES})")
            with session.get(url, stream=True, timeout=HTTP_TIMEOUT) as r:
                status_code = str(r.status_code)
                if r.status_code == 404:
                    raise DownloadError(f"404 Not Found: {url}")
                if r.status_code >= 500:
                    raise requests.HTTPError(f"server {r.status_code}")
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)); got = 0
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        if chunk:
                            fh.write(chunk); got += len(chunk)
                if total and got < total:
                    raise requests.ConnectionError(f"short read: {got} of {total} bytes")
            if not zipfile.is_zipfile(tmp):
                head = tmp.read_bytes()[:200]; tmp.unlink(missing_ok=True)
                raise DownloadError(f"payload from {url} is not a valid ZIP (HTTP {status_code}); head={head!r}")
            tmp.replace(dest)
            sha, size = _sha256_and_size(dest)
            log(f"    OK: {dest.name} ({size:,} bytes)")
            return dest, status_code, size, sha
        except DownloadError:
            tmp.unlink(missing_ok=True); raise
        except (requests.RequestException, OSError) as exc:
            last_exc = exc; tmp.unlink(missing_ok=True)
            if attempt == MAX_RETRIES:
                break
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            log(f"    transient error ({exc}); retrying in {wait:.0f}s"); time.sleep(wait)
    raise DownloadError(f"Failed after {MAX_RETRIES} attempts: {url} ({last_exc})")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =============================================================================
# Build the ordered source list for the requested departments + national files.
# Antioquia + national URLs come from FALLBACK_URLS (confirmed-good). Other
# departments use the URL template as the extensibility seam (accented names need
# discovery — out of scope for the narrow run; a 404 is recorded, never fatal).
# =============================================================================
def dept_url(year, election, dept_slug):
    if dept_slug == slugify(PRIORITY_DEPARTMENT):
        return {("2023", "Territoriales"): FALLBACK_URLS["2023_territoriales"],
                ("2019", "Territoriales"): FALLBACK_URLS["2019_territoriales"],
                ("2022", "Congreso"): FALLBACK_URLS["2022_congreso"]}.get((str(year), election))
    base = {("2023", "Territoriales"): f"{ORIGIN}/comprimidos/MMV_TERRITORIALES2023_",
            ("2019", "Territoriales"): f"{ORIGIN}/comprimidosTwo/MMV_TERRITORIALES2019_",
            ("2022", "Congreso"): f"{ORIGIN}/congreso22/MMV_CONGRESO_2022_"}.get((str(year), election))
    return f"{base}{dept_slug}.zip" if base else None


def build_sources(departments_request, year_set):
    if departments_request == "ALL":
        # Narrow run: ALL is not the supported workload yet; fall back to the priority dept.
        depts = [PRIORITY_DEPARTMENT]
        log("  note: departments=ALL not exercised in the narrow run; using ANTIOQUIA. "
            "Extend via discover_sources (see nb_00) when widening the workload.")
    else:
        depts = departments_request
    # priority dept first
    depts = sorted(set(depts), key=lambda d: (norm_name(d) != norm_name(PRIORITY_DEPARTMENT), norm_name(d)))

    sources = []
    for d in depts:
        ds = slugify(d)
        for year, election in [(2023, "Territoriales"), (2022, "Congreso"), (2019, "Territoriales")]:
            if year in year_set:
                sources.append(dict(year=year, election=election, scope="department",
                                    department=d, round_label="",
                                    url=dept_url(year, election, ds)))
    # national presidential (department-independent; parsed once, filtered later)
    nat = [(2022, "1v", "2022_presidente_1v"), (2022, "2v", "2022_presidente_2v"),
           (2018, "1v", "2018_presidente_1v"), (2018, "2v", "2018_presidente_2v")]
    for year, rl, key in nat:
        if year in year_set:
            sources.append(dict(year=year, election=f"Presidencial {rl}", scope="national",
                                 department=None, round_label=rl, url=FALLBACK_URLS[key]))
    return sources


sources = build_sources(departments_request, year_set)
for s in sources:
    tag = s["department"] or "national"
    log(f"  [{s['year']}] {s['election']:<16} {tag:<14} -> {s['url']}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =============================================================================
# Download + extract + transcode each source into the staging area.
# =============================================================================
def source_key(s) -> str:
    if s["scope"] == "national":
        return f"{s['year']}_PRESIDENTE_{(s['round_label'] or '').upper()}".rstrip("_")
    return f"{s['year']}_{slugify(s['election'])}_{slugify(s['department'])}"


def stage_source(session, s):
    """Returns an acquire-manifest dict for one source (None-safe; never raises out)."""
    rec = dict(s); rec["key"] = source_key(s); rec["members"] = []
    rec["http_status"] = ""; rec["bytes"] = 0; rec["sha256"] = ""; rec["note"] = ""
    if not s["url"]:
        rec["note"] = "no URL (discovery needed for this department)"
        log(f"  SKIP {rec['key']}: {rec['note']}"); return rec
    try:
        path, status, size, sha = download_zip(session, s["url"], Path(MMV_RAW_ZIPS))
    except DownloadError as exc:
        rec["http_status"] = str(exc).split(":")[0][:40]; rec["note"] = f"download failed: {exc}"
        log(f"  ERROR {rec['key']}: {exc}"); return rec
    rec["http_status"], rec["bytes"], rec["sha256"] = status, size, sha

    stage_dir = Path(MMV_RAW_NORMALIZED) / rec["key"]
    stage_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        data = [n for n in names
                if n.lower().endswith((".csv", ".txt", ".xlsx", ".xls"))
                and "hash" not in n.rsplit("/", 1)[-1].lower()]
        cx = [n for n in data if n.lower().endswith((".csv", ".xlsx", ".xls"))]
        if cx:
            data = cx
        for member in data:
            raw = zf.read(member)
            base = member.rsplit("/", 1)[-1]
            if member.lower().endswith((".xlsx", ".xls")):
                out = stage_dir / base                       # pass xlsx through for spark-excel
                out.write_bytes(raw)
                rec["members"].append(dict(member=base, format="xlsx", delimiter="",
                                           encoding="binary", staged=str(out)))
            else:
                text, enc = decode_bytes(raw)                # byte-transcode -> clean UTF-8
                delim = detect_delimiter(text)
                out = stage_dir / (base.rsplit(".", 1)[0] + ".csv")
                out.write_text(text, encoding="utf-8", newline="")
                rec["members"].append(dict(member=base, format="csv", delimiter=delim,
                                           encoding=enc, staged=str(out)))
            log(f"    staged {rec['key']}/{base}  [{rec['members'][-1]['format']}; "
                f"{rec['members'][-1]['encoding']}]")
    return rec


# =============================================================================
# Manual sources (data that exists but is NOT published on the Registraduria site,
# e.g. 2018 Senado). Drop files in  Files/raw/mmv/manual/<KEY>/  alongside a tiny
# meta.json describing the source; this runs on EVERY acquire so manual sources are
# durable (they survive re-runs, unlike a one-off manifest edit). zip/csv/xlsx are
# all handled with the same transcoding cascade as online sources.
#
#   Files/raw/mmv/manual/2018_CONGRESO_ANTIOQUIA/
#       <your file(s)>.zip|.csv|.xlsx
#       meta.json   ->  {"year":2018,"election":"Congreso","scope":"department",
#                        "department":"ANTIOQUIA","round_label":""}
# =============================================================================
MMV_RAW_MANUAL = f"{FILES_ROOT}/raw/mmv/manual"


def stage_manual_dir(folder: Path):
    """Stage one manual-source folder -> acquire-manifest rec (None if no meta/members)."""
    meta_path = folder / "meta.json"
    if not meta_path.exists():
        log(f"  SKIP manual {folder.name}: no meta.json"); return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    s = dict(year=int(meta["year"]), election=meta.get("election", "Congreso"),
             scope=meta.get("scope", "department"), department=meta.get("department"),
             round_label=meta.get("round_label", ""), url="")
    key = source_key(s)
    stage_dir = Path(MMV_RAW_NORMALIZED) / key
    stage_dir.mkdir(parents=True, exist_ok=True)
    rec = dict(s); rec["key"] = key; rec["members"] = []
    rec["http_status"] = "manual"; rec["bytes"] = 0; rec["sha256"] = ""
    rec["note"] = meta.get("note", "manual upload (not published on the Registraduria site)")

    inputs = []  # (container_path, inner_name|None)
    for p in sorted(folder.iterdir()):
        if p.name == "meta.json":
            continue
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as zf:
                for n in zf.namelist():
                    if n.endswith("/") or "hash" in n.rsplit("/", 1)[-1].lower():
                        continue
                    if n.lower().endswith((".csv", ".txt", ".xlsx", ".xls")):
                        inputs.append((p, n))
        elif p.suffix.lower() in (".csv", ".txt", ".xlsx", ".xls"):
            inputs.append((p, None))

    for container, inner in inputs:
        raw = zipfile.ZipFile(container).read(inner) if inner else container.read_bytes()
        base = (inner or container.name).rsplit("/", 1)[-1]
        if base.lower().endswith((".xlsx", ".xls")):
            out = stage_dir / base
            out.write_bytes(raw)
            rec["members"].append(dict(member=base, format="xlsx", delimiter="",
                                       encoding="binary", staged=str(out)))
        else:
            text, enc = decode_bytes(raw)
            delim = detect_delimiter(text)
            out = stage_dir / (base.rsplit(".", 1)[0] + ".csv")
            out.write_text(text, encoding="utf-8", newline="")
            rec["members"].append(dict(member=base, format="csv", delimiter=delim,
                                       encoding=enc, staged=str(out)))
        log(f"    staged manual {key}/{base}  [{rec['members'][-1]['format']}; "
            f"{rec['members'][-1]['encoding']}]")
    if not rec["members"]:
        rec["note"] = "manual folder had no stageable .csv/.xlsx members"
        log(f"  WARN manual {key}: {rec['note']}")
    return rec


session = make_session()
os.makedirs(MMV_CONTROL, exist_ok=True)
acquire_manifest = []
for i, s in enumerate(sources):
    log(f"\n=== acquire {source_key(s)} ===")
    acquire_manifest.append(stage_source(session, s))
    if i + 1 < len(sources):
        time.sleep(POLITE_DELAY)   # be polite to the Registraduria server

# Manual drop-folder sources (durable; appended every run, replacing any same-key online rec).
manual_root = Path(MMV_RAW_MANUAL)
if manual_root.exists():
    for folder in sorted(p for p in manual_root.iterdir() if p.is_dir()):
        log(f"\n=== acquire manual {folder.name} ===")
        rec = stage_manual_dir(folder)
        if rec:
            acquire_manifest = [m for m in acquire_manifest if m.get("key") != rec["key"]]
            acquire_manifest.append(rec)

ctrl_path = Path(MMV_CONTROL) / "mmv_acquire.json"
ctrl_path.write_text(json.dumps(acquire_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
log(f"\nWrote acquire manifest: {ctrl_path}  ({len(acquire_manifest)} sources)")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
