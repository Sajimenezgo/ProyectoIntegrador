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

# # nb_ecv_01_acquire  —  ECV Phase 1: acquire online
# 
# **Driver-side.** For one `year`, download the ECV microdata (and its data dictionary, when
# published) from Medellín's open-data portals into `Files/raw/ecv/<year>/`, idempotently
# (cache hit → skip; retries + backoff). URLs come from the **`ECV_SOURCES`** registry in
# `nb_00_shared_utils` — the single re-pointable place if a link moves.
# 
# If a year has no configured URL (e.g. 2019, whose per-year file isn't published as a clean
# link), the notebook logs it and continues — the file can be set in `ECV_SOURCES` or uploaded
# manually to `Files/raw/ecv/<year>/`. Phase 2 (`nb_ecv_02_convert`) then reads whatever landed.

# PARAMETERS CELL ********************

year = 2018       # the Data Pipeline ForEach passes one year per invocation

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
from pathlib import Path

year = int(year)
raw_dir = Path(ECV_RAW) / str(year)
raw_dir.mkdir(parents=True, exist_ok=True)
cfg = ECV_SOURCES.get(year, {})
log(f"ECV acquire — year={year}  ({cfg.get('note', 'no registry entry')})")

acquired = {"year": year, "files": [], "note": cfg.get("note", "")}

# Filename patterns that are auxiliary (not the microdata base file).
_AUX = ("diccionario", "ficha", "instructivo", "serie", "campos", "formato",
        "requerimiento", "foginf", "acceso", "~$")
_EXT = (".csv", ".xlsx", ".xls", ".dta", ".txt")


def fetch(kind, url):
    if not url:
        log(f"  {kind}: no URL configured for {year} — upload to {raw_dir} or set ECV_SOURCES[{year}].")
        return
    dest = raw_dir / url.rstrip("/").split("/")[-1]
    try:
        status, nbytes = download_file(url, dest)
        acquired["files"].append({"kind": kind, "url": url, "file": dest.name,
                                  "status": status, "bytes": nbytes})
    except Exception as exc:
        log(f"  ERROR acquiring {kind} for {year}: {exc}")
        acquired["files"].append({"kind": kind, "url": url, "file": "",
                                  "status": "error", "bytes": 0, "error": str(exc)})


# Honor pasted local files: if a real (non-auxiliary) data file is already staged for
# this year, this is "use my locals" mode — skip the remote pull (acquire is the pilot).
existing = [f for f in os.listdir(raw_dir)
            if f.lower().endswith(_EXT) and not any(a in norm_key(f) for a in _AUX)]
if existing:
    acquired["note"] = f"local data already present -> skipped remote download: {existing}"
    log("  " + acquired["note"])
else:
    fetch("microdata", cfg.get("microdata"))
    fetch("dictionary", cfg.get("dictionary"))

(raw_dir / "_acquire.json").write_text(json.dumps(acquired, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
ok = [f for f in acquired["files"] if f["status"] != "error"]
log(f"  acquired {len(ok)} file(s) into {raw_dir} (existing local: {bool(existing)})")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
