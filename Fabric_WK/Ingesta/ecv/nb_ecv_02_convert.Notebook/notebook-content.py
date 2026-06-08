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

# # nb_ecv_02_convert  —  ECV Phase 2: locate base file + enforce CSV
# 
# **Driver-side.** For one `year`, find the base microdata file under `Files/raw/ecv/<year>/`
# (whatever Phase 1 downloaded, or a manual upload), skipping diccionario / ficha / instructivo /
# serie / campos / formato auxiliaries, read it (csv/xlsx — `.dta`/`.txt` are future seams), and
# write a clean UTF-8 **`microdata.csv`** to `Files/raw/ecv/normalized/<year>/`. The year's data
# dictionary is converted likewise to `dictionary.csv` (landed as a reference table in Phase 3).
# A small `_meta.json` records the chosen source file and row/column counts for the manifest.
# 
# In-scope years: 2018 (.csv), 2019 (.xlsx), 2022 (.csv), 2023 (.csv).

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
import io
import json
from pathlib import Path
import pandas as pd

year = int(year)
src_dir = Path(ECV_RAW) / str(year)
out_dir = Path(ECV_RAW_NORMALIZED) / str(year)
out_dir.mkdir(parents=True, exist_ok=True)
log(f"ECV convert — year={year}  src={src_dir}")

# Filename patterns that are NOT the microdata base file.
_AUX = ("diccionario", "ficha", "instructivo", "serie", "campos", "formato",
        "requerimiento", "foginf", "acceso", "~$")
_DICT = ("diccionario",)
_DATA_EXT = (".csv", ".xlsx", ".xlsm", ".xls", ".dta", ".txt")


def _is_aux(name: str) -> bool:
    n = norm_key(name)
    return any(a in n for a in _AUX)


def list_files():
    try:
        return sorted(os.listdir(src_dir))
    except FileNotFoundError:
        return []


def pick_base(files):
    """Authoritative base file: prefer csv > xlsx > dta > txt among non-auxiliary files."""
    cand = [f for f in files if f.lower().endswith(_DATA_EXT) and not _is_aux(f)]
    order = {".csv": 0, ".xlsx": 1, ".xlsm": 1, ".xls": 1, ".dta": 2, ".txt": 3}
    cand.sort(key=lambda f: order.get(os.path.splitext(f)[1].lower(), 9))
    return cand[0] if cand else None


def pick_dictionary(files):
    cand = [f for f in files if any(d in norm_key(f) for d in _DICT)
            and f.lower().endswith((".xlsx", ".xlsm", ".xls", ".csv"))]
    return cand[0] if cand else None

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def read_any(path: Path) -> "pd.DataFrame":
    """Read csv/xlsx into a string-typed DataFrame, reusing the encoding cascade for csv."""
    ext = path.suffix.lower()
    if ext in (".csv", ".txt"):
        text, enc = decode_bytes(path.read_bytes())
        delim = detect_delimiter(text)
        log(f"    csv read [{enc}; delim={delim!r}]")
        # C engine (default) — far faster/leaner than the python engine on the large
        # files (the 2018 base is ~359 MB); sep is always a single char here.
        return pd.read_csv(io.StringIO(text), sep=delim, dtype=str, keep_default_na=False)
    if ext in (".xlsx", ".xlsm", ".xls"):
        log(f"    excel read ({path.name})")
        return pd.read_excel(path, dtype=str, engine="openpyxl").fillna("")
    if ext == ".dta":
        raise NotImplementedError("`.dta` (Stata) reader is a future seam (add pyreadstat).")
    raise ValueError(f"unsupported ECV format: {ext}")


def to_csv_utf8(df, dest: Path):
    df.to_csv(dest, index=False, encoding="utf-8")


files = list_files()
base = pick_base(files)
dct = pick_dictionary(files)
meta = {"year": year, "base_file": base, "dictionary_file": dct,
        "microdata_rows": 0, "microdata_cols": 0, "note": ""}

if not files:
    meta["note"] = f"no files under {src_dir} — run nb_ecv_01_acquire or upload to Files/raw/ecv/{year}/"
    log("  " + meta["note"])
elif not base:
    rar = [f for f in files if f.lower().endswith((".rar", ".7z"))]
    if rar:
        meta["note"] = (f"only a compressed archive present ({rar}); Fabric has no unrar — "
                        f"extract it and stage the .xlsx/.csv in Files/raw/ecv/{year}/")
    else:
        meta["note"] = f"no base microdata file found among {files}"
    log("  " + meta["note"])
else:
    log(f"  base file: {base}")
    mdf = read_any(src_dir / base)
    meta["microdata_rows"], meta["microdata_cols"] = int(mdf.shape[0]), int(mdf.shape[1])
    # Sanity guard: a real ECV microdata file is wide (hundreds of columns) with many
    # rows. Flag a suspiciously thin/empty file (e.g. a stray pivot fragment) instead of
    # silently landing garbage — but still write it so the issue is visible downstream.
    if meta["microdata_cols"] < 5 or meta["microdata_rows"] == 0:
        meta["note"] = (f"SUSPECT base file: only {meta['microdata_rows']} rows x "
                        f"{meta['microdata_cols']} cols — not a full microdata table?")
        log("  WARNING: " + meta["note"])
    to_csv_utf8(mdf, out_dir / "microdata.csv")
    log(f"  -> microdata.csv  ({meta['microdata_rows']:,} rows x {meta['microdata_cols']} cols)")
    if dct:
        try:
            ddf = read_any(src_dir / dct)
            to_csv_utf8(ddf, out_dir / "dictionary.csv")
            log(f"  dictionary: {dct} -> dictionary.csv ({ddf.shape[0]} rows)")
        except Exception as exc:
            meta["note"] = f"dictionary convert failed: {exc!r}"
            log("  " + meta["note"])
    else:
        log("  (no dictionary file found)")

(out_dir / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
log(f"  wrote {out_dir/'_meta.json'}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
