#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Generate a HEK SWPC flare parquet for timeseries generation.

Usage:
  scripts/generate_hek_flare_catalog.sh [START_DATE] [END_DATE]

Defaults:
  START_DATE = 1996-01-01T00:00:00
  END_DATE   = today's UTC date at 00:00:00

Environment:
  ARCAFF_DATA_FOLDER  Data root. Defaults to /ARCAFF/data.
  HEK_WINDOW_DAYS     Days per HEK request. Defaults to 90.
  HEK_MAX_RETRIES     Retry count per request window. Defaults to 5.
  HEK_RETRY_SLEEP     Base seconds between retries. Defaults to 30.
  HEK_REQUEST_SLEEP   Seconds to sleep after each successful request. Defaults to 1.
  HEK_CACHE_DIR       Per-window cache dir. Defaults to ${ARCAFF_DATA_FOLDER}/flare_files/.hek_swpc_cache.

Output:
  ${ARCAFF_DATA_FOLDER}/flare_files/hek_swpc_<START_DATE>-<END_DATE>_dev.parq
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

START_DATE="${1:-1996-01-01T00:00:00}"
END_DATE="${2:-$(date -u +%Y-%m-%dT00:00:00)}"
DATA_DIR="${ARCAFF_DATA_FOLDER:-/ARCAFF/data}"
OUT_DIR="${DATA_DIR}/flare_files"
CACHE_DIR="${HEK_CACHE_DIR:-${OUT_DIR}/.hek_swpc_cache}"
WINDOW_DAYS="${HEK_WINDOW_DAYS:-90}"
MAX_RETRIES="${HEK_MAX_RETRIES:-5}"
RETRY_SLEEP="${HEK_RETRY_SLEEP:-30}"
REQUEST_SLEEP="${HEK_REQUEST_SLEEP:-1}"

export ARCAFF_DATA_FOLDER="${DATA_DIR}"

cd "${REPO_ROOT}"

python - "${START_DATE}" "${END_DATE}" "${OUT_DIR}" "${CACHE_DIR}" "${WINDOW_DAYS}" "${MAX_RETRIES}" "${RETRY_SLEEP}" "${REQUEST_SLEEP}" <<'PY'
from pathlib import Path
import time
import sys

import numpy as np
import astropy.units as u
from astropy.table import Table, unique, vstack
from astropy.time import Time
from sunpy.net import Fido
from sunpy.net import attrs as a

from arccnet.catalogs.flares.hek import HEKFlareCatalog

start, end, out_dir, cache_dir, window_days, max_retries, retry_sleep, request_sleep = sys.argv[1:9]
out_dir = Path(out_dir)
cache_dir = Path(cache_dir)
out_dir.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)

window_days = int(window_days)
max_retries = int(max_retries)
retry_sleep = int(retry_sleep)
request_sleep = int(request_sleep)

if window_days <= 0:
    raise ValueError("HEK_WINDOW_DAYS must be positive")


def cache_name(index, start_time, end_time):
    safe_start = start_time.isot.replace("-", "").replace(":", "").replace(".", "")
    safe_end = end_time.isot.replace("-", "").replace(":", "").replace(".", "")
    return cache_dir / f"{index:04d}_{safe_start}_{safe_end}.parq"


def fetch_window(hek, start_time, end_time):
    result = Fido.search(a.Time(start_time, end_time), *hek.query)
    flares = result["hek"]
    flares.meta = None

    # Match HEKFlareCatalog.search cleanup, but per window so successful
    # windows can be cached and reused if a later HEK request drops.
    for col_to_remove in ["refs", "event_probability", "event_avg_rating", "event_importance"]:
        if col_to_remove in flares.colnames:
            flares.remove_column(col_to_remove)

    if len(flares) == 0:
        return None

    table = Table(flares.as_array())
    col_to_remove = []
    for col in table.colnames:
        if np.all(table[col] == None):  # noqa: E711
            col_to_remove.append(col)
    if col_to_remove:
        table.remove_columns(col_to_remove)
    return table


def load_or_fetch_window(hek, index, start_time, end_time):
    path = cache_name(index, start_time, end_time)
    empty_marker = path.with_suffix(".empty")

    if empty_marker.exists():
        print(f"[{index:04d}] cached empty {start_time.isot} - {end_time.isot}", flush=True)
        return None

    if path.exists():
        try:
            table = Table.read(path, format="parquet")
            print(f"[{index:04d}] cached {len(table)} flares {start_time.isot} - {end_time.isot}", flush=True)
            return table
        except Exception as error:
            print(f"[{index:04d}] removing unreadable cache {path}: {error}", flush=True)
            path.unlink(missing_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            print(
                f"[{index:04d}] querying HEK attempt {attempt}/{max_retries}: "
                f"{start_time.isot} - {end_time.isot}",
                flush=True,
            )
            table = fetch_window(hek, start_time, end_time)
            if table is None:
                empty_marker.touch()
                print(f"[{index:04d}] wrote empty marker", flush=True)
            else:
                table.write(path, format="parquet", overwrite=True)
                print(f"[{index:04d}] cached {len(table)} flares", flush=True)
            if request_sleep > 0:
                time.sleep(request_sleep)
            return table
        except Exception as error:
            if attempt == max_retries:
                raise
            sleep_for = retry_sleep * attempt
            print(
                f"[{index:04d}] HEK request failed: {type(error).__name__}: {error}. "
                f"Sleeping {sleep_for}s before retry.",
                flush=True,
            )
            time.sleep(sleep_for)

    raise RuntimeError("unreachable")


hek = HEKFlareCatalog(catalog="swpc")
start_time = Time(start)
end_time = Time(end)
if end_time <= start_time:
    raise ValueError("END_DATE must be later than START_DATE")

tables = []
cur = start_time
index = 0
while cur < end_time:
    nxt = min(cur + window_days * u.day, end_time)
    table = load_or_fetch_window(hek, index, cur, nxt)
    if table is not None and len(table) > 0:
        tables.append(table)
    cur = nxt
    index += 1

if not tables:
    raise RuntimeError(f"No HEK flares found between {start} and {end}")

query = vstack(tables, join_type="outer")
dedupe_keys = [
    key
    for key in ["event_starttime", "event_endtime", "event_peaktime", "ar_noaanum", "fl_goescls"]
    if key in query.colnames
]
if dedupe_keys:
    query = unique(query, keys=dedupe_keys)

catalog = hek.create_catalog(query)

out = out_dir / f"hek_swpc_{start}-{end}_dev.parq"
catalog.write(out, format="parquet", overwrite=True)
print(f"Wrote {out} ({len(catalog)} flares)")
PY
