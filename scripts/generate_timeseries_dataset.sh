#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Generate the SDO time-series flare dataset.

Usage:
  scripts/generate_timeseries_dataset.sh [--dry-run] [START_DATE] [END_DATE]

Defaults:
  START_DATE = 2010-05-13
  END_DATE   = 2026-07-07T00:00:00

Environment:
  ARCAFF_DATA_FOLDER       Base data folder. Defaults to /ARCAFF/data.
  RUN_NAME                 Dataset folder name under ${ARCAFF_DATA_FOLDER}/timeseries.
  DATA_ROOT                Exact output root. Overrides ARCAFF_DATA_FOLDER/RUN_NAME.
  INPUT_DATA_ROOT          Shared input root containing flare_files. Defaults to ARCAFF_DATA_FOLDER.
  HEK_FILE                 HEK parquet file name.
  SRS_FILE                 SRS parquet file name.
  TS_CORES                 Processing workers. Defaults to 48.
  TS_DOWNLOAD_WORKERS      Prefetched JSOC records. Defaults to 1.
  TS_MAX_DRMS_CONNECTIONS  Max concurrent DRMS sessions. Defaults to 1.
  TS_POINTING_CHUNK_DAYS   Days per AIA pointing-table JSOC query. Defaults to 180.
  JSOC_PENDING_RETRIES     Retries when JSOC reports pending exports. Defaults to 20.
  JSOC_PENDING_SLEEP       Seconds between pending-export retries. Defaults to 120.
  TS_GROUP_BY_TIME         Reuse one JSOC download for ARs with the same time window. Defaults to True.
  TS_USE_MANIFEST          Reuse cached raw-file manifests on restart. Defaults to True.
  RUN_LOG                  Full console log path. Defaults under ${DATA_ROOT}/logs.

Output:
  ${DATA_ROOT}/04_final/data
  ${DATA_ROOT}/logs/timeseries_generation_status.csv
EOF
}

DRY_RUN=0
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
elif [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

START_DATE="${1:-2010-05-13}"
END_DATE="${2:-2026-07-07T00:00:00}"
BASE_DATA_DIR="${ARCAFF_DATA_FOLDER:-/ARCAFF/data}"
INPUT_DATA_DIR="${INPUT_DATA_ROOT:-${BASE_DATA_DIR}}"

date_tag() {
    local value="${1%%T*}"
    value="${value//-/}"
    echo "${value}"
}

START_TAG="$(date_tag "${START_DATE}")"
END_TAG="$(date_tag "${END_DATE}")"
RUN_NAME="${RUN_NAME:-arcaff-timeseries-${START_TAG}-${END_TAG}}"
DATA_DIR="${DATA_ROOT:-${BASE_DATA_DIR}/timeseries/${RUN_NAME}}"
HEK_FILE="${HEK_FILE:-hek_swpc_1996-01-01T00:00:00-2026-07-07T00:00:00_dev.parq}"
SRS_FILE="${SRS_FILE:-srs_processed_catalog.parq}"
YEAR_START="${YEAR_START:-${START_DATE:0:4}}"
YEAR_END="${YEAR_END:-${END_DATE:0:4}}"
TS_CORES="${TS_CORES:-48}"
TS_DOWNLOAD_WORKERS="${TS_DOWNLOAD_WORKERS:-1}"
TS_MAX_DRMS_CONNECTIONS="${TS_MAX_DRMS_CONNECTIONS:-1}"
TS_POINTING_CHUNK_DAYS="${TS_POINTING_CHUNK_DAYS:-180}"
JSOC_PENDING_RETRIES="${JSOC_PENDING_RETRIES:-20}"
JSOC_PENDING_SLEEP="${JSOC_PENDING_SLEEP:-120}"
TS_GROUP_BY_TIME="${TS_GROUP_BY_TIME:-True}"
TS_USE_MANIFEST="${TS_USE_MANIFEST:-True}"
LOG_DIR="${DATA_DIR}/logs"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/timeseries_$(date -u +%Y%m%dT%H%M%SZ).log}"

if [[ -z "${JSOC_EMAIL:-}" ]]; then
    echo "JSOC_EMAIL is not set. Set it to your JSOC-registered email before running." >&2
    exit 1
fi

prepare_input() {
    local file_name="$1"
    local dest="${DATA_DIR}/flare_files/${file_name}"
    local src="${INPUT_DATA_DIR}/flare_files/${file_name}"

    if [[ -f "${dest}" ]]; then
        return
    fi

    if [[ -f "${src}" ]]; then
        if [[ "${DRY_RUN}" == "1" ]]; then
            echo "Would link input: ${dest} -> ${src}"
        else
            mkdir -p "${DATA_DIR}/flare_files"
            ln -sf "${src}" "${dest}"
        fi
        return
    fi

    echo "Required input not found: ${dest}" >&2
    if [[ "${dest}" != "${src}" ]]; then
        echo "Looked for shared input under: ${INPUT_DATA_DIR}/flare_files" >&2
    fi
    exit 1
}

prepare_input "${HEK_FILE}"
prepare_input "${SRS_FILE}"

if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${LOG_DIR}"
    exec > >(tee -a "${RUN_LOG}") 2>&1
    echo "Run log:   ${RUN_LOG}"
fi

CONFIG_ROOT="$(mktemp -d)"
trap 'rm -rf "${CONFIG_ROOT}"' EXIT
mkdir -p "${CONFIG_ROOT}/ARCCnet"

cat > "${CONFIG_ROOT}/ARCCnet/arccnetrc" <<EOF
[paths]
data_folder = ${DATA_DIR}
data_root = ${DATA_DIR}

[drms]
cores = ${TS_CORES}
download_workers = ${TS_DOWNLOAD_WORKERS}
max_drms_connections = ${TS_MAX_DRMS_CONNECTIONS}
jsoc_pending_retries = ${JSOC_PENDING_RETRIES}
jsoc_pending_sleep = ${JSOC_PENDING_SLEEP}

[timeseries]
duration_hours = 6
timesteps = 6
long_lim_degrees = 65
pointing_chunk_days = ${TS_POINTING_CHUNK_DAYS}
year_start = ${YEAR_START}
year_end = ${YEAR_END}
sdo_start_date = ${START_DATE}
catalog_end_date = ${END_DATE}
hek_file = ${HEK_FILE}
srs_file = ${SRS_FILE}
keep_no_flare = True
samples_per_year = -1
aia_degradation_correction = True
resume = True
make_animations = False
group_by_time = ${TS_GROUP_BY_TIME}
use_time_window_manifest = ${TS_USE_MANIFEST}
EOF

echo "Data root: ${DATA_DIR}"
echo "Input root: ${INPUT_DATA_DIR}"
echo "HEK file:  ${DATA_DIR}/flare_files/${HEK_FILE}"
echo "SRS file:  ${DATA_DIR}/flare_files/${SRS_FILE}"
echo "Years:     ${YEAR_START}-${YEAR_END}"
echo "Output:    ${DATA_DIR}/04_final/data"
echo "Status:    ${DATA_DIR}/logs/timeseries_generation_status.csv"
echo "Grouped:   ${TS_GROUP_BY_TIME}"
echo "Manifest:  ${TS_USE_MANIFEST}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo
    echo "Temporary ARCCnet config:"
    cat "${CONFIG_ROOT}/ARCCnet/arccnetrc"
    exit 0
fi

cd "${REPO_ROOT}"
export ARCAFF_DATA_FOLDER="${DATA_DIR}"
export XDG_CONFIG_HOME="${CONFIG_ROOT}"

python -m arccnet.data_generation.timeseries.drms_pipeline
