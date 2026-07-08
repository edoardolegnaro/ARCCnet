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
  ARCAFF_DATA_FOLDER       Data root. Defaults to /ARCAFF/data.
  DATA_ROOT                Alias for ARCAFF_DATA_FOLDER if set.
  HEK_FILE                 HEK parquet under ${ARCAFF_DATA_FOLDER}/flare_files.
  SRS_FILE                 SRS parquet under ${ARCAFF_DATA_FOLDER}/flare_files.
  TS_CORES                 Processing workers. Defaults to 4.
  TS_DOWNLOAD_WORKERS      Prefetched JSOC records. Defaults to 3.
  TS_MAX_DRMS_CONNECTIONS  Max concurrent DRMS sessions. Defaults to 6.

Output:
  ${ARCAFF_DATA_FOLDER}/04_final/data
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
DATA_DIR="${DATA_ROOT:-${ARCAFF_DATA_FOLDER:-/ARCAFF/data}}"
HEK_FILE="${HEK_FILE:-hek_swpc_1996-01-01T00:00:00-2026-07-07T00:00:00_dev.parq}"
SRS_FILE="${SRS_FILE:-srs_processed_catalog.parq}"
YEAR_START="${YEAR_START:-${START_DATE:0:4}}"
YEAR_END="${YEAR_END:-${END_DATE:0:4}}"
TS_CORES="${TS_CORES:-4}"
TS_DOWNLOAD_WORKERS="${TS_DOWNLOAD_WORKERS:-3}"
TS_MAX_DRMS_CONNECTIONS="${TS_MAX_DRMS_CONNECTIONS:-6}"

if [[ -z "${JSOC_EMAIL:-}" ]]; then
    echo "JSOC_EMAIL is not set. Set it to your JSOC-registered email before running." >&2
    exit 1
fi

for input in "${DATA_DIR}/flare_files/${HEK_FILE}" "${DATA_DIR}/flare_files/${SRS_FILE}"; do
    if [[ ! -f "${input}" ]]; then
        echo "Required input not found: ${input}" >&2
        exit 1
    fi
done

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

[timeseries]
duration_hours = 6
timesteps = 6
long_lim_degrees = 65
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
EOF

echo "Data root: ${DATA_DIR}"
echo "HEK file:  ${DATA_DIR}/flare_files/${HEK_FILE}"
echo "SRS file:  ${DATA_DIR}/flare_files/${SRS_FILE}"
echo "Years:     ${YEAR_START}-${YEAR_END}"
echo "Output:    ${DATA_DIR}/04_final/data"

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
