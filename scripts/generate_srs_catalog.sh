#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Generate the processed NOAA SRS catalog needed by timeseries generation.

Usage:
  scripts/generate_srs_catalog.sh [START_DATE] [END_DATE]

Defaults:
  START_DATE = 2010-05-13T00:00:00
  END_DATE   = 2026-07-07T00:00:00

Environment:
  ARCAFF_DATA_FOLDER  Base data folder. Defaults to /ARCAFF/data.
  RUN_NAME            Dataset folder name under ${ARCAFF_DATA_FOLDER}/timeseries.
  DATA_ROOT           Exact output root. Overrides ARCAFF_DATA_FOLDER/RUN_NAME.

Output:
  ${DATA_ROOT}/03_processed/metadata/noaa_srs/srs_processed_catalog.parq
  ${DATA_ROOT}/flare_files/srs_processed_catalog.parq -> ../03_processed/metadata/noaa_srs/srs_processed_catalog.parq
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

START_DATE="${1:-2010-05-13T00:00:00}"
END_DATE="${2:-2026-07-07T00:00:00}"
BASE_DATA_DIR="${ARCAFF_DATA_FOLDER:-/ARCAFF/data}"

date_tag() {
    local value="${1%%T*}"
    value="${value//-/}"
    echo "${value}"
}

START_TAG="$(date_tag "${START_DATE}")"
END_TAG="$(date_tag "${END_DATE}")"
RUN_NAME="${RUN_NAME:-arcaff-timeseries-${START_TAG}-${END_TAG}}"
DATA_DIR="${DATA_ROOT:-${BASE_DATA_DIR}/timeseries/${RUN_NAME}}"

export ARCAFF_DATA_FOLDER="${DATA_DIR}"

cd "${REPO_ROOT}"

python -m arccnet.cli.main catalog generate ar_catalog \
    --data-root "${DATA_DIR}" \
    --start-date "${START_DATE}" \
    --end-date "${END_DATE}"

SRS_PATH="${DATA_DIR}/03_processed/metadata/noaa_srs/srs_processed_catalog.parq"
if [[ ! -f "${SRS_PATH}" ]]; then
    echo "Expected SRS catalog was not created: ${SRS_PATH}" >&2
    exit 1
fi

mkdir -p "${DATA_DIR}/flare_files"
ln -sf "../03_processed/metadata/noaa_srs/srs_processed_catalog.parq" \
    "${DATA_DIR}/flare_files/srs_processed_catalog.parq"

echo "SRS catalog: ${SRS_PATH}"
echo "Timeseries link: ${DATA_DIR}/flare_files/srs_processed_catalog.parq"
