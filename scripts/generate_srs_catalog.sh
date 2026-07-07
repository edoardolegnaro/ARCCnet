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
  ARCAFF_DATA_FOLDER  Data root. Defaults to /ARCAFF/data.
  DATA_ROOT           Alias for ARCAFF_DATA_FOLDER if set.

Output:
  ${ARCAFF_DATA_FOLDER}/03_processed/metadata/noaa_srs/srs_processed_catalog.parq
  ${ARCAFF_DATA_FOLDER}/flare_files/srs_processed_catalog.parq -> ../03_processed/metadata/noaa_srs/srs_processed_catalog.parq
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
DATA_DIR="${DATA_ROOT:-${ARCAFF_DATA_FOLDER:-/ARCAFF/data}}"

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
