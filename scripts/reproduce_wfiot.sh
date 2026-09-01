#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/reproduce_wfiot.sh {smoke|static|dynamic|compaction|scaling|scaling-rss|all} [REPEATS] [OUT_ROOT]

Runs isolated repetitions of a WF-IoT artifact target and aggregates results.

Examples:
  scripts/reproduce_wfiot.sh static
  scripts/reproduce_wfiot.sh dynamic 5 results/repeats_dynamic5
  OVERWRITE_REPEATS=1 scripts/reproduce_wfiot.sh smoke 2 /tmp/etar-repeat-smoke

Notes:
  - Defaults: REPEATS=5, OUT_ROOT=results/repeats.
  - Existing run directories are not overwritten unless OVERWRITE_REPEATS=1.
  - scaling and scaling-rss include the million-row HNSW cases and can be very slow.
USAGE
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 2
fi

TARGET="$1"
REPEATS="${2:-5}"
OUT_ROOT="${3:-results/repeats}"

if ! [[ "${REPEATS}" =~ ^[0-9]+$ ]] || [[ "${REPEATS}" -lt 1 ]]; then
  echo "REPEATS must be a positive integer, got: ${REPEATS}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_ROOT="${ARTIFACT_ROOT}/configs"
RUNNER="${ARTIFACT_ROOT}/scripts/run_experiment.sh"

cd "${ARTIFACT_ROOT}"
export CONDA_ENV="${CONDA_ENV:-etar-wfiot}"
export PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/etar-xdg-cache}"
mkdir -p "${XDG_CACHE_HOME}" "${OUT_ROOT}"

targets_for() {
  case "$1" in
    smoke) echo "smoke" ;;
    static) echo "static" ;;
    dynamic) echo "dynamic" ;;
    compaction) echo "compaction" ;;
    scaling) echo "scaling" ;;
    scaling-rss) echo "scaling-rss" ;;
    all) echo "static dynamic compaction scaling scaling-rss" ;;
    *)
      echo "unknown target: $1" >&2
      usage
      exit 2
      ;;
  esac
}

config_for() {
  case "$1" in
    smoke) echo "wfiot_smoke.yaml" ;;
    static) echo "wfiot_static.yaml" ;;
    dynamic) echo "wfiot_dynamic.yaml" ;;
    compaction) echo "wfiot_compaction_stress.yaml" ;;
    scaling) echo "wfiot_scaling.yaml" ;;
    *)
      echo "no direct config for target: $1" >&2
      exit 2
      ;;
  esac
}

run_standard_target() {
  local target="$1"
  local repeat="$2"
  local run_name
  printf -v run_name "run_%02d" "${repeat}"
  local run_dir="${OUT_ROOT}/${target}/${run_name}"
  if [[ -e "${run_dir}" ]]; then
    if [[ "${OVERWRITE_REPEATS:-0}" != "1" ]]; then
      echo "repeat output exists: ${run_dir}" >&2
      echo "set OVERWRITE_REPEATS=1 to replace it" >&2
      exit 1
    fi
    rm -rf "${run_dir}"
  fi
  echo "[repeat] ${target} ${repeat}/${REPEATS} -> ${run_dir}"
  "${RUNNER}" "${CONFIG_ROOT}/$(config_for "${target}")" --out "${run_dir}"
}

ensure_scaling_data() {
  local required="results/wfiot_scaling/data/wfiot_scaling_dense_gaussian_N1000000_d256_q50_s777.hdf5"
  if [[ ! -f "${required}" ]]; then
    echo "[repeat] generating shared scaling data for scaling-rss"
    "${RUNNER}" "${CONFIG_ROOT}/wfiot_scaling.yaml" --stage data
  fi
}

run_scaling_rss_target() {
  local repeat="$1"
  local run_name
  printf -v run_name "run_%02d" "${repeat}"
  local run_dir="${OUT_ROOT}/scaling-rss/${run_name}"
  if [[ -e "${run_dir}" ]]; then
    if [[ "${OVERWRITE_REPEATS:-0}" != "1" ]]; then
      echo "repeat output exists: ${run_dir}" >&2
      echo "set OVERWRITE_REPEATS=1 to replace it" >&2
      exit 1
    fi
    rm -rf "${run_dir}"
  fi
  ensure_scaling_data
  echo "[repeat] scaling-rss ${repeat}/${REPEATS} -> ${run_dir}"
  conda run --no-capture-output -n "${CONDA_ENV}" env \
    PYTHONPATH="${PYTHONPATH}" \
    PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
    OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
    NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
    VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS}" \
    BLIS_NUM_THREADS="${BLIS_NUM_THREADS}" \
    XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
    python analysis/profile_scaling_fresh_rss.py --out "${run_dir}"
}

for target in $(targets_for "${TARGET}"); do
  for repeat in $(seq 1 "${REPEATS}"); do
    if [[ "${target}" == "scaling-rss" ]]; then
      run_scaling_rss_target "${repeat}"
    else
      run_standard_target "${target}" "${repeat}"
    fi
  done
done

echo "[repeat] aggregating repeated results"
conda run --no-capture-output -n "${CONDA_ENV}" env \
  PYTHONPATH="${PYTHONPATH}" \
  PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" \
  python analysis/aggregate_repeated_results.py --root "${OUT_ROOT}" --out "${OUT_ROOT}/summary"
