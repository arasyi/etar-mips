#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {smoke|static|dynamic|compaction|scaling|scaling-rss|all}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_ROOT="${ARTIFACT_ROOT}/configs"
RUNNER="${ARTIFACT_ROOT}/scripts/run_experiment.sh"

if [[ ! -x "${RUNNER}" ]]; then
  echo "missing executable runner: ${RUNNER}" >&2
  exit 1
fi

cd "${ARTIFACT_ROOT}"
export CONDA_ENV="${CONDA_ENV:-etar-wfiot}"

run_one() {
  local config="$1"
  shift || true
  "${RUNNER}" "${CONFIG_ROOT}/${config}" "$@"
}

run_scaling_rss() {
  local required_data="results/wfiot_scaling/data/wfiot_scaling_dense_gaussian_N1000000_d256_q50_s777.hdf5"
  if [[ ! -f "${required_data}" ]]; then
    run_one wfiot_scaling.yaml --stage data
  fi
  PYTHONPATH="${PYTHONPATH:-src}" \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  BLIS_NUM_THREADS=1 \
  XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/etar-xdg-cache}" \
  conda run --no-capture-output -n "${CONDA_ENV}" env \
    PYTHONPATH="${PYTHONPATH:-src}" \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    BLIS_NUM_THREADS=1 \
    XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/etar-xdg-cache}" \
    python analysis/profile_scaling_fresh_rss.py --out results/wfiot_scaling_fresh_rss
}

case "$1" in
  smoke)
    run_one wfiot_smoke.yaml
    ;;
  static)
    run_one wfiot_static.yaml
    ;;
  dynamic)
    run_one wfiot_dynamic.yaml
    ;;
  compaction)
    run_one wfiot_compaction_stress.yaml
    ;;
  scaling)
    run_one wfiot_scaling.yaml
    ;;
  scaling-rss)
    run_scaling_rss
    ;;
  all)
    run_one wfiot_static.yaml
    run_one wfiot_dynamic.yaml
    run_one wfiot_compaction_stress.yaml
    run_one wfiot_scaling.yaml
    run_scaling_rss
    ;;
  *)
    echo "unknown target: $1" >&2
    exit 2
    ;;
esac
