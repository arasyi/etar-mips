#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: scripts/run_experiment.sh CONFIG.yaml [--out OUT_DIR] [--stage STAGE]" >&2
  exit 2
fi

CONFIG="$1"
shift
CONDA_ENV="${CONDA_ENV:-etar-wfiot}"
OUT_DIR=""
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--out" && $((i + 1)) -lt ${#ARGS[@]} ]]; then
    OUT_DIR="${ARGS[$((i + 1))]}"
  fi
done

export PYTHONPATH="${PYTHONPATH:-src}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/etar-xdg-cache}"
mkdir -p "${XDG_CACHE_HOME}"

CMD=(conda run --no-capture-output -n "${CONDA_ENV}" env \
  PYTHONPATH="${PYTHONPATH}" \
  PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
  NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
  VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS}" \
  BLIS_NUM_THREADS="${BLIS_NUM_THREADS}" \
  XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
  python -m etar run-config "${CONFIG}" "$@")

if [[ -n "${OUT_DIR}" ]]; then
  mkdir -p "${OUT_DIR}/logs"
  "${CMD[@]}" 2>&1 | tee "${OUT_DIR}/logs/run_experiment.log"
else
  "${CMD[@]}"
fi
