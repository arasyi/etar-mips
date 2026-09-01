#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: scripts/build_cpp.sh [--clean] [--no-conda]" >&2
  echo "" >&2
  echo "Builds the ETAR C++ Python extension in place." >&2
  echo "" >&2
  echo "Environment variables:" >&2
  echo "  CONDA_ENV   Conda environment to use, default: etar-wfiot" >&2
  echo "  PYTHON      Python executable when --no-conda is used, default: python" >&2
}

CLEAN=0
USE_CONDA=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      CLEAN=1
      shift
      ;;
    --no-conda)
      USE_CONDA=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${PYTHONPATH:-src}"

if [[ "${CLEAN}" -eq 1 ]]; then
  rm -rf build
  rm -f src/etar/_etar_cpp*.so
  rm -f src/etar/_etar_storage_cpp*.so
fi

if [[ "${USE_CONDA}" -eq 1 ]]; then
  CONDA_ENV="${CONDA_ENV:-etar-wfiot}"
  conda run -n "${CONDA_ENV}" env PYTHONPATH="${PYTHONPATH}" python setup.py build_ext --inplace
else
  PYTHON_BIN="${PYTHON:-python}"
  env PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}" setup.py build_ext --inplace
fi

ls -lh src/etar/_etar_cpp*.so src/etar/_etar_storage_cpp*.so
