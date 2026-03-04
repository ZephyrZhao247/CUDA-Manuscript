#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

if ! command -v clang-format >/dev/null 2>&1; then
  echo "error: clang-format is not installed or not on PATH" >&2
  exit 1
fi

if ! command -v find >/dev/null 2>&1; then
  echo "error: find is not installed or not on PATH" >&2
  exit 1
fi

mapfile -d '' cu_files < <(
  find "${repo_root}" \
    -path "${repo_root}/third_party" -prune -o \
    -type f -name '*.cu' -print0
)

if [[ "${#cu_files[@]}" -eq 0 ]]; then
  echo "No .cu files found under ${repo_root}"
  exit 0
fi

echo "Formatting ${#cu_files[@]} .cu file(s) under ${repo_root}..."
clang-format -i "${cu_files[@]}"
echo "Formatted ${#cu_files[@]} .cu file(s)."
