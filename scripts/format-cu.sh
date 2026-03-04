#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

if ! command -v clang-format >/dev/null 2>&1; then
  echo "error: clang-format is not installed or not on PATH" >&2
  exit 1
fi

if ! command -v rg >/dev/null 2>&1; then
  echo "error: rg (ripgrep) is not installed or not on PATH" >&2
  exit 1
fi

mapfile -d '' cu_files < <(rg --files -g '*.cu' -0 "${repo_root}")

if [[ "${#cu_files[@]}" -eq 0 ]]; then
  echo "No .cu files found under ${repo_root}"
  exit 0
fi

clang-format -i "${cu_files[@]}"
echo "Formatted ${#cu_files[@]} .cu file(s)."
