script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

export CUTLASS_HOME="${repo_root}/third_party/cutlass"
