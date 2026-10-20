#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "${FSIS_PYTHON:-python3}" "$script_dir/tools/manage_installation.py" remove "$@"
