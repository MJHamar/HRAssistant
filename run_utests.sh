#!/usr/bin/env bash
set -euo pipefail

# Re-install the package in editable mode and run tests
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python -m pip install --upgrade pip >/dev/null
python -m pip install -e .[dev]

pytest -q
