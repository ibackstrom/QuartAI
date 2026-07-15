#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Create it and install requirements first." >&2
  exit 1
fi

OUTPUT="qllm/runs/overnight-50m"
mkdir -p "$OUTPUT"

if command -v caffeinate >/dev/null 2>&1; then
  KEEP_AWAKE=(caffeinate -i)
else
  KEEP_AWAKE=()
fi

echo "Starting the 300M-token overnight experiment."
echo "Results and log: $OUTPUT"
"${KEEP_AWAKE[@]}" .venv/bin/python -u -m qllm.overnight --device auto \
  2>&1 | tee "$OUTPUT/overnight.log"
