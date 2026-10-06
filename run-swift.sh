#!/usr/bin/env bash
# Run Swift 1.5 IQ3_XXS with the shared Strata container launcher.
# All run.sh options are supported, including --release to select another release.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATA_RELEASE="${STRATA_RELEASE:-swift}" exec "$ROOT/run.sh" "$@"
