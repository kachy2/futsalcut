#!/usr/bin/env bash
# Copy files generated on the test host back into the local working tree.
# Usage: tools/pull.sh <repo-relative path>...   e.g. tools/pull.sh eval/reports eval/labels
set -euo pipefail
. "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/.."
[ $# -gt 0 ] || { echo "usage: $0 <path>..." >&2; exit 2; }
ssh "$HOST" "cd ~/$REMOTE_DIR && tar -czf - $(printf '%q ' "$@")" | tar -xzf -
echo "pulled: $*"
