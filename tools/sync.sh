#!/usr/bin/env bash
# Copy the local working tree to the test host (overlay copy via tar over ssh; remote-only files are kept).
# Usage: tools/sync.sh
set -euo pipefail
. "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/.."
tar -czf - \
  --exclude=.git --exclude=.claude --exclude=node_modules --exclude=__pycache__ --exclude='*.pyc' --exclude=.pytest_cache \
  --exclude=web/dist --exclude=web/dist-bench --exclude=web/test-results --exclude=web/playwright-report \
  . | ssh "$HOST" "mkdir -p ~/$REMOTE_DIR && tar -xzf - -C ~/$REMOTE_DIR"
echo "synced -> $HOST:~/$REMOTE_DIR"
