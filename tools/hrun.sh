#!/usr/bin/env bash
# Run a command inside a project container on the test host.
# Usage: tools/hrun.sh <py|web> <command...>
#   tools/hrun.sh py python -m futsalcut.cli probe /src/<video>.mp4
# Mounts: repo -> /work, ~/$DATA_DIR -> /data, ~/$SRC_DIR -> /src (read-only); see tools/env.sh.
set -euo pipefail
. "$(dirname "$0")/env.sh"
[ $# -ge 2 ] || { echo "usage: $0 <py|web> <command...>" >&2; exit 2; }
case "$1" in
  py)  IMAGE=futsalcut-py ;;
  web) IMAGE=futsalcut-web ;;
  *) echo "unknown image: $1" >&2; exit 2 ;;
esac
shift
CMD=$(printf '%q ' "$@")
ssh "$HOST" "docker run --rm -i --init --user \$(id -u):\$(id -g) -e HOME=/tmp \
  -e PYTHONPATH=/work/python/src \
  -v \$HOME/$REMOTE_DIR:/work -v \$HOME/$DATA_DIR:/data -v \$HOME/$SRC_DIR:/src:ro \
  -w /work $IMAGE bash -c $(printf '%q' "$CMD")"
