# Settings shared by the helper scripts (sourced, not run). Put your own values in tools/local.env
# (not tracked), for example:
#   FC_HOST=${FC_HOST:-mybox}
#   FC_SRC_DIR=${FC_SRC_DIR:-videos/futsal}
# shellcheck disable=SC1091
[ -f "$(dirname "${BASH_SOURCE[0]}")/local.env" ] && . "$(dirname "${BASH_SOURCE[0]}")/local.env"
HOST=${FC_HOST:-devhost}                 # ssh host that runs Docker
REMOTE_DIR=${FC_REMOTE_DIR:-futsalcut}    # copy of this repo on the host (relative to ~), mounted at /work
DATA_DIR=${FC_DATA_DIR:-futsalcut_data}   # caches, outputs, models (relative to ~), mounted at /data
SRC_DIR=${FC_SRC_DIR:-videos}             # original recordings (relative to ~), mounted read-only at /src
