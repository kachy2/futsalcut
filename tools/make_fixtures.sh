#!/usr/bin/env bash
# Regenerate the Python<->TypeScript parity fixture inside the futsalcut-py container on the test host.
# Usage: tools/make_fixtures.sh   (then tools/pull.sh web/tests/fixtures to copy it back)
set -euo pipefail
cd "$(dirname "$0")/.."
tools/sync.sh >/dev/null
tools/hrun.sh py bash -c "
  test -f /data/synthetic/syn_h264_gop1.mp4 || python eval/runners/make_synthetic.py --out /data/synthetic
  python -m futsalcut.cli sample /data/synthetic/syn_h264_gop1.mp4 --out /data/cache --thumbs >/dev/null
  python eval/make_parity_fixture.py --cache /data/cache/syn_h264_gop1.luma640x360.u8 \
    --zone '0.05,0.30 0.95,0.30 1,1 0,1' --model python/src/futsalcut/models/ensemble_v1.json \
    --out web/tests/fixtures/synthetic
"
