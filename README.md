# FutsalCut — keep only the play from static-camera futsal/soccer recordings

[한국어](README.ko.md) · **Web app: https://kachy2.github.io/futsalcut/** (Korean page: [/ko/](https://kachy2.github.io/futsalcut/ko/))

A free, open-source tool for phone-on-a-tripod futsal/soccer recordings (1–2 h, 4K): it keeps only the moments
with **2+ players in the camera-side zone** and joins them into one MP4 **in original quality and with audio,
without re-encoding**.

- **No install** — runs in the browser (Chrome/Edge on a PC recommended)
- **No upload** — your video never leaves the device; analysis and saving happen in the browser
- **Fast** — analyzes keyframes only (≈1 per second) and copies compressed packets, so a 2-hour 4K game takes minutes
- **Adjustable** — auto (recommended) / sensitivity / target length, a drawn zone, and advanced options
  (lead-in before each scene, algorithm choice, side-by-side comparison, quick calibration)
- **Split recordings** — files split every 4 GB (e.g. Galaxy phones) are processed as one game

## How to use
1. Select the game video (select all split files together).
2. Draw the **zone where 2+ players mean "keep"** on the first frame (camera-side half, without the halfway area).
3. Review the kept scenes on the timeline; adjust sensitivity or target length if needed.
4. Click **Save as MP4**.

## How it works
1. Decode keyframes only (usually 1 per second) with WebCodecs and shrink them to 640-px luma images.
2. Compare with a ±60 s temporal-median background to find moving people inside the zone; compute
   perspective-corrected "person equivalents" and counts.
3. A calibrated ensemble of scorers gives each moment a probability; a person detector (D-FINE-N, Apache-2.0)
   is used only for uncertain moments.
4. Merge short gaps, add a lead-in before each scene (5 s by default, so you see how the attack developed) and a
   2 s tail, snap to keyframes, and copy the compressed packets into the output.

Evaluation: [docs/g1_report.md](docs/g1_report.md) · Algorithm contract: [docs/algorithms.md](docs/algorithms.md)

## Status
Early release. It has been evaluated on **one** 12-minute 4K H.264 futsal recording from a tripod-mounted Android phone:
all 7 episodes of 3 s or longer were kept, with about 0.9 F1 per second of footage ([report](docs/g1_report.md)).
Other cameras and angles, HEVC (iPhone) and full 1–2 hour games are still being validated, so please
[tell us how it went](https://github.com/kachy2/futsalcut/issues/new/choose) (the app's diagnostics are enough; no video needed).

## Browser support
| Environment | Status |
|---|---|
| Chrome / Edge (Windows, Mac) | Recommended. H.264 everywhere; HEVC depends on GPU/extension |
| Safari 26+ (Mac) | Analysis works; saving large files planned |
| Firefox | H.264 analysis only; saving large files planned |
| Mobile | Experimental |

## For developers
- `web/` — Svelte 5 + Vite app (analysis/export workers, TypeScript core that reproduces the Python reference bit for bit)
- `python/` — reference implementation (sampler, features, ensemble, lossless remuxer), Docker image `futsalcut-py`
- `eval/` — metrics, algorithm tournament, baseline comparison with existing tools and AI-proposed scripts
  (the recordings, labels and the proposed scripts themselves stay private)
- Tests — `npm test` (Vitest incl. Python parity), `npx playwright test` (headless Chrome end-to-end)

## License
MIT. Bundled components keep their licenses: see [LICENSE](LICENSE) and [web/public/models/NOTICE.md](web/public/models/NOTICE.md).
