# Baseline comparison: the 8 AI proposals, DVR-Scan, and our candidates

Track A runs each script as-is (paths + headless fix only) on a 60-s 4K excerpt (05:00-06:00) on the test
host (2 vCPU, no AVX). Track B re-implements each script's scoring idea on the shared 1 Hz cache with
identical post-processing, so quality differences come from the idea, not from incidental bugs.

| proposal | method | track A: status | wall (60 s) | est. 2 h video | output usable? | idea-level CV F1 (track B) |
|---|---|---|---|---|---|---|
| claude_full | 2 s chunks, YOLOv4→MobileNet→HOG fallback, OpenCV re-encode | ok | 479 s | 16.0 h | 2 s, audio NO | hog_full 0.14 |
| claude_simple | 2 s chunks, MOG2 + HOG, OpenCV re-encode | ok | 58 s | 1.9 h | 0 s, audio NO, **empty file** | mog2_blobs_full 0.69, hog_full 0.14 |
| gemini | 3 s chunks, frame differencing every frame, top-N, OpenCV re-encode | ok | 213 s | 7.1 h | 15 s, audio NO | framediff_full 0.70 |
| chatgpt | 2 s chunks, YOLOv8n person count (no ROI), ffmpeg concat list | ok | 284 s | 9.5 h | 12 s, audio yes; cut list of 6 x 2 s chunks rendered with ffmpeg concat; inpoints not on keyframes -> 17+ duplicate-timestamp warnings | dfine_full_count 0.42 |
| grok | MOG2 blobs + differencing every frame, ffmpeg copy | ok | 340 s | 11.3 h | no output | mog2_blobs_full 0.69, framediff_full 0.70 |
| perplexity | 2 fps samples, HOG + differencing, 8 s chunks, ffmpeg copy | ok | 87 s | 2.9 h | 17 s, audio yes | hog_full 0.14, framediff_full 0.70 |
| qwen | YOLOv5 on every frame, OpenCV re-encode | not runnable as-is | - | ~200.0 h (est.) | no output (not runnable as-is: torch.hub asks interactively to trust the repo, then downloads code and pip-installs requirements at runtime; estimate = YOLOv5s on every frame at ~3.4 s/frame on this host) | dfine_full_count 0.42 |
| deepseek | MobileNet-SSD (Caffe prototxt + TF .pb mismatch) | not runnable | - | - | - | - |
| liner | concept only, no code | not runnable | - | - | - | - |

Reference rows (track B):

| candidate | AP | CV F1 [95% CI] | kept |
|---|---|---|---|
| dvrscan_t0.15 | 0.503 | 0.58 [0.33, 0.75] | 23.2% |
| dvrscan_t0.5 | 0.631 | 0.72 [0.46, 0.85] | 18.6% |
| fg_zone_w_gain | 0.956 | 0.88 [0.73, 0.96] | 9.1% |
| pe_foot | 0.932 | 0.87 [0.71, 0.93] | 9.4% |
| count_foot | 0.922 | 0.85 [0.73, 0.94] | 9.2% |
| dfine_count | 0.953 | 0.84 [0.70, 0.91] | 11.3% |
| framediff_zone | 0.940 | 0.79 [0.57, 0.93] | 9.1% |
| anchor_keep_all | 0.129 | 0.14 [0.04, 0.25] | 100.0% |
