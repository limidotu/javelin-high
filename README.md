# Javelin highlight reel

Turn a gym volleyball MP4 into a 16:9 highlight reel. A person watches the reel, then posts it or rejects it.

## What the job does

1. `gemini-2.5-flash` scans the local MP4 in 10-minute windows.
2. A window starts as one 10-minute 1 fps clip. If Vertex rejects it, the job splits that window into two 5-minute clips. If a 5-minute clip fails, the job splits only that half into 60 s clips.
3. The job keeps up to 75 plays by score. It does not pad with junk.
4. FFmpeg cuts 20 s 16:9 clips. There is no crop and no ball follow.
5. `gemini-3.8-flash` scores clips in batches of 10 with high thinking. It sets a trim near 15 s (12 to 18 s).
6. One high-score rally can run to 40 s. The model watches that clip alone.
7. FFmpeg concatenates 8 to 10 clips in time order.

One command runs the full file. A dead 10-minute window is split, then scanned. A 429 streak still stops the scan.

If the 2.5 scan cannot cover the whole file, the job still builds a reel from finished windows. If 3.8 review stops early, the job fills the reel from unreviewed 2.5 clips.

## Setup

Python 3.12 or later. Vertex AI on Google Cloud. FFmpeg comes from `imageio-ffmpeg`.

```
pip install -r requirements.txt
gcloud auth application-default login
```

Set the Vertex project in `pipeline/gemini_defaults.py`. The default project is `gen-lang-client-0442353446`. Location is `global`.

Do not put API keys in git. A local `.env` file stays off git.

## Run a game

1. Drop one complete MP4 into `pipeline/inbox`.
2. In `pipeline`, run `python run_job.py`. The job creates `pipeline/generations` if that folder is missing.
3. Watch `pipeline/generations/gen_N/reel.mp4`. Post it or reject it.

A 2-hour file takes about 15-40 minutes. Time depends on scan coverage and Vertex quota.

If the MP4 is already in `pipeline/inbox/done`, that same command still finds it.

Use a known file instead of the inbox:

```
python run_job.py --source source.mp4
```

Reuse a scan from a prior job:

```
python run_job.py --skip-scan --scan-dir generations/gen_N/scan
```

Extend the longest high-score clip to 40 s:

```
python run_job.py --from-gen 8 --extend-cool --gen 10
```

## Layout

- `pipeline/run_job.py` — drop-folder job.
- `pipeline/scan_local.py` — 2.5 window scan on the local file.
- `pipeline/review_clips.py` — 3.8 batch review and 40 s long-rally pass.
- `pipeline/inbox/` — drop folder.
- `pipeline/generations/gen_N/` — one job output. Git ignores this folder.

## Tests

```
cd pipeline
python -m unittest test_pipeline
```

## What git ignores

Source MP4 files, `.env`, `pipeline/generations/` outputs, and scan JSON from a match. Do not commit those files.
