# Javelin highlight reel

Turn a gym volleyball MP4 into a 16:9 highlight reel. A person watches the reel, then posts it or rejects it.

## What the job does

1. `gemini-2.5-flash` scans the local MP4 in 10-minute windows.
2. The job keeps up to 75 plays by score. It does not pad with junk.
3. FFmpeg cuts 20 s 16:9 clips. There is no crop and no ball follow.
4. `gemini-3.8-flash` scores clips in batches of 10 and sets a 10-20 s trim.
5. One high-score rally can run to 40 s. The model watches that clip alone.
6. FFmpeg concatenates about 8 clips in time order.

If the 2.5 scan dies mid-file, the job still builds a reel from finished windows.

## Setup

Python 3.12 or later. Vertex AI on Google Cloud. FFmpeg comes from `imageio-ffmpeg`.

```
pip install -r requirements.txt
gcloud auth application-default login
```

Set the Vertex project in `proof/gemini_defaults.py`. The default project is `gen-lang-client-0442353446`. Location is `global`.

Do not put API keys in git. `proof/.env` stays local.

## Run a game

1. Drop one complete MP4 into `proof/inbox`.
2. In `proof`, run `python run_job.py`.
3. Watch `proof/gen_N/reel.mp4`. Post it or reject it.

A 2-hour file takes about 15-40 minutes. Time depends on scan coverage and Vertex quota.

Use a known file instead of the inbox:

```
python run_job.py --source source.mp4
```

Reuse an existing 2.5 scan:

```
python run_job.py --skip-scan --scan-dir scan
```

Extend the longest high-score clip to 40 s:

```
python run_job.py --from-gen 8 --extend-cool --gen 10
```

## Layout

- `proof/run_job.py` — MVP job.
- `proof/scan_local.py` — 2.5 window scan on the local file.
- `proof/review_clips.py` — 3.8 batch review and 40 s long-rally pass.
- `proof/inbox/` — drop folder.
- `proof/scan/` — example 2.5 timestamps for the Sept 12 match.
- `proof/legacy/` — old YouTube and 9:16 trial scripts.
- `proof/follow_cam.py` — 9:16 follow camera. Not in the MVP path.

## Tests

```
cd proof
python -m unittest test_v1 test_follow_cam
```

## What git ignores

Source MP4 files, `proof/.env`, `proof/gen_*` outputs, and YOLO weights. Do not commit those files.
