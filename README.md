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

Set the Vertex project in `pipeline/gemini_defaults.py`. The default project is `gen-lang-client-0442353446`. Location is `global`.

Do not put API keys in git. A local `.env` file stays off git.

## Run a game

1. Drop one complete MP4 into `pipeline/inbox`.
2. In `pipeline`, run `python run_job.py`.
3. Watch `pipeline/gen_N/reel.mp4`. Post it or reject it.

A 2-hour file takes about 15-40 minutes. Time depends on scan coverage and Vertex quota.

Use a known file instead of the inbox:

```
python run_job.py --source source.mp4
```

Reuse a scan from a prior job:

```
python run_job.py --skip-scan --scan-dir gen_N/scan
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
- `pipeline/gen_N/` — one job output. Git ignores this folder.

## Tests

```
cd pipeline
python -m unittest test_pipeline
```

## What git ignores

Source MP4 files, `.env`, `pipeline/gen_*` outputs, and scan JSON from a match. Do not commit those files.
