# Feasibility: 2-hour gym game to a 2-3 minute 9:16 highlight reel

**Verdict: possible with caveats.**

A pipeline can ingest a ~2 hour game, return event times, crop to 9:16, and join clips. Vendors document long-video ingest and timestamps. They do not document broadcast-quality volleyball "kill" or basketball "dunk" detectors for a distant gym camera. A person must review cuts. Generative video APIs are the wrong tool for this job.

**Minimum viable stack**

1. A file the rights holder supplies (not an unofficial YouTube download).
2. Cheap local prefilter with FFmpeg `scdet` and `silencedetect`.
3. Twelve Labs Pegasus 1.5 sports segments, or Gemini Flash on clipped windows with timestamps.
4. Rank the candidate clips. Keep about 8-12 clips.
5. FFmpeg `crop` to 9:16, then the FFmpeg concat demuxer to a 2-3 minute reel.

**Example input**

| Field | Value | Source |
| --- | --- | --- |
| URL | https://www.youtube.com/watch?v=sbGPehFw7n8 | User |
| Title | Sept 12 - Saturday intermediate javelin drop in | [YouTube oEmbed](https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=sbGPehFw7n8&format=json) |
| Channel | Javelin Ottawa | [YouTube oEmbed](https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=sbGPehFw7n8&format=json) |
| Duration | 6632 s (1 h 50 m 32 s) | Public watch page `lengthSeconds` |
| Description | Empty in the public watch-page JSON this research pass parsed | Public watch page |

The title names a drop-in session, not a broadcast match. Treat the footage as amateur gym video.

Cost numbers below use this duration (6632 s, about 111 min) and a round 2-hour case. Prices are list rates on 15 Sep 2026. They change.

---

## Proof run on this file (15 Sep 2026)

A local prototype cut a 9:16 reel from a 2-minute slice (08:00–10:00). No paid vision API ran. Motion peaks plus FFmpeg crop/concat are enough to prove the **edit** half. They are not enough to prove "cool moment" quality.

| Artifact | Path |
| --- | --- |
| 2 min 360p sample | `proof/sample.mp4` |
| Peak frames | `proof/out/peaks/` |
| 7 s clips | `proof/out/clips/` |
| 28 s 9:16 reel | `proof/out/vertical_reel.mp4` |
| JSON report | `proof/out/report.json` |
| Storyboard (74 sheets, ~90 s each) | `proof/out/storyboard/` |

Motion at 8 fps found four peaks. A person labeled each peak frame:

| Source time | Motion | Human label | Keep |
| --- | --- | --- | --- |
| 8:16 | 5.95 | Walk / reset | no |
| 8:31 | 6.08 | Idle between plays | no |
| 9:24 | 6.53 | Rally, ball over net | yes |
| 9:32 | 7.38 | Net contact, ball in the 9:16 crop | yes |

Precision of motion-only keep: 2/4. Gemini 3.6 Flash then scored the four 7 s clips and kept **0**. It rejected walking, a failed attack, a reset, and a net error. That matches the editorial rule (both teams at their best, no funny misses).

The Developer API then returned `429 RESOURCE_EXHAUSTED` on a YouTube window: AI Studio prepay is CA$0. Google AI Pro does not fund that wallet. About $83 of Google Developer Program credits sit on the Firebase Payment billing account. Official billing docs: prepay users must buy a minimum of **$5** of AI Studio credits before those GCP credits apply.

Run the detector again:

Storyboard sheets at 0:00, 1:00:00, and 1:49:12 are volleyball. This file has no basketball. The same stack still applies to a basketball drop-in.

Run the detector again:

```
python proof/detect_highlights.py
```

---

## 1. Video-understanding APIs: ingest limits

### Google Gemini (Files API + video)

Gemini accepts video. Official input methods:

| Method | Max size | Recommended use |
| --- | --- | --- |
| Files API | 20 GB paid / 2 GB free | Files over 100 MB, videos over 10 min, reuse |
| Cloud Storage registration | 2 GB per file | Long videos, persistent reuse |
| Inline data | under 100 MB | Short clips under 1 min |
| YouTube URLs | N/A | Public YouTube videos only |

Source: [Gemini video understanding](https://ai.google.dev/gemini-api/docs/video-understanding).

The Files API page also says: store up to 20 GB per project, **2 GB per file**, delete after 48 hours. Source: [Files API](https://ai.google.dev/gemini-api/docs/files).

Those two pages disagree on paid Files API size (20 GB vs 2 GB per file). Plan for a 2 GB per-file cap unless you confirm the video table. A 1080p two-hour file often exceeds 2 GB. Split the file or use Cloud Storage.

**Duration / tokens**

- Models with a 1M context window can process up to **3 hours** at low media resolution, or **1 hour** at high media resolution. Source: [Gemini video understanding, Technical details](https://ai.google.dev/gemini-api/docs/video-understanding).
- Default **static** mode samples **1 FPS**. Audio is 1 Kbps, single channel. Timestamps land every second. Fast action can lose detail. Source: same page.
- You can set custom `fps` and `start_offset` / `end_offset` in static mode. Source: same page, "Customize video processing".
- Gemini 3.8 / 3.7 / 3.6 Flash and 3.5 Flash Lite support **agentic** mode. The model loads transcript, frames, and audio on demand. Google claims up to 88% fewer tokens on long-form content. Source: same page.

**YouTube URL ingest (preview)**

You can pass a public YouTube URL as video input. Limits:

- Free tier: at most 8 hours of YouTube video per day.
- Paid tier: no length quota on this feature. Context limits still apply.
- Public videos only. Not private. Not unlisted.
- Gemini 2.5 and later: at most 10 videos per request.

The feature is in preview and currently at no charge. Pricing can change. Source: [Gemini video understanding, YouTube URLs](https://ai.google.dev/gemini-api/docs/video-understanding).

This path can **analyze** the example video. It does not return pixels you can cut.

**Token math (static mode)**

- Low resolution: about 66 tokens per frame + 32 audio tokens per second ≈ **100 tokens/s**.
- High resolution: 258 tokens per frame + 32 audio ≈ **300 tokens/s**.

Source: [Gemini video understanding](https://ai.google.dev/gemini-api/docs/video-understanding) and [token guide](https://ai.google.dev/gemini-api/docs/tokens).

For 6632 s at low static: about 663k tokens. At high static: about 2.0M tokens.

**Cost (Gemini 3.7 Flash, paid, through 31 Dec 2026)**

Input $0.75 per 1M tokens. Output $3.75 per 1M tokens. Source: [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing).

| Mode | Input tokens (approx) | Input cost |
| --- | --- | --- |
| Static, low | 663k | about $0.50 |
| Static, high | 2.0M | about $1.49 |
| Agentic, if 88% fewer | about 80k-240k | about $0.06-$0.18 |

Gemini 3.5 Flash Lite input is $0.30 per 1M tokens and supports agentic video. Source: [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing). Low static on Flash Lite is about $0.20 for this file.

Agentic savings assume the model does not load every second. A prompt that asks for **all** highlights may load most of the video. Treat the 88% figure as a best case, not a guarantee.

### Google Cloud Video Intelligence (Vertex / classic Video AI)

This is a classic annotation API, not an LLM.

| Limit | Value | Source |
| --- | --- | --- |
| Video size | 50 GB | [Quotas](https://cloud.google.com/video-intelligence/quotas) |
| Videos per request | 1 | same |
| Length | up to 3 hours | same |
| Features | label, shot change, explicit, face, speech, OCR, object tracking, logo, person | [videos.annotate](https://cloud.google.com/video-intelligence/docs/reference/rest/v1/videos/annotate) |

Pricing is per minute. Partial minutes round up. First 1,000 minutes per feature per month are free. After that (stored video): label $0.10/min, shot $0.05/min, object tracking $0.15/min, person $0.10/min, OCR $0.15/min. Source: [Video Intelligence pricing](https://cloud.google.com/video-intelligence/pricing).

One 111-minute file sits inside the monthly free bucket if the project is under 1,000 minutes. After the free bucket: labels plus shots ≈ $16.65. Add object tracking: about $33 more.

Labels are general (sport, ball, person). The API does not name volleyball kills.

### Twelve Labs (Pegasus / Marengo)

**Pegasus 1.5** (video-to-text, segmentation):

- Duration 4 s to **2 hours**. File size ≤ **2 GB**. Resolution 360×360 to 5184×2160.
- Aspect ratio between 1:1 and 1:2.4, or 2.4:1 and 1:1 (16:9 and 9:16 are allowed).
- Context window 261,120 tokens (input + output). Max output 98,304 tokens.
- Sync analyze: max **1 hour**. Async analyze: max **2 hours**.
- Direct URLs must be raw media files. Hosting-platform links are not supported.

Sources: [Pegasus model card](https://docs.twelvelabs.io/docs/concepts/models/pegasus), [async analyze](https://docs.twelvelabs.io/docs/guides/segment-videos).

The example video is 1 h 50 m. That is under 2 hours. File size may still exceed 2 GB at 1080p.

**Marengo** (embeddings / search):

- Duration 4 s to **4 hours**. File size ≤ **4 GB**.

Source: [Marengo model card](https://docs.twelvelabs.io/docs/concepts/models/marengo).

**Pricing (Developer, pay as you go)**

- Marengo video indexing: **$2.50 per hour**.
- Infrastructure: **$0.09 per hour** per month.
- Search: **$4 per 1,000 queries**.
- Pegasus Analyze: **$1.75 per hour** of input video. Output text **$7.50 per 1M tokens**.
- Segment bills the time window times the number of segment definitions.

Source: [Twelve Labs pricing](https://www.twelvelabs.io/pricing).

For 1.84 hours: Analyze ≈ $3.22. Marengo index ≈ $4.60. Segment with 4 definitions ≈ $12.88.

### OpenAI (GPT-4o / GPT-4.1 vision frames)

OpenAI documents **no native video file input** on the Responses API. The official cookbook extracts frames and sends images. Source: [OpenAI cookbook: video with vision](https://developers.openai.com/cookbook/examples/gpt_with_vision_for_video_understanding). The Node SDK issue close states the same: extract frames, process audio separately. Source: [openai-node#1778](https://github.com/openai/openai-node/issues/1778) (vendor SDK repo).

Vision request limits:

- File types: PNG, JPEG, WEBP, non-animated GIF.
- Payload up to **512 MB** per request.
- Up to **1,500 images** per request.

Source: [Images and vision](https://developers.openai.com/api/docs/guides/images-vision).

At 1 frame per second, 6632 frames need at least 5 requests. The 512 MB cap often forces fewer, smaller JPEGs per call.

Sora generates new video. It does not understand a 2-hour file. 720p Sora 2 is $0.10 per second of **output**. Source: [OpenAI pricing](https://platform.openai.com/docs/pricing). A 180 s generated clip would be about $18 and would not be the original game.

### Anthropic Claude vision

Claude accepts images, not video files. Animated GIFs use only the first frame. Source: [Claude vision](https://platform.claude.com/docs/en/build-with-claude/vision).

Limits:

- 100 images per request on 200k-context models.
- 600 images per request on other models.
- 10 MB per image (direct API). 32 MB request size on standard endpoints.

A 2-hour game at 1 FPS cannot fit in one request. You must sample and chunk.

### AWS Rekognition Video

Stored video:

- Up to **10 GB** and **6 hours**.
- H.264 in MPEG-4 or MOV.
- Up to 20 concurrent jobs per account.
- Results kept 7 days.

Sources: [Guidelines and quotas](https://docs.aws.amazon.com/rekognition/latest/dg/limits.html), [StartLabelDetection](https://docs.aws.amazon.com/rekognition/latest/APIReference/API_StartLabelDetection.html).

Stored-video label detection list price: **$0.10 per minute**. Shot detection: **$0.05 per minute**. Free tier: 60 minutes per month for 12 months. Source: [Rekognition pricing](https://aws.amazon.com/rekognition/pricing/).

For 111 minutes of labels: about $11.10. Labels plus shots: about $16.65.

### Azure AI Video Indexer

- Device upload: **2 GB**. URL upload: **30 GB**. The URL must be a media file, **not a YouTube page**.
- Duration: **6 hours** (12 hours for Basic Audio).
- Output transcode: 720p MP4 H.264 + AAC.
- Trial account: up to **2,400 minutes** of free indexing on the website.

Sources: [Support matrix](https://learn.microsoft.com/en-us/azure/azure-video-indexer/avi-support-matrix), [What is Azure AI Video Indexer?](https://learn.microsoft.com/en-us/azure/azure-video-indexer/video-indexer-overview).

The paid pricing page did not load in this research pass. Use the Azure pricing page before you budget. The trial covers this one file.

Azure lists **content creation** (trailers, highlight reels) as a documented use of scenes, shots, people, and labels. Source: [overview, Content creation](https://learn.microsoft.com/en-us/azure/azure-video-indexer/video-indexer-overview). That is generic insight, not sport-specific events.

### Groq / Llama vision

Groq vision is **images only**. Current models: `qwen/qwen3.6-27b` (max **5** images, 20 MB request) and `qwen/qwen3.8-27b` (max **3** images). Each image counts as 2048 input tokens. Source: [Groq Images and Vision](https://console.groq.com/docs/vision).

Not useful as a 2-hour video model. You could score a handful of candidate frames after a prefilter.

### Comparison (one ~2 h 1080p file)

| Vendor | Native video | Max duration | Hard size | Time precision (documented) | Fit for this file |
| --- | --- | --- | --- | --- | --- |
| Gemini | Yes | 3 h low / 1 h high | 2-20 GB (see conflict) | 1 s default | Yes, with caveats |
| Twelve Labs Pegasus | Yes | 2 h | 2 GB | `start_time` / `end_time` seconds | Yes if file ≤ 2 GB |
| Twelve Labs Marengo | Yes | 4 h | 4 GB | clip `start` / `end` seconds | Yes |
| Vertex Video Intelligence | Yes | 3 h | 50 GB | Duration with up to 9 fractional digits | Yes (generic labels) |
| Rekognition Video | Yes | 6 h | 10 GB | ms, sampled at 2 FPS | Yes (generic labels) |
| Azure Video Indexer | Yes | 6 h | 2 GB device / 30 GB URL | insight start/end | Yes if not a YouTube URL |
| OpenAI | Frames only | Bounded by 1500 images + 512 MB | 512 MB/request | Only if you stamp frames yourself | Chunked only |
| Claude | Frames only | Bounded by 100-600 images + 32 MB | 32 MB/request | Same | Chunked only |
| Groq | Frames only | 3-5 images | 20 MB | Same | Prefilter only |

---

## 2. Can they return timestamps precise enough to cut?

**Yes at about 1 second. Not frame-accurate for a spike.** Pad each cut by 1-2 seconds.

### Gemini

You can ask about times in `MM:SS`. Prompts can request timestamps for salient moments. File API static mode adds timestamps every second. Above 1 FPS, use `MM:SS.sss`. Offsets over one hour use `H:MM:SS`. Source: [Gemini video understanding](https://ai.google.dev/gemini-api/docs/video-understanding) and [Vertex video understanding](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/video-understanding).

Google states that fast action may lose detail at 1 FPS. A volleyball kill is often under 1 second. Default sampling can miss the contact frame. Raise `fps` on short candidate windows after a prefilter.

### Twelve Labs

Pegasus lists **temporal grounding**: "Accurately identifies timestamps of specific events." Source: [Pegasus](https://docs.twelvelabs.io/docs/concepts/models/pegasus).

Segment mode returns `start_time` and `end_time` in seconds. Example print: `[45.0s - ...]`. Source: [Segment videos](https://docs.twelvelabs.io/docs/guides/segment-videos).

Marengo search returns `start` and `end` in seconds per clip. Source: [Search quickstart](https://docs.twelvelabs.io/docs/get-started/quickstart/search).

Timestamp formats include `seconds`, `hh:mm:ss`, and `hh:mm:ss.fff`. Source: [Release notes](https://docs.twelvelabs.io/docs/get-started/release-notes).

Sports is a documented segment use: "Detect sports highlights: Find scoring plays and key moments." Source: [Segment videos](https://docs.twelvelabs.io/docs/guides/segment-videos). Vendors do not publish a measured F1 for amateur volleyball kills.

### Vertex Video Intelligence

Shot change and object tracks use `startTimeOffset` / `endTimeOffset` / `timeOffset` as durations with up to nine fractional digits (example `"3.5s"`). Source: [AnnotateVideoResponse](https://cloud.google.com/video-intelligence/docs/reference/rest/v1/AnnotateVideoResponse), [Detect shot changes](https://cloud.google.com/video-intelligence/docs/analyze-shots).

Precision is high for **shots** and **tracked objects**. Labels are not "kill" or "3-pointer".

### Rekognition

`Timestamp` is milliseconds from start. AWS says it is **not guaranteed** accurate to the first frame. `TIMESTAMPS` aggregation uses **2 FPS** sampling. That rate can change. Source: [Calling Video operations](https://docs.aws.amazon.com/rekognition/latest/dg/api-video.html), [LabelDetection](https://docs.aws.amazon.com/rekognition/latest/APIReference/API_LabelDetection.html).

Good enough to find a window. Not enough to cut on the ball contact without pad.

### Azure Video Indexer

Insights include `start` / `end` (example `0:01:21.82`). Observed people include bounding boxes and start/end. Audio effects include crowd reactions (cheering, clapping, booing) on Advanced Audio. Source: [overview](https://learn.microsoft.com/en-us/azure/azure-video-indexer/video-indexer-overview), [keywords insight example](https://learn.microsoft.com/en-us/azure/azure-video-indexer/keywords-insight).

Crowd cheer is a useful highlight proxy when speech is noise.

### OpenAI / Claude / Groq

No vendor video clock. You attach times when you sample frames (for example frame 120 at 1 FPS = 00:02:00). Precision equals your sample rate.

**Cut rule:** use model times as windows, not as edit points. Expand each window by 1-2 s. Snap to nearby scene changes from FFmpeg `scdet`.

---

## 3. Recommended pipeline

**Recommended: cheap prefilter, then a video model on short windows. Native long-video models are optional, not required.**

This is what vendor docs support, not a blog recipe.

### Stage A — local prefilter (cheap)

FFmpeg `scdet` sets `lavfi.scd.time` when a scene change crosses a threshold (default 10, good range 8-14). Source: [FFmpeg scdet](https://ffmpeg.org/ffmpeg-filters.html#scdet).

FFmpeg `silencedetect` logs `silence_start`, `silence_end`, and duration when volume stays below a noise floor. Inverse: loud non-silence is a crowd or whistle candidate. Source: [FFmpeg silencedetect](https://ffmpeg.org/ffmpeg-filters.html#silencedetect).

Azure Advanced Audio detects cheering, clapping, and booing in non-speech segments. Source: [Azure Video Indexer overview](https://learn.microsoft.com/en-us/azure/azure-video-indexer/video-indexer-overview).

Vertex shot detection returns shot boundaries with sub-second offsets. Source: [Detect shot changes](https://cloud.google.com/video-intelligence/docs/analyze-shots).

Do not send 2 hours of 1080p at high FPS into an LLM first. Gemini itself says set **low FPS for long videos** and higher FPS only when you need fine time analysis. Source: [Gemini video understanding](https://ai.google.dev/gemini-api/docs/video-understanding).

### Stage B — video model on candidates

Two documented patterns:

1. **Native long video.** Gemini agentic mode is the documented default for long-form (lectures, sports matches, archives). Static mode is for short clips under 5 minutes or full-clip frame inspection. Source: [Agentic video guide](https://aistudio.google.com/learn/agentic-video-understanding-with-gemini) and [Gemini video understanding](https://ai.google.dev/gemini-api/docs/video-understanding). Twelve Labs Pegasus segment mode is documented for sports plays. Source: [Segment videos](https://docs.twelvelabs.io/docs/guides/segment-videos).

2. **Clip then score.** Gemini `start_offset` / `end_offset` analyze a window (example 1200-1500 s). Source: [Gemini video understanding](https://ai.google.dev/gemini-api/docs/video-understanding). OpenAI's cookbook samples frames and does not send every frame. Source: [OpenAI cookbook](https://developers.openai.com/cookbook/examples/gpt_with_vision_for_video_understanding).

For gym sports, combine them:

1. Run FFmpeg loudness + scene change. Keep the top N peaks (for example 40-80 windows of 8-12 s).
2. Send those windows to Gemini static at 4-8 FPS, or to Pegasus with sport segment definitions.
3. Ask for JSON: event type, start, end, score, reason.
4. Rank. Keep 8-12 clips that sum to 120-180 s.

### What not to do

- Do not use Rekognition labels alone as "kill" detectors. Labels are generic.
- Do not use OpenAI/Claude as the only pass over 7,000 full-resolution frames. Cost and payload limits fight you.
- Do not use Groq vision as the long-video stage (3-5 images per request).

---

## 4. Vertical 9:16 crop

### Center crop (proven, dumb)

FFmpeg `crop` defaults `x` to `(in_w-out_w)/2` and `y` to `(in_h-out_h)/2` (center). `x` and `y` evaluate **per frame**, so a time-varying window is valid. Source: [FFmpeg crop](https://ffmpeg.org/ffmpeg-filters.html#crop).

Example 9:16 from landscape, using input height: `crop=ih*9/16:ih` (center, because x/y default). Then `scale=1080:1920` if you need that size.

Remotion `Video` supports `cropLeft` / `cropRight` / `cropTop` / `cropBottom` ratios and `objectFit`. Source: [Remotion Video](https://www.remotion.dev/docs/media/video).

### MediaPipe AutoFlip (smart crop, legacy)

AutoFlip reframes to an `aspect_ratio` such as 9:16. It detects faces and objects, then chooses stationary, panning, or tracking crops. Google ended support for this **legacy** solution on **1 Mar 2023**. Source: [AutoFlip docs](https://github.com/google/mediapipe/blob/master/docs/solutions/autoflip.md).

Treat AutoFlip as research / maintenance risk, not a product API.

### MediaPipe Tasks (current)

Pose Landmarker and Object Detector run in `VIDEO` mode with `detect_for_video(image, timestamp_ms)` and return landmarks or boxes. Source: [Pose Landmarker (Python)](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python), [Object Detector](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector).

On a distant gym camera, people are small. Official Rekognition guidance: a face must be at least 40×40 in a 1920×1080 image. Source: [Rekognition quotas](https://docs.aws.amazon.com/rekognition/latest/dg/limits.html). Pose on a far court is research-grade, not proven.

### Gemini pointing

`gemini-robotics-er-2-preview` returns points `[y, x]` normalized 0-1000 and bounding boxes. It can track an object in video frames. Input token limit on the robotics overview is **131,072**. Source: [Spatial reasoning](https://ai.google.dev/gemini-api/docs/robotics-spatial), [Robotics overview](https://ai.google.dev/gemini-api/docs/robotics-overview).

131k tokens is far below a 2-hour video. Use pointing **only on short highlight clips** to pick a crop center.

### SAM 2

SAM 2 is a self-hosted video tracker (`propagate_in_video`). Large checkpoint: 39.5 FPS on an A100 (compiled benchmark). Source: [facebookresearch/sam2 README](https://github.com/facebookresearch/sam2/blob/main/README.md).

That is a research model, not a hosted "smart crop" API. Tracking one subject through 2 hours at full frame rate needs a GPU and prompts. Use it on the 8-12 kept clips, not on the whole game.

### Vertex object / person tracking

Object and person tracks include `normalizedBoundingBox` and `timeOffset`. Source: [AnnotateVideoResponse](https://cloud.google.com/video-intelligence/docs/reference/rest/v1/AnnotateVideoResponse). Azure observed people also return boxes. Source: [overview](https://learn.microsoft.com/en-us/azure/azure-video-indexer/video-indexer-overview).

Boxes can drive FFmpeg `crop` x/y. A wide gym shot often boxes many people. The "subject" is the play, not one face.

**Practical crop for v1:** center 9:16. If the court sits in the middle, that is enough. Add pose or SAM 2 later on clips that fail.

---

## 5. Editing: compile, do not generate

**Use FFmpeg (or MoviePy / Remotion on top of FFmpeg). Do not use a generative video model to rebuild the game.**

### Deterministic editors (right tool)

- **FFmpeg concat demuxer** joins files in order. Same codecs and time base required. `file`, `duration`, `inpoint`, `outpoint` directives. Source: [FFmpeg concat demuxer](https://ffmpeg.org/ffmpeg-formats.html#concat).
- **MoviePy** `concatenate_videoclips` plays clips one after another. `method='chain'` or `'compose'`. Source: [concatenate_videoclips](https://zulko.github.io/moviepy/reference/reference/moviepy.video.compositing.CompositeVideoClip.concatenate_videoclips.html).
- **Remotion** `<Video trimBefore trimAfter from durationInFrames>` plus `<Series>` sequences clips on a timeline. Source: [Remotion Video](https://www.remotion.dev/docs/media/video), [Series](https://www.remotion.dev/docs/series).

There is no official "editing LLM" that outputs a finished reel from a 2-hour file. The model returns times. FFmpeg cuts.

### Generative APIs (wrong tool)

These APIs **create new pixels**. They do not compile existing footage.

| Product | What the API does | Duration | Why it is wrong here |
| --- | --- | --- | --- |
| Runway Gen-4.5 | text/image to video | 2-10 s | New clip, not the game. Source: [Runway API](https://docs.dev.runwayml.com/api.md) |
| Runway video-to-video | edit an input video | about 10 s | Input is a short clip, not 2 hours. Source: [Runway changelog](https://docs.dev.runwayml.com/api-details/api_changelog/) |
| OpenAI Sora 2 | generate video | billed per second | $0.10/s at 720p. Source: [OpenAI pricing](https://platform.openai.com/docs/pricing) |
| Gemini Veo | generate video | billed per second | $0.05-$0.40/s. Source: [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing) |
| Kling Open Platform | text-to-video / image-to-video | short generated clips | Official getting-started is generation, not NLE compile. Source: [Kling API overview](https://kling.ai/document-api/guides/get-started/overview) |

Video-to-video on a 8 s highlight can add style. It is optional and expensive. It is not how you make the reel.

---

## 6. Cost ballpark: one 2-hour 1080p game → 3-minute reel

Assumptions: 6632 s (example video) or 7200 s (round 2 h). 1080p file already on disk. No download cost. Local FFmpeg is $0 API. Human review is unpaid here.

### Path A — cheapest documented (MVP)

| Step | Service | Ballpark |
| --- | --- | --- |
| Prefilter | FFmpeg scdet + silencedetect | $0 |
| Score 60 windows × 10 s at 4 FPS | Gemini 3.5 Flash Lite | about $0.20-$1 |
| Or one agentic/static pass on the full file | Gemini 3.7 Flash low | about $0.50 (static) |
| Crop + concat | FFmpeg | $0 |
| **Total API** | | **about $0.50-$2** |

### Path B — Twelve Labs sports segments

| Step | Ballpark |
| --- | --- |
| Pegasus Analyze, full file | about $3.22 |
| Or Segment with 3 definitions | about $9.66 |
| Optional Marengo index + a few searches | about $4.60 + $0.01 |
| FFmpeg | $0 |
| **Total** | **about $3-$15** |

### Path C — classic video AI

| Step | Ballpark |
| --- | --- |
| Vertex labels + shots (under 1,000 min/month free) | $0 |
| Same, after free bucket | about $16.65 |
| Rekognition labels | about $11.10 |
| Azure trial (≤ 2,400 min) | $0 on trial |
| **Total** | **$0-$17** plus a later LLM rank if you still need "kill" vs "rally" |

### Path D — OpenAI frames (not recommended as the only pass)

7,000 JPEGs, 1,500 per request, 512 MB cap. Token cost depends on model and `detail`. Do not budget this without a sample `count_tokens` / usage call. It is usually tens of dollars at GPT-4o-class high detail, less at a cheap vision model with heavy downsampling.

### Path E — generate a fake reel (wrong)

Sora 2, 180 s, 720p: 180 × $0.10 = **$18**, and the output is not the game. Source: [OpenAI pricing](https://platform.openai.com/docs/pricing).

**Honest range for a real reel of original footage: about $1 to $15 in APIs, plus compute and review time.** Classic video AI without the free tier can reach about $20. Generative video does not replace this.

File size: if the MP4 exceeds 2 GB, add transcode or split cost (local). Twelve Labs Pegasus and some upload paths cap at 2 GB.

---

## 7. Failure modes: amateur gym footage

Vendors do not publish a gym-volleyball accuracy card. These risks follow from documented limits.

| Risk | Why it hurts | Source of the limit |
| --- | --- | --- |
| Distant camera | Faces and people fall below detector size. Rekognition: min face 40×40 in 1080p. | [Rekognition quotas](https://docs.aws.amazon.com/rekognition/latest/dg/limits.html) |
| Fast play | Gemini default 1 FPS. Rekognition labels at 2 FPS. A spike/kill can sit between samples. | [Gemini video](https://ai.google.dev/gemini-api/docs/video-understanding), [Rekognition video](https://docs.aws.amazon.com/rekognition/latest/dg/api-video.html) |
| Poor lighting | No vendor SLA. Low contrast hurts OCR, pose, and tracking. | Inference from OCR/pose dependence |
| No broadcast overlay | No scorebug. Azure/Gemini OCR and "on-screen text" in Pegasus have nothing to read for score. | [Pegasus features](https://docs.twelvelabs.io/docs/concepts/models/pegasus), [Azure OCR](https://learn.microsoft.com/en-us/azure/azure-video-indexer/video-indexer-overview) |
| Crowd noise | Speech models hear crowd, not play-by-play. Azure crowd-reaction audio is the useful signal. Speech transcription is not. | [Azure audio effects](https://learn.microsoft.com/en-us/azure/azure-video-indexer/video-indexer-overview) |
| Fixed wide shot | Center crop may keep the court. AutoFlip/face-weighted crop may track a nearby spectator. | [AutoFlip](https://github.com/google/mediapipe/blob/master/docs/solutions/autoflip.md) |
| Volleyball vs basketball | Classic APIs emit "sports", "person", "ball". "Kill", "block", "ace", "dunk", "3-pointer" need an LLM or custom Pegasus segment text. | [Video Intelligence features](https://cloud.google.com/video-intelligence/docs/reference/rest/v1/videos/annotate), [Segment videos](https://docs.twelvelabs.io/docs/guides/segment-videos) |
| Many similar rallies | Models can over-recall every rally. Ranking and a duration cap are required. | Product design, not a vendor guarantee |
| This example video | "Intermediate javelin drop in" is a gym session, not a TV feed. | [oEmbed title](https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=sbGPehFw7n8&format=json) |

**Volleyball-specific:** contact events are short. Default 1 FPS is a known miss. Score windows at 4-8 FPS.

**Basketball-specific:** a dunk is also short, but crowd spikes and ball trajectory may be easier. Still not a Rekognition label named "dunk".

Do not claim automatic sport IQ equal to a human editor. That claim is not in any cited model card.

---

## 8. YouTube: official API vs unofficial download

This section states vendor rules. It does not tell you how to bypass them.

### Official: YouTube Data API

`videos.list` returns metadata (`snippet` title/description, `contentDetails.duration`, statistics, player embed). It does **not** return media bytes or stream URLs. `fileDetails` is owner-only. Source: [videos.list](https://developers.google.com/youtube/v3/docs/videos/list), [Videos resource](https://developers.google.com/youtube/v3/docs/videos).

### Official: Gemini YouTube URL (preview)

Gemini can **analyze** a **public** YouTube URL. It does not give you an MP4 to edit. Source: [Gemini video understanding](https://ai.google.dev/gemini-api/docs/video-understanding).

### Official: Terms

YouTube Terms: you may not "access, reproduce, **download**, distribute…" Content except as the Service authorizes, or with prior written permission from YouTube and rights holders. Source: [YouTube Terms of Service](https://www.youtube.com/t/terms).

Developer Policies: API clients must not let users download videos for offline play outside YouTube Premium. They must not separate audio or modify audiovisual content via the API. Source: [Developer policies guide](https://developers.google.com/youtube/terms/developer-policies-guide).

API Developer Policies: do not download, import, backup, cache, or store copies of YouTube audiovisual content without YouTube's prior written approval. Source: [YouTube API Services - Developer Policies](https://developers.google.com/youtube/terms/developer-policies).

### Unofficial download tools

Tools such as yt-dlp are not a YouTube API. This document does not instruct their use. They conflict with the download restriction in the Terms above.

### Compliant input for cutting

1. The channel owner exports or records their own file (camera, YouTube Studio download of **their** upload, or Google Takeout where offered).
2. A user uploads a file they have the right to process.
3. Gemini may analyze a public URL for timestamps only. Cutting still needs a lawful file.

Azure Video Indexer rejects YouTube web pages as upload URLs. Source: [Support matrix](https://learn.microsoft.com/en-us/azure/azure-video-indexer/avi-support-matrix). Twelve Labs rejects video-hosting-platform links for URL ingest. Source: [Pegasus input notes](https://docs.twelvelabs.io/docs/concepts/models/pegasus).

---

## Proven today vs research-grade

### Proven (vendor-documented, shippable)

- Ingest ~2 hours of video (Gemini, Pegasus ≤ 2 h, Marengo ≤ 4 h, Video Intelligence ≤ 3 h, Rekognition ≤ 6 h, Azure ≤ 6 h).
- Return timestamps at ~1 s (Gemini, Twelve Labs) or finer for shots/tracks (Video Intelligence).
- Analyze a **public** YouTube URL in Gemini (preview), without a local file.
- Cut and join with FFmpeg crop + concat. MoviePy and Remotion wrap the same idea.
- Cheap audio/scene prefilter with FFmpeg.
- Azure crowd-reaction audio as a highlight hint.
- Cost in the low dollars for Gemini, or low tens for Twelve Labs / Rekognition.

### Research-grade or fragile

- Automatic volleyball kill / block / ace labels on a distant gym camera (no published gym benchmark in the cited docs).
- MediaPipe AutoFlip (legacy, support ended 1 Mar 2023).
- SAM 2 as a production 2-hour crop service (self-host GPU, prompts per object).
- Gemini robotics pointing on a full game (131k input tokens).
- Frame-accurate cuts at 1 FPS.
- One-click "YouTube URL in, 9:16 MP4 out" without a lawful media file.

### Not the right product

- Runway, Kling, Sora, Veo as the compiler of existing footage.

---

## Verdict

**Possible with caveats.**

The example file is 1 h 50 m 32 s. That is inside Gemini, Pegasus async, Marengo, Video Intelligence, Rekognition, and Azure duration caps. The hard parts are (1) a lawful file to cut, (2) 1 FPS miss on fast contact, (3) no scorebug, (4) 9:16 crop of a wide gym shot, (5) 2 GB upload caps.

**Minimum viable model stack**

| Layer | Pick | Fallback |
| --- | --- | --- |
| Ingest | User-supplied MP4 | Gemini public YouTube URL for analysis only |
| Prefilter | FFmpeg `scdet` + `silencedetect` | Azure Advanced Audio crowd reactions |
| Events | Twelve Labs Pegasus 1.5 segments **or** Gemini 3.5/3.7 Flash on windows | Vertex shots + LLM rank |
| Crop | FFmpeg center `crop` to 9:16 | Pose/SAM 2 / Gemini points on kept clips only |
| Edit | FFmpeg concat demuxer | MoviePy or Remotion |
| Review | Human pass on the 3-minute output | Required for v1 |

Do not start with generative video. Do not start with OpenAI/Claude/Groq as the long-video engine.

**Next measurement (about 2-4 hours of work):** run Gemini on 10 random 15-second windows from a lawful copy. Count true plays vs false crowd motion. If precision is poor, add Pegasus segment definitions before you build the editor.
