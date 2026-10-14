# Work

## Current preset

The page and `run_job.py` use Players' best light by default.
Argon receives the original reel with 8 fps sampling, HIGH thinking, and temperature 0.7.
The prompt uses each reel's duration, shot boundaries, sport, and target length.
The normal-speed renderer matches the selected reference video.

## Checks

- The reference reel generates the exact winning prompt.
- The saved winning decisions retain every interval and crop position.
- Invalid plans return to Argon before rendering, with a limit of three attempts.
- A failed render cannot replace a complete video.
- A fresh live run produced seven plays on its first attempt.
- The live output is 60 seconds at 1080x1920 and passed the picture and audio checks.

The live sample and its request records are in `pipeline/generations/gen_14/preset_verification/`.
The earlier ten-prompt comparison stays in `pipeline/generations/gen_14/prompt_trials/`.
These local media folders stay outside git.

## Limits

Model decisions depend on the footage. The renderer still uses one fixed crop per play.
The `polish_locked.py` script remains a manual rerender tool for older output.
