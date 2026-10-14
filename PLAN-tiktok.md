# TikTok cut

The 16:9 reel stays as it is. The vertical file uses the tested Players' best light preset.

## Rules

- Keep the existing reel and prior generations.
- Keep the reel order. Do not add a teaser card.
- Use normal speed, original sound, and no titles.
- Use the selected reference video's crop and blurred background.
- Argon chooses each interval and crop position. GPT does not choose the edit.
- Get shot bounds from `keeps.json` when present.

## Input and prompt

Upload the original reel to private Cloud Storage. Sample it at 8 fps with HIGH thinking and temperature 0.7.
Build `prompts/players_best_light.txt` with the current duration, shot bounds, sport, and target.
For the reference volleyball reel, this generates the exact winning prompt.

## Timing and publication

Check that every source shot has one decision with a unique index.
Keep each selected interval inside its source shot. Keep crop positions between 0 and 1.
The selected durations must total the target within 0.05 seconds.
Ask Argon to correct invalid plans, with a limit of three attempts.
Do not use `fit_pace` to rescue a short or long plan. It only aligns an accepted plan to the target.
Publish the video after the renderer checks picture duration and audio synchronization.
Save the prompt, attempts, settings, source hash, and accepted decisions beside the video.
