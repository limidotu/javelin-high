# TikTok cut

The 16:9 reel stays as it is. The vertical file is a 60 second polish of that reel.

## Rules

- Do not overwrite `reel.mp4` or the old `tiktok.mp4`.
- Keep the reel order. Do not open with a teaser card.
- Play every frame at normal speed. No slow motion and no speed-up.
- No on-screen titles.
- Scale the picture up and crop the ceiling so the players read on a phone. Keep a blurred background.
- Screen time is 60 seconds. Argon chooses the in and out. The fitter only trims or extends inside each shot.
- Shot bounds come from `keeps.json` when that file is present.

## Fit

Argon marks impact, in, out, center, label, and score inside each locked shot. `fit_to_target` assigns a budget that sums to the target. High scores get a higher floor. The window stays inside the shot and always contains the contact. A final conform pass makes the file duration 60 seconds.
