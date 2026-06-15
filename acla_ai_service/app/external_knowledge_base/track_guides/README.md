# Track guide knowledge (keyed coaching lookup)

One `.md` file per track. Read by the track guide agent through
`track_guide(track_id, corner=None)`.

This folder is for actionable driving guidance: braking references, line,
gear, kerb use, throttle timing, and common risks. Keep factual circuit
history and venue notes in `../tracks/`.

## File format

Use the same shape as `../tracks/`: frontmatter for track metadata, overview
prose for whole-lap advice, and one `## <corner name>` section per corner.
When a corner is requested, the API returns just that section as
`corner_detail`.
