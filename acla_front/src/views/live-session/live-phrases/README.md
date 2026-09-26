# Live phrases

Open Live Session, expand the right sidebar, and select **Live phrases**. The
panel lists all 18 corner-position sentences and their conditions, including
inactive rules, above the latest 50 triggered sentences. It keeps listening while
the sidebar is folded or the Assistant tab is selected. Leaving the live screen
unmounts the panel.

`LivePhrases` registers the named `live-phrases` handle using the dashboard's
component directory. It connects exclusively to `LiveSessionHandle`:

- `subscribeTelemetry` forwards new frames and session/stream reset events.
  It does not replay a retained frame as fresh input.
- `getTrackVisionDetection` and `subscribeTrackVision` bridge the named
  `visualization:track-vision` component, including late mounts and removal.

`PhraseEngine` is a deterministic local processor with no chat, API, speech,
world-position, lap-position, or circuit-map dependency. `PHRASE_RULES` drives
both the catalog and evaluation. Only fresh live telemetry frames can emit;
timers only expire inputs. Session and stream resets clear the history and
invalidate previous vision. Track Vision computes visual geometry before publishing
each result, independently of whether Live Phrases is mounted.

Screen analysis, model requirements, and camera calibration belong to
[Track Vision](../track-vision/README.md#screen-analysis). Live Phrases reads the
published `analysis` fields (corner direction, driver position, opponent position,
and whether a car is ahead); it never inspects masks, boxes, or image geometry.

The catalog describes each combination of corner direction and both positions,
for example: "Left-hand corner: you are on the inside; the opponent ahead is on
the outside." It does not infer passing clearance or corner exit from screen
space. Unknown geometry produces no corner-position phrase.

Telemetry only gates live driving and speed of at least 30 km/h. Speed falls
back to the magnitude of all three velocity channels in m/s, converted to km/h.
G-forces and brake input are not read or required by Live Phrases. Missing speed
is never treated as zero. Published Track Vision analysis must be no more than
2 s old; telemetry must be live and no more than 1.5 s old.
Conditions must hold for 0.8 s, with an 8 s cooldown and a 0.5 s clear period
before repeating. Capture and telemetry gaps restart the continuous hold.
