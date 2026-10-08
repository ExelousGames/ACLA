# Live phrases

Open Live Session, expand the right sidebar, and select **Live phrases**. The
panel lists all overtaking guides and their conditions, including inactive
rules, above the latest 50 triggered sentences. It keeps listening while the
sidebar is folded or the Assistant tab is selected.

Each condition is a `PhraseCondition` instance with a boolean `conditionFit`.
The panel shows **Met**, **Not met**, or **Missing input** for every condition.
Each engine snapshot includes freshly evaluated condition instances, including
conditions in lower-priority guides; priority, hold and cooldown affect the
guide's status separately. Missing or expired inputs never count as a fit.

Open **Track Vision** and **Live Map** in Add Visualization. Apply Track Vision's
camera calibration and use a saved circuit map with a middle line and centerline
tags: `corner` plus `slow` or `fast`, and `straight` or `long straight`.
Legacy `slow corner` and `fast corner` tags also work. Untagged sections are
unknown, not implicitly straights; missing or conflicting speed tags withhold
guides that require a specific corner speed. Tag an enclosing region with
`consecutive corners` and tag the individual `corner` segments inside it to
describe a linked sequence. The enclosing region is not itself a corner.

The catalog includes:

- **Slipstream:** a visible opponent on a tagged straight, at least 80 km/h.
- **Outbraking on the inside:** a slow corner entry, player inside and opponent
  off the inside.
- **Around the outside:** a fast corner entry, player outside and opponent inside.
- **Switchback:** entry or middle of a slow corner, player outside and opponent inside;
  the advice depends on the opponent running wide and an opening appearing.
- **Better exit pass:** the exit of a tagged slow or fast corner, with both
  positions known; use any exit-speed advantage once a passing lane is clear.
- **Setting up the next corner:** player outside and opponent inside during entry
  or mid-corner, with a linked mapped turn in the opposite direction. Establish
  overlap and stay alongside so the outside becomes the inside for the next turn.
- **Linked corners in the same direction:** preserve balance and room for the
  next apex, then build the passing run from the final exit.
- **Exit into another corner:** prioritize positioning for the remaining corners
  of a tagged sequence before attempting an exit-speed pass.
- **Direction change within a corner:** transition smoothly through an S-bend
  and prioritize its final exit.
- **Tightening corner:** reserve grip and delay full throttle as curvature grows.
- **Opening corner:** progressively unwind steering and build traction as curvature eases.
- **Hairpin exit:** finish rotating the car before building the exit-speed run.
- **Pressure and a feint:** entry to a tagged slow or fast corner without either
  of the above inside/outside arrangements. Show an attack before braking and
  exploit the opening or compromised exit if the opponent responds.

`LivePhrases` registers the `live-phrases` handle and connects through
`LiveSessionHandle`. `subscribeTelemetry` supplies new frames and session/stream
resets without treating retained frames as fresh. `getTrackVisionDetection` /
`subscribeTrackVision` bridge `visualization:track-vision`;
`getLiveCircuitMap` / `subscribeLiveCircuitMap` bridge
`visualization:live-trajectory-map`, including late mounts, map loads and removal.

`PhraseEngine` is deterministic and local, with no chat, API or speech calls.
It reads the published Track Vision `birdsEyeScene`, the same calibrated flat-road projection
rendered in the BEV tab, never raw masks, image boxes or depth-based `analysis`/`geometry`.
It works with relative depth. Live Phrases determines the visible bend from consistent BEV
road-edge curvature, then converts boundary distances into inside/middle/outside positions.
The driver origin is compared with the nearest visible road slice, and the nearest individual
opponent ahead is compared with the boundaries at its projected distance. Interpolation stays
within visible boundary sections, without bridging gaps or extrapolating unseen edges.
Missing, straight or conflicting curvature leaves corner-relative positions unknown.
Traffic must lie between visible boundaries; car packs establish traffic ahead but do not
supply an individual position. Unplaced traffic cannot establish an empty road. These are
flat-road estimates from applied camera calibration, not metric depth measurements.
The map resolver uses normalized lap position, map tags and centerline geometry.
Entry includes the approach within 150 m (capped at 3% of the lap) and the first
35% of the tagged corner. The middle extends to 70%, followed by the exit.
Geometry is prepared once per map. Each corner's centerline is clipped to its
tagged bounds, ordered in lap direction and resampled at equal distances (at
least 5 m, at most 64 intervals). Accumulated heading changes distinguish bends,
hairpins (at least 135 degrees), and S-bends (substantial curvature in both
directions). Curvature spread through the corner distinguishes tightening and
opening shapes when one half bends over 1.6 times as much as the other. Straight,
degenerate or insufficient geometry leaves shape unknown. Speed still comes
from tags, not from shape or the driver's current speed.

Consecutive-corner regions collect fully contained corner segments, ordered from
the region's entry, and classify their direction sequence as alternating, same
direction or mixed. Unknown child directions leave the sequence shape unknown.
An explicit region defines which corners are linked regardless of the internal
gap; otherwise the nearby-corner fallback uses the 150 m / 3% maximum gap. The
final member never links back to the first or to a corner outside that region.
Gaps outside corner approach range show sequence context without inventing a
corner phase or a straight. Nested regions use the smallest containing region.
Turn-sign comparisons work with mirrored ACC/iRacing coordinates. Lap wraparound
is supported for geometry and regions. Shapes and phases are map estimates, not
measured racing lines or apexes. The panel displays the shape and sequence position.

The first matching rule in the catalog takes priority, so linked-corner guidance
supersedes an outside pass or switchback, and intermediate sequence exits
supersede final-exit attacks. Rules require a visible opponent and
at least 30 km/h (80 for slipstream). All corner guides require known individual
positions. The wording does not claim that vision confirms overlap, passing
clearance, relative speed or the opponent's intent.

Only new live telemetry frames emit. Timers and map/vision updates only evaluate
or expire inputs. Telemetry expires after 1.5 s and Track Vision after 2 s.
Conditions must hold for 0.8 s, with an 8 s per-guide cooldown and 0.5 s clear
period before repeating. Capture/telemetry gaps and section changes restart the
hold; session/stream resets clear history and invalidate previous vision.
Speed falls back to the magnitude of all three velocity channels, converted to
km/h. G-forces and brake input are not required.
