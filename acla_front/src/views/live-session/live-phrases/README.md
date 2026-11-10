# Live phrases

Live Phrases copies the reference lap and centerline map published by Live Map.
All phrase rules use these local copies, including corner geometry, straight
membership, speed corrections, advance cues and the braking exit from movement
guidance. The telemetry forecast is limited to a fixed **three seconds**. Generic
closures have a `preparationTimeMs` (default **1000 ms**); phrase entry uses the
prepared clip duration, with no reaction allowance by default. Once triggered,
every phrase uses the same immediate playback path.
The position model supplies lateral checks
and never selects reference lookup progress.
**Live Map must remain open to supply the reference and map**. Lateral predictions
must match its current source reference. Copies refresh when Live Map publishes
a replacement; removing Live Map clears them and invalidates retained location
predictions.

**Driving aggression** selects **0–100% braking adjustment**, defaulting to **0%
(Reference)**, which preserves the original lap exactly. The calculated braking profile
preserves the reference corner minimum speed, its original position, and peak
deceleration. Every adjusted speed is calculated backward from that fixed minimum;
the endpoint never moves toward turn-in. Braking already completed before the
corner's direction-change range needs no adjustment. The reference trajectory
geometry remains unchanged. Direction-change checks are limited
to the start and end of each centerline corner, excluding entry-position adjustments.

Directional speech is enabled only at **100% (Direction guide)**: **Turn in**,
**Hold line**, **More left/right**, **Unwind**, and all straight-line movement
variants. Below 100%, Steering still supplies the reference-corner overlay and
pedal coaching, including **Ease brake**, without waiting for a silent turn cue.

The map below Driving aggression previews the adjusted reference even with detection
disabled. Orange marks decreasing-speed intervals and blue marks the same trajectory
direction-change ranges used by guidance, including overlaps and start/finish crossings.
The preview refreshes with the slider, Live Map's reference lap, and centerline corner bounds.

Rates use `(startSpeed² - endSpeed²) / (2 × distance)` in m/s units. The terminal
release's lowest positive deceleration is retained. Increasing the slider moves the
start of that low-rate section toward the reference trajectory's direction-change
start: `adjustedStart = originalStart + aggression / 100 × (turnStart - originalStart)`.
At **100%**, the release reaches minimum deceleration at direction-change start.
Speed continues falling gently from there to the original minimum speed at its
original position; the low-rate section is not a constant-speed coast.

Release intervals shorten in proportion to how far that boundary advances through
the original braking-to-low-rate span. Their deceleration values are retained, and
the terminal low-rate section stretches backward to fill the remaining distance.
The peak hold's length compensates for the resulting change in braking work.
Starting at the original minimum, integrate each interval backward with
`previousSpeed² = nextSpeed² + 2 × deceleration × adjustedDistance`.
This moves the braking lead-in earlier while retaining the initial rise and peak
rate. The earlier curve joins the preceding acceleration where their speeds meet.
Adjustment is limited when the available peak hold or approach cannot accommodate
the requested extension without changing a preceding corner's minimum.
Constant-deceleration profiles and releases already at their lowest rate before
turn-in remain unchanged. Separate braking inputs form one corner braking sequence
when a coast or acceleration gap lasts at most **0.5 s** at reference speed, braking
resumes within the same trajectory direction-change range, and the later input
reaches a lower minimum. The combined calculation includes every braking input and
any speed gained in the gap, retaining the final minimum speed and its position.
Light braking below the terminal rate no longer discards the adjustment. Longer
gaps and distinct direction-change ranges keep their braking sequences separate.

Reference brake pressure follows the earlier lead-in, adjusted peak hold and shortened release,
then stays at zero through the original minimum-speed section. Its rise
allows **Brake** to speak on the adjusted approach. Adjustments use physical XYZ
distance and wrap at start/finish. Every change rebuilds the complete private lap
from its untouched baseline, replacing speed and brake targets at the original
samples without adding points or changing normalized positions. Braking interpolates
squared speed between these samples, so cue boundaries and transitions in deceleration
have the original reference's sampling resolution. An interval crossing a profile
boundary averages the deceleration across that interval. Reference acceleration, throttle,
steering, racing-line geometry, centerline geometry and tags retain their original
profiles. Missing or invalid speed leaves the lap unchanged. Missing or invalid
brake data is preserved; the adjustment does not invent a missing brake channel.
Each adjustment rebuilds from the untouched local baseline, clears current cues
and movement progress, and keeps detection and prepared speech running.
The baseline is used only to rebuild the active reference. Every condition and
cue inside **Brake zone** and **Throttle zone** uses the active, aggression-adjusted reference for its
speed, controls, geometry and pedal responses. Missing active-reference data
never falls back to the original lap. Live Map's source reference is retained
only to identify replacement laps and validate position-model captures.

Reference-line guidance has five root closures: **Brake zone**, **Throttle zone**,
**Steering**, **Corner exit**, and **Turn to reference line**.
Generic closures check their own conditions within `preparationTimeMs`, defaulting
to one second. The look-ahead retains a shared buffer of **31 prediction frames**, from 0
through 3000 ms at 100 ms intervals. Every closure and phrase can inspect the full
buffer through `context.predictionFrames`; each frame contains its `offsetMs`,
`intendedTelemetry`, deceleration and brake-release state. `context.forPreparation(offsetMs)`
provides the guidance for a selected time, including exact times between stored frames.
Telemetry updates rebuild the buffer; timer evaluations reuse it. Reference changes
invalidate it, and stale or inactive telemetry exposes an empty buffer. Guidance and
child measurements are still evaluated only when their parent allows them.
Each preparation window checks its stored frames and exact endpoint. Phrase entry
uses the first speech action's prepared duration plus `reactionAllowanceMs`
(default 0), capped at the fixed three-second forecast. Explicit child action
conditions use their own action's duration. Pedal zones use the longest child lead
so the parent does not shorten a phrase's entry window. Child conditions, confirmation clocks,
steering comparisons and pedal measurements start only after their parent enters.
Pedal-zone entry never forecasts child phrases. **More brake** retains its measured
state comparison and continuous hold. The first-order prediction is
`intended = current + smoothedFirstDerivative × lookaheadSeconds`.
The current sample and latest eligible anchor estimate the rate using their actual elapsed time;
anchors are at least 100 ms apart to avoid amplifying packet jitter. Each channel's rate
uses exponential smoothing with a 150 ms time constant and elapsed-time weighting,
updated once per telemetry frame. The first valid rate seeds the filter; missing or invalid
channel history clears it. Current measurements remain immediate, while changes to the
predicted trend settle gradually. Normalized lap position and orientation are unwrapped
before differentiation and rate smoothing. Brake and throttle pressure,
steering, world/local velocity, player XYZ position and speed are projected into
`context.intendedTelemetry`.
Player coordinates are matched by car ID, not array slot. Reference targets, pedal-zone
boundaries, Steering ranges, corner selection and straight membership use that point.
Transitions use the same point so an outgoing guide can yield to the next guide early,
except Brake zone exit, which uses current measured position and speed, and Steering's
range exit, which uses the current measured position. Steering's positive
reference-acceleration exit checks the next 200 ms of the shared lookahead buffer.
The input layer provides this preview before the existing closure tree runs; there
is no delay from the preview itself, clock offset or eager child traversal. **More brake**
uses current telemetry and its own 500 ms continuous entry hold, described below.
Other closure entry and exit checks share the intended telemetry, including both pedal pressures.
Reference pedal and speed comparisons and steering corrections use the intended controls
and motion. Deceleration uses the speed prediction's derivative at the selected horizon:
`deceleration = -smoothedSpeedFirstDerivative / 3.6` in m/s².
It becomes zero when predicted speed reaches the zero floor.
Brake-release detection uses the same smoothed brake-pressure derivative at that horizon;
a predicted drop greater than one percentage point per second counts as releasing.
Lateral errors remain observed inputs. Capture validation,
cooldowns, measurement intervals and speech deadlines use real timestamps. Until a channel
has enough history it retains its measured value; normalized position falls back to
`liveSpeed / 3.6 × 0.5` metres along the reference's XYZ distance. Speed stays nonnegative,
steering stays within [-1, 1], and pedal pressures stay within [0, 1]. Missing or invalid pedal samples prevent prediction
for that channel; the current measurement is retained.
Zero live speed gives zero advance; invalid speed, position or reference geometry withholds
guidance. Session/stream/car/track changes, pauses, stale or non-increasing timestamps,
frozen progress and implausible position jumps clear derivative history and smoothing. Timer evaluations
reuse the current prediction. Overtaking guides retain live progress.
Brake zone groups **Brake**, **More brake**, **Release the brake**, and **Hold brake pressure**;
Throttle zone groups **Throttle**, **More throttle**, and
**Less throttle**. Each zone enters from the active Live Phrases reference interpolated
at the shared decision position, even when the driver's pedals already match. A brake
zone starts where reference brake begins rising from zero and ends where reference
brake decreases to zero or reference throttle begins rising from zero. A throttle interval starts at that throttle rise and
ends when reference throttle first reaches 100%, decreases to zero, or reference
braking starts. Throttle zone entry additionally requires reference steering to stay
constant from the decision position through that endpoint. Any increase or reduction
anywhere in the remaining interval withholds entry, including a reversal with zero
net change. Entry becomes available after the last steering change; constant nonzero
steering qualifies. This uses the active reference's unsmoothed steering across
start/finish, independently of live steering and mapped corner bounds. Missing or
invalid reference steering withholds throttle entry. There is no minimum pedal
pressure for zone entry. A partial pedal release stays in its zone until zero is reached.
After release to zero, that pedal's zone stays ended until a new reference rise from zero.
The throttle zone also ends at full throttle and waits for a new rise from zero.
The latest opposite-pedal onset switches zones even when both pedals overlap.
Simultaneous onsets give brake priority.

Only zones longer than 500 ms qualify. Duration covers the complete interval from
onset to the zone's end, using physical reference distance and interpolated reference speed,
including start/finish crossings. References without a usable speed channel use
travel time at intended speed. This is a duration filter, not an entry hold: qualifying
zones enter immediately at the start of the interpolated rise when their steering
requirement is met. A short opposite zone still ends the preceding zone, but cannot itself enter. A pedal with no ending boundary
occupies the whole cyclic lap; all-zero and constant-full-throttle references have no pedal zones.
Driving aggression recalculates this brake profile together with the braking speed
profile. Calculated braking demand alone does not extend its bounds. Live pedal
inputs do not change zone membership. With reference speed available, live speed
does not change the pedal interval itself. Both pedal zones wait for any active
sibling closure to exit. Brake zone also defers while its direction-start speed exit
condition is met. The closure tree selects one active child at each level.
Both reference pedal channels must be available to determine zone membership.
Zone boundaries do not depend on mapped corner bounds: braking can start before
corner entry and throttle zones can continue onto straights.

Brake and throttle correction phrases return to their zone after speaking. The zone stays selected while
quiet or giving corrections, and only gives corrections for its own pedal.
Live pedal errors never switch zones. The throttle-zone exit checks the
reference brake onset, reference full throttle, or reference throttle release to zero even during a
selected speech action or speech cooldown; missing normalized telemetry position withholds
speech and waits for confirmed boundary evidence. Throttle-zone exit uses its predicted decision position.
Brake-zone entry uses its preparation window. Reference brake release and throttle onset do not exit an active Brake zone.
Brake zone returns to root after any eligible **Release the brake** cue finishes, when current speed is strictly below the
active Live Phrases reference speed at the start of the upcoming or active trajectory
direction-change range within the mapped centerline corner bounds. Before turn-in,
the target comes from the first matching range ahead in the same reference braking
interval. The speed target is interpolated at that range's clipped start, rather
than at the current position. This can exit before reaching the range; reaching
the range alone, or matching its start speed exactly, does not exit. Missing or
invalid reference speed withholds this exit. Brake zone cannot re-enter while the
speed condition remains met, and remains active until its own exit action runs,
including while no brake correction is speaking. Steering's own
entry condition still uses direction-range membership independently of speed.
Steering can enter from root when its direction range is active; an active Brake zone
must first meet an exit condition. Throttle zone returns to root
at reference full throttle, release to zero, or an earlier brake onset. Throttle pickup beyond the corner can enter the throttle zone.
Movement cues wait until the active pedal zone has exited and a mapped straight
allows movement guidance. The former **Prepare to brake** cue is removed;
**Brake** speaks once per brake-zone entry when braking is needed now. Pedal changes,
expired inputs, and map refreshes cannot repeat it within the same zone; a confirmed
zone exit rearms it for the next entry. Brake-zone and brake-phrase entry have no
minimum live speed. **Prepare to accelerate** is also removed;
**Throttle** speaks when throttle is needed now.
Open Live Map with a reference lap for the current game/car/track. Brake and
steering guidance and zone membership project from live normalized position without requiring
Track Vision's position model. Run the model for lateral checks used by movement
and throttle cues. These cues do not require opponents, BEV reconstruction, or
camera calibration.
Movement guidance also requires Live Map straight tags. Steering and **Release the brake** require a known
current or upcoming mapped corner direction; other reference cues do not require circuit tags.

`phrase-reference-guidance.ts` interpolates the reference's raw XYZ distance and
`expert_optimal_speed`, `expert_optimal_steering`, `expert_optimal_brake`, and `expert_optimal_throttle`
channels at the decision position projected from `Graphics_normalized_car_position`. Steering becomes eligible from the start
of a continuous trajectory direction-change range within a mapped centerline corner,
and **Turn in** speaks immediately on entry when Driving aggression is 100%.
It uses intended speed to locate reference brake rises from zero up to
1.8 s ahead, including start/finish crossings, to suppress **Throttle** before braking.
**Throttle** and **More throttle** compare the applied pedal with demand for the adjusted speed profile
and never request more throttle at full pedal. Missing or invalid control channels disable their
cues rather than being treated as zero.

**Brake** and **Hold brake pressure** first compare intended speed with the active, aggression-adjusted
reference speed interpolated at the shared decision position. At least
5 km/h below that speed suppresses **Brake** and **Hold brake pressure**. At least
5 km/h above it requests **Brake** when the pedal is not applied and suppresses
**Hold brake pressure**. This applies at every braking-adjustment setting, including 0%.

Within that speed tolerance, the required deceleration is the local
speed-profile slope multiplied by intended speed, plus the signed speed error in m/s
divided by 1.5 s, clamped to zero. Even errors below 5 km/h contribute so that
matching deceleration while still too fast does not count as matching the target.
The comparison band is 10% of required deceleration, with a 0.1 m/s² minimum.

- **Brake**: estimated demand is at least 10%, the driver is below 10% pedal,
  and speed is not at least 5 km/h below the local reference.
- **More brake**: current measured deceleration stays strictly below the active
  reference deceleration at the current lap position for 500 ms continuously.
  Current position must be in a reference brake zone, current brake pressure must
  be at least 10% and below full pressure, and the pedal must not be releasing.
  It uses neither lookahead nor the speed-error correction or 10% tolerance band.
  Reference deceleration is `(startSpeed² - endSpeed²) / (2 × distance)` for the
  current reference interval, with speed in m/s, and must be positive.
  Measured deceleration is the speed decrease between consecutive telemetry frames
  divided by their actual elapsed time; pedal release uses those same frames.
  Confirmation begins with the first measured deficit. Matching or exceeding the
  reference, leaving the zone, releasing the pedal, invalid required data, or a
  telemetry discontinuity resets the hold. Timer evaluations cannot supply a new
  measurement or speak the phrase. Existing cooldown and rearming rules still apply.
- **Hold brake pressure**: both predicted speed and predicted deceleration are within their bands.
- **Release the brake**: predicted speed matches or falls below the active reference
  turn-in speed at the clipped start of the upcoming or active direction-change range.
  The upcoming range is selected within the mapped corner even when reference braking
  ends before turn-in; already-passed ranges within that corner are skipped.
  It requires at least 10% predicted brake pressure and an active Brake-zone closure;
  the predicted position may be past the reference braking interval.
  It takes priority over the other brake corrections and finishes before the speed-based
  zone exit. Matching the speed exactly triggers the cue but does not itself exit the zone.
  Required deceleration does not control this cue; missing turn-in speed withholds it.

**Hold brake pressure** requires at least 10% live brake. **More brake** confirms
independently of the predicted guidance; the closure tree selects one phrase to speak per frame.

For the other brake cues, rate comparisons use predicted deceleration at the phrase's decision horizon,
derived from the same two speed samples as predicted speed. If that history is
unavailable, deceleration falls back to the measured rate, matching the other
lookahead channels' measured-value fallback. Brake-release detection likewise uses
`releaseRate = -brakeFirstDerivative` and the
same two pressure samples as the pedal preview. Rates above 0.01 pedal units
per second count as releasing, unless projected pressure is already zero or lies
above full pressure before clamping. A valid prediction of no release overrides
measured release evidence. Without valid pressure history, measured detection is
the fallback. Predictions refresh on telemetry frames; timer evaluations reuse
the latest prediction.

Fallback deceleration and fallback pedal release are measured together at 500 ms intervals
while **Brake zone** is active. Fallback deceleration is the speed decrease in m/s
divided by actual elapsed seconds. A pedal reduction greater than one percentage point
per second counts as releasing. The first active frame starts the interval; the
first frame at least 500 ms later supplies the measurement and starts the next
interval. Measurements persist between checks. Missing or invalid speed,
non-increasing timestamps, gaps over 1.5 s, pauses, resets, and game, track, car,
session or stream changes clear both measurements. Missing or invalid brake
samples also clear release evidence, including its previous pedal baseline.
**Hold brake pressure** waits for valid prediction history or a fallback interval;
**Brake** and **Release the brake** do not require it. Existing cooldown and rearming rules
prevent repeated speech while a cue stays matched.

The initial pedal demand uses the active reference's braking response. Each
reference decelerating interval gives `(startSpeed² - endSpeed²) / (2 × distance)`
in m/s units; dividing by mean brake pressure estimates response per unit pedal.
The nearest valid interval supplies that response, including adjusted
braking. Required deceleration divided by this response gives a pedal target
clamped to 0–100%, with initial demand suppressed at least 5 km/h below the local target.
Without a usable response, initial demand falls back to reference pressure with the
same underspeed suppression. Without reference speed, only
reference-pressure-based **Brake** remains available.
Pace cues compare predicted motion directly with the speed profile, independently
of reference brake pressure. They work across start/finish and with the adjusted
reference, and are spoken only while the Brake-zone closure is active. Steering uses its
direction-change entry condition independently of the target speed.

**Steering** is a root closure. Direction-change ranges are clipped to the mapped
centerline corner's start and end. The decision position must be inside that clipped
range; turning to adjust the car's position before entry does not qualify.
Entry becomes eligible as soon as the prepared Turn in clip's decision position reaches the start of that
direction-change range. Its head is the later of the first geometric bend and the
centerline corner start. A shallow lead-in inside the corner can qualify before the
turn reaches the confirmation threshold below. A decision point inside the range still qualifies if it skips
the exact start boundary.
Entry does not compare live speed with the reference speed and has no minimum live
speed. Unknown corner direction or bounds withhold entry.

Trajectory headings use the horizontal tangent from 5 raw coordinate units behind to 5 ahead (XY
for ACC, XZ for iRacing), interpolated between samples using the shortest angular
difference. This avoids false plateaus when caution inserts reference samples.
An absolute heading change of **1 degree per 10 raw coordinate units** confirms a turn,
normalized by raw coordinate distance to account for sample spacing. These distances
are not converted to metres. The head follows the first incoming/outgoing segment
direction change in that continuous turn, rather than the centered tangent's earlier
anticipation or the point where curvature crosses the threshold. The tail still ends
when curvature falls below that threshold, independently of peak curvature.
Flat headings and reversals separate turns. A near-straight
section between two stronger bends splits them into separate ranges, even when they turn
in the same direction. Uniformly near-straight bends below the minimum do not qualify.
Direction changes and interpolation wrap across start/finish, including a duplicated
endpoint; ranges join across that boundary only if the turn qualifies continuously.

At 100% Driving aggression, **Turn in** speaks immediately once per Steering entry, without a reference steering
threshold. Changes in reference steering do not cancel a triggered **Turn in** cue.
Steering captures the corner-limited trajectory direction-change range on entry and
exits to root when the current measured position reaches its end: the earlier of the curvature
boundary described above and the centerline corner end. Excluded near-straight tails
cannot enter Steering. Lookahead can enter Steering before the car reaches the range;
it remains active until measured progress passes the captured end, including across
start/finish, or the reference starts gaining speed within the next **200 ms**.
The acceleration exit checks now and the forecast frames through 200 ms, independently
of closure preparation and speech duration. Positive acceleration comes from the active
reference speed profile, not live acceleration or an increase in acceleration while
still slowing. Either exit can leave **Steering** while **Turn in** finishes playing. Positive acceleration in that same
window also withholds entry, preventing immediate re-entry after an early exit.
Either exit rearms Steering for the next qualifying range,
without a timed clear period or entry cooldown. Steering evaluates its own direction-change
condition independently of Brake-zone exit. Direction corrections inside Steering
speak once per continuous deviation and rearm when the car realigns.

Reference steering changes, steering smoothing, live steering, orientation, velocity
and position-model error do not determine the trajectory range. Missing reference steering
still permits entry, **Turn in** at 100%, and exit. Missing or invalid reference
speed does not prevent entry or range exit; it only withholds the acceleration exit.
Future reference speed does not determine direction-change range membership.
Missing trajectory direction withholds entry and range exit. Missing or stale telemetry withholds speech;
expired position-model captures do not prevent **Turn in**.

At 100% Driving aggression, the spoken corner sequence uses short instructions: **Turn in** to start the turn,
**Ease brake** when braking should taper, **Hold line** through a stable loaded
reference arc, and **Unwind** as the reference releases steering. **Hold line**
also requires the driver's predicted trajectory to be aligned. These are separate
cues, so a short corner can skip an unsupported intermediate instruction.

**Corner exit** speaks **Unwind** only with evidence of steering release after the
final substantial peak in the same mapped turn. Acceleration alone does not prove
the wheel should unwind. Missing or contradictory reference steering suppresses
these phase cues. The map's fixed entry/middle/exit percentages do not locate an
apex and are not used as apex calls. At 100%, an eligible exit cue releases Steering
and prevents a new **Turn in** during that unwind. Below 100%, this exit is disabled
so coasting through an unwind keeps the reference overlay active. **Unwind** speaks once per reference
range, including across unloading plateaus; the next distinct corner does not
inherit a two-second cooldown. Pedal cues also use shorter speech: **Off brake**,
**Hold brake**, **Squeeze gas**, **More gas**, and **Ease gas**. Pedal closure names
stay unchanged.

This wording follows the concise directive style in the
[Porsche Club of America instructor training manual](https://rmr.pca.org/files/Instructors/PCAInstTrng.pdf).
[Driver61's racing-line guide](https://driver61.com/uni/racing-line/) organizes the
corner around braking, turn-in, apex and exit; its
[corner-exit guide](https://driver61.com/uni/corner-exit/) connects opening the steering
with building throttle. The software applies those principles when supported by
the active reference, without prescribing a fixed path through every corner.

**Ease brake** speaks once per Steering entry at every aggression setting when the
driver still applies at least 10% brake and the active phrase reference's deceleration
is decreasing by more than 0.01 m/s³. At 100%, it follows **Turn in**; below 100%,
it can speak without a directional cue. The rate comes from the change between adjacent
reference deceleration intervals divided by their reference travel time; it does not
use live car speed or brake-pedal release speed. Missing speed data, constant or
increasing deceleration, and a released brake do not trigger this instruction.
The phrase instance selects one of four prepared timing inputs: 1 for easing up to
0.5 m/s³, 2/3 up to 1 m/s³, 1/3 up to 2 m/s³, and 0 above 2 m/s³. The curve in
`curves/slowly-release-brake.json` stretches slower easing more. These conditions
govern entry; they are not checked again before playing a triggered phrase.

After **Turn in**, nested **More right** and **More left** corrections enter only when
**Driving aggression = 100** (**Direction guide**), as do **Turn in** and **Hold line**.
These corrections use the remaining reference curve from
the capture's projected reference position through the trimmed end of its continuous Steering
direction-change range, including start/finish wraparound. Shift its starting point
left or right using the position model's lateral reference error, then taper that
shift to zero at Steering's end. The rest of the reference's bends are retained.
An absolute lateral error below **3 coordinate units** produces no starting-point
shift. These are position-model units, not metres. This tolerance does not silence
direction corrections: local velocity can still indicate deviation from the unshifted curve.
At exactly 3 units or above, apply the full lateral shift.

The model error points from the driver to the reference in car-local right/forward/up
coordinates. Rotate only its lateral component into world coordinates using the
capture's heading, pitch and roll. Forward and vertical model errors do not shift
the starting point. For remaining reference curve `R(s)` of length `L`, calculate:

`adjustedCurve(s) = R(s) - worldLateralError * (1 - s / L)`

Its current tangent is the reference tangent plus `worldLateralError / L`; compare
against that tangent, rather than a chord across the corner. Use the intended local
velocity, rotated by the intended orientation and projected onto the simulator's ground
plane. ACC supplies forward velocity on Y and body X opposite the model's right
axis; iRacing supplies forward on Z and right on X. Local vertical velocity is
included in the rotation for pitched or banked cars. World velocity fields do not
drive this comparison. The signed angle between normalized velocity and the adjusted
tangent determines the required correction: **More right** or **More left**,
accounting for ACC's and iRacing's ground-plane conventions in either corner direction.
The separate 20° angular tolerance keeps small direction errors silent.
First-order normalized progress determines the preparation point within the three-second forecast. Once that point is selected,
velocity magnitude and reference speed do not change the endpoint or angular comparison.
Steering entry and corrections have no minimum live speed.
After **Turn in**, the same direction deviation must persist continuously for **500 ms**
before its correction can speak. Realignment, a change of correction direction,
missing or expired inputs, a capture gap over 500 ms, telemetry discontinuity, or leaving
Steering resets the hold. This hold uses elapsed live time in addition to the existing
preparation window. Once confirmed, a correction speaks only once during that
continuous deviation. A valid **aligned** comparison rearms the cues; a subsequent
deviation must pass a new 500 ms hold. Missing data, capture gaps, pauses, and switching
correction sides without observed realignment do not rearm a spoken cue. Leaving
Steering or resetting its reference/session starts a new correction opportunity.

Direction comparison uses the shared intended normalized position. The lateral error
still comes from a validated capture (at most 500 ms old) and is rotated with that
capture's orientation; the intended orientation rotates only the intended velocity.
Missing or invalid error, orientation or local velocity,
unknown corner direction, an unavailable remaining Steering range, zero vectors,
and a desired tangent opposing the velocity direction
withhold new corrections. Once triggered, correction audio uses the common playback
path without checking the direction, input freshness or Steering scope again.

Throttle demand uses the active reference's throttle and acceleration.
For accelerating intervals, `(endSpeed² - startSpeed²) / (2 × distance)` in m/s
units divided by mean throttle estimates acceleration per unit of pedal. While
the active profile is accelerating, the local speed slope and interpolated
throttle align this response to the decision position, including throttle ramps.
Elsewhere the nearest valid accelerating interval supplies the response, wrapping
at start/finish. The adjusted speed slope multiplied by intended speed gives the new
acceleration demand; speed errors of at least 5 km/h add or subtract a correction
over 1.5 s. Reference throttle is adjusted by the change from that same reference's
acceleration divided by the response, then clamped to 0–100%. This preserves
maintenance throttle on flat-speed sections and accepts matching speed and
throttle at every caution setting.
The response is an estimate from the reference, not measured live acceleration.

- **Throttle**: demand is at least 15% and the driver is below 15% pedal.
- **More throttle**: the applied pedal is at least 15% and at least 15 percentage points below demand.
  Required braking and reference braking within the next 1.8 s do not suppress this cue.
- **Less throttle**: speed is at least 5 km/h above the interpolated local reference,
  at least 15% throttle is applied, and the driver is at least 5 source units outside
  the reference line and moving farther outward. It uses the local speed profile
  adjusted by Driving aggression, independently of calculated throttle demand.

Signed horizontal curvature of the local reference, sampled 10 m before and after
the normalized lap position, identifies the inside and outside of the bend using each
game's coordinate convention. Curvature below 0.0015/m is treated as straight.
Two distinct, validated position captures at most 500 ms apart must show the driver
at least 5 source units on the outside and moving farther outward at more than
0.05 source units/s. A steady outside offset, inward movement, and missing or stale
position data stay quiet.
Reference replacement, caution changes, calibration changes and session resets
clear the position history.

Only one throttle cue can match at a time. **Throttle** and **More throttle** require
the driver to be within 2 m of the line with brake below 10%. **Throttle** also requires
no braking required or imminent. **More throttle** uses the calculated throttle demand
even when braking is required or imminent, but remains withheld at speeds at least
5 km/h above the local reference. **Less throttle** can still be suggested away from
the line during outward overspeed, with brake below 10%.
Throttle corrections are spoken only inside the reference throttle zone.
Overspeed can request less throttle there, but cannot switch to a brake correction.
Demand below 15% withholds **Throttle** and **More throttle**.
Without usable speed or acceleration response, demand falls back to reference
throttle, or zero when usable speed shows the driver at least 5 km/h above target.
Missing reference throttle/brake or live pedal data withholds throttle instructions.

**Turn to reference line** replaces **Increase steering**. Its **Move to the left**
and **Move to the right** children each contain **Small movement**, **Medium
movement**, and **A little bit more**.
These are executable closures with their own speech actions and inspectable
conditions. Each speech action randomly picks from three short wordings, preserving
the direction and movement size (for example, “Move to the left”, “Shift to the
left”, or “Bring it to the left”). The position model's car-local X offset selects the side: negative
is left, positive is right. No reference steering channel is required.

Movement requires **Driving aggression = 100** and the shared decision position to be inside the current
Live Map's `straight` or `long straight` centerline bounds, with no overlapping
corner. Bounds include start/finish wraparound and remain eligible during the
map's corner-approach phase. An untagged section is not a straight. The aligned
reference brake channel must be available, with no reference braking or active pedal zone,
and speed must be less than 5 km/h above target when reference speed is available.
Movement children return to their parent after speaking; **Turn to reference
line** stays selected between movement cues and exits to root when reference
braking starts or speed exceeds the target by at least 5 km/h, even if the driver's
brake input already matches. Brake-pressure corrections also end movement guidance.
Leaving the straight or losing required inputs also exits and clears movement
progress, so a later straight cannot inherit old movement feedback. Its exit waits
for any active movement child to return. Other reference cues wait for this scope to exit.

Each movement phrase's already-spoken flag is cleared when **Turn to reference
line** exits. Returning from a movement child keeps that flag for the current
visit. On re-entry, the same phrase can speak again once its existing 2 s
cooldown has elapsed.

Movement starts at 3 source units of lateral error. Small corrections are from
3 to below 7 units (“Move to the left/right”), and medium corrections are from
7 through 13 units (“Further to the left/right”). There is no large movement tier.
After a cue, fresh captures spanning at least 0.75 s estimate the
closing rate. “A little bit more” requires measurable progress above 0.05 units/s
but more than 3 s estimated to reach the line's 0.35-unit tolerance. Adequate
progress stays quiet. Reaching that tolerance or crossing the line silently
completes the movement. Movement cues are at least 2 s apart. Capture gaps over 500 ms, missing
or mismatched inputs, reference/calibration changes and session/stream resets
clear movement history. Repeated captures never add progress evidence.

Reference cues require running telemetry no older than 1.5 s and a matching reference
lap. Lateral checks additionally require position captures no older than 500 ms,
paired with the same session/stream/game/car/track and selected reference lap.
Position jumps and remote reference offsets suppress lateral checks only; brake,
steering and zone lookup continue from normalized position. **Throttle** and **More throttle**
require being within 2 source units laterally of the line with brake below 10%.
Only **Throttle** additionally requires no current braking demand or reference braking
in the next 1.8 s. Distances use the reference's
source coordinates, assumed to be metres, and constant current speed for ETA;
the thresholds and lead time still need validation during simulator driving.

Only one child closure can be active at each level. An active closure keeps control
until its own exit action runs, including while waiting for a speech condition.
Brake zone, Throttle zone, Steering, Corner exit, movement and overtaking closures all wait for
an active sibling to exit. A parent's exit action also waits for its active child
to return. Pedal cues run only inside their matching zone. Reference prompts
have no shared spacing delay. Steering entry has no cooldown; Corner exit speaks once
per range without a timed cooldown; other reference closures retain their 2 s cooldown.
**Brake** rearms only after leaving its zone; **Steering** rearms when its captured
trajectory direction-change range ends or reference acceleration is positive within 200 ms;
other cues require a 0.5 s clear period before rearming. Short
clips prepare first. The audible clip duration plus any configured reaction allowance
determines the phrase entry forecast. Playback does not recheck duration, deadlines,
entry conditions or driving scope. Event deadlines do not release a speaking action.
Live chat retains higher audio priority.

Live phrases runs a closure tree instead of a flat sentence-rule dispatcher.
`closure.ts` defines the three building blocks:

- `Closure<Context>(name, description, condition, children)` has an entry predicate and an
  ordered list of child closures or actions.
- `Action<Context>(name, description, run, condition?)` runs any lambda, including one returning
  a promise. Omitting its condition makes it immediately eligible.
- `State<Context>(root)` starts at root and remembers its current closure or
  action and the full path. `step(context)` delegates to the root's `evaluate`
  method; every active closure evaluates its immediate children through the
  `Evaluable<Context>` interface. A step runs at most one action. Actions run once per visit. A pending
  async action holds the state until it settles; failures propagate to the caller.

Child conditions, confirmation holds and action measurements start only after their
parent becomes active. There is no global pass over phrase or action conditions.
Unchecked closures and actions are shown as **Not evaluated**; a checked entry
predicate that fails is **Not matched**. Entered closures are always **Active**,
including while waiting for an action or after their entry conditions change.
Phrase readiness can further distinguish **Missing input**, **Confirming**,
**Cooldown**, or **Waiting** (another phrase was selected or this phrase has not
rearmed). Actions retain their **Running**, **Completed**, and **Failed** execution
statuses. Leaving a parent clears its subtree's measurements and hold timers.
Fallback brake deceleration and pedal release, throttle outward-motion
history and reference-line movement progress follow the same scope boundaries.
Cooldown timestamps remain independent of these measurements.

Entry conditions determine whether a closure can be entered. Once entered, the
state stays inside even if the entry condition becomes false or no children match.
Phrase closures refresh their entry results for display and rearming inside their
own evaluation method; this does not make their entry conditions an exit condition.
`Action.exitToParent(condition?)` returns one level and clears the exited subtree's
action memory, allowing another visit while preserving the parent's completed actions.
`Action.exitToRoot(condition?)` returns directly to root and clears all completed
actions. With no condition, either exit runs as soon as the state reaches it.
Each state owns its traversal memory.

```ts
const root = new Closure<{ ready: boolean }>('root', 'Select an eligible guide.', () => true, [
    new Closure('example', 'Run the example when ready.', ({ ready }) => ready, [
        new Action('do something', 'Log entry into the example.', () => console.log('Entered example')),
        Action.exitToRoot(),
    ]),
]);
const state = new State(root);
state.step({ ready: true });  // Enters example and runs its lambda.
state.step({ ready: false }); // Runs exit to root, independent of entry conditions.
```

Both classes require a display `name` and a `description` explaining their purpose.
`createPhraseRoot` turns `PHRASE_DEFINITIONS` (names, descriptions, sentence text and entry
conditions) into the root and its nested guide closures.
Each sentence closure contains one or more **say phrase** lambdas. Steering stays
entered after speaking and exits to root when its captured trajectory direction-change range ends
or reference acceleration is positive within the next 200 ms. Other sentence
closures have an exit action eligible **when playback of the trimmed WAV finishes**. Movement and pedal sentences exit
to their parent; overtaking sentences exit to root. Optional `additionalActions` add separate
speech actions; each action's `actionConditions` gate speech independently of entry conditions.
Optional `exitConditions` use the same AND/OR condition structure. On a phrase definition
they leave its scope, including while its own speech is waiting, once any active child
closure has returned; on an additional speech action
they release that action. They do not require its entry conditions to become false.
Any speech action can keep its existing `sentence` and add `sentenceAlternatives`,
for example `{ sentence: 'Move to the left.', sentenceAlternatives: ['Shift to the left.'] }`.
Each execution chooses one wording with equal probability, including the original;
consecutive executions may repeat. Selection happens only when the action runs;
checking conditions or inspecting the tree does not choose a wording. Action Details shows the alternatives.
All wordings are prepared for speech, and the chosen sentence is published once
for audio, history and overlays. Omitting alternatives keeps the single-sentence behavior.
Each speech action runs once per scope visit. Its lambda publishes the event used by speech, history
and overlays; there is no parallel sentence dispatcher. Hold, cooldown, clear
period and fresh-frame checks gate entry into these sentence closures. The first
matching guide reserves priority while waiting for those checks. The exit runs on
a playback-boundary update or the next eligible engine step; state then considers the enclosing scope again. Session/stream
resets and detection restarts create a fresh state at root.

The UI uses a folder explorer: `State.snapshot(context)` projects the whole
runtime tree to plain `NodeSnapshot` data, and `ClosureTree` displays one directory
at a time with a breadcrumb path and a Back button that moves up one level.
Contents, Conditions and Details tabs show only the inspected node's information;
opening an action selects Details. Navigation follows snapshot IDs independently
of the live engine location, so updates preserve the inspected directory and tab.
`PhraseSnapshot.root` is that recursive snapshot, including root-level
actions and every nested closure in declaration order. The legacy `closures`
summary remains for phrase events, timing, overlays and speech; it does not drive
the tree UI. The plain `state` still contains `current`, `description`, `kind` and
a `path` of display names.

Each node exposes `id`, `kind`, `name`, `description`, `current`, `onPath`, `status`,
`conditions`, `fields` and `children`. Actions also expose `execution` with an
`idle`, `running`, `completed` or `failed` status and an error message on failure.
Execution belongs to each `State` and resets on exit to root; taking a snapshot
never runs a condition, description callback, inspector or action. It copies saved
evaluation results and static descriptions. Async results appear on the next engine snapshot (the panel
polls every 250 ms). Action failures remain available to inspect without an
unhandled rejection in the telemetry subscription. Direct `State.step` callers
still receive thrown errors/rejections unless they pass the optional
`onActionError` callback to `step`. Condition exceptions always propagate.

Add a closure or action anywhere in the tree using the same constructors. Wrap
new predicates with `describeCondition(predicate, description)` to expose
their last measured result. For grouped conditions or predicates with
side effects, pass a **read-only** description callback returning condition data
instead; the phrase adapter reuses its locally evaluated conditions this way.
An optional third argument supplies static condition descriptions to display before
evaluation, with `conditionFit: null` and `inputMissing: false`.
The renderer handles condition groups structurally, without importing phrase
condition classes or checking action names. An opaque, undecorated predicate is
never called just to display it: the UI labels its result as last checked, or
`Not evaluated` until traversal checks it. Human-readable condition meaning cannot
be inferred from an arbitrary JavaScript callback.

Optional node metadata (the last constructor argument) supplies an explicit `id`
and a read-only `inspect(context, state)` callback with extra labeled `fields` or
a domain `status`, recorded when the node is evaluated. Static `fields` can be
supplied directly in metadata so they remain visible before activation.
`priority` orders sibling evaluation only when no child closure is active. An
entered child owns traversal until it exits; sibling conditions and enclosing exit
actions are not evaluated during that visit. All declared fields render automatically, including false and
zero. Explicit IDs must be unique; otherwise IDs derive from tree position below
the nearest explicit ID, independent of names. Keep the tree structure fixed for
the lifetime of a `State`. New snapshots copy conditions, fields and execution
records so later runtime updates do not modify earlier snapshots.

```ts
interface RecordingContext { ready: boolean; channel: string; save: () => void }
const recorder = new Closure<RecordingContext>(
    'diagnostics', 'Optional nested diagnostics.', describeCondition(() => true, 'Always eligible'), [
        new Action('record sample', 'Save the current sample.', ({ save }) => save(),
            describeCondition(({ ready }) => ready, 'Recorder is ready'), {
                id: 'record-sample',
                inspect: ({ channel }) => ({ fields: { Channel: channel } }),
            }),
        Action.exitToRoot(),
    ],
);
// Include recorder in any closure's children; no renderer changes are needed.
```

`PhraseEngine(phrases, createRoot?)` also accepts a root factory for composing
custom trees around the generated catalog. Phrase-specific timing status binds
to a closure's metadata ID, independent of its depth or root-child position.
The speech action's description remains the sentence used by TTS and overlays.

The first child, **Chicane overtake**, enters immediately when the upcoming mapped
slow corner has at least one following corner in its labeled consecutive-corners
sequence and its next linked corner turns the opposite way. The entry condition
is displayed as **Upcoming mapped consecutive-corners sequence >= 1**. Entry has
no opponent-distance or arrival-time limit and does not require a motion rate;
it still needs a fresh opponent lap position. Arrival uses the nearest car
ahead in `Graphics_normalized_positions`, excluding `Graphics_player_car_id`,
and its forward normalized-position change per second. At least two distinct
position samples are required; repeated packets retain the rate for up to 1.5 s.
Opponent changes, stale telemetry, pauses and resets discard the rate. The map
lookahead for this estimate is not limited by the normal 150 m / 3% approach range.

Its first child, **Inside the Long Straight**, uses the same driver-position entry,
exit-to-parent behavior, and exit-speed advice as the child of **One Tight Slow Corner**
described below. It offers only the “Leave room. Pass on exit.” phrase, with
the same three conditions, and speaks once per straight visit. Leaving the straight
returns to **Chicane overtake** so its timed speech actions can run.

At the parent, speech waits until the opponent's estimated corner entry is
less than the prepared action's duration plus any configured reaction allowance away.
Without playback timing, the ETA threshold is 500 ms. Track Vision then selects between two speech actions:

- Opponent on the left before a mapped left turn, or on the right before a mapped right turn:
  “Go wide. Aim for second apex.”
- Opponent on the left before a mapped right turn, or on the right before a mapped left turn:
  “Brake early. Hold inside.”

The upcoming corner's direction
comes from the same map segment used for the entry ETA. Neither car needs to be
detected in a corner, and the player's position is not required.
After either speech action finishes playback, the next action exits to root.
While waiting, its status remains **Active**; losing the entry conditions
does not exit an entered closure. As with other closures, a session/stream reset
or detection restart returns to root. Missing opponent positions or a valid rate
prevent the timed speech actions. Track Vision supplies the opponent's side position;
map labels and geometry determine the upcoming slow corner's direction and its
opposite linked turn, using ACC's normalized X/Y ground plane or iRacing's X/Z
ground plane and the game's display orientation.

The next root child, **One Tight Slow Corner**, enters immediately when the next corner is labeled slow
on the Live Map (`corner` with `slow`, or `slow corner`) **and** is outside every
Live Map `consecutive corners` label. Even the last or only corner in a labeled
region is excluded. Fast, unlabeled, or conflicting speed tags do not qualify.
This entry check has no opponent-distance or arrival-time limit and does not require an opponent
motion rate; it still needs a fresh opponent lap position.

Its first child, **Inside the Long Straight**, enters when the driver's lap position
is inside a Live Map segment tagged `long straight`. It remembers that specific
segment and waits until the driver leaves its bounds, then runs **exit to parent**
to return to **One Tight Slow Corner**. A plain `straight` tag does not qualify.
Membership follows the tag bounds, including start/finish wraparound, independently
of corner-approach labels. Missing map or fresh driver position data withholds the
exit and speech. Inside this child, estimated opponent distance must be at most 2 m
**and** the opponent must be outside the upcoming turn (at the left edge before a
right turn, or at the right edge before a left turn) to trigger its **say phrase** action:
“Inside pass possible.” The phrase runs
once per visit; the child stays selected until the driver leaves the straight.

This child also offers:
“Leave room. Pass on exit.”
It requires all three conditions: the player's average exit speed for the upcoming
slow corner is greater than the nearest car ahead's, the estimated opponent gap
is at most 2 m, and the corner leads onto a mapped `long straight`. The exit
straight must begin at the exit or within the map's 150 m / 3% tagging tolerance,
and cannot cross another corner. This action has priority over the inside-overtake
phrase when both match; only one phrase runs per visit to this child.

Open **Live Event Log** to collect exit observations for each car and corner.
Exit speed is the forward normalized lap-position change per second between the
two samples bracketing the mapped exit, using `Graphics_game_time` (elapsed seconds).
It is independent of player physics and corner duration, and supports finish-line
wraparound. A captured exit is valid even when the corner entry was not observed.
The event log displays exit speed as percent of a lap per second and averages the
middle 75% using its existing rule: drop 12.5% at each end, rounded down with at
least one dropped from each end. Three valid exits per car are required. Speeds
are trimmed independently of corner times. Missing history, equal/slower player
speed, or a different map/corner withholds this action. Completed records survive
stream restarts; a session reset clears them.

Its next child, **Inside the Corner**, enters when the driver's lap position is
inside a mapped corner's bounds (`corner`, `slow corner`, or `fast corner`). The
corner-approach phase alone does not qualify. It remembers that specific corner
and runs **exit to parent** when the driver leaves it, returning to **One Tight
Slow Corner** even if another corner overlaps it. Missing map or fresh driver
position data withholds the exit. Corner bounds include start/finish wraparound.

While either child is active, the parent's speech action waits. At the parent,
opponent in the middle, with the driver at the right edge before a right turn or
at the left edge before a left turn, triggers “They defend. Stay close.”

Speech can run as soon as its position conditions match; this guide has no extra
arrival-time delay. The parent exits to root after its defending phrase runs. Chicane overtake
retains priority when its more specific entry conditions match.

Open Live Session, expand the right sidebar, and select **Live phrases**. The
panel lists all overtaking guides as collapsed rows with live status and condition/action
counts, above the latest 50 triggered sentences. Expand a closure, its entry or speech
conditions, nested condition groups, and individual actions to inspect their details.
Expansion choices survive live updates, and deeper condition groups stop adding indentation.
The root catalog can also be collapsed while the current state path remains visible.
Detection starts disabled; click
**Enable detection** at the top of the panel to start listening. While enabled,
it keeps listening when the sidebar is folded or the Assistant tab is selected.
**Disable detection** stops listening and clears triggered sentences and active
phrase cards. Enabling again starts fresh with new telemetry.

Enabling detection prepares every catalog sentence and declared instance variant through
`POST /voice/phonemes` followed by `POST /voice/synthesize/prosody`, at a default 1.2× speed,
one synthesis at a time. Completed clips stay cached while the panel is
mounted. The Speech indicator shows **Preparing** until all requests finish,
then turns green with **Ready** only when every catalog instance has received
its audio. Failed requests leave it **Incomplete**; disabling detection shows
**Disabled**. Enabling again reuses cached clips and retries missing ones.
Each new phrase event immediately attempts playback once through the common
`audioManager` path, using type `voice` and priority 25. A new phrase replaces an
earlier phrase at that priority. Live chat's default priority 50 can interrupt
or suppress a phrase; discarded phrases are not queued for
replay. Speech works independently of overlay visibility.

Entry forecasting and phrase occupancy use separate clocks. `setSpeechDurations`
supplies audible clip durations keyed by the complete instance cache key; the
longest prepared wording/variant determines that action's entry forecast window. No
telemetry callback synthesizes audio or changes the prepared clip. `reactionAllowanceMs`
can add an allowance per speech action; the default is 0 ms.

`PhraseEvent.speech` records `durationMs`, `preparationTimeMs`, `reactionAllowanceMs`,
`startedAt`, and `releaseAt`. Prepared PCM16 WAV packs store `speechBounds` by
scanning forward from the file start for the first frame above 0.01 and backward
from the file end for the last, checking every channel. Live phrase preparation trims
the outer silence once, keeping every PCM frame between those endpoints. The resulting
WAV's duration is the audible duration used for entry lookahead, and its bounds run from
zero to that duration. `startedAt` is set once when playback reaches its first frame;
`releaseAt` is set by actual playback completion. Pauses inside a sentence, buffering,
the estimated duration, and event deadlines do not release it. Entirely silent clips use `speechBounds: null`
and finish naturally; older packs or unsupported WAV encodings without bounds
use the meter for onset and natural completion for release. Completion,
rejection, cancellation, failure, or a silent clip ending
also releases the action. Steering and long-straight scopes keep their driving
exit rules, while the released speech action allows later actions to run.
Without a connected playback owner, the standalone engine retains immediate action
completion for detection-only use.

Each completed clip immediately updates the entry forecast durations and becomes
available for playback while the rest of the startup batch prepares. A triggered
event waits only if its own clip is missing; the latest waiting event plays when
that clip arrives, without rechecking duration, deadlines or entry conditions.
Session/stream resets stop phrase speech and discard pending events. Disabling
detection or unmounting also aborts preparation and stops owned playback. Speech
errors appear in the panel without stopping rule detection or overlay cards.

### Runtime phrase instances and prosody curves

`phrase.ts` provides the `Phrase` prototype and its `createInstance(settings)` factory.
Every executed speech action creates a `PhraseInstance` with independent, immutable
copies of its text, voice settings, float inputs and three amplify curves. Instances
live only in memory and are attached to their phrase events; no instance is saved
to local storage, project files or the backend. Alternative wordings and additional
actions each have their own speech prototype. Rule conditions and closure state
stay in the engine.

Curve points are edited in project JSON files, with no curve controls in the UI.
`curves/default.json` supplies three zero curves for unconfigured phrases.
`curves/reference-brake.json` is a configured example attached to **Brake**.
Import another phrase's JSON in `phrase-engine.ts` and assign it to that action's
`amplifyCurves`. Each file contains `timing`, `energy` and `pitch` arrays of
`{ "position": number, "value": number }` points. Positions run from 0 to 1 across
the selected wording's complete indexed phoneme range. Each curve needs 2–64
points, endpoints at 0 and 1, and strictly increasing positions. The backend
interpolates between points. Rebuild/reload the frontend after saving JSON edits.

An action's optional `speechSettings` sets its default voice, language, speed and
named `inputs`. Inputs default to zero, so even the Brake example initially applies
neutral controls. Set `speechSettings.inputs` for fixed settings, or supply an
`instanceSettings(context)` strategy on the action to compute overrides when it
actually speaks. This callback runs once per execution, after choosing the wording;
it is not called during inspection or catalog preparation. Declare every possible
override in the action's `preloadSettings`; runtime playback never generates audio.
**Ease brake** uses this mechanism to select its timing from the active
reference's deceleration change rate.

For explicit startup preparation:

```ts
const phrase = new Phrase('reference-brake:say:0', 'Brake.', brakeCurves);
const instance = phrase.createInstance({
    inputs: { timing: 0.5, energy: 0.75, pitch: -0.25 },
    speed: 1.2,
});
const packs = new Map<string, TtsPack>();
await loadAllPhraseSpeech([instance], packs, signal);
const voice = packs.get(instance.cacheKey);
```

The shared `components/tts` function `synthesizeProsodyTts(curves, inputs, speech, signal?)`
also works without a phrase. Each curve is `{ setting, points }` with setting
`timing`, `energy` or `pitch`; `inputs[i]` is the float multiplier for `curves[i]`.
The phrase adapter emits these arrays in timing/energy/pitch order. At every point:

- Timing multiplier = `1 + curve.value × input` (allowed range 0.25–4).
- Energy adjustment = `curve.value × input` dB (allowed range −12–12).
- Pitch adjustment = `curve.value × input` semitones (allowed range −12–12).

Timing changes duration relative to the model's prediction at the requested speed;
energy and pitch add to the model's predictions in their respective units. Finite
negative and fractional inputs are supported. Invalid curves, mismatched input
counts and amplified values outside backend limits are rejected, without clipping.
The helper previews phonemes for the exact text/language, sends the returned
`expected_phonemes` and index range to synthesis, then returns a WAV `TtsPack`
with its measured duration. `previewTtsPhonemes` is also exported for inspecting
indexed phonemes independently. The same abort signal covers both requests.

Speech caching includes text, voice, language, speed, curves and input values.
`getPhraseSpeechInstances()` enumerates all wordings and declared instance settings.
`loadAllPhraseSpeech(instances, packs, signal, onPrepared?)` requests that entire batch when
detection is enabled, deduplicates identical audio identities, serializes model work,
and returns completion status and any errors. Its optional callback runs after
each newly prepared clip is cached, making it available before the batch finishes.
Completed clips, including all release variants, are reused across detection toggles;
unmounting releases the cache. Failed clips can be retried on the next enable.
Telemetry events only read prepared audio. An instance still missing when preparation
finishes is skipped without requesting audio or substituting a different delivery.

Every triggered phrase also publishes a pop-out card (eight seconds for overtaking,
or until the reference cue's deadline) through the
existing overlay addon contract. Enable the overlay in the Assistant to see it.
Live phrases reuse an active live overlay presentation, or create a local live
presentation on the first trigger when none exists; no AI conversation is needed.
The shared overlay visibility setting is respected. Other session modes do not
receive live phrase cards.

`LivePhraseOverlay` owns one registered `MutableAiOverlayComponent` handle per
catalog rule, event deduplication, presentation scoping and cleanup. Session/stream
resets clear the cards, and unmounting releases all handles and any locally owned
presentation. Expired events never replay when a presentation changes.
All phrase display components, styles and future overlay graphs belong in
`live-phrases/graphs/`. `graphs/LivePhraseDisplay` implements the renderer and its
snapshot validation; the floating overlay only registers and displays it.

Entering **Steering** immediately inserts a separate `live_steering_corner` addon,
independently of speech preparation or phrase events. `PhraseSnapshot.steeringCorner`
identifies the same mapped corner used by the steering guidance, including an
approaching corner before its entry. The card uses the analyst presentation's
styling and displays the Live Map reference line plus the Circuit Map's saved left
and right boundaries, clipped to that corner with interpolated endpoints and
start/finish wrapping. Missing boundaries are reported instead of synthesized.
The trajectory uses a vehicle-height perspective from the corner entry, aligned
with the reference's initial travel direction. Distant geometry converges toward
the horizon; paths and the road surface are clipped behind the camera.
The card stays visible while Steering is entered, without republishing on every
telemetry frame, and clears on exit, stale input, reset or detection shutdown.
`graphs/SteeringCornerDisplay` owns its rendering and snapshot validation.

Each leaf condition is a `PhraseCondition` instance with a boolean `conditionFit`
after measurement, or `null` before it can be evaluated.
The panel shows **Met**, **Not met**, **Missing input**, or **Not evaluated**.
Only conditions reached through active parents are measured. Browsing an inactive
branch never starts its measurements or timers. Missing or expired inputs never
count as a fit; an inactive condition is not reported as missing input.

Configure the connector before each condition in `PHRASE_DEFINITIONS` using the fourth
argument of `condition(input, operator, value, connector)`. It accepts `'and'`
(the default) or `'or'`; the first condition's connector is ignored. The panel
displays **AND** or **OR** between conditions using the same configuration as the
engine. AND is evaluated before OR, so this example means `(A AND B) OR (C AND D)`:

```ts
conditions: [
    condition('speed', '>=', 80),                         // A
    condition('phase', '=', 'straight', 'and'),           // B
    condition('speed', '>=', 30, 'or'),                   // C
    condition('phase', '=', 'entry', 'and'),              // D
],
```

For direct construction, pass the connector as the fifth argument:
`new PhraseCondition('phase', '=', 'entry', undefined, 'or')`.

Use `conditionGroup(conditions, connector)` to add parentheses. A group can contain
conditions and other groups at any depth. Its connector joins the **whole group**
to the previous condition or group; it defaults to `'and'`. The first connector
inside every group is ignored, just like the first connector at the rule level.
Groups are evaluated first, with AND before OR within each group. For example,
`(A OR (B AND C))` is:

```ts
conditions: [
    conditionGroup([
        condition('speed', '>=', 80),                    // A
        conditionGroup([
            condition('speed', '>=', 30),                // B
            condition('phase', '=', 'entry'),            // C
        ], 'or'),
    ]),
],
```

To express `(A OR B) AND C`, group A and B instead:

```ts
conditions: [
    conditionGroup([
        condition('phase', '=', 'entry'),                // A
        condition('phase', '=', 'middle', 'or'),          // B
    ]),
    condition('speed', '>=', 30),                        // C
],
```

For direct construction, use `new PhraseConditionGroup(conditions, connector)`.
Both rule definitions and evaluated snapshots retain the recursive
`PhraseConditionNode` tree, with `PhraseCondition` leaves and `PhraseConditionGroup`
nodes. Each group has its own `conditionFit` and `inputMissing`; an empty group
never matches. Every descendant is evaluated, even when an OR alternative already
matches. The panel displays indented, parenthesized groups with their own status
and each leaf's status. Missing inputs inside a satisfied group remain visible on
their leaves but do not add a **Waiting for** message.

Evaluated snapshots retain each connector. A matching OR alternative can satisfy
a rule even if another alternative has missing inputs; individual missing
conditions remain visible, but a satisfied rule does not show **Waiting for**.
Live telemetry is always required, and priority, hold and cooldown still apply.
Existing catalog rules keep their AND behavior unless their connectors change.

Open **Track Vision** and **Live Map** in Add Visualization. Apply Track Vision's
camera calibration and use a saved circuit map with a middle line and centerline
tags: `corner` plus `slow` or `fast`, and `straight` or `long straight`.
Legacy `slow corner` and `fast corner` tags also work. Untagged sections are
unknown, not implicitly straights; missing or conflicting speed tags withhold
guides that require a specific corner speed. Tag an enclosing region with
`consecutive corners` and tag the individual `corner` segments inside it to
describe a linked sequence. The enclosing region is not itself a corner.

The catalog includes:

- **Slipstream:** an individual visible opponent within an estimated 10 m on a
  tagged straight, at least 80 km/h, with a steady or increasing gap and the
  player laterally offset rather than already tucked directly behind.
- **Pressure and a feint:** entry to a tagged slow or fast corner without the
  player inside and opponent off the inside, or the player outside and opponent
  inside. Show an attack before braking and
  exploit the opening or compromised exit if the opponent responds.

`LivePhrases` registers the `live-phrases` handle and connects through
`LiveSessionHandle`. `subscribeTelemetry` supplies new frames and session/stream
resets without treating retained frames as fresh. `getTrackVisionDetection` /
`subscribeTrackVision` bridge `visualization:track-vision`;
`getLiveCircuitMap`, `getLiveReferenceLap` and `subscribeLiveCircuitMap` bridge
`visualization:live-trajectory-map`, including late mounts, map/reference loads,
reference changes and removal.
`getLiveCornerSummaries` reads the currently mounted Event Log on each telemetry
event and supplies exit-speed comparisons only for matching map and corner bounds.

`PhraseEngine` is deterministic and local, with no chat, API or speech calls.
For overtaking, it reads the published Track Vision `birdsEyeScene`, the same calibrated flat-road projection
rendered in the BEV tab, never raw masks, image boxes or depth-based `analysis`/`geometry`.
It works with relative depth. Live Phrases keeps two separate conditions for each car:
whether it is in a left or right turn corner, and whether it is near the left edge,
middle of the track, or right edge. Left and right follow the direction of travel.
Turn direction comes from consistent BEV road-edge curvature at each car's road slice;
track position comes from boundary distances independently of that curvature.
The driver origin is compared with the nearest visible road slice, and the nearest individual
opponent ahead is compared with the boundaries at its projected distance. Interpolation stays
within visible boundary sections, without bridging gaps or extrapolating unseen edges.
Missing, straight or conflicting curvature leaves turn direction unknown while supported
track positions remain available. Inside/outside tactics combine the two conditions:
left is inside a left turn and right is inside a right turn. These tactics require
both cars to have the same turn direction.
Traffic must lie between visible boundaries; car packs establish traffic ahead but do not
supply an individual position. Unplaced traffic cannot establish an empty road. These are
flat-road estimates from applied camera calibration, not metric depth measurements.
Slipstream uses the nearest on-track individual opponent's planar distance from
the vehicle origin. A lateral offset of at most 1 m in either direction counts
as directly behind. Closing is estimated by comparing the distance in successive
fresh captures; a decreasing distance withholds the suggestion. A single capture,
missing individual opponent, capture gap, calibration change or session reset
withholds motion-dependent advice until new distance comparisons are available.
Repeated or out-of-order captures do not count as new motion evidence. The
comparison estimates the nearest opponent's gap and does not track car identity.
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

The first matching rule in the catalog takes priority, with Chicane overtake and
One Tight Slow Corner before Slipstream and Pressure and a feint.
The other guides require a visible opponent and
at least 30 km/h (80 for slipstream). All corner guides require known individual
positions. The wording does not claim that vision confirms overlap, passing
clearance or the opponent's intent; the slipstream gap trend is a visual estimate.

Only new live telemetry frames emit. Timers and map/vision updates only evaluate
or expire inputs. Telemetry expires after 1.5 s and overtaking vision after 2 s;
reference-position captures expire after 0.5 s.
Other overtaking guides' entry conditions must hold for 0.8 s. Overtaking guides use an 8 s per-guide
cooldown and 0.5 s clear period before repeating. Capture/telemetry gaps and section changes restart the
hold; session/stream resets clear history and invalidate previous vision.
Overtaking speed falls back to the magnitude of all three velocity channels,
converted to km/h. G-forces and brake input are not required for overtaking.
