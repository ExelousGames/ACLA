# Live phrases

Live Phrases copies the reference lap and centerline map published by Live Map.
All phrase rules use these local copies, including corner geometry, straight
membership, speed corrections, advance cues and the braking exit from movement
guidance. Reference channels are read at live telemetry's normalized lap position;
the position model supplies only lateral checks and never shifts reference lookup.
**Live Map must remain open to supply the reference and map**. Lateral predictions
must match its current source reference. Copies refresh when Live Map publishes
a replacement; removing Live Map clears them and invalidates retained location
predictions.

**Driving aggression** selects **0–50% caution relative to the reference**, defaulting
to **0% (Reference)**, which preserves the original lap exactly. At 20% caution,
each corner's minimum **speed** is 20% lower and local deceleration is capped at
80% of the reference rate at the same normalized progress through its braking zone.
Rates use `(startSpeed² - endSpeed²) / (2 × distance)` in m/s units. The braking
distance is derived from those reduced rates and the lower minimum, rather than
increased by a fixed percentage or determined by stretching speed values.
Braking starts earlier and reaches the lower minimum at the original position.
Reference brake pressure uses the same distance stretch, preserving its pressure
pattern through the braking section. Its rise from zero therefore moves earlier
with caution, allowing **Brake** to speak on the stretched approach. Overlapping
approaches use the strongest mapped brake pressure.
The acceleration section keeps its original start and end positions, rising from
the lower minimum back to the reference exit speed without stretching the throttle
zone. Adjustments use physical XYZ distance and wrap at start/finish. Mapped rate
boundaries add samples along the existing reference polyline, interpolating local
throttle and steering and the stretched brake profile. A backward speed calculation enforces local deceleration caps,
including the tightest cap in overlapping braking zones; a nearby corner can
therefore lower an earlier target further. Braking interpolates squared speed so
the rate stays feasible between samples. Reference throttle, steering,
racing-line geometry, centerline geometry and tags retain their original profiles. Missing or
invalid speed leaves the lap unchanged. Missing or invalid brake data is preserved;
caution does not invent brake pressure when the reference has none.
Each adjustment rebuilds from the untouched local baseline, clears current cues
and movement progress, and keeps detection and prepared speech running.
The baseline is used only to rebuild the active reference. Every condition and
cue inside **Brake zone** and **Throttle zone** uses the active, aggression-adjusted reference for its
speed, controls, geometry and pedal responses. Missing active-reference data
never falls back to the original lap. Live Map's source reference is retained
only to identify replacement laps and validate position-model captures.

Reference-line guidance has four root closures: **Brake zone**, **Throttle zone**,
**Steering**, and **Turn to reference line**.
Brake zone groups **Brake**, **More brake**, **Release the brake**, **Release brake slower**,
**Release brake faster**, and **Hold brake pressure**;
Throttle zone groups **Throttle**, **More throttle**, and
**Less throttle**. Each zone enters from the active Live Phrases reference interpolated
at normalized lap position, even when the driver's pedals already match. A brake
zone starts where reference brake begins rising from zero and ends where reference
brake decreases to zero or reference throttle begins rising from zero. A throttle interval starts at that throttle rise and
ends when reference throttle first reaches 100%, decreases to zero, or reference
braking starts. Throttle zone entry additionally requires reference steering to stay
constant from the current position through that endpoint. Any increase or reduction
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
travel time at live speed. This is a duration filter, not an entry hold: qualifying
zones enter immediately at the start of the interpolated rise when their steering
requirement is met. A short opposite zone still ends the preceding zone, but cannot itself enter. A pedal with no ending boundary
occupies the whole cyclic lap; all-zero and constant-full-throttle references have no pedal zones.
Driving aggression stretches this brake profile together with the braking speed
profile. Calculated braking demand alone does not extend its bounds. Live pedal
inputs do not change zone membership. With reference speed available, live speed
does not change the pedal interval itself. Both pedal zones defer while Steering is
active. Brake zone also defers while its direction-change exit condition
is met. The closure tree's priority and transition rules select a single active zone.
Both reference pedal channels must be available to determine zone membership.
Zone boundaries do not depend on mapped corner bounds: braking can start before
corner entry and throttle zones can continue onto straights.

Brake and throttle correction phrases return to their zone after speaking. The zone stays selected while
quiet or giving corrections, and only gives corrections for its own pedal.
Live pedal errors never switch zones. The enclosing exit checks the
opposite reference pedal onset, reference full throttle, or its own reference pedal release to zero even during a
selected speech action or speech cooldown; missing normalized telemetry position withholds
speech and waits for confirmed boundary evidence.
Fresh telemetry crossing the zone's reference endpoint exits the zone,
even if the reference-position capture has expired.
Brake zone also returns to root from the start of an active Live Phrases reference
trajectory direction-change range when it turns toward an overlapping mapped corner.
This exit uses the range direction, without reference or live steering changes,
a steering threshold, or a 500 ms heading-change check. Brake zone cannot re-enter
while this range condition remains met. Steering uses the same range-start condition
for entry. Neither transition requires live speed to be equal to or below the local
reference speed, or a usable reference speed channel. **Turn** additionally requires
the reference steering threshold below.
Steering can enter independently of either pedal zone. Throttle zone returns to root
at reference full throttle, release to zero, or an earlier brake onset. Throttle pickup beyond the corner can enter the throttle zone.
Movement cues wait until the active pedal zone has exited and a mapped straight
allows movement guidance. The former **Prepare to brake** cue is removed;
**Brake** speaks once per brake-zone entry when braking is needed now. Pedal changes,
expired inputs, and map refreshes cannot repeat it within the same zone; a confirmed
zone exit rearms it for the next entry. **Prepare to accelerate** is also removed;
**Throttle** speaks when throttle is needed now.
Open Live Map with a reference lap for the current game/car/track. Brake and
steering guidance and zone membership use live normalized position without requiring
Track Vision's position model. Run the model for lateral checks used by movement
and throttle cues. These cues do not require opponents, BEV reconstruction, or
camera calibration.
Movement guidance also requires Live Map straight tags. Steering requires a known
current or upcoming mapped corner direction; other reference cues do not require circuit tags.

`phrase-reference-guidance.ts` interpolates the reference's raw XYZ distance and
`expert_optimal_speed`, `expert_optimal_steering`, `expert_optimal_brake`, and `expert_optimal_throttle`
channels at `Graphics_normalized_car_position`. Steering becomes eligible from the start
of a continuous trajectory direction-change range toward an overlapping mapped corner,
with a 0.08 local steering threshold for **Turn**.
It uses live speed to locate reference brake rises from zero up to
1.8 s ahead, including start/finish crossings, to suppress **Throttle** before braking.
**Throttle** and **More throttle** compare the applied pedal with demand for the adjusted speed profile
and never request more throttle at full pedal. Missing or invalid control channels disable their
cues rather than being treated as zero.

Brake-zone guidance uses the active, aggression-adjusted speed profile to guide
braking toward the target speed. The required deceleration is the local
speed-profile slope multiplied by live speed, plus the signed speed error in m/s
divided by 1.5 s, clamped to zero. Even errors below 5 km/h contribute so that
matching deceleration while still too fast does not count as matching the target.
The comparison band is 10% of required deceleration, with a 0.1 m/s² minimum.

- **Brake**: estimated demand is at least 10% and the driver is below 10% pedal.
- **More brake**: measured deceleration is below the required band, the pedal is
  not being released, and the applied brake is at least 10% but below full pressure.
- **Release brake slower**: the pedal is being released and deceleration is below
  the required band. Retain more braking while approaching the target speed.
- **Release brake faster**: deceleration exceeds the required band. Reduce pressure
  faster to avoid dropping below the target speed.
- **Hold brake pressure**: measured deceleration is within the required band.
- **Release the brake**: no further deceleration is needed. A small positive demand
  still uses pace guidance rather than telling the driver to release completely.

The three pace cues each use three words and require at least 10% live brake.
They replace **Ease the brake**, which previously depended only on matching speed
and the remaining zone's net speed decrease. Only one brake cue can match per frame.

Live deceleration and pedal release are measured together at 500 ms intervals
while **Brake zone** is active. Deceleration is the speed decrease in m/s divided
by actual elapsed seconds. A pedal reduction greater than one percentage point
per second counts as releasing. The first active frame starts the interval; the
first frame at least 500 ms later supplies the measurement and starts the next
interval. Measurements persist between checks. Missing or invalid speed,
non-increasing timestamps, gaps over 1.5 s, pauses, resets, and game, track, car,
session or stream changes clear both measurements. Missing or invalid brake
samples also clear release evidence, including its previous pedal baseline.
**More brake** and the three pace cues wait for a valid interval; **Brake** and
**Release the brake** do not require it. Existing cooldown and rearming rules
prevent repeated speech while a cue stays matched.

The initial pedal demand uses the active reference's braking response. Each
reference decelerating interval gives `(startSpeed² - endSpeed²) / (2 × distance)`
in m/s units; dividing by mean brake pressure estimates response per unit pedal.
The nearest valid interval supplies that response, including earlier Cautious
braking. Required deceleration divided by this response gives a pedal target
clamped to 0–100%. Without a usable response, initial demand falls back to reference
pressure (or zero at least 5 km/h below target). Without reference speed, only
reference-pressure-based **Brake** and **Release the brake** remain available.
Pace cues compare measured motion directly with the speed profile, independently
of reference brake pressure. They work across start/finish and with the adjusted
reference, and are spoken only inside its brake zone. Steering uses its
direction-change entry condition independently of the target speed.

**Steering** is a root closure. First, the active reference trajectory from the current
normalized position to the end of its continuous direction change must be partially
or fully inside the current or upcoming mapped corner. Only the remaining section
counts; overlap behind the current point and touching a boundary alone do not qualify.
Entry becomes eligible as soon as live normalized progress reaches the start of that
direction-change range, including gentle turns. There is no timed lookahead or minimum
heading-change angle for entry. A telemetry frame inside the range still qualifies if it skips
the exact start boundary.
Entry does not compare live speed with the reference speed. The existing minimum
live speed of 20 km/h still applies. Unknown corner direction or bounds withhold entry.

Trajectory headings use the horizontal tangent from 5 m behind to 5 m ahead (XY
for ACC, XZ for iRacing), interpolated between samples using the shortest angular
difference. This avoids false plateaus when caution inserts reference samples.
Flat headings and reversals bound a continuous direction change. Its near-straight
exit tail is then trimmed: the range ends at the last segment whose absolute heading
change per metre reaches 10% of that range's peak, capped at 1 degree per 10 m.
This preserves uniformly shallow bends and meaningful gentle exits after sharp turns,
while small residual heading changes no longer extend Steering far onto a straight.
Only the trailing weak section is removed; the original start and easing before a later
apex remain. Direction changes and interpolation wrap across start/finish, including
a duplicated endpoint; exit trimming uses the whole joined range.

Preparation is silent. **Turn** speaks once per entry while the direction-change range
overlaps the corner and local reference steering reaches magnitude 0.08 in the corner direction.
Steering captures the continuous trajectory direction-change range on entry and
exits to root only when live normalized progress leaves that range, using the trimmed
endpoint described above. The discarded tail cannot re-enter Steering. Exit does not depend on
mapped corner bounds, reference speed, or completion of **Turn**.
Leaving that captured range also rearms Steering for the next qualifying range,
without a timed clear period or entry cooldown. Steering evaluates its own direction-change
condition independently of Brake-zone exit. Direction corrections inside Steering
retain their 2 s speech interval.

Reference steering changes, steering smoothing, live steering, orientation, velocity
and position-model error do not determine entry or exit. Missing reference steering
still permits silent entry and exit, but withholds **Turn**. Missing or invalid reference
speed withholds the entry speed comparison. Future reference speed does not determine
direction-change range membership. Missing trajectory direction withholds entry and exit. Missing or stale telemetry withholds speech;
expired position-model captures do not prevent **Turn**.

After **Turn**, nested **Too wide** and **Too tight** corrections compare the driver's
normalized horizontal world velocity with the direction to a reference endpoint
10 m ahead along the reference polyline, interpolating across samples and start/finish.
The position model's error points from the driver to the reference in car-local
right/forward/up coordinates. Rotate it into world coordinates using the capture's
heading, pitch and roll and the simulator's axis convention, then calculate:

`driverStart = referenceStart - worldReferenceError`

`desiredDirection = normalize(referenceEnd - driverStart)`

The endpoint remains on the reference. Both lateral and forward error shift the
start; vertical error is also rotated before projecting onto the horizontal plane.
The signed angle between normalized velocity and desired direction is classified
relative to the mapped corner's turn sign: outside is **Too wide**, inside is
**Too tight**. A 3° tolerance is silent. Neither velocity magnitude, reference speed
nor live speed changes the endpoint or the angular comparison. The existing speed
condition still gates entry into Steering, but is not reapplied to its corrections.
Corrections can repeat every 2 seconds while needed, sharing that interval with **Turn**.

Direction comparison uses reference progress, error, orientation and velocity from
the same validated capture (at most 500 ms old), so a delayed error is never rotated
using a newer heading. Missing or invalid error, orientation or world velocity,
unknown corner direction, zero vectors, and targets behind the velocity direction
withhold corrections. Pending correction audio is discarded if its direction no
longer matches, required data expires, or the driver leaves Steering.

Throttle demand uses the active reference's throttle and acceleration.
For accelerating intervals, `(endSpeed² - startSpeed²) / (2 × distance)` in m/s
units divided by mean throttle estimates acceleration per unit of pedal. While
the active profile is accelerating, the local speed slope and interpolated
throttle align this response to the current position, including throttle ramps.
Elsewhere the nearest valid accelerating interval supplies the response, wrapping
at start/finish. The adjusted speed slope multiplied by live speed gives the new
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
movement**, **A little bit more**, and **Enough movement**.
These are executable closures with their own speech actions and inspectable
conditions. Each speech action randomly picks from three short wordings, preserving
the direction and movement size (for example, “Move to the left”, “Shift to the
left”, or “Bring it to the left”). The position model's car-local X offset selects the side: negative
is left, positive is right. No reference steering channel is required.

Movement requires the driver's normalized lap position to be inside the current
Live Map's `straight` or `long straight` centerline bounds, with no overlapping
corner. Bounds include start/finish wraparound and remain eligible during the
map's corner-approach phase. An untagged section is not a straight. The aligned
reference brake channel must be available, with no reference braking or active pedal zone,
and speed must be less than 5 km/h above target when reference speed is available.
Movement children return to their parent after speaking; **Turn to reference
line** stays selected between movement cues and exits to root as soon as reference
braking starts or speed exceeds the target by at least 5 km/h, even if the driver's
brake input already matches. Brake-pressure corrections also end movement guidance.
Leaving the straight or losing required inputs also exits and clears movement
progress, so a later straight cannot inherit an old completion cue. Other
reference cues retain their interrupt priority.

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
progress stays quiet. “OK, that's enough” requires an earlier movement cue and
reaching that tolerance or crossing the line; it does not fire merely because
the driver started on the line. Movement cues are at least 2 s apart, except
completion can speak as soon as its conditions match. Capture gaps over 500 ms, missing
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

Brake zone and Throttle zone can interrupt waiting overtaking closures when their
reference zone starts, even when no speech cue matches. Pedal cues run only
inside their matching zone. Steering can interrupt a pedal zone when its entry
conditions match. Brake zone can interrupt Throttle zone. Both pedal zones wait
while Steering is active, so they cannot end its captured trajectory range early. Reference prompts
have no shared spacing delay. Steering entry has no cooldown; other reference closures
retain their 2 s cooldown.
**Brake** rearms only after leaving its zone; **Steering** rearms when its captured
trajectory direction-change range ends;
other cues require a 0.5 s clear period before rearming. Short
clips prepare first. Upcoming cues expire 100 ms before the predicted action;
corrections expire after 1.5 s. Speech is discarded if its prepared clip cannot
finish by that deadline or its conditions clear, and overlays use the same
deadline. A queued steering prompt is discarded when its
reference phase changes or Steering exits. Live chat retains higher audio priority.

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
An inactive subtree is shown as **Not evaluated**, and leaving its parent clears
its measurements and hold timers. Brake deceleration, throttle outward-motion
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
entered after speaking and exits to root when its captured trajectory direction-change range ends. Other sentence
closures have an exit action eligible after a phrase action runs. Movement and pedal sentences exit
to their parent; overtaking sentences exit to root. Optional `additionalActions` add separate
speech actions; each action's `actionConditions` gate speech independently of entry conditions.
Any speech action can keep its existing `sentence` and add `sentenceAlternatives`,
for example `{ sentence: 'Move to the left.', sentenceAlternatives: ['Shift to the left.'] }`.
Each execution chooses one wording with equal probability, including the original;
consecutive executions may repeat. Selection happens only when the action runs;
checking conditions or inspecting the tree does not choose a wording. Action Details shows the alternatives.
All wordings are prepared for speech, and the chosen sentence is published once
for audio, history and overlays. Omitting alternatives keeps the single-sentence behavior.
Only one speech action runs per entry. Its lambda publishes the event used by speech, history
and overlays; there is no parallel sentence dispatcher. Hold, cooldown, clear
period and fresh-frame checks gate entry into these sentence closures. The first
matching guide reserves priority while waiting for those checks. The exit runs on
the next engine step; state then considers root's children again. Session/stream
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
`priority` orders sibling evaluation; `interrupt` lets an enclosing exit or guide
replace an entered child. Brake zone uses `yieldToSiblings` to let Steering take
over when its active children have no action to run. All declared fields render automatically, including false and
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
described below. It offers only the “From pervious laps statistic...” phrase, with
the same three conditions, and speaks once per straight visit. Leaving the straight
returns to **Chicane overtake** so its timed speech actions can run.

At the parent, speech waits until the opponent's estimated corner entry is
strictly less than 0.5 s away. Track Vision then selects between two speech actions:

- Opponent on the left before a mapped left turn, or on the right before a mapped right turn:
  “Go wide in the first turn. then take second apex if possible”.
- Opponent on the left before a mapped right turn, or on the right before a mapped left turn:
  “brake early, Hold inside”.

The upcoming corner's direction
comes from the same map segment used for the entry ETA. Neither car needs to be
detected in a corner, and the player's position is not required.
After either speech action runs, the next action exits to root.
While waiting, its status is **Waiting for action**; losing the entry conditions
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
“Opponent didnt defend, overtake from inside is possible here”. The phrase runs
once per visit; the child stays selected until the driver leaves the straight.

This child also offers:
“From pervious laps statistic, you generally have better exit speed. long straight after this corner, consider back up a little bit to have space to overtake at exit.”
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
at the left edge before a left turn, triggers “Opponent is defending, Pressure is on”.

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

Enabling detection also prepares every catalog sentence, including alternative actions, through the system TTS
service at 1.2× speed, one request at a time. Completed clips stay cached while the panel is
mounted. The Speech indicator shows **Preparing** until all requests finish,
then turns green with **Ready** only when every catalog sentence has received
its audio. Failed requests leave it **Incomplete**; disabling detection shows
**Disabled**. Enabling again reuses cached clips and retries missing ones.
Each new phrase event attempts playback once through `audioManager`,
using type `voice` and priority 25, below live chat's default priority 50. Live
chat can interrupt or suppress a phrase; discarded phrases are not queued for
replay. Speech works independently of overlay visibility.

If a triggered clip is still being prepared, only the latest event can play once
ready. Overtaking clips require an active rule and an event less than eight
seconds old; reference cues require matching conditions and enough time to finish
before their shorter deadline.
Session/stream resets stop phrase speech and discard pending events. Disabling
detection or unmounting also aborts preparation and stops owned playback. Speech
errors appear in the panel without stopping rule detection or overlay cards.

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
