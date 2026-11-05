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
pattern through the braking section. Its 10% threshold therefore moves earlier
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

Reference-line guidance has three root closures: **Brake zone**, **Throttle zone**,
and **Turn to reference line**.
Brake zone groups **Brake**, **More brake**, **Release the brake**, **Ease the brake**, and **Steering**;
Throttle zone groups **Throttle**, **More throttle**, and
**Less throttle**. Each zone enters from the active Live Phrases reference interpolated
at normalized lap position, even when the driver's pedals already match. A brake
zone enters at interpolated local reference brake pressure of at least 0.10 and
stays active until that pressure drops below 0.01 (1%). Exactly 1% remains inside.
Driving aggression stretches this brake profile together with the braking speed
profile. Calculated braking demand alone does not extend its bounds. Live speed
and pedal inputs do not change zone membership.
Brake zones take precedence over throttle zones.
Otherwise interpolated local reference throttle of at least 0.15 identifies a throttle zone.
The reference brake channel must be available. With a matching, tagged centerline
map, throttle zones are limited to actual corner bounds, including start/finish
wraparound. Reference throttle on a straight or an untagged gap cannot enter or
keep the throttle zone active. Once entered, the throttle zone stays selected
until the driver leaves the corner, even after the reference throttle zone ends.
Without usable map tags, it waits for confirmed corner-exit evidence.
Braking can begin before mapped corner entry.

Brake and throttle correction phrases return to their zone after speaking. The zone stays selected while
quiet or giving corrections, and only gives corrections for its own pedal.
Live speed or pedal errors never switch zones. The enclosing exit checks the
interpolated brake boundary or mapped corner exit for throttle even during a
selected speech action or speech cooldown; missing normalized telemetry position withholds
speech and waits for confirmed boundary evidence.
Fresh telemetry crossing the mapped corner end exits the throttle zone,
even if the reference-position capture has expired.
Brake zone returns to root when it ends. Steering cues run only inside
Brake zone. Throttle zone returns to root only when the driver leaves
the corner. Throttle pickup beyond the corner does not enter the throttle zone.
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
channels at `Graphics_normalized_car_position`. Steering sums the reference values
over the next 0.8 s of travel, with a 0.08 local magnitude threshold for **Turn**.
It uses live speed to locate reference brake onsets at the 0.10 threshold up to
1.8 s ahead, including start/finish crossings, to suppress acceleration before braking.
**Throttle** and **More throttle** compare the applied pedal with demand for the adjusted speed profile
and never request more throttle at full pedal. Missing or invalid control channels disable their
cues rather than being treated as zero.

Demand for **Brake** and **Release the brake** uses the active reference's brake pressure and deceleration.
For each decelerating interval, `(startSpeed² - endSpeed²) / (2 × distance)` in
m/s units gives deceleration; dividing by its mean brake pressure estimates the
response per unit of pedal. The nearest valid interval around the lap supplies
this response, including when Cautious starts braking before the local brake onset.
The adjusted speed profile's local slope multiplied by live speed gives the
deceleration needed to follow it. Speed errors of at least 5 km/h add or subtract
a correction over 1.5 s. Dividing the resulting demand by the active reference's
braking response gives a pedal target clamped to 0–100%. The response is a reference-based
estimate, not a measurement of the live car's deceleration.

- **Brake**: demand is at least 10% and the driver is below 10% pedal.
- **More brake**: speed is at least 5 km/h above the interpolated target and live
  deceleration is at least 10% lower than the reference, with at least 10% brake
  applied and the pedal below 100%.
- **Release the brake**: demand is below 10% while the driver is applying at least 10%.
- **Ease the brake**: speed is equal to or below the interpolated target, with at
  least 10% brake applied, and the remaining brake zone decelerates overall.
  The active reference's speed at the zone's interpolated 1% brake exit must be
  lower than its current speed. This includes zones crossing start/finish and
  permits local flat or accelerating intervals if the remaining zone has a net
  speed decrease. It does not require a live deceleration measurement.

**More brake** and **Ease the brake** use the aggression-adjusted speed profile at
normalized position. For **More brake**, reference deceleration comes from that
profile's speed and local slope, independently of reference brake pressure. Its
percentage comparison requires positive reference deceleration and live
deceleration at most 90% of it. When reference deceleration is zero, **More brake**
is disabled. **Ease the brake** checks the remainder of the zone instead of the local slope.
Live deceleration is the speed decrease in m/s
divided by elapsed seconds between consecutive active telemetry frames. Missing or
invalid speed, non-increasing timestamps, gaps over 1.5 s, pauses, resets, or changes
of game, track, car, session or stream clear this measurement. No pedal comparison
is used as a fallback for **More brake**.

**More brake** uses deceleration between consecutive telemetry frames without a
confirmation hold. **Ease the brake**, **Brake**, and **Release the brake** also
do not require this hold.

Only one brake cue can match at a time; speed/deceleration corrections take priority
over the demand-based cues. Full pedal never receives **More brake**. Without usable
reference speed or braking response, **Brake** and **Release the brake** demand
falls back to reference pressure; if speed is available and the driver is at least
5 km/h below target, demand is zero. Brake corrections are
spoken only inside the reference brake zone, including release corrections when
the driver is too slow while still inside that zone.

**Steering** is a nested closure inside Brake zone. From the current normalized
position, it adds all signed active reference steering values over the next 0.8 s
of travel at live speed, including start/finish crossings. Entry requires a nonzero
sum in the same direction as the current or upcoming corner in the copied Live Map.
Live speed must also be equal to or below the aggression-adjusted live phrase
reference speed interpolated at the current normalized position. Missing or invalid
reference speed withholds entry. This speed condition is checked only on entry.
Unknown corner direction withholds entry. The sum includes interpolated values at both
window boundaries and every reference sample strictly between them; a duplicated
start/finish sample counts once. Flat segments and reversals are allowed, and
opposite steering values cancel. Live steering and live deceleration do not
affect the sum. Preparation is silent. **Turn** speaks only while the window sum
qualifies and interpolated local reference steering reaches magnitude 0.08 in
the direction of that sum.

The action runs at most once per entry. The closure stays entered after **Turn**
and exits when the brake zone ends (reference brake pressure below 1%). Missing or stale
telemetry withholds speech; expired position-model captures do not. Missing reference
steering disables the steering sequence.

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
- **More throttle**: the applied pedal is at least 15 percentage points below demand.
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

Only one throttle cue can match at a time. Applying or adding throttle requires
the driver to be within 2 m of the line, with no braking required or imminent and
no applied brake. Braking sets throttle demand to zero. **Less throttle** can still
be suggested away from the line during outward overspeed, with no applied brake.
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
reference brake channel must be available and below the 0.10 onset threshold,
and speed must be less than 5 km/h above target when reference speed is available.
Movement children return to their parent after speaking; **Turn to reference
line** stays selected between movement cues and exits to root as soon as reference
braking starts or speed exceeds the target by at least 5 km/h, even if the driver's
brake input already matches. Brake-pressure corrections also end movement guidance.
Leaving the straight or losing required inputs also exits and clears movement
progress, so a later straight cannot inherit an old completion cue. Other
reference cues retain their interrupt priority.

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
steering and zone lookup continue from normalized position. Applying or adding
throttle requires being within 2 source units laterally of the line, with no current
braking or reference braking in the next 1.8 s. Distances use the reference's
source coordinates, assumed to be metres, and constant current speed for ETA;
the thresholds and lead time still need validation during simulator driving.

Brake zone and Throttle zone can interrupt waiting overtaking closures when their
reference zone starts, even when no speech cue matches. Pedal cues run only
inside their matching zone; steering cues run inside Brake zone. Reference prompts
have no shared spacing delay. The 2 s per-closure cooldown still applies.
**Brake** rearms only after leaving its zone;
other cues require a 0.5 s clear period before rearming. Short
clips prepare first. Upcoming cues expire 100 ms before the predicted action;
corrections expire after 1.5 s. Speech is discarded if its prepared clip cannot
finish by that deadline or its conditions clear, and overlays use the same
deadline. A queued steering prompt is discarded when its
reference phase changes. Live chat retains higher audio priority.

Live phrases runs a closure tree instead of a flat sentence-rule dispatcher.
`closure.ts` defines the three building blocks:

- `Closure<Context>(name, description, condition, children)` has an entry predicate and an
  ordered list of child closures or actions.
- `Action<Context>(name, description, run, condition?)` runs any lambda, including one returning
  a promise. Omitting its condition makes it immediately eligible.
- `State<Context>(root)` starts at root and remembers its current closure or
  action and the full path. `step(context)` descends through the first eligible
  children and runs at most one action. Actions run once per visit. A pending
  async action holds the state until it settles; failures propagate to the caller.

Entry conditions are checked only before entering a closure. Once entered, the
state stays inside even if the entry condition becomes false or no children match.
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
entered after speaking and relies on the enclosing Brake zone exit. Other sentence
closures have an exit action eligible after a phrase action runs. Movement and braking sentences exit
to their parent; acceleration and overtaking sentences exit to root. Optional `additionalActions` add separate
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
never runs an action. Async results appear on the next engine snapshot (the panel
polls every 250 ms). Action failures remain available to inspect without an
unhandled rejection in the telemetry subscription. Direct `State.step` callers
still receive thrown errors/rejections unless they pass the optional
`onActionError` callback to `step`. Condition exceptions always propagate.

Add a closure or action anywhere in the tree using the same constructors. Wrap
new pure predicates with `describeCondition(predicate, description)` to expose
their current result automatically. For grouped conditions or predicates with
side effects, pass a **read-only** description callback returning condition data
instead; the phrase adapter reuses its already evaluated conditions this way.
The renderer handles condition groups structurally, without importing phrase
condition classes or checking action names. An opaque, undecorated predicate is
never called just to display it: the UI labels its result as last checked, or
`Not evaluated` until traversal checks it. Human-readable condition meaning cannot
be inferred from an arbitrary JavaScript callback.

Optional node metadata (the last constructor argument) supplies an explicit `id`
and a read-only `inspect(context, state)` callback with extra labeled `fields` or
a domain `status`. All declared fields render automatically, including false and
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

Each leaf condition is a `PhraseCondition` instance with a boolean `conditionFit`.
The panel shows **Met**, **Not met**, or **Missing input** for every condition and group.
Each engine snapshot includes freshly evaluated condition instances, including
conditions in lower-priority guides; priority, hold and cooldown affect the
guide's status separately. Missing or expired inputs never count as a fit.

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
