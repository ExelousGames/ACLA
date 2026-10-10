# Live phrases

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
state cannot move to its parent or a sibling closure, even if the entry condition
becomes false or no children match. It stays inside until an explicit
`Action.exitToRoot(condition?)` runs. With no condition, that action returns
directly to root as soon as the state reaches it. Returning to root clears the
completed actions, allowing a fresh visit. Each state owns its traversal memory.

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
conditions) into one root with a separate child closure for every guide.
Each child contains one or more **say phrase** lambdas and an **exit to root** action, which
is eligible after any phrase action runs. Optional `additionalActions` add alternative
sentences; each action's `actionConditions` gate speech independently of entry conditions.
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

The first child, **Chicane overtake**, enters immediately when the
visible opponent is within 2 m and its estimated arrival at a slow corner is
within 2 s. That corner must belong to a labeled consecutive-corners region and
its next linked corner must turn the opposite way. Arrival uses the nearest car
ahead in `Graphics_normalized_positions`, excluding `Graphics_player_car_id`,
and its forward normalized-position change per second. At least two distinct
position samples are required; repeated packets retain the rate for up to 1.5 s.
Opponent changes, stale telemetry, pauses and resets discard the rate. The map
lookahead for this estimate is not limited by the normal 150 m / 3% approach range.

Inside this closure, speech waits until the opponent's estimated corner entry is
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
prevent this timed guide. Track Vision supplies the opponent's side position;
map labels and geometry determine the upcoming slow corner's direction and its
opposite linked turn, accounting for the game's X/Z coordinate orientation.

The next root child, **One Tight Slow Corner**, enters immediately when the
estimated opponent distance is at most 2 m **and** the upcoming corner is outside
every Live Map `consecutive corners` label. Even the last or only corner in a
labeled region is excluded. This entry check does not require a slow tag, an
arrival-time limit, or an opponent motion rate; it still needs a mapped upcoming
corner and a fresh opponent lap position. It selects between two speech actions:

- Opponent at the left edge before a right turn, or at the right edge before a left turn:
  “Opponent didnt defend, overtake from inside is possible here”
- Opponent in the middle, with the driver at the right edge before a right turn or
  at the left edge before a left turn:
  “Opponent is defending, Pressure is on”.

Speech can run as soon as its position conditions match; this guide has no extra
arrival-time delay. It exits to root after either phrase runs. Chicane overtake
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
ready, while its rule remains active and it is less than eight seconds old.
Session/stream resets stop phrase speech and discard pending events. Disabling
detection or unmounting also aborts preparation and stops owned playback. Speech
errors appear in the panel without stopping rule detection or overlay cards.

Every triggered phrase also publishes an eight-second pop-out card through the
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
- **Outbraking on the inside:** a slow corner entry, player inside and opponent
  off the inside.
- **Around the outside:** a fast corner entry, player outside and opponent inside.
- **Switchback:** entry or middle of a slow corner, player outside and opponent inside;
  the advice depends on the opponent running wide and an opening appearing.
- **Better exit pass:** the exit of a tagged slow or fast corner, with both
  positions known; use any exit-speed advantage once a passing lane is clear.
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

The first matching rule in the catalog takes priority, so linked-corner guidance
supersedes an outside pass or switchback, and intermediate sequence exits
supersede final-exit attacks. The other guides require a visible opponent and
at least 30 km/h (80 for slipstream). All corner guides require known individual
positions. The wording does not claim that vision confirms overlap, passing
clearance or the opponent's intent; the slipstream gap trend is a visual estimate.

Only new live telemetry frames emit. Timers and map/vision updates only evaluate
or expire inputs. Telemetry expires after 1.5 s and Track Vision after 2 s.
Other guides' entry conditions must hold for 0.8 s. All guides use an 8 s per-guide
cooldown and 0.5 s clear period before repeating. Capture/telemetry gaps and section changes restart the
hold; session/stream resets clear history and invalidate previous vision.
Speed falls back to the magnitude of all three velocity channels, converted to
km/h. G-forces and brake input are not required.
