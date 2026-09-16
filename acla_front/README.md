# Kestrel Motorsport Analyst Frontend & Desktop Companion

This package contains the React UI and the Electron desktop shell that executes all Python automation located in `src/py-scripts/`.

## Prerequisites

- Node.js 18+
- npm 9+
- Python 3.10 or newer (system interpreter used to create virtual environments)

## Python environments for desktop scripts

All Electron-only Python entry points live under `src/py-scripts/`. We manage their dependencies with
virtual environments so that development and production builds are reproducible.

| Mode | Virtual environment location | How it is created |
| --- | --- | --- |
| Development | `acla_front/.venv/py-scripts` | `npm run start:electron` (via pre script) or `npm run setup:python -- --mode=dev` |
| Production (packaged app) | `acla_front/.venv/py-scripts-prod` (bundled as `resources/python-env`) | `npm run build:electron` (via pre script) or `npm run setup:python -- --mode=prod` |

The managed environments install everything from `src/py-scripts/requirements.txt`, including
scikit-learn and pandas.

### One-time manual setup

```bash
# Install Node dependencies
npm install

# Prepare the development Python environment (optional – start:electron runs it automatically)
npm run setup:python -- --mode=dev
```

> **Tip:** Set `PYTHON=/path/to/python3` if your system Python command is named differently.

### Run the desktop app in development

```bash
npm run start:electron
```

The pre-script will:

1. Create/refresh `.venv/py-scripts`
2. Install every package in `src/py-scripts/requirements.txt`
3. Launch the React dev server and Electron shell

### Build the production bundle

```bash
npm run build:electron
```

This command:

1. Creates/refreshes the production env in `.venv/py-scripts-prod`
2. Installs requirements inside that env
3. Builds the React bundle and packages Electron via `electron-builder`

The packaged app ships with the Python runtime under `resources/python-env`. The Electron
main process automatically selects that interpreter; override it with the environment variable
`ACLA_PYTHON_PATH=/absolute/path/to/python` if needed.

### Additional scripts

- `npm run setup:python` — Run the Python environment bootstrap in on-demand mode.
- `npm run electron` — Launch Electron pointing at the compiled build output.

### Live telemetry dataset

[`src/data/live-telemetry-dataset.js`](src/data/live-telemetry-dataset.js) is the
authoritative table of application telemetry fields and value types. `Physics_*`,
`Graphics_*`, and `Static_*` are shared application field groups used across
simulators. Every simulator reader maps its native data to this contract and
must produce rows accepted by the table. Register new fields before emitting them and
document their units and meanings in [`tmp/telemetry-fields.md`](tmp/telemetry-fields.md).
Readers must omit values their simulator cannot supply.

The writer, live view, recorded-file reader, preload bridge, and renderer all
validate against this dataset. Renderer field types also derive from the table.
Rows retain their flat field names and values through recording and upload;
transport metadata stays outside the row. Unknown fields, wrong types, raw SDK
objects, and legacy aliases are rejected. There is no legacy catalog or fallback.

### iRacing live recording

Uploading an iRacing recording creates two sessions with matching name prefixes:
`iracing_live` contains the app's saved live samples and `iracing_recorded` contains
samples converted from native `.ibt` files. These values are also saved in
`game_recorded_from`; the backend must accept both source values before the updated
desktop app is used. Existing `iracing` sessions remain supported.

Enable iRacing disk telemetry while driving and exit the car before uploading so
the `.ibt` files are finalized. The desktop app searches the Windows Documents
`iRacing/telemetry` folder for files overlapping the app recording's time window,
matching track, car and driver. Multiple matching stints are combined. Ambiguous
or missing matches open a file picker; selected files must match the recording and
belong to one simulator session. Conversion runs in a worker, streams bounded
batches through the dedicated
[`IRacingIBTAdapter`](electron/recording/readers/iracing/iracing-ibt-adapter.js), and
preserves the native files. This adapter has its own channel allowlist and coverage
table, reuses the common iRacing mappings, and adds 36 mappings for brake-line
pressure, tire surface temperatures, wheel speed, suspension velocity, ride height,
oil measurements, fuel/manifold pressure, ABS force reduction, and player coordinates/identity. Thirty new shared
Physics fields preserve measurements with different meanings from existing fields;
surface temperatures are separate from core temperatures, and wheel speed is in
m/s rather than rad/s. Additional mappings require matching SDK units in the file.
Player `Lat`/`Lon`/`Alt` are converted through WGS84 Earth-centered coordinates to
track-referenced XYZ in meters: **X east, Y up, Z north**. The origin is the session's
`TrackLatitude`/`TrackLongitude`/`TrackAltitude`, shared across laps and stints.
Only the player's coordinate slot is populated; missing geographic channels or
reference metadata leave position absent. This geographic position frame differs
from the simulator-local frame used by the existing heading and world-velocity fields.
See [recorded-file field units and availability](tmp/telemetry-fields.md#iracing-recorded-file-fields).
Missing or incomplete native data keeps the local draft available for retry.
Both uploads must finish before the app recording and converted temporary file
are deleted. Retrying in the current upload flow skips an already completed version.

On Windows, launch an iRacing session and start recording from the live-session view.
The recorder waits for the simulator while disconnected and resumes when telemetry
returns. It uses the existing managed Python runtime and requires no extra Python
packages or native compilation. The `electron/**/*` and `py-scripts` packaging rules
include the reader, adapter, and capture script in desktop builds.

The capture process follows the [iRacing SDK](https://forums.iracing.com/discussion/62/iracing-sdk)
shared-memory protocol. It opens the map read-only, waits on the SDK event, copies
the newest row, and checks the tick again before accepting the snapshot. Reference
implementation: iRacing-authored [structure definitions](https://github.com/vipoo/irsdk/blob/master/irsdk_defines.h)
and [capture algorithm](https://github.com/vipoo/irsdk/blob/master/irsdk_utils.cpp)
in a public mirror of the official SDK; the official forum requires member access.

Processing runs in four OS processes: SDK capture, reader/adapter, writer, and live
view. `IRacingAdapter.adapt()` in
[`iracing-adapter.js`](electron/recording/readers/iracing/iracing-adapter.js) converts
the data **before** either consumer receives it. Raw SDK names and YAML never enter
saved rows or uploads. `IRACING_FIELD_COVERAGE` accounts for all 270 registered fields;
100 have mappings, conditional on the car/session exposing the required source.
Other fields remain absent under the [standard contract](tmp/telemetry-fields.md).

Conversions include m/s to km/h, kPa to psi, lap seconds to integer milliseconds,
gear indexing (reverse/neutral/first become 0/1/2), normalized steering input,
front brake bias as a fraction, and standard session/flag enums. Session YAML is
parsed only when its SDK update counter changes. Metadata resets on reconnect.
Physics is omitted outside the cockpit and during replay to avoid recording stale
or spectator data as driver input. Cold pit tire measurements are not live tire
pressure/core temperature; SDK repair time is not body damage. Live world positions
and other channels without equivalent standard semantics remain absent.

The additional mappings cover body acceleration, local and world velocity, angular velocity,
orientation, fitted tire compound, fuel consumption and remaining laps, completed
sector times, on-track relative gaps, and distance traveled. Motion units, axis/sign
conventions, and calculation/reset rules are in
[`tmp/telemetry-fields.md`](tmp/telemetry-fields.md#motion-and-calculated-fields).
Sector and gap values are interpolated estimates. Fuel averages require a complete
observed non-pit lap and confirmed non-electric metadata. Stint distance and used
fuel require an observed stationary pit start or refuel; joining mid-stint leaves
these totals absent. Missing data, towing, replay, session/driver changes, and gaps
over one second invalidate history. Relative history is bounded to 180 seconds.
The motion basis is covered by synthetic conversion tests; controlled ACC/iRacing
captures are still needed to validate physical signs and gravity behavior across
simulators. Mapping counts do not establish per-car live availability.

There is one outstanding capture request and at most 120 unacknowledged frames per
consumer. The writer acknowledges after disk commit; the view acknowledges after
delivery. Capture pauses when either consumer falls behind and reports an error if
it remains stalled for five seconds. IRSDK exposes the latest few ticks, not a durable
history: slow consumers can therefore cause missed SDK ticks. This bounds recorder
memory rather than promising lossless capture under arbitrary load. Game FPS impact
still needs measurement with iRacing running on the target machine.

Recorder checks:

```bash
python -m unittest discover -s src/py-scripts/tests -p test_iracing_sdk.py -v
node node_modules/react-scripts/bin/react-scripts.js test --watchAll=false --runInBand --runTestsByPath src/common/__tests__/iracing-mappings.test.js src/common/__tests__/iracing-recording.test.js src/common/__tests__/live-telemetry-dataset.test.js src/common/__tests__/recording-architecture.test.js
```

### Troubleshooting

- Delete `acla_front/.venv/` and re-run `npm run setup:python -- --mode=dev` if packages become inconsistent.
