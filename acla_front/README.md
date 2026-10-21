# Kestrel Motorsport Analyst Frontend & Desktop Companion

This package contains the React UI and the Electron desktop shell that executes all Python automation located in `src/py-scripts/`.

## Prerequisites

- Node.js 18+
- npm 9+
- Python 3.10 or newer (system interpreter used to create virtual environments)

## ACC per-car track positions

ACC recording supplements shared memory with the local UDP Broadcasting API.
`Graphics_normalized_positions` contains native car IDs mapped to lap fractions
from 0 to 1, for example `{"7":0.25,"1052":0.8}`. iRacing emits the same field
from `CarIdxLapDistPct`; the existing player-only `Graphics_normalized_car_position`
is unchanged. Missing or stale car positions are omitted.

Enable broadcasting in `Documents\Assetto Corsa Competizione\Config\broadcasting.json`
with a nonzero `updListenerPort` (ACC's spelling), for example:

```json
{
  "updListenerPort": 9000,
  "connectionPassword": "your-password",
  "commandPassword": ""
}
```

Preserve any existing settings and restart ACC after changing its configuration.
The extractor reads the port and connection password from that file automatically,
including redirected Documents folders. Set `ACLA_ACC_BROADCAST_CONFIG` to use a
different config file. It only connects to `127.0.0.1` and never sends driving,
camera, or replay controls. No additional Python dependencies are required.
If broadcasting is disabled or unavailable, shared-memory recording continues
and the new field is omitted. UDP samples are requested every 100 ms; individual
car positions expire after two seconds without an update.

## AI Self-Driving Training (Windows desktop)

The sidebar's **AI Self-Driving Training** page trains a local, continuous-action
actor–critic neural policy against the running simulator. Start live telemetry,
add **Live Map** with a matching reference lap, and add **Track Vision** with
position and segmentation models and screen capture enabled. The training page
reports any missing inputs. Live Session stays mounted when switching tabs.

### Wheel / joystick output (ACC and iRacing)

Select **Wheel / joystick (vJoy)** for a simulator using wheel controls. This creates
standard DirectInput axes independently of the physical wheel brand (MOZA, Fanatec,
Logitech, Thrustmaster, etc.). It does not intercept a physical wheel or automatically
replace that wheel's bindings. The simulator must be bound to the virtual device
for AI driving; restore your normal controls when returning to manual driving.

1. Install the signed Windows 10/11 [vJoy driver](https://github.com/BrunnerInnovation/vJoy/releases).
   It is separate from the bundled Xbox driver. In **Configure vJoy**, enable a free
   device with **X**, **Y**, and **Z** axes, then select its device number in the app.
2. Connect before launching the simulator (or restart it if the device is missing).
   Map **X → steering**, **Y → throttle**, **Z → brake**. Steering uses the full signed
   range; each pedal has its own independent axis. Check that neutral shows zero
   throttle/brake and centered steering; invert a pedal in the simulator if needed.
3. **ACC:** use **Options → Controls → Wheel**, save your existing preset, and create
   a separate training preset with those vJoy axes.
4. **iRacing:** use **Configure Controls**, or **Settings → Driving → Input Calibration**,
   and calibrate the vJoy steering and pedal axes. Follow the
   [iRacing calibration guide](https://support.iracing.com/support/solutions/articles/31000133335).
   Keep your manual controller configuration available for switching back.

The **Test … in 5 seconds** buttons allow time to switch to the simulator's controls
screen. They hold the chosen axis for 10 seconds and then release it, even while the
simulator has focus. Pedals test full travel. For steering calibration, use **Full
left**, **Full right**, then **Center**; for iRacing's 90° left measurement, use
**20% left** to calibrate a 900° total range. Accept the measurement while the timed
input is active. **Stop**, Escape in the app, disconnect, or leaving the training
page cancels a pending/active test. The hold buttons release when the app loses focus.

**Connected** confirms only that the output device accepted initialization. Check
the simulator's input meters before training. This bridge does not relay physical
wheel input or force feedback. Both simulators use the same output backend; their
bindings and calibration are configured separately.

No extra Python package is needed for vJoy: the bridge loads the installed SDK DLL
matching the Python architecture from `Program Files\vJoy\x64` (or `x86`). For a
custom installation, set `ACLA_VJOY_DLL` to the absolute `vJoyInterface.dll` path
before launching the app. Missing drivers/axes, busy devices, and rejected writes
are reported as errors. The watchdog and disconnect reset pedals to zero and
steering to center before relinquishing the device.

### Xbox gamepad output

The virtual Xbox controller is included with the Windows application. Choose
**Xbox gamepad (XInput)** when using the simulator's gamepad controls. Python setup
installs [vgamepad](https://github.com/yannbouteiller/vgamepad) into the managed
development and production environments. The Windows application installer bundles
[ViGEmBus 1.22.0](https://github.com/nefarius/ViGEmBus/releases/tag/v1.22.0) and installs
it when no working controller driver is present. Approve the Windows administrator
prompt and restart Windows if setup requests it. No separate pip command or
internet connection is needed on the target PC during application installation.

`electron-builder` downloads and verifies the pinned driver installer before
packaging, caching it in `.venv/controller-driver`. Building the app does not
install a system driver on the build machine. Ordinary app startup does not
install drivers either. Uninstalling the app leaves the shared driver available
for other applications.

Python setup uses the checksum-pinned upstream vgamepad 0.1.3 GitHub archive,
because that release is not published on PyPI. It supports skipping the legacy
driver installer while building the Python package. Build machines need GitHub
access; a Git executable is not required.

For development, run `npm run setup:python` and install the official ViGEmBus
release linked above once. If `ACLA_PYTHON_PATH` points to a custom Python
environment, install the controller requirements there as well:

```powershell
$env:VGAMEPAD_SKIP_VIGEMBUS_INSTALL = 'true'
& $env:ACLA_PYTHON_PATH -m pip install -r scripts/controller-requirements.txt
```

If Connect reports a missing driver in an installed application, rerun application
setup to retry driver installation.

Connect the controller and use the timed tests to bind right trigger to throttle,
left trigger to brake, and left stick X to steering in the simulator.

### Training

Once the simulator responds to the selected output, start a
practice session with automatic shifting configured, position the car near the
reference line, then select **Start training**. The policy only controls these
three axes; repositioning, restarting laps and changing gears are not automated.
**Run trained model** uses deterministic inference without updating weights.

The observation includes track-frame velocity XYZ (m/s), speed (km/h), game/car/map
category encodings, cyclic normalized lap position, rotation (radians), current
steering/pedals, the position model's reference XYZ delta (m), and a 4×4 pooled
segmentation mask per class. Telemetry is paired with the image before inference;
stale or mismatched sessions are rejected. Policies have a 32-unit tanh hidden
layer, a three-dimensional Gaussian actor and a TD(0) value head. Positive
progress close to the reference and smooth reductions in position error earn
reward; abrupt controls, excessive yaw/attitude changes, pedal overlap, distance
from the line and standing still incur penalties. New policies are untrained;
successful driving is an empirical training outcome, not a built-in capability.

Only observations captured after a controller acknowledgement can train that
action, and each image is learned from at most once. Session changes, telemetry
older than 500 ms, images older than 1.5 s, invalid inputs and large position jumps
stop the episode. A separate Python watchdog releases controls after 600 ms
without a command and requires reconnection. **Stop** and Escape in the app release
controls; Escape is not a global simulator hotkey. Checkpoints save locally every
50 learning steps and at stop, and can be imported/exported as JSON. A checkpoint
must match the current game, car, map and segmentation classes.

## Python environments for desktop scripts

All Electron-only Python entry points live under `src/py-scripts/`. We manage their dependencies with
virtual environments so that development and production builds are reproducible.

| Mode | Virtual environment location | How it is created |
| --- | --- | --- |
| Development | `acla_front/.venv/py-scripts` | `npm run start:electron` (via pre script) or `npm run setup:python -- --mode=dev` |
| Production (packaged app) | `acla_front/.venv/py-scripts-prod` (bundled as `resources/python-env`) | `npm run build:electron` (via pre script) or `npm run setup:python -- --mode=prod` |

The managed environments install everything from `src/py-scripts/requirements.txt`, including
scikit-learn, pandas and the Windows virtual controller package.

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

Uploading an iRacing live recording creates one session containing the app's saved
live samples, with `game_recorded_from` set to `iracing_live` and the original session
name. Live-session uploads do not search for, convert, or upload native `.ibt` files.
The app recording is deleted after the upload completes; a failed upload keeps the
local draft available for retry. Existing `iracing` sessions remain supported.

Native `.ibt` files can be opened separately for local recorded-session analysis.
Enable iRacing disk telemetry while driving and exit the car before opening a file
so it is finalized. Conversion runs in a worker, streams bounded
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
