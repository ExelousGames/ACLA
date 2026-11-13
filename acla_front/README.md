# Kestrel Motorsport Analyst Frontend & Desktop Companion

This package contains the React UI and the Electron desktop shell that executes all Python automation located in `src/py-scripts/`.

## Prerequisites

- Node.js 18+
- npm 9+
- Python 3.10 or 3.11 (system interpreter used to create virtual environments;
  the pinned desktop NumPy packages require Python below 3.12)

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

The sidebar's **AI Self-Driving Training** page trains a local, mixed-action
actor–critic neural policy against the running simulator. Start live telemetry,
add **Live Map** with a matching reference lap, and add **Track Vision** with
screen capture enabled. Self-driving does not require a position model. The training
page reports any missing inputs. Live Session stays mounted when switching tabs.

Track Vision supports temporal GRU position exports: it reads the model's saved
window length and keeps aligned image, reference-lap and telemetry history automatically.
Single-frame position models remain supported. See the
[position model usage](src/views/live-session/track-vision/README.md#position-model)
for the ONNX input contract and history reset behavior.

### Steering wheel output (ACC and iRacing)

AI driving uses **Steering wheel (vJoy)** output with standard DirectInput axes
independently of the physical wheel brand (MOZA, Fanatec, Logitech, Thrustmaster,
etc.). It does not intercept a physical wheel or automatically
replace that wheel's bindings. The simulator must be bound to the virtual device
for AI driving and controller takeover during training. Restore your normal controls
when driving outside the training page.

1. The Windows x64 app installer includes the signed Windows 10/11
   [vJoy driver](https://github.com/BrunnerInnovation/vJoy/releases/tag/v2.2.2.0)
   and installs it automatically with administrator approval. Restart Windows if
   prompted. For development, run `npm run setup:all` to install vJoy. In **Configure vJoy**,
   enable a free device with **X**, **Y**, and **Z** axes and at least **2 buttons**,
   then select its device number in the app.
2. Connect before launching the simulator (or restart it if the device is missing).
   Map **X → steering**, **Y → throttle**, **Z → brake**, **button 1 → shift up**,
   and **button 2 → shift down**. Steering uses the full signed
   range; each pedal has its own independent axis. Check that neutral shows zero
   throttle/brake and centered steering; invert a pedal in the simulator if needed.
3. **ACC:** save your existing controls preset, then select **Install ACC preset**
   in AI Self-Driving Training. In **Options → Controls → Wheel**, load the displayed
   **ACLA AI Training - vJoy N** preset. Restart ACC if the new preset is not listed.
   The app reads ACC's saved device GUID, binds X/Y/Z to steering/throttle/brake
   and buttons 1/2 to shift up/down,
   and saves the training preset under your Windows Documents folder in
   `Assetto Corsa Competizione/Customs/Controls`. It leaves active controls and
   existing presets intact; conflicting driving axes and shift bindings on other
   devices are removed only in the new preset. If ACC has not recorded the selected vJoy device yet, enable it, restart
   ACC, save your controls, and retry. Keep this preset loaded for controller takeover.
4. **iRacing:** use **Configure Controls**, or **Settings → Driving → Input Calibration**,
   and calibrate the vJoy steering and pedal axes. Follow the
   [iRacing calibration guide](https://support.iracing.com/support/solutions/articles/31000133335).
   Keep your manual controller configuration available for switching back.

The **Test … in 5 seconds** buttons allow time to switch to the simulator's controls
screen. They hold the chosen axis for 10 seconds, or press a shift button once for
0.3 seconds, and then release it, even while the simulator has focus. Pedals test
full travel. For steering calibration, use **Full
left**, **Full right**, then **Center**; for iRacing's 90° left measurement, use
**20% left** to calibrate a 900° total range. Accept the measurement while the timed
input is active. **Stop**, Escape in the app, disconnect, or leaving the training
page cancels a pending/active test. The hold buttons release when the app loses focus.
If ACC steering and pedals work but shifts do not, check that both shift bindings
use the same vJoy device as the axes. Install and load a fresh training preset to
replace old or duplicate shift bindings, then use the timed shift tests with ACC focused.
Shift controls are available for manual testing, simulator binding and learned
gear changes. Disable automatic gears in the simulator during autonomous training
so gear changes come from the policy.

**Connected** confirms only that the output device accepted initialization. Check
the simulator's input meters before training. Controller takeover can relay physical
wheel input; force feedback is not relayed. Both simulators use the same output backend; their
bindings and calibration are configured separately.

No extra Python package is needed for vJoy: the bridge loads the installed SDK DLL
matching the Python architecture from `Program Files\vJoy\x64` (or `x86`). For a
custom installation, set `ACLA_VJOY_DLL` to the absolute `vJoyInterface.dll` path
before launching the app. Missing drivers/axes, busy devices, and rejected writes
are reported as errors. Disconnect and bridge failures reset pedals to zero and
steering to center, and release both shift buttons before relinquishing the device.

### Shared human / AI control

The control panel has two input sources, **Human takeover** and **AI driver**, feeding
one shared vJoy output. The control owner indicator shows which source currently
owns steering, pedals and shifts. Both sources use the same simulator bindings.

After connecting vJoy, use **Human takeover** to map physical steering,
throttle and brake axes from a physical gamepad, wheel, joystick or separate pedals.
USB and Bluetooth controllers recognized by SDL are supported without a brand-specific
mapping preset. Center the wheel or stick and release pedals, triggers and buttons,
click **Detect** beside a control, then follow its prompt: steer right, press a pedal
or trigger, or press and hold a button until
detected. The device, axis/button and axis direction are filled in automatically.
Detection ignores small axis jitter and already-held buttons, and can be cancelled
or retried after its 15-second timeout. Manual selections remain available.
Map shift paddles and one **Take over / Return to AI** button to switch while the simulator
has focus. Each press switches control once; release before pressing again. The UI offers
one button that changes between **Take over** and **Return to AI**. Existing mappings use
the previous Take over binding, or the Return to AI binding if no Take over binding was saved.
Check the live axis values,
invert pedals if they read 1 at rest, and calibrate ranges in Windows controller
settings. Enable controller takeover, then choose **Start human driving** to drive
without a model, telemetry or screen capture. You can also enable takeover before
starting AI training or evaluation. Mappings save
locally; enabling takeover is explicit each time the page opens.

Takeover is latched and owns steering, both pedals and shifts until an explicit
return to AI. Starting training or evaluation while you are driving arms the AI
and keeps human ownership; use **Return to AI** when ready. A return requires an
active AI session with matching, valid observations. A blocked return press is consumed;
release and press again when AI is ready. Recovery of missing observations never
transfers ownership automatically.
The native bridge polls SDL joystick input (HID, DirectInput and XInput) while the
simulator has focus, and excludes vJoy input devices to prevent feedback. It exposes
all reported axes and buttons, plus hat/D-pad directions as extra buttons (up, right,
down, left). Connecting or disconnecting devices refreshes detection automatically.
SDL and its Python bindings are installed by `npm run setup:python` and bundled in
the desktop build. After upgrading from the old WinMM reader, detect your bindings
again because device and axis identifiers have changed.
Controllers must expose independent pedal axes; combined-trigger configurations
need a suitable Windows driver configuration. Force feedback is not forwarded.

The design uses a **single-writer arbiter with revocable ownership leases**:

```text
Physical controller → Human input adapter ─┐
                                          ├→ Control selector → Output lease → vJoy → Simulator
Live observations → AI driving policy ─────┘                          ↑
Calibration tests (only while stopped) ───────────────────────────────┘
```

`DrivingControlSelector` handles latched ownership and physical button edges.
`DrivingControlOutput` serializes every write, including calibration and Stop.
Each handoff revokes the old lease, drops its queued commands, and releases the
outgoing controls before writing the new driver's action. Stop revokes ownership
immediately and queues neutral after any in-flight write, so a late acknowledgement
cannot restart driving. The Electron/Python bridge releases controls on disconnect or failure.
`DrivingTrainingLoop` schedules the active source and keeps learning separate from
output ownership. Human control updates on the control clock even between camera
frames. Training samples acknowledged
actions once per fresh observation: AI actions use actor–critic learning, while
human actions supervise the existing actor through behavior cloning. Handover
discards the pending transition and waits for a fresh observation before learning
or resuming AI output, avoiding mixed-driver reward attribution. Human controller
state and driver identity are **not added to model inputs**; the existing telemetry
controls still describe what the simulator received. Evaluation does not update weights from either source.

Disconnecting any mapped device while takeover is enabled stops the session,
even if AI currently owns the controls. Telemetry/capture loss while AI is driving
neutralizes vJoy controls and retries automatically without ending the session or
resetting model progress. Fresh, matching observations resume AI driving. During human control, missing or
mismatched AI observations pause learning and discard the pending transition;
physical input keeps driving vJoy. Valid observations can resume learning, but
returning to AI still requires an explicit handoff. Stop, Escape in the app and
disconnect release both sources. Calibration tests are exclusive and cannot run
during either human or AI driving.

### Training

Once the simulator responds to the virtual wheel, start a
practice session with manual gears configured and both shift buttons bound,
position the car on track, and choose **New model** to select a save
location (or **Open model** to resume a file). Then select **Start training**. The policy
outputs one continuous throttle/brake value from −1 (full brake), through 0 (both
pedals released), to 1 (full throttle), plus steering and the choice to hold gear,
shift up or shift down. Positive pedal values apply throttle; negative values apply
brake, so AI output never presses both pedals together. Human demonstrations and
future action targets use throttle minus brake. Create and train a new model for
this output layout; older separate-pedal checkpoints are not converted.
Repositioning and restarting laps are not automated.
**Run trained model** uses deterministic inference without updating weights.

The AI executes its action plan on the control timer: by default, steps start at
0, 0.5, 1, 1.5, 2, 2.5 and 3 seconds after the first control acknowledgement.
The current plan continues during learning and inference. A new output replaces
the entire plan immediately and starts its own clock. A late control tick skips
expired steps; after the final step, the AI holds its pedals and steering until
the next output or input interruption. Each planned shift is pulsed only once.
Training explores every plan slot. Observed reward is shared among the steps
that actually controlled the car, weighted by their acknowledged execution time.
This includes the outgoing plan's actions while a replacement was computing;
skipped and superseded steps receive no actor gradient. Human demonstrations
continue to teach future plan slots from later acknowledged human actions.

Gear choices learn from the same driving reward as the continuous controls.
Each shift presses one paddle for 100 ms (released on the next control tick)
with at least 500 ms between shifts. Downshifts are allowed in every gear,
including neutral and reverse. Both shift directions are masked during cooldown
before sampling the immediate action and again when executing later plan steps.
Training retains the execution mask, so a suppressed shift receives no gradient.
Button releases continue even while waiting for a new camera frame.

The observation includes car-local velocity XYZ (m/s), speed (km/h), game/car/map
category encodings, cyclic normalized lap position,
yaw rate (rad/s, normalized by π and clamped to ±1), g-force XYZ
(in g, divided by `gForceScale`, default 5, and clamped to ±1), current
steering/pedals, current gear, engine RPM,
and a **384 × 216 RGB camera image** from the same Track Vision capture.
G-force retains the telemetry adapter's axis order, signs and gravity contribution,
and is included in both the current observation and the five-second history.
The camera is the only visual input: segmentation labels, depth grids, reconstructed
track ribbons and traffic boxes are not policy inputs or checkpoint matching keys.
Camera observations remain available if the segmentation, depth or position models are
missing or fail; the driving page does not request CPU depth downloads.
The camera is copied before asynchronous inference or overlays, preserving the full
capture's aspect ratio with centered black letterboxing when it is not 16:9.
A learned CNN uses four overlapping stride-2 convolutions: a 5 × 5 kernel with
16 channels, then 3 × 3 kernels with 32, 64 and 64 channels. Symmetric zero padding
is 2 pixels in the first layer and 1 in the others; each applies leaky ReLU (negative
slope 0.01). The feature maps are 192 × 108, 96 × 54, 48 × 27 and 24 × 14.
Adaptive average pooling retains an 8-column × 4-row × 64-channel grid, flattened
in Y/X/channel order. A learned linear projection maps its 2,048 values to 256 visual
features, with separate weights for each spatial position so left and right remain distinct.
A 128-unit LSTM reads these features and telemetry in chronological order, from the
oldest available history frame through the current observation. Input, forget,
candidate and output gates maintain hidden and cell states within that sequence.
Actor–critic, human demonstrations and delayed action-plan training all backpropagate
through the complete sequence, updating convolution and projection weights/biases as
well as LSTM input weights, recurrent weights and gate biases. Visual gradients accumulate
across all camera timesteps before updating the shared encoder.
CNN and LSTM inference, learning-gradient calculations, backpropagation and weight
updates run on a hardware WebGPU device. The CPU sends demonstration/plan targets or
reward, elapsed time and the sampled actions with per-slot execution weights. The GPU calculates target transforms,
masked shift probabilities, the TD advantage and clipped gradients without intermediate
readbacks. Weights, activations and training predictions stay on the GPU; only control
decisions read the small action/value output. Each training tick waits once for GPU
completion, without copying results, to keep queued work bounded during human driving.
Repeated images share convolution activations within a decision;
each later decision re-encodes images with the current learned weights. Pending
decisions retain their own activations until training and release.
Saving the model reads the current trained weights back after the entire training tick.
Training and evaluation require GPU acceleration and report an error if unavailable;
the CPU inference/training implementation has been removed. Human driving remains
available without WebGPU. Run `npx electron scripts/check-driving-gpu.cjs` for real-GPU
gradient checks using finite differences, readback checks and a full training-tick
benchmark, without connecting to simulator controls.

New models receive up to five seconds of observations, including game/car/map, telemetry,
controls, reference targets and full-resolution camera frames, in 17 elapsed-time bins
of 300 ms (the oldest bin spans the final 200 ms), using the newest observation in each
bin, followed by the current observation. Each history slot has presence and
age inputs; missing bins are skipped when assembling the LSTM sequence. Raw RGB bytes
are stored separately from scalar telemetry and shared across pending samples. The same
learned CNN processes current and historical frames. Every decision starts with zero
recurrent state. Saved configurations declare the history features.
History resets on start/stop, human/AI handover, invalid inputs, session/reference/context
changes, backwards timestamps and gaps over 1.5 seconds. The observation panel shows the
384 × 216 camera resolution. Earlier square-image CNN and pooled-image LSTM checkpoints
and configurations are incompatible with this architecture. Create a new CNN–LSTM model and retrain
from scratch; existing files are not automatically migrated.
Heading, pitch and roll angles are not required for training or used as policy features.
Telemetry is paired with the image before inference;
stale or mismatched sessions are rejected. New policies have a 128-unit LSTM,
a three-dimensional Gaussian actor, a three-way categorical shifting head
and a TD(0) value head. **Driving aggression** shapes the reference lap's speed goals:
AI training owns its speed-profile calculation independently of Live Phrases.
0–50 matches the braking-speed calculation and tagged corner boundaries used by Live Phrases,
with `liveBrakingAdjustment = 100 - 2 × trainingAggression`. Training 0 corresponds to
Live Phrases 100%, 25 to 50%, and 50 to the unchanged reference at 0%.
Without tagged corners, both use the unchanged reference speed profile.
Above 50, every reference speed is multiplied by `1 + (trainingAggression - 50) / 100`:
75 targets 25% higher speed, and 100 targets 50% higher speed. Targets are interpolated
between the original samples, including across start/finish. Lap and corner snapshots
are retained for pending actions; changing the selected lap or map requires a fresh episode.
The selected lap must contain raw XYZ coordinates, `expert_optimal_speed` (km/h),
`expert_optimal_throttle` and `expert_optimal_brake` (both between 0 and 1).
Aggression, reference speed and target speed are included in observations and memory.
The acceleration/deceleration reward credits measured speed changes toward the goal
on every fresh forward-driving transition, including held pedals. It uses one fixed
goal at the destination position for both speed readings:
`gapClosed = abs(previousSpeed - targetSpeed) - abs(currentSpeed - targetSpeed)`.
This credits useful acceleration or braking even far from the goal. Moving away is
penalized, and overshooting reduces credit. Both faster and slower speeds also incur
a penalty for their remaining difference from the goal, including while holding speed.
The correction rate uses elapsed telemetry time, not camera processing time:
`correction = clamp(gapClosed / (3.6 × telemetrySeconds × accelerationTolerance), -1, 1)`.
The speed-following penalty is `min(abs(currentSpeed - targetSpeed) / speedTolerance, 1)`.
With default weights, the combined acceleration/deceleration reward is
`100 × forwardProgress × (correction - speedPenalty)`.
`accelerationTolerance` defaults to 3 m/s² and `speedTolerance` to 20 km/h.
Holding the target speed has no penalty; holding an incorrect speed is penalized.
Correction credit can be outweighed by the remaining speed penalty.
Forward progress is capped at 0.02 per observation; stationary or reverse driving
earns no timing bonus. Both ends of a transition use the aggression that produced
the action, even if the slider changes before scoring. There is no separate raw-speed
or pedal-matching reward. The previous speed-window and pedal-change-gated reward
are removed. Reference error is not a policy or history input, and distance from the
reference line does not affect rewards. Line-following and approach rewards and
penalties have been removed, along with their configuration settings. Create a new
model and retrain; earlier reference-offset and pedal-event aggression checkpoints are rejected without migration.

The normalized-position progress reward uses the signed lap-position change, corrected
for start/finish crossings, divided by elapsed telemetry seconds. With
`rate = abs(positionChange / telemetrySeconds)`, it adds
`0.25 × progressWeight × cappedSignedProgress × rate / (rate + 0.01)`.
Faster forward progress earns more and negative progress incurs a penalty. Zero speed
or unchanged normalized position instead incurs `-0.25 × progressWeight × 0.01 × telemetrySeconds`,
or -0.25 reward per second with default weights. This stationary penalty scales with
elapsed telemetry time so its accumulated value is independent of sampling frequency.
Positive rewards also require nonzero speed and a non-reverse
gear. Signed progress uses the existing 0.02-per-observation cap. At 0.01 lap/second
the rate factor is one half; it approaches 1 as the rate increases, keeping the
bonus below `0.25 × progressWeight × cappedForwardProgress`.
Distance weighting keeps accumulated reward independent of sampling frequency
at a constant rate below the existing progress cap. Camera processing delays do
not affect the measured rate. The reward appears separately under **Reward signal**
and contributes to the total used for learning.
**Impact penalty** monitors live horizontal local velocity independently of camera
and GPU processing. A bounded history checks telemetry intervals 20–500 ms apart
inside each training transition, so a brief bounce can be detected even when AI
images are more than 500 ms apart or the car is already sliding by the next image.
Only telemetry up to the image's paired timestamp is used. The strongest detected
impact in the interval contributes one penalty; later intervals do not reuse it.
A shared elliptical allowance combines ordinary
acceleration/braking up to 2.5 g with steering-related sideslip. The full horizontal
velocity change is scored against that allowance, with velocity noise removed once,
so diagonal jolts do not disappear into separate speed and direction allowances.
Smaller excess impulses now reach the
50% confidence threshold sooner; confidence is heuristic, not a collision probability.
At every speed, speed loss with the brake below 5% uses a 3 m/s² coasting allowance
and a more sensitive impulse score. With steering also below 5%, this score combines
speed loss and sideways deflection to detect glancing impacts. Acceleration adds no
extra evidence to that score. Brake and steering inputs anywhere inside an interval
are retained, so a pedal press or steering correction between images is respected.
Missing telemetry, session changes and pauses do not create synthetic impacts.
Detected impacts subtract `5 × confidence × severity`, with severity reaching 1 at
10 m/s of excess velocity change. Vertical bumps and car damage telemetry are excluded;
spins and extreme braking can still resemble impacts.
**Off-track penalty** subtracts `5 × telemetrySeconds` when the current image's paired
telemetry is off track. iRacing uses `Graphics_player_track_surface === 0` (`irsdk_OffTrack`);
pit states are not off track, and missing/invalid or not-in-world values are unavailable.
ACC checks all four tyre contact points against the matching Live Map's left and right
boundaries in adapter X/Y coordinates. The penalty applies only when all four tyres
are outside; touching a boundary counts as on track. Missing tyre coordinates, missing
or mismatched maps, and unusable boundary sections leave this penalty unavailable.
Boundary interpolation handles start/finish wrapping but does not bridge gaps larger
than 2% of a lap. Detection and its penalty appear separately under **Reward signal**.
The penalty scales with elapsed telemetry time, including while stationary, and does
not add policy inputs or change the saved model format.
The total reward combines acceleration/deceleration timing, normalized-position progress,
off-track and impact penalties, clamped to the configured total limit. Yaw rate
remains a policy input but does not incur an instability penalty. The removed reward
settings `yawWindowMs`, `yawDeadband` and `yawPenalty` are no longer accepted in model
JSON. New policies are untrained;
successful driving is an empirical training outcome, not a built-in capability.

Only observations captured after a controller acknowledgement can train that
action, and each image is learned from at most once. Session changes, telemetry
older than 500 ms, images older than 1.5 s, invalid inputs and large position jumps
pause learning and discard the pending transition while the loop waits and retries.
AI controls stay neutral until fresh inputs match the model, original session and
reference. Recovery starts a new learning transition without using the interrupted sample.
Pauses between training commands keep the virtual controller connected with its
last applied inputs. **Stop** and Escape in the app release
controls; Escape is not a global simulator hotkey. Select **New model** to choose a
model file with the desktop save dialog, or **Open model** to resume an existing file.
The complete model saves back to that file every 50 learning steps by default and
at stop. **Save model** saves immediately. The selected path is shown on the page. A model
must match the current game, car, map and observation configuration.
All three local velocity and g-force components and yaw rate must be present in telemetry. Gear and RPM do not gate observation validity, so stalled-engine readings can still be used. Missing or non-finite gear/RPM readings use neutral gear and zero RPM as policy inputs. Pitch and roll rates are not used or required.

### Self-contained model JSON

The settings above are defaults for a new model, defined in
[`driving-model-defaults.json`](src/views/ai-driving-training/driving-model-defaults.json).
Each model file contains `format`, `algorithm`, `configuration`, and `model`.
The desktop dialogs label these files **Kestrel self driving model** and default to
`kestrel-self-driving-model.json`. The `format` is `kestrel-self-driving-model`.
`configuration` owns the ordered input features and their normalization/encodings,
camera resolution/resizing, CNN layers/padding/activation, spatial pooling and projection size,
history feature selection and timing, LSTM size,
action plan times, shift timing, exploration and learning parameters, reward
parameters, input freshness limits, checkpoint frequency, and aggression.
`model` owns the game/car/track, per-timestep input count, learning step count,
convolution weights (`cameraWeights`: layers, output-channel rows, then flattened
kernel Y/X/input-channel weights and a final bias), linear projection weights
(`cameraProjectionWeights`: output rows, pooled Y/X/channel columns and a final bias),
LSTM gate weights (`lstmWeights`: input/forget/candidate/output rows, with input features,
previous hidden values and bias columns) and the actor/critic output matrix (`outputWeights`). An untrained document has `model: null`.
The format and algorithm names identify supported operations; there is no numeric model version.

Training and evaluation reload and validate JSON from the selected file every time
they start. Choose a file before starting training. Observations, memory,
inference, learning and the UI use the saved configuration. Changing the defaults
only affects newly created models. Files preserve configuration and weights together;
configuration-only files can also be opened and saved before training. Writes are
serialized and replace the file only after the complete new contents have been written.
The app remembers the selected path while it is running; after restarting, use
**Open model** to select it again. Model files require the desktop app.
Validation checks supported encodings, finite values, resource limits, timing,
feature names/order and the matrix dimensions derived from the configuration.
Missing settings are rejected rather than filled with current defaults.

Models are never saved in browser local storage. The old `acla.self-driving.model`,
numbered checkpoints and separate aggression setting are deleted when the training
page loads, without migration or backup. Select **New model** to choose a file and
start again with the default configuration and fresh weights.

## Python environments for desktop scripts

All Electron-only Python entry points live under `src/py-scripts/`. We manage their dependencies with
virtual environments so that development and production builds are reproducible.

| Mode | Virtual environment location | How it is created |
| --- | --- | --- |
| Development | `acla_front/.venv/py-scripts` | `npm run start:electron` (via pre script) or `npm run setup:python -- --mode=dev` |
| Production (packaged app) | `acla_front/.venv/py-scripts-prod` (bundled as `resources/python-env`) | `npm run build:electron` (via pre script) or `npm run setup:python -- --mode=prod` |

The managed environments install everything from `src/py-scripts/requirements.txt`, including
scikit-learn and pandas. Steering wheel output uses the vJoy driver bundled with the Windows installer
and does not require an additional Python package.

### One-command setup

With Node.js, npm and Python installed, run from `acla_front`:

```bash
npm run setup:all
```

This installs Node dependencies (including development/build tools), the desktop
Python packages in `.venv/py-scripts`, and the vision packages from
`scripts/vision-requirements.txt` in a separate `.venv/track-vision` environment.
It also prepares the local vision runtime and downloads/exports the bundled depth
models using that vision environment. The first run requires internet access and
can take several minutes; subsequent runs reuse existing environments and models.

On Windows x64, it downloads and verifies the pinned vJoy installer, then installs
the driver with a Windows administrator prompt. Existing driver/SDK installations
are preserved. Setup waits for installation to finish, reports failures, and tells
you if a restart is needed without restarting Windows automatically. Enable the
device's X/Y/Z axes in **Configure vJoy** and configure simulator bindings as above.
Other operating systems skip the vJoy driver; macOS also skips desktop Python setup.

`setup:all` prepares the development environment. `build:electron` still prepares
the separate production Python environment when packaging the app.

### Individual setup commands

```bash
# Install Node dependencies
npm install

# Prepare the development Python environment (optional – start:electron runs it automatically)
npm run setup:python -- --mode=dev

# Prepare the isolated vision Python environment and depth models
npm run setup:python -- --mode=vision
npm run setup:vision-models
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
4. Downloads and verifies the pinned vJoy 2.2.2.0 installer in the Windows
   `beforePack` hook, then includes it and its MIT license in `resources/vjoy`

The Windows installer targets x64, installs for all users, and requests administrator
access to install the driver. End users do not need an internet connection for vJoy
setup. Existing installations with both the driver service and SDK in the standard
location are preserved. Driver failures stop setup with an error and a log at
`%TEMP%\Kestrel-vJoy-setup.log`; restart requests appear on the finish page without
forcing a reboot (silent setup returns 3010). Uninstalling Kestrel leaves the shared
vJoy driver installed. Simulator axis bindings still need to be configured as above.

Use `npm run setup:vjoy` to cache the driver before an offline build. The binary is
git-ignored; every Windows packaging run checks its SHA-256 and fails if it differs
from the pinned release. Direct `electron-builder` invocations use the same hook.

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
