# Detailed Extensible Recording Process Architecture

## Scope and Decisions

This phase implements only the ACC-specific live telemetry reader. ACC and every future game use the same game-independent writing and saving pipeline; adding a game adds a reader, not another writing or saving implementation. The architecture keeps simulator selection separate from reading capability so the product can continue to show entries for iRacing, Assetto Corsa, and future simulators without implying that their telemetry reader exists.

Terminology is strict throughout this plan:

- **Reading** means acquiring telemetry from a running game, mapping its raw fields into the standard telemetry fields, validating the standard sample, and applying game-specific state and session-continuity rules. Reading is game-specific; ACC has its own reading process, and every future game supplies its own reader.
- **Writing and saving** mean accepting validated samples, assigning transport-only commit sequences, batching writes, creating and finalizing the JSONL file, and reporting committed samples. Writing and saving are shared across every game. Transport metadata is never merged into a saved telemetry row.
- **Collecting** and **recording** both mean the complete read-plus-save lifecycle. Neither term is used for the game-specific reading step alone.

The key scope decisions are:

- Each game has its own telemetry reader because games expose telemetry through different SDKs, shared-memory layouts, processes, and runtimes. Readers must not be treated as one shared game-neutral implementation.
- ACC uses its dedicated Python reader because `pyaccsharedmemory` and the existing `ACCMemoryExtractor.py` are Python-based. The ACC reader's JavaScript utility wrapper only connects that Python reader to the shared worker protocol.
- iRacing, Assetto Corsa (`ac`), and any future simulator entries are catalog/UI entries only in this phase. Do not add an SDK dependency, telemetry probe, reader, field mapper, fixture set, or native-module packaging work for them. Their future readers must use the existing shared writing and saving pipeline unchanged.
- The application has one game identifier type, `DesktopGame`. Recording support is not encoded in a second game union or capability list. A start request resolves the selected game's reader entry point; a missing entry point, runtime, script, or other required launch path returns `{ ok: false, error: { type: 'unsupported-recording-game', message } }`. No worker, message channel, file, Python process, SDK handle, or recording directory is created when resolution fails.
- Each game-specific reader owns telemetry acquisition, mapping and any required unit/coordinate conversion from its raw SDK/shared-memory fields into the standard telemetry fields, standard-contract validation, state mapping, and session-continuity rules. Once it has produced the standard telemetry object, it must not rename fields or change values. It does not own commit sequencing, write batching, file creation, serialization, or finalization.
- The `RecordingSessionManager`, reader protocol, writer and saving path, recorded-file reader, view worker, preload bridge, renderer integration, and startup/shutdown lifecycle are shared. Every game-specific reader sends validated events to the same writer and view implementations; adding a game must not create game-specific writing or saving paths.
- The existing flat output of the modified `src/py-scripts/ACCMemoryExtractor.py` is adopted as the application-wide standard. The exhaustive 240-key contract is recorded in [`telemetry-fields.md`](telemetry-fields.md). ACC already emits these names. Every future reader maps its source fields to this same contract, omits unsupported standard fields, and must not introduce aliases such as `speedKph`, `lapNumber`, `timestampMs`, or `worldPosition`. No game persists an envelope around a sample.
- Adding telemetry reading for another simulator requires a separate implementation plan and release gate. This plan preserves the reader boundary needed for that later work but does not preselect a language, SDK, or transport for it.

The existing desktop simulator catalog is the single source of game identifiers:

```ts
type DesktopGame = 'ac' | 'acc' | 'iracing'; // existing detection/UI entries
```

Membership in `DesktopGame` means the application can identify and present the simulator; it does not promise that live recording exists. A game becomes recordable when its reader entry point and all required launch paths are shipped. Every accepted recording session carries the immutable selected `DesktopGame`; in this phase every successful session is ACC because ACC is the only reader present. Downstream processes consume both the shared transport contract and the same standard flat telemetry schema for every game. A later reader reuses the same writing, saving, recorded-file reading, viewing, upload, and lifecycle implementation without introducing another telemetry schema or extending a second game union.

## Process and Module Location

The Electron shell integration points already exist and are extended in place:

```text
public/
└── electron.js                 # Electron bootstrap, IPC registration, and saved-file reader launcher
src/
└── common/
    └── preload.js              # renderer-facing live-recording and saved-file adapters
```

Place the shared pipeline and the implemented ACC reader at:

```text
electron/
└── recording/
    ├── recording-session-manager.js
    ├── recording-protocol.js
    ├── telemetry-contract.js
    ├── readers/
    │   ├── reader-contract.js
    │   └── acc/
    │       ├── acc-reader-config.js
    │       ├── acc-reader-worker.js
    │       └── acc-python-reader.js
    └── workers/
        ├── recorded-file-reader-worker.js # offline reader for finalized JSONL files
        ├── writer-worker.js
        └── view-worker.js
```

The `electron/recording/` tree above is the target location introduced by this plan; it does not exist in the current codebase yet. In particular, `recorded-file-reader-worker.js` is a planned Node utility-process entry point, not the name of an existing module. The current saved-file implementation is `src/py-scripts/read_telemetry_data.py`, called by `LiveSessionContext.readRecordedTelemetry()` through the generic Python bridge. The target worker replaces that Python path for JSONL reading; it does not replace `acc-reader-worker.js`, which reads live telemetry from the running game.

`recording-protocol.js` defines a transport-safe failure result for recording startup instead of custom JavaScript error classes:

```ts
type RecordingStartFailure = {
  ok: false;
  error: {
    type:
      | 'malformed-recording-game'
      | 'unknown-recording-game'
      | 'unsupported-recording-game';
    message: string;
  };
};
```

`malformed-recording-game` covers a missing or non-string input, `unknown-recording-game` covers a string outside `DesktopGame`, and `unsupported-recording-game` covers a missing reader entry point, runtime, script, SDK, or other required launch path for a known game. Consumers branch on `error.type`; `error.message` is descriptive and must not be parsed. The main process returns this plain object through Electron IPC, the preload validates and forwards it unchanged, and the renderer does not reconstruct or receive a custom `Error` subclass.

`public/electron.js` remains the Electron bootstrap. After `app.whenReady()`, it registers the recording IPC handlers, but it does **not** create a recording manager, utility process, Python extractor, message channel, or recording file. The first `startRecordingSession` request for a known `DesktopGame` from the active live-workspace renderer lazily creates the single, side-effect-free `RecordingSessionManager`. The manager resolves the requested reader before starting workers; in this phase only ACC resolution succeeds. Add `electron/**/*` to Electron Builder's `files` list.

The manager constructor must be side-effect free. Only `RecordingSessionManager.startSession()` may resolve a reader and then fork the complete recording pipeline, and only after the IPC boundary confirms that the request belongs to the active live-workspace renderer and contains a known `DesktopGame`. Application launch, window creation, route navigation, React provider mounting, game-process detection, and draft restoration must produce zero reader, writer, view, or ACC extractor processes. A failed reader resolution may leave an idle manager object, but it creates no pipeline resource. The manager may remain idle after recording, but none of those processes may remain.

```text
                     Electron main process
                RecordingSessionManager
                lifecycle and ownership only
                            │
      ┌─────────────────────┼──────────────────────┐
      │                     │                      │
 ACC Reader ── frames ──► Shared Writer ── progress ──► Shared View
      │                     │                      ▲       │
 Python reader         writes/saves JSONL          │       │
      └──────────── frames and states ─────────────┘       │
                                                          │
                                                direct MessagePort
                                                          ▼
                                             Preload bridge
                                      `src/common/preload.js`
                                                          │
                                                validated callbacks
                                                          ▼
                                            Live workspace renderer
```

The selected live telemetry reader is game-specific. For ACC, `acc-reader-worker.js` starts and owns the Python reader; its Python child handle is private reader state. Telemetry never passes through the Electron main process. A game without a shipped reader never enters this diagram because reader entry-point resolution returns an `ok: false` result before any pipeline resource is created.

When another simulator's recording support is implemented, it supplies a different reader that may use a different language, SDK, process model, or transport. That reader must map its raw data to [`telemetry-fields.md`](telemetry-fields.md), implement the shared reader protocol, and emit validated `SourceFrame` events whose `sample` uses only the standard names. The same message-channel roles, writer and saving behavior, view worker, renderer port, manager, and shutdown sequence remain in use.

## Shared Telemetry and Reader Contract

Control and lifecycle fields are transport metadata, not telemetry fields. Each reader populates the standard `Graphics_status` field and also maps that status to a game-neutral transport state so the shared workers can coordinate lifecycle. The pipeline does not remove, rename, or replace `Graphics_status` in the sample. Define the transport types in `telemetry-contract.js` and mirror them in renderer TypeScript declarations:

```ts
type TelemetrySourceState =
  | 'connecting'
  | 'live'
  | 'paused'
  | 'replay'
  | 'unavailable'
  | 'completed'
  | 'fatal';

type TelemetryJsonValue =
  | string
  | number
  | boolean
  | null
  | TelemetryJsonValue[]
  | { [key: string]: TelemetryJsonValue };

// Runtime validation restricts keys and value types to telemetry-fields.md.
type StandardTelemetrySample = Record<string, TelemetryJsonValue>;

type SourceFrame = {
  game: DesktopGame;
  sourceState: TelemetrySourceState;
  recordable: boolean;
  sessionIdentity: string | null;
  sample: StandardTelemetrySample;
};
```

`sample` is a `StandardTelemetrySample`: a flat object whose keys are limited to the exhaustive catalog in [`telemetry-fields.md`](telemetry-fields.md). Examples include the already established `Physics_speed_kmh`, `Graphics_completed_lap`, and `Graphics_normalized_car_position`. All consumers and all games use those exact names. `Static_track`, `Static_car_model`, and `Graphics_session_type` remain the static/session fields; the pipeline does not create `track`, `car`, or `sessionType` aliases. A game that cannot provide a standard field omits it instead of creating a replacement.

`game`, `sourceState`, `recordable`, and `sessionIdentity` exist only in the in-memory worker message. `sample` is the only part written to disk or sent as a telemetry row during upload. The writer must serialize it as one compact, flat JSON object without adding an envelope, schema version, sequence, timestamp, session identity, game, or duplicated source tick:

```ts
type PersistedTelemetrySample = StandardTelemetrySample;
```

The shared writer persists only frames with `recordable: true`, `sourceState: 'live'`, the session's accepted `sessionIdentity`, and a sample that passes the standard exact-key/type validator. Status-only events and reader-control messages such as ACC's `{"available":false}` are lifecycle information and never persisted. All games therefore use the same flat row shape. Draft restoration, validation, deletion, analysis, and upload consume those rows unchanged. Upload metadata carries the selected `game_recorded_from` outside the standard telemetry rows.

Each game-specific reader conforms to this lifecycle interface:

```ts
interface TelemetryReader {
  readonly game: DesktopGame;
  start(emit: (event: ReaderEvent) => void): Promise<void>;
  pause(): void;
  resume(): void;
  stop(): Promise<void>;
}

type ReaderEvent =
  | { type: 'frame'; frame: SourceFrame }
  | { type: 'state'; state: TelemetrySourceState; reason?: string }
  | { type: 'complete'; reason: string }
  | { type: 'fatal'; error: string };
```

`pause()` and `resume()` apply the shared writer's backpressure commands to the game-specific telemetry transport; they do not represent the game's pause state. `stop()` is idempotent, removes subprocess or SDK listeners, and resolves only after the reader can no longer emit. None of these control operations mutates `frame.sample`.

## Manager Construction and State

`RecordingSessionManager` receives injected Electron dependencies and one lazy reader-config resolver:

```js
new RecordingSessionManager({
  utilityProcess,
  MessageChannelMain,
  getMainWindow,
  getReaderLaunchConfig,
  recordingDirectory: path.join(app.getPath('userData'), 'acla-temp'),
});
```

All five properties are required. The constructor validates shapes and that `recordingDirectory` is a non-empty absolute path, then stores them without invoking callbacks, forking, creating channels/directories, or resolving Python.

`getReaderLaunchConfig(game)` is called only after a start request has passed renderer, live-session, and `DesktopGame` validation. It resolves the game-specific worker entry point and everything that worker needs to launch. Its shared output is:

```ts
type ReaderLaunchConfig = {
  game: DesktopGame;
  readerEntryPath: string;
  readerOptions: Record<string, unknown>;
};

type ReaderLaunchConfigResult =
  | { ok: true; config: ReaderLaunchConfig }
  | {
      ok: false;
      error: {
        type: 'unsupported-recording-game';
        message: string;
      };
    };
```

For ACC, `readerOptions` contains `{ runtime: 'python', pythonExecutable, scriptDirectory, scriptName: 'ACCMemoryExtractor.py' }`. The shared manager treats these options as opaque and passes them to the resolved reader.

The resolver follows a convention such as `electron/recording/readers/<game>/<game>-reader-worker.js` and checks that the resolved entry is a packaged regular file. It then asks the discovered reader's co-located launch-config producer to resolve that reader's required runtime paths. A missing reader entry, Python executable, script directory, script, SDK, or equivalent launch piece returns `{ ok: false, error: { type: 'unsupported-recording-game', message } }`. The manager checks `ok` before creating a session ID, worker, channel, directory, or file and passes the failure result upward unchanged. This is the only recording-support check: there is no duplicate supported-game union, `game === 'acc'` guard, or capability registry. The ACC reader wrapper verifies that config `game` is ACC and that its opaque options declare the Python runtime before starting the Python reader.

The manager owns worker handles and the complete recording lifecycle only. It does not implement game telemetry reading, receive telemetry, write samples, or retain the writer's path.

```ts
type ManagedRecordingSession = {
  sessionId: string;
  game: DesktopGame;
  status: 'starting' | 'running' | 'stopping' | 'terminated';
  ownerWebContentsId: number;
  reader: UtilityProcess;
  writer: UtilityProcess;
  view: UtilityProcess;
  readyWorkers: Set<'reader' | 'writer' | 'view'>;
  writtenSamples: number;
  stopPromise: Promise<StopResult> | null;
};
```

`game` is retained because it authorizes/scopes control and lifecycle replies; it is not inferred from process names or file paths. `filePath`, source handles, `sessionIdentity`, transferred ports, and file streams remain excluded. Starting while another session is starting, running, or stopping returns a conflict, regardless of game.

## Startup and Shutdown

Startup sequence:

1. Validate that the request came from the current main live-workspace renderer, then validate `config.game` before manager creation. A missing, non-string, or otherwise malformed value returns `{ ok: false, error: { type: 'malformed-recording-game', message } }`; a well-formed string outside `DesktopGame` returns the same shape with `type: 'unknown-recording-game'`. Verify that the selected game matches the renderer's active detected game and live session.
2. Resolve the selected game's reader launch config. If the reader entry point or any required launch path is missing, return `{ ok: false, error: { type: 'unsupported-recording-game', message } }`. This failure creates no session ID, worker, channel, directory, or file.
3. Generate an opaque, globally unique `sessionId`; do not construct a filename.
4. Fork the shared writer and view utilities plus the resolved game-specific reader utility with `stdio: 'pipe'` and separate service names. A synchronous fork failure terminates utilities already created by this attempt.
5. Attach lifecycle/error/exit listeners before initialization.
6. Construct four `MessageChannelMain` pairs: reader/writer, reader/view, writer/view, and view/renderer.
7. Initialize writer with `sessionId`, `game`, and the recording directory; transfer its reader-input and view-progress endpoints. The writer chooses the filename and opens it exclusively. No schema version or envelope field is added to telemetry rows.
8. Initialize view with `sessionId` and `game`, transfer its reader-input, writer-progress-input, and renderer-output endpoints, and transfer the renderer endpoint through `webContents.postMessage`.
9. Wait for writer file-open readiness and the view/renderer direct-port handshake.
10. Initialize the selected reader with `sessionId`, `game`, the matching reader launch config, and the remaining writer/view endpoints.
11. In this phase the resolved ACC reader starts its Python telemetry process, accepts its initial session identity, and reports ready only after the Python process emits a valid, recordable live frame.
12. Resolve with `{ ok: true, sessionId, game, filePath, startedAt }`, relaying the writer-owned path without adding it to manager state.

If the reader cannot emit a recordable frame within the startup deadline, the entire group is shut down and the exclusively created empty file is deleted by the writer before startup rejects. Once a path has been published, cleanup never silently deletes it.

Shutdown is idempotent and ordered:

1. Assign the shared `stopPromise` and mark the session `stopping`.
2. Ask the reader to stop accepting source data.
3. The reader stops and awaits its game-specific telemetry transport, sends end-of-stream directly to writer and view, and acknowledges.
4. Writer finishes queued writes, closes JSONL, and reports `{ filePath, writtenSamples }`.
5. View flushes its final display batch, closes ports, and acknowledges.
6. Manager waits for all three acknowledgements, with one five-second forced-cleanup deadline.
7. Kill remaining workers, remove listeners, set `terminated`, and clear the active reference.
8. Emit one ended result to the owning renderer.

An unexpected exit or fatal event from any worker uses the same group shutdown. On application quit, `public/electron.js` prevents the first quit only when the lazy manager has an active session, awaits shutdown, then quits through the existing guard.

## Game-Specific Readers

### Reader protocol responsibilities

There is no shared game-neutral live telemetry reader implementation. Each game has a dedicated reader, but every reader must implement the same lifecycle and port protocol so it can use the shared writer and view. A reader's responsibilities are:

- Verify that its game and reader launch config agree before opening the game's telemetry source.
- Validate every source event against the shared contract.
- Accept the first recordable frame's non-empty `sessionIdentity` and use the game's continuity rules to complete the session before frames from another identity can enter the pipeline.
- Map raw source values to the names, types, meanings, units, and coordinate conventions in [`telemetry-fields.md`](telemetry-fields.md), omitting unavailable standard fields and rejecting attempts to add other telemetry keys.
- Emit validated frame and state events directly to both the shared writer and shared view without renaming standard telemetry keys, changing mapped values, assigning commit sequences, batching writes, creating files, or serializing rows.
- Apply `pause` and `resume` commands from the shared writer to its telemetry transport without owning the shared backpressure thresholds.
- Treat reader completion and fatal events as lifecycle signals, never as telemetry rows.

Game-specific telemetry acquisition, raw-to-standard field mapping, state mapping, and continuity rules stay inside that game's reader. Standard field validation, commit sequencing, write batching, backpressure thresholds, serialization, file ownership, and finalization stay shared. Every reader emits the same transport event shapes and the same standard sample field names. The writer independently enforces one game and one accepted `sessionIdentity` without persisting either value inside the telemetry row.

### ACC Python reader

`acc-reader-worker.js` is the ACC-specific utility wrapper. It owns `acc-python-reader.js`, which starts the Python subprocess because ACC telemetry reading depends on `pyaccsharedmemory`. Together they form the ACC reader. It:

- Launches `ACCMemoryExtractor.py` in stream-only mode with the supplied executable/script directory.
- Parses stdout with a line reader so split or combined chunks are valid.
- Validates the flattened object against [`telemetry-fields.md`](telemetry-fields.md) and places that same already-standard object on `SourceFrame.sample`; ACC requires no additional field mapper, alias projection, or unit conversion.
- Reads `Graphics_status` without removing or replacing it, and separately maps its live/pause/replay/off states plus the `{"available":false}` control line to shared source states.
- Applies the shared writer's pause/resume commands to Python stdout.
- Keeps the extractor alive while ACC is paused, replaying, temporarily unavailable, or reports `ACC_OFF`.
- Applies the existing seven-field ACC continuity classifier after an unavailable frame. A continuity break emits `complete`; `ACC_OFF` alone does not.
- Terminates and awaits the Python reader process during `stop()`.

The shared writer, view, recorded-file reader, renderer bridge, and manager do not know that ACC uses Python. A future game's live telemetry reader may use Node.js, Python, a native SDK, or another runtime without changing those shared components.

## Deferred Simulator Entries

iRacing, Assetto Corsa, and future simulators remain visible wherever the product presents its supported or planned simulator catalog. In this phase those entries are capability placeholders, not telemetry implementations.

The boundary is enforced as follows:

- Keep the existing `DesktopGame`, detection, label, and limited-workspace entries for iRacing and Assetto Corsa. Future entries follow the same UI path.
- Do not add a second game type, `game === 'acc'` implemented-reader guard, or simulator capability registry. The presence and successful resolution of the conventional reader entry point is the recording capability.
- A start request for any known `DesktopGame` follows the same API. ACC resolves successfully. In this phase iRacing, Assetto Corsa, and future catalog entries receive `{ ok: false, error: { type: 'unsupported-recording-game', message } }` from the missing reader path or another missing launch dependency.
- The renderer handles that failure result and presents a clear “recording coming soon” state. It does not predict support from a second allowlist or parse `error.message`.
- The Electron IPC handler still distinguishes malformed values with `error.type: 'malformed-recording-game'` and unknown simulator IDs with `error.type: 'unknown-recording-game'` before resolution. For a known game, the manager checks the resolver output and passes through a missing-piece failure with `error.type: 'unsupported-recording-game'`. Exact error messages are not API contracts.
- Do not create placeholder reader entry points or dormant implementations for iRacing or another simulator.
- Do not install or evaluate an iRacing SDK, create an iRacing reader or field-mapping file, add captured SDK fixtures, or change ASAR/native-module packaging in this phase.
- Process detection may continue to recognize existing `DesktopGame` entries for navigation and UI messaging. Recognition is not a live-telemetry reading probe and must not start telemetry access.

When another simulator is prioritized, add its conventional reader entry point and co-located launch-config producer without changing the generic resolver or introducing another game type. If it is not already in the catalog, add its identifier once to `DesktopGame`. Its separate plan must choose and validate the SDK/transport, define the mapping from raw source fields to the existing standard fields and its session identity rules, add reader and fixture tests, prove its standard samples work with the unchanged shared writer and saving path, and prove the packaged runtime. It must not define a new telemetry field set. The manager, reader protocol, standard telemetry contract, writer and saving path, recorded-file reader, view, renderer bridge, upload path, and lifecycle are reused; only the game-specific live telemetry reader, raw-to-standard mapping, and launch requirements are newly implemented.

## Writer Utility

There is one game-independent writing and saving implementation shared by every game. Its contract accepts validated `SourceFrame` messages using the standard telemetry fields without branching into per-game write paths; ACC is the only live telemetry reader implemented in this phase. The shared writer never reads telemetry from a game, maps raw fields, or invents telemetry field names. It:

- Receives `game` at initialization and rejects events whose frames disagree.
- Accepts only `recordable: true`, `sourceState: 'live'` frames with the session's first accepted non-empty `sessionIdentity`; state-only and non-recordable events are never persisted.
- Chooses a unique JSONL path below the assigned recording directory and opens it with exclusive-create semantics.
- Retains the path as writer-owned state; the manager never mirrors it in the managed session.
- Validates the transport shape, one game, and one session identity, then assigns monotonically increasing commit sequence numbers for acknowledgements only. Those numbers are not written into telemetry rows.
- Batches writes at 100 ms or 30 samples, whichever occurs first.
- Owns the shared backpressure policy: pause the reader above 300 uncommitted samples and resume it below 120.
- Writes each batch with one stream operation and acknowledges only after the callback succeeds.
- Reports commit sequence ranges and committed counts directly to the view, and lifecycle summaries to the manager.
- Ends the stream only after all earlier writes complete and reports the same path in `ready` and `finalized`.
- On failed startup before the path is published, closes and removes its empty file when instructed to roll back.
- Treats serialization, containment, stream, field-contract, identity, or commit-sequence errors as fatal.

For every accepted frame, regardless of game, the stream operation serializes exactly `frame.sample`. It does not serialize the surrounding `SourceFrame`, and it produces the same standard flat object shape that the current upload endpoint already accepts.

Filename generation may include the game for diagnostics, but correctness and game detection never depend on parsing the filename.

## Recorded-File Reading

The recorded-file reader is the offline consumer of a JSONL file that the shared writer has already closed. It is unrelated to acquiring telemetry from a running simulator. Its job is to open an existing recording, identify its persisted format, parse and validate it incrementally, and provide rows and progress to validation, restoration, analysis, or upload consumers.

### Exact location, ownership, and migration

| Concern | Current implementation | Target implementation in this plan |
| --- | --- | --- |
| File parser | `src/py-scripts/read_telemetry_data.py` | `electron/recording/workers/recorded-file-reader-worker.js` |
| Launcher/owner | `LiveSessionContext.readRecordedTelemetry()` calls the generic `runPythonScript` bridge and filters global Python messages by `shellId` | `public/electron.js` validates the saved-file request, forks one short-lived utility for its `readId`, and owns cancellation/cleanup in a saved-read registry |
| Renderer bridge | `src/common/preload.js` exposes generic Python start/message/end functions | `src/common/preload.js` exposes only the saved-file start, event, and cancel operations and validates events before callbacks run |
| Current caller | `LiveAnalysisSessionRecording.handleUpload()` reads the whole file before chunking the HTTP upload | Upload, draft restoration, and analysis receive the same standard flat rows; upload forwards bounded chunks without changing any telemetry key or value |

The target worker is deliberately outside `readers/` because that directory contains game-specific **live** telemetry readers. It is also not owned by `RecordingSessionManager` and is not a fourth member of the reader/writer/view group. `public/electron.js` creates it only in response to an authorized request for a finalized or restored file. No recorded-file worker exists at application startup, while live recording is active, or merely because a draft path was discovered.

```text
finalized/restored JSONL file
             │
             ▼
recorded-file-reader-worker.js     public/electron.js owns only the read lifecycle
  parse + validate + chunk         (readId, utility handle, renderer owner)
             │
             ▼ private, validated saved-read events
      src/common/preload.js
             │
             ├──► validation / draft restoration / analysis
             └──► upload chunk preparation
```

### What the worker does

For each read, `recorded-file-reader-worker.js`:

1. Receives an immutable `readId`, an absolute `filePath`, the recording's `game`, and a purpose (`validate` or `consume`). The game comes from authoritative recording/draft metadata because the standard field names are deliberately identical across games. `public/electron.js` first verifies the requesting renderer and ensures the resolved path is a regular file inside the configured recording directory; the worker repeats the path and regular-file checks before opening it.
2. Opens the file read-only with Node `fs.createReadStream` and a line reader. It never opens a game SDK, starts Python, creates a `RecordingSessionManager`, or touches the live reader/writer/view ports.
3. Uses the first non-empty parsed row to identify the accepted `standard-flat` format from its `Physics_*`, `Graphics_*`, and `Static_*` fields. Format validation comes from row content, never from the filename. The worker does not infer the game from telemetry keys because every game uses the same keys. Empty files are reported as valid filesystem objects with zero telemetry rows, not as successful upload inputs.
4. Parses every non-empty line. A malformed JSON line is a read failure with its one-based row number and byte position; it is not silently skipped. Every key and value must satisfy [`telemetry-fields.md`](telemetry-fields.md), every saved row must represent live telemetry, and the continuity validator selected for the request's game must hold across the file. For ACC this is the established seven-field classifier. An unknown key, renamed alias, invalid value type, reader-control line, or continuity break makes the file invalid.
5. In `validate` mode, reads through the file and returns only a summary. In `consume` mode, emits bounded chunks of the parsed flat objects plus throttled `{ rowsRead, bytesRead, totalBytes }` progress. It never projects, renames, wraps, supplements, or converts a row.
6. Emits one terminal summary containing `format: 'standard-flat'`, the immutable request `game`, `rowCount`, and `totalBytes`, closes the stream, closes its ports, and exits. Cancellation or failure follows the same cleanup path and emits no successful terminal result.

It does **not** append or rewrite rows, assign sequences, repair malformed data, delete the file, initiate an HTTP upload, or choose `game_recorded_from`. The writer owns file creation and transport-only commit sequencing, deletion remains an explicit lifecycle operation, and the saved-file request receives its immutable game from the same authoritative recording/draft metadata used by upload.

### Saved-file read protocol and lifecycle

The narrow renderer-facing API is separate from the live recording port:

```ts
type RecordedFileReadRequest = {
  filePath: string;
  game: DesktopGame;
  purpose: 'validate' | 'consume';
};

type RecordedFileReadEvent =
  | { type: 'format'; readId: string; format: 'standard-flat'; game: DesktopGame }
  | { type: 'chunk'; readId: string; rows: StandardTelemetrySample[] }
  | { type: 'progress'; readId: string; rowsRead: number; bytesRead: number; totalBytes: number }
  | { type: 'complete'; readId: string; format: 'standard-flat'; game: DesktopGame; rowCount: number; totalBytes: number }
  | { type: 'error'; readId: string; message: string; row?: number; byteOffset?: number };

startRecordedFileRead(request: RecordedFileReadRequest): Promise<{ readId: string }>;
cancelRecordedFileRead(readId: string): Promise<void>;
onRecordedFileReadEvent(callback: (event: RecordedFileReadEvent) => void): () => void;
```

`public/electron.js` creates one `MessageChannelMain` and one read registry entry per accepted request. It transfers one endpoint to the utility and the other to the owning preload, waits for the preload's ready acknowledgement, and only then tells the worker to open the file. Row chunks and progress travel directly over that private port; the main process retains only `{ readId, utility, ownerWebContentsId }` and never receives file rows. Cancellation still goes through the main-process handler so another renderer cannot cancel the read. The preload keeps the raw port private, validates `readId` and every payload, strips transport details, and fans events out only to that read's callbacks. Bounded chunks prevent the utility or bridge from holding the entire file; upload forwards those exact chunks without rebuilding one full in-memory array or changing their rows.

The reader may start only after the live writer has reported finalization, or for a restored draft that has no active writer. Reader cancellation, renderer destruction, and application quit close the stream and terminate the utility. Success, cancellation, and failure all remove the `readId` registry entry. None of these paths constructs or wakes `RecordingSessionManager`.

The existing Python file reader may remain temporarily only as an implementation fallback while the Node worker is introduced. The completion gate for this phase is that the Node worker reads the existing flat recordings as `standard-flat`, after which `LiveSessionContext.readRecordedTelemetry()` stops launching `read_telemetry_data.py`. Every future game reuses this streaming file-reader mechanism and the same standard field validator; only its continuity validator may be game-specific.

## Preload Bridge

The preload bridge is the existing `src/common/preload.js`. `public/electron.js` supplies that file as `webPreferences.preload` when it creates both the main window and the floating-chat window, and `package.json` already includes it in Electron Builder's `files`. Extend this file in place with the recording API; do not put the bridge in `electron/recording/`, because those modules belong to the main process and utility processes rather than to a renderer's preload context. The BrowserWindow configuration must explicitly retain `contextIsolation: true` and `nodeIntegration: false`.

The script runs before React in Electron's isolated preload context. It can import `contextBridge` and `ipcRenderer`; the React renderer cannot. It adds only the narrow recording functions below to the existing `window.electronAPI` object created with `contextBridge.exposeInMainWorld`. It must never expose `ipcRenderer`, Node APIs, a worker handle, or a raw `MessagePort` to React.

Its responsibilities are:

| Responsibility | Required behavior |
| --- | --- |
| Commands | `startRecordingSession(config)` invokes the main-process `recording-session-start` handler, and `stopRecordingSession(sessionId, reason)` invokes `recording-session-stop`. `startRecordedFileRead(request)` and `cancelRecordedFileRead(readId)` invoke the separate saved-file handlers. These calls transport requests and results; the preload constructs neither the recording manager nor a utility process. |
| View-port intake | Listen for the main process's `recording-view-port` transfer, take `event.ports[0]`, validate its `{ sessionId, game }` descriptor, attach handlers, and retain the port in preload-private state. Only one port may be active for the owning renderer. |
| Direct updates | Receive view batches directly from the shared view worker over that port. Validate the outer message type, immutable `sessionId` and `game`, frame/state/count transport payload, and the standard sample against the field catalog before invoking renderer callbacks. Telemetry batches do not pass through `ipcMain`. |
| Handshake | After handlers and validation are installed, send the ready acknowledgement over the private port. The main-process start request does not resolve until the view reports that this handshake succeeded. |
| Subscriptions | `onRecordingViewUpdate(callback)` registers callbacks for validated direct-port batches. `onRecordingSessionEnded(callback)` listens for the main process's `recording-session-ended` lifecycle event. Both return idempotent unsubscribe functions, strip the Electron event object, and deliver only validated application data. One bad or unsubscribed callback must not prevent delivery to the others. |
| Saved-file events | Accept the `recorded-file-read-port` transfer for a preload-owned `readId`, install validation and close handlers, acknowledge readiness, and keep its raw port private. `onRecordedFileReadEvent(callback)` validates format/chunk/progress/terminal payloads and exposes only application data. Offline chunks never enter the live view port. |
| Errors | Validate the main process's `{ ok: false, error: { type, message } }` response and forward it unchanged from the exposed start call. Renderer code branches on `error.type`, never parses `error.message`, and never receives or reconstructs a custom error class. Unexpected IPC transport failures may still reject with ordinary `Error` instances. |
| Cleanup | For live recording, a validated terminal message stops view dispatch, closes the port, deletes its session entry, and delivers the ended result once. For saved-file reads, complete, error, cancellation, or an unexpected port close removes only that `readId`, closes its port, and stops its utility. Navigation or renderer destruction cleans up both kinds of private port. Cleanup is idempotent. |

The recording port is deliberately private because it is a capability: possession permits receipt of live session data. React receives callbacks, not the capability itself. Although the floating-chat window currently loads the same preload file and therefore sees the same function names, that does not authorize it to record. The handlers in `public/electron.js` must accept start/stop requests only when `event.sender` is the current main-window `webContents`, the renderer owns the active live workspace, and the request matches the immutable session and game. Calls from the overlay or stale/replaced renderers are rejected before lazy manager creation.

The start and delivery sequence is:

1. React registers its view-update and ended callbacks, then calls `window.electronAPI.startRecordingSession({ game })`.
2. Preload invokes the main-process start handler. The main process validates the sender and game, lazily creates the pipeline, and transfers the view/renderer endpoint with `webContents.postMessage('recording-view-port', descriptor, [port])`.
3. Preload validates and stores the transferred port, installs its message and close handlers, and sends the ready acknowledgement. The raw port never crosses `contextBridge`.
4. The shared view worker sends batches over the direct port. Preload validates each batch and fans it out to the registered callbacks; React updates `SessionIntelligence` and UI state.
5. Stop still travels through request/response IPC because the main process owns lifecycle coordination. After shutdown, the main process sends one `recording-session-ended` event to the owning preload; the validated result is delivered once to ended subscribers, and preload releases the session port and callback state.

The preload bridge does **not** read ACC shared memory, parse a recorded file itself, spawn Python or a utility, rename or transform game fields, batch or persist telemetry, choose a filename, own the recording file, calculate committed counts, or decide whether a simulator has a reader. Those responsibilities remain with the live reader, recorded-file reader, shared writer/view utilities, and authoritative main-process validation described above.

## View Utility and Renderer

There is one shared view worker for every `DesktopGame` that has a reader. In this phase it receives `SourceFrame` messages and states from the ACC Python reader; future game-specific readers use the same port and batching path. It receives commit sequence ranges and committed counts directly from the shared writer. It buffers display frames for 100 ms, tracks the latest unchanged standard sample/source state/commit sequence/committed count, and sends one renderer message containing every standard sample for `SessionIntelligence` plus the latest sample for React. Shared UI data always comes from standard keys such as `Static_track` and `Static_car_model`, regardless of game. It flushes immediately for paused, resume, replay, unavailable, completed, fatal, and final events.

The preload bridge described above keeps the transferred port private and is the only renderer-side endpoint of the view/renderer channel. The view worker does not call React or `ipcRenderer` directly. An unexpected renderer-port closure is a group failure; normal finalization sends a terminal message before closing it.

The recording API becomes:

```ts
startRecordingSession(config: {
  game: DesktopGame;
}): Promise<
  | {
      ok: true;
      sessionId: string;
      game: DesktopGame;
      filePath: string;
      startedAt: number;
    }
  | RecordingStartFailure
>;

stopRecordingSession(
  sessionId: string,
  reason: 'manual' | 'complete' | 'upload' | 'discard' | 'reset'
): Promise<{
  sessionId: string;
  game: DesktopGame;
  filePath: string;
  writtenSamples: number;
  reason: string;
}>;

onRecordingViewUpdate(callback): () => void;
onRecordingSessionEnded(callback): () => void;

startRecordedFileRead(request: RecordedFileReadRequest): Promise<{ readId: string }>;
cancelRecordedFileRead(readId: string): Promise<void>;
onRecordedFileReadEvent(callback): () => void;
```

`startRecordingSession` accepts the single `DesktopGame` type. It resolves with an `ok: false` result using `error.type: 'malformed-recording-game'` for malformed input and `error.type: 'unknown-recording-game'` for a well-formed string outside that type. For a known game, it asks the resolver for the reader launch config; a missing entry point or launch dependency produces `error.type: 'unsupported-recording-game'`. These expected failures are plain data, contain no partial success fields, and cross Electron IPC without custom error serialization. `LiveAnalysisSessionRecording` calls this API for the active `DesktopGame`, checks `result.ok`, handles `unsupported-recording-game` as a recording-unavailable outcome, and never maintains its own support list. A successfully resolved reader's unchanged samples always use the shared writing and saving path. In this phase, the only successful game is ACC. The component stores the returned session ID/path/game from the `ok: true` result and awaits stop before upload, discard, or reset. It no longer owns Python shell IDs, Python listeners, or renderer write queues.

`LiveSessionContext` must:

- Track the shared source state plus the latest standard flat sample. The state may be derived from `Graphics_status`, but `Graphics_status` remains present and unchanged in the sample.
- Track the active `RecordingSessionManager` session ID and immutable recording game.
- Tick `SessionIntelligence` with every unchanged flat sample in a view batch and commit only the latest flat sample to React.
- Keep all existing consumers on the standard `Physics_*`, `Graphics_*`, and `Static_*` names; future game readers make those consumers reusable by emitting the same names rather than aliases.
- Update recorded count only from writer-committed summaries.
- Map `live → RECORDING`, non-recordable temporary states → `HOLDING`, and a matching-identity return to live → `RECORDING`.
- Map completion or worker failure to `UPLOAD_READY`, preserving the published partial file.
- Remove the renderer-owned writer session, append queue, pending acknowledgements, `appendTelemetrySample`, and `finalizeRecordingWrites`.
- Keep flat-row reading, validation, draft restoration, upload, and deletion.

`LiveSessionDetectionManager` uses the ACC live-telemetry availability probe and must not start a second probe while a recording session exists, including `HOLDING`. ACC may retain its generic Python checker. Non-ACC process detection may select the appropriate catalog entry and limited workspace, but it must not perform live telemetry reading checks or create the recording manager.

Generic Python IPC remains available for ACC detection, temporary file-reading fallback if still needed, analysis, and unrelated scripts. No live telemetry reader is implemented for iRacing or other simulators in this phase.

## Termination and Continuity Rules

The selected live telemetry reader, its telemetry runtime, the shared writer, and the view terminate together when:

- The user stops recording.
- Upload begins.
- The user discards or resets the live session.
- The ACC reader identifies a different ACC session after reconnect.
- Any reader/runtime/worker fails or exits unexpectedly.
- The renderer owning the session closes or is replaced.
- The application quits.

ACC pause/replay/`ACC_OFF` conditions are non-recordable temporary conditions; none alone creates or destroys a worker. An ACC session resumes in the same file only when the ACC reader proves the same `sessionIdentity`. There are zero reader/writer/view workers at startup, exactly one ACC reader/shared-writer/shared-view group while the read-plus-save session is active or in `HOLDING`, and zero pipeline resources after final shutdown. Catalog-only simulator entries always have zero reader, writer, or view resources.

A later **Start Recording** always creates three new utilities, four new channels, one new game-specific reader instance, a new session ID, and a new JSONL file. Restored drafts never restart workers.

## Test Plan

- Manager tests:
  - constructor validation is side-effect free
  - startup/navigation/draft restoration creates no reader, writer, or view resource
  - ACC, mismatched detected-game, and malformed request validation
  - missing, non-string, and otherwise malformed game values return `{ ok: false, error: { type: 'malformed-recording-game', message } }` before manager construction
  - well-formed unknown simulator IDs return the same failure shape with `type: 'unknown-recording-game'` at the same boundary
  - all contracts use `DesktopGame`; no second game union or separate supported-game list exists
  - `iracing`, `ac`, and every other recognized game without a reader ask the resolver for launch config and receive `{ ok: false, error: { type: 'unsupported-recording-game', message } }`
  - the manager passes that resolver failure through before creating a session ID or any pipeline resource
  - no recording startup path constructs or depends on a custom `Error` subclass
  - main-process and preload/renderer tests assert the failure object's `ok`, `error.type`, and `error.message` survive IPC unchanged
  - unsupported requests resolve/check reader config but create no directories, files, processes, workers, or channels
  - a valid ACC request creates exactly three forks and four channels with the correct endpoints
  - reader config/game mismatch rolls back startup
  - path relay without manager path ownership
  - readiness, stale session/game rejection, conflicts, idempotent stop, worker failure, renderer destruction, and quit
- Shared reader-contract tests applied to the ACC reader and every future reader:
  - every reader emits only names and types from `telemetry-fields.md`; unavailable fields are omitted and no per-game aliases are allowed
  - transport-frame validation while the standard telemetry sample retains its mapped names and values
  - frame/state delivery to the shared writer and view without reader-side commit sequencing or write batching
  - game-specific session continuity completes before another identity enters the pipeline
  - shared writer pause/resume commands are applied to the game-specific reading transport
  - reader completion/fatal propagation and stop-before-end-of-stream ordering
- ACC reader tests:
  - lazy Python spawn, split stdout lines, invalid JSON, and cleanup
  - exhaustive 240-key standard allowlist, field types, enum integer values, nested flattening, and array shapes from `telemetry-fields.md`
  - exact pass-through: no renamed keys, aliases, unit conversions, wrapper fields, or added timestamps
  - `{"available":false}` is handled as control state and never enters a saved/uploaded row
  - pause/replay/off/unavailable classification
  - seven-field reconnect continuity and break detection
- Deferred simulator entry tests:
  - simulator catalog keeps iRacing, Assetto Corsa, and any currently configured future entries visible
  - ACC start resolves its reader; every entry without a reader returns an `ok: false` result with `error.type: 'unsupported-recording-game'` and is presented as coming soon
  - renderer behavior is driven by the discriminated start result and has no hard-coded recording-support allowlist
  - process detection of a non-ACC entry starts no telemetry probe, reader, SDK, or Python process; a start attempt may construct only the side-effect-free manager before reader resolution fails
- Writer tests:
  - status/non-recordable filtering, accepted identity, shared commit sequencing and batching, unique exclusive path, game/field-contract validation, contiguous flat JSONL, committed acknowledgements, final flush, and stream failures
  - serialized lines from every game are byte-equivalent after JSON parse to the standard `frame.sample` and contain no transport metadata
  - shared pause/resume thresholds and direct committed-progress delivery to view
  - the same writer and saving behavior accepts every implemented game's standard sample without a game-specific write path
  - unpublished empty-file startup rollback versus preservation after path publication
- Recorded-file reader/upload tests:
  - target worker location and launcher are packaged; no recorded-file worker exists at startup or during live writing
  - authorized finalized/restored paths start exactly one short-lived read utility without constructing `RecordingSessionManager`
  - existing ACC rows and every newly recorded game's rows have the same standard readable/uploadable shape
  - content-based `standard-flat` validation and exact unchanged row output
  - authoritative game metadata selects continuity validation; telemetry keys are never used to infer the game
  - a requested game that disagrees with authoritative recording/draft metadata is rejected before the worker starts
  - unknown/renamed keys, invalid types, control lines, continuity breaks, and malformed JSON are rejected with row/byte diagnostics
  - the Node reader reports bounded progress/chunks and honors explicit cancellation, renderer destruction, and application quit
  - upload consumes unchanged flat chunks without accumulating the entire file in renderer memory
- View/preload/React tests:
  - `src/common/preload.js` is the configured bridge and exposes only the named narrow live-recording and saved-file functions, never raw `ipcRenderer`, Node APIs, worker handles, or transferred ports
  - port descriptor, session/game, message-type, transport-payload, and standard sample validation; ready is acknowledged only after handlers are installed
  - Electron event stripping, callback isolation, idempotent unsubscribe, one terminal delivery, and idempotent cleanup
  - overlay, stale-renderer, duplicate-port, mismatched-session, malformed-message, and unexpected-close rejection causes group cleanup without creating a second manager
  - 100 ms batching, every frame reaches intelligence, only latest commits to React
  - shared state/static/count propagation for ACC
  - paused/resume/completion/failure transitions with the same session ID
  - upload waits for writer finalization and restored drafts start no utilities
- Packaging gates:
  - Jest and React production build
  - Electron packaged smoke test includes `electron/recording/**/*`
  - ACC packaged test resolves bundled Python only when the ACC reader starts
  - packaged app contains no new iRacing SDK/native telemetry dependency from this work
