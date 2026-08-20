# Detailed Extensible Recording Process Architecture

## Scope and Decisions

This phase implements only the ACC-specific live telemetry reader. ACC and every future game use the same game-independent writing and saving pipeline; adding a game adds a reader, not another writing or saving implementation. The architecture keeps simulator selection separate from reading capability so the product can continue to show entries for iRacing, Assetto Corsa, and future simulators without implying that their telemetry reader exists.

Terminology is strict throughout this plan:

- **Reading** means acquiring telemetry from a running game, mapping its raw fields, normalizing them, and applying game-specific session-continuity rules. Reading is game-specific; ACC has its own reading process, and every future game supplies its own reader.
- **Writing and saving** mean validating normalized frames, assigning persisted sequences, batching writes, creating and finalizing the JSONL file, and reporting committed samples. Writing and saving are shared across every game.
- **Collecting** and **recording** both mean the complete read-plus-save lifecycle. Neither term is used for the game-specific reading step alone.

The key scope decisions are:

- Each game has its own telemetry reader because games expose telemetry through different SDKs, shared-memory layouts, processes, and runtimes. Readers must not be treated as one shared game-neutral implementation.
- ACC uses its dedicated Python reader because `pyaccsharedmemory` and the existing `ACCMemoryExtractor.py` are Python-based. The ACC reader's JavaScript utility wrapper only connects that Python reader to the shared worker protocol.
- iRacing, Assetto Corsa (`ac`), and any future simulator entries are catalog/UI entries only in this phase. Do not add an SDK dependency, telemetry probe, reader, normalizer, fixture set, or native-module packaging work for them. Their future readers must use the existing shared writing and saving pipeline unchanged.
- A recording request for a recognized simulator whose reader has not shipped must throw a structured error with code `READER_NOT_IMPLEMENTED` and message `Telemetry reader has not been implemented for this game yet.` This happens before a manager, worker, message channel, file, Python process, SDK handle, or recording directory is created. A malformed or unknown simulator ID returns `INVALID_RECORDING_GAME` at the same boundary.
- Each game-specific reader owns telemetry acquisition, raw-field mapping, normalization, and session-continuity rules before emitting the shared normalized contract. It does not own persisted sequencing, write batching, file creation, serialization, or finalization.
- The `RecordingSessionManager`, reader protocol, writer and saving path, recorded-file reader, view worker, preload bridge, renderer integration, and startup/shutdown lifecycle are shared. Every game-specific reader must send normalized events to the same writer and view implementations; adding a game must not create game-specific writing or saving paths.
- Adding telemetry reading for another simulator requires a separate implementation plan and release gate. This plan preserves the reader boundary needed for that later work but does not preselect a language, SDK, or transport for it.

The existing desktop simulator catalog remains broader than the set of recordable simulators:

```ts
type DesktopGame = 'ac' | 'acc' | 'iracing'; // existing detection/UI entries
type RecordingGame = 'acc';                  // implemented in this phase
```

Future simulator identifiers may be added to `DesktopGame` without being added to `RecordingGame`. A simulator is added to `RecordingGame` only when its reader, normalization, continuity behavior, tests, and packaging gate are implemented and its normalized events pass through the unchanged shared writer and saving path. Every accepted recording session carries the immutable `game: 'acc'` value in this phase; downstream processes consume the neutral contract so a later reader can reuse the same writing, saving, recorded-file reading, viewing, and lifecycle implementation without redesigning the pipeline.

## Process and Module Location

Place the shared pipeline and the implemented ACC reader at:

```text
electron/
└── recording/
    ├── recording-session-manager.js
    ├── recording-protocol.js
    ├── telemetry-contract.js
    ├── normalize-telemetry.js
    ├── readers/
    │   ├── reader-contract.js
    │   └── acc/
    │       ├── acc-reader-worker.js
    │       ├── acc-python-reader.js
    │       └── acc-normalizer.js
    └── workers/
        ├── recorded-file-reader-worker.js
        ├── writer-worker.js
        └── view-worker.js
```

`public/electron.js` remains the Electron bootstrap. After `app.whenReady()`, it registers the recording IPC handlers, but it does **not** create a recording manager, utility process, Python extractor, message channel, or recording file. The first valid ACC `startRecordingSession` request from the active live-workspace renderer lazily creates the single `RecordingSessionManager` and starts the workers. Add `electron/**/*` to Electron Builder's `files` list.

The manager constructor must be side-effect free. Only `RecordingSessionManager.startSession()` may fork the complete recording pipeline, and only after the IPC boundary confirms that the request belongs to an active, recordable ACC live session. Application launch, window creation, route navigation, React provider mounting, game-process detection, and draft restoration must produce zero reader, writer, view, or ACC extractor processes. The manager may remain as an idle main-process object after recording, but none of those processes may remain.

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
                                            Live workspace renderer
```

The selected live telemetry reader is game-specific. For ACC, `acc-reader-worker.js` starts and owns the Python reader; its Python child handle is private reader state. Telemetry never passes through the Electron main process. Catalog-only simulator entries never enter this diagram because their requests stop at the IPC capability check.

When another simulator's recording support is implemented, it supplies a different reader that may use a different language, SDK, process model, or transport. That reader must implement the shared reader protocol and emit `NormalizedSourceFrame` events. The same message-channel roles, writer and saving behavior, view worker, renderer port, manager, and shutdown sequence remain in use.

## Shared Telemetry and Reader Contract

The pipeline must not expose `ACC_STATUS` as its cross-process status type. Define recording-specific types in `telemetry-contract.js` and mirror them in renderer TypeScript declarations:

```ts
type RecordingGame = 'acc';

type TelemetrySourceState =
  | 'connecting'
  | 'live'
  | 'holding'
  | 'replay'
  | 'unavailable'
  | 'completed'
  | 'fatal';

type NormalizedTelemetry = {
  timestampMs: number;
  speedKph: number | null;
  throttle: number | null;
  brake: number | null;
  clutch: number | null;
  steering: number | null;
  gear: number | null;
  rpm: number | null;
  lapNumber: number | null;
  lapDistancePct: number | null;
  lapTimeMs: number | null;
  worldPosition: { x: number; y: number; z: number } | null;
};

type NormalizedSourceFrame = {
  game: RecordingGame;
  sourceTick: number | string | null;
  capturedAt: number;
  sourceState: TelemetrySourceState;
  recordable: boolean;
  sessionIdentity: string | null;
  telemetry: NormalizedTelemetry;
  staticData: {
    track: string | null;
    car: string | null;
    sessionType: string | null;
  };
  sourceData: Record<string, unknown>;
};
```

`sourceData` preserves a versioned, serializable allowlist of ACC-specific fields needed for existing analysis, future features, and upload fidelity. It is not an unbounded dump of extractor output. The field remains namespaced by game so a future simulator reader cannot accidentally masquerade as ACC. Shared live components and new analysis code consume `telemetry`; explicitly game-specific features consume `sourceData` only after checking `game`.

During migration, `normalize-telemetry.js` also creates a renderer-only compatibility projection for consumers that still expect ACC-style flat keys such as `Physics_speed_kmh`, `Graphics_completed_laps`, and `Graphics_normalized_car_position`. That projection must be covered by contract tests and must not become the persisted neutral schema. New code uses the neutral names.

The JSONL container remains one compact JSON object per recorded sample. New recordings use `schemaVersion: 2` and persist the normalized frame without worker-only fields:

```ts
type PersistedTelemetrySampleV2 = {
  schemaVersion: 2;
  game: RecordingGame;
  sequence: number;
  capturedAt: number;
  sourceTick: number | string | null;
  sessionIdentity: string;
  telemetry: NormalizedTelemetry;
  staticData: NormalizedSourceFrame['staticData'];
  sourceData: Record<string, unknown>;
};
```

The shared writer persists only frames with `recordable: true`, `sourceState: 'live'`, and the session's accepted `sessionIdentity`. Status-only events are sent to the view but not persisted. The recorded-file reader must accept both legacy flat ACC JSONL rows and version 2 rows. Draft restoration, validation, deletion, and upload continue to work for legacy files; upload metadata's `game_recorded_from` must match the row game for version 2 files. A mixed-game or mixed-identity file is invalid.

Each game-specific reader conforms to this lifecycle interface:

```ts
interface TelemetryReader {
  readonly game: RecordingGame;
  start(emit: (event: ReaderEvent) => void): Promise<void>;
  pause(): void;
  resume(): void;
  stop(): Promise<void>;
}

type ReaderEvent =
  | { type: 'frame'; frame: NormalizedSourceFrame }
  | { type: 'state'; state: TelemetrySourceState; reason?: string }
  | { type: 'complete'; reason: string }
  | { type: 'fatal'; error: string };
```

`pause()` and `resume()` apply the shared writer's backpressure commands to the game-specific telemetry transport; they do not represent the game's pause state. `stop()` is idempotent, removes subprocess or SDK listeners, and resolves only after the reader can no longer emit.

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

`getReaderLaunchConfig(game)` is called only after a start request has passed renderer, live-session, and implemented-reader validation. It returns:

```ts
type ReaderLaunchConfig = {
  game: 'acc';
  runtime: 'python';
  pythonExecutable: string;
  scriptDirectory: string;
  scriptName: 'ACCMemoryExtractor.py';
};
```

The resolver lazily resolves the development/packaged Python executable and script directory after an ACC request passes validation. It is never called for iRacing, Assetto Corsa, or a future catalog-only simulator entry. The ACC reader wrapper verifies that config `game` and `runtime` match ACC before starting the Python reader.

The manager owns worker handles and the complete recording lifecycle only. It does not implement game telemetry reading, receive telemetry, write samples, or retain the writer's path.

```ts
type ManagedRecordingSession = {
  sessionId: string;
  game: 'acc';
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

1. Validate that the request came from the current main live-workspace renderer and that `config.game` is a known `DesktopGame`; an unknown value returns `INVALID_RECORDING_GAME`. If the recognized game does not have an implemented reader, throw `READER_NOT_IMPLEMENTED` with message `Telemetry reader has not been implemented for this game yet.` before manager creation. For ACC, verify that it matches the renderer's active detected game and has a recordable live session.
2. Generate an opaque, globally unique `sessionId`; do not construct a filename.
3. Resolve the ACC reader launch config. A failure creates no worker or file.
4. Fork the shared writer and view utilities plus the game-specific ACC reader utility with `stdio: 'pipe'` and separate service names. A synchronous fork failure terminates utilities already created by this attempt.
5. Attach lifecycle/error/exit listeners before initialization.
6. Construct four `MessageChannelMain` pairs: reader/writer, reader/view, writer/view, and view/renderer.
7. Initialize writer with `sessionId`, `game`, `schemaVersion: 2`, and the recording directory; transfer its reader-input and view-progress endpoints. The writer chooses the filename and opens it exclusively.
8. Initialize view with `sessionId` and `game`, transfer its reader-input, writer-progress-input, and renderer-output endpoints, and transfer the renderer endpoint through `webContents.postMessage`.
9. Wait for writer file-open readiness and the view/renderer direct-port handshake.
10. Initialize the ACC reader with `sessionId`, `game`, the matching reader launch config, and the remaining writer/view endpoints.
11. The ACC reader starts its Python telemetry process, accepts its initial session identity, and reports ready only after the Python process emits a valid, recordable live frame.
12. Resolve with `{ sessionId, game, filePath, startedAt, schemaVersion: 2 }`, relaying the writer-owned path without adding it to manager state.

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
- Emit normalized frame and state events directly to both the shared writer and shared view without assigning persisted sequences, batching writes, creating files, or serializing rows.
- Apply `pause` and `resume` commands from the shared writer to its telemetry transport without owning the shared backpressure thresholds.
- Treat reader completion and fatal events as lifecycle signals, never as telemetry rows.

Game-specific telemetry acquisition, raw-field mapping, normalization, and continuity rules stay inside that game's reader. Persisted sequencing, write batching, backpressure thresholds, serialization, file ownership, and finalization stay in the shared writer. Each reader must emit only the shared normalized event shapes; the writer independently enforces that a saved recording contains one game and one accepted `sessionIdentity`.

### ACC Python reader

`acc-reader-worker.js` is the ACC-specific utility wrapper. It owns `acc-python-reader.js`, which starts the Python subprocess because ACC telemetry reading depends on `pyaccsharedmemory`. Together they form the ACC reader. It:

- Launches `ACCMemoryExtractor.py` in stream-only mode with the supplied executable/script directory.
- Parses stdout with a line reader so split or combined chunks are valid.
- Converts flattened ACC data through `acc-normalizer.js` into `NormalizedSourceFrame`.
- Maps ACC live/pause/replay/off/unavailable states to the shared source states.
- Applies the shared writer's pause/resume commands to Python stdout.
- Keeps the extractor alive while ACC is paused, replaying, temporarily unavailable, or reports `ACC_OFF`.
- Applies the existing seven-field ACC continuity classifier after an unavailable frame. A continuity break emits `complete`; `ACC_OFF` alone does not.
- Terminates and awaits the Python reader process during `stop()`.

The shared writer, view, recorded-file reader, renderer bridge, and manager do not know that ACC uses Python. A future game's live telemetry reader may use Node.js, Python, a native SDK, or another runtime without changing those shared components.

## Deferred Simulator Entries

iRacing, Assetto Corsa, and future simulators remain visible wherever the product presents its supported or planned simulator catalog. In this phase those entries are capability placeholders, not telemetry implementations.

The boundary is enforced as follows:

- Keep the existing `DesktopGame`, detection, label, and limited-workspace entries for iRacing and Assetto Corsa. Future entries follow the same UI path.
- Use the minimal `game === 'acc'` implemented-reader guard in this phase; do not introduce a simulator capability registry solely for one working reader.
- Present ACC recording as available. Present iRacing, Assetto Corsa, and future entries as coming soon until their readers actually ship.
- A non-ACC live workspace may show the detected simulator and a clear “recording coming soon” state, but it must not show an enabled **Start Recording** control.
- Renderer guards provide immediate UX, while the Electron IPC handler independently throws `READER_NOT_IMPLEMENTED` with message `Telemetry reader has not been implemented for this game yet.` for every recognized non-ACC game. The main-process guard is authoritative.
- Do not create placeholder reader entry points, conditional dynamic loads, or dormant implementations for iRacing or another simulator.
- Do not install or evaluate an iRacing SDK, create an iRacing source/normalizer file, add captured SDK fixtures, or change ASAR/native-module packaging in this phase.
- Process detection may continue to recognize existing `DesktopGame` entries for navigation and UI messaging. Recognition is not a live-telemetry reading probe and must not start telemetry access.

When another simulator is prioritized, extend `RecordingGame` and replace the single-game guard with explicit capability metadata as part of that simulator's reader work. Its separate plan must choose and validate the SDK/transport, define normalization and session identity rules, add reader and fixture tests, prove its normalized events work with the unchanged shared writer and saving path, and prove the packaged runtime before changing the entry from coming soon to recording-enabled. The manager, reader protocol, writer and saving path, recorded-file reader, view, renderer bridge, and lifecycle are reused; only the game-specific live telemetry reader is newly implemented.

## Writer Utility

There is one game-independent writing and saving implementation shared by every game. Its contract accepts normalized frames from any implemented game without branching into per-game write paths; ACC is the only live telemetry reader implemented in this phase. The shared writer never reads telemetry from a game. It:

- Receives `game` and `schemaVersion` at initialization and rejects events whose frames disagree.
- Accepts only `recordable: true`, `sourceState: 'live'` frames with the session's first accepted non-empty `sessionIdentity`; state-only and non-recordable events are never persisted.
- Chooses a unique JSONL path below the assigned recording directory and opens it with exclusive-create semantics.
- Retains the path as writer-owned state; the manager never mirrors it in the managed session.
- Validates schema, one game, and one session identity, then assigns monotonically increasing persisted sequences.
- Batches writes at 100 ms or 30 samples, whichever occurs first.
- Owns the shared backpressure policy: pause the reader above 300 uncommitted samples and resume it below 120.
- Writes each batch with one stream operation and acknowledges only after the callback succeeds.
- Reports persisted sequence ranges and committed counts directly to the view, and lifecycle summaries to the manager.
- Ends the stream only after all earlier writes complete and reports the same path in `ready` and `finalized`.
- On failed startup before the path is published, closes and removes its empty file when instructed to roll back.
- Treats serialization, containment, stream, schema, identity, or sequence errors as fatal.

Filename generation may include the game for diagnostics, but correctness and game detection never depend on parsing the filename.

## Recorded-File Reading

`recorded-file-reader-worker.js` is the shared reader for saved normalized recordings. It is distinct from each game's live telemetry reader and is an independent, short-lived utility, not a fourth member of the active recording group. It starts only for an explicit saved-file read, validation, or upload operation after the writer has finalized, and it terminates after success, cancellation, or failure.

The reader uses Node `fs.createReadStream` plus a line reader, detects legacy ACC versus version 2 from parsed content rather than the filename, and emits bounded chunks with row and byte progress. It validates every version 2 row's schema, game, sequence, and session identity before returning it. It can project normalized rows into the temporary legacy flat shape for existing renderer consumers, but the upload path retains the canonical version 2 data and game metadata.

The current Python `read_telemetry_data.py` may remain temporarily for legacy ACC compatibility. The target state is the Node reader for legacy ACC and the shared version 2 format, since JSONL parsing does not require a Python dependency. Reader cancellation, renderer destruction, and application quit must close the stream and terminate its utility without creating a `RecordingSessionManager`. Because every game saves the same version 2 schema, the same recorded-file reader handles future games without a game-specific file-reading path.

## View Utility and Renderer

There is one shared view worker for all recording games. In this phase it receives normalized frames and states from the ACC Python reader; future game-specific readers use the same port and batching path. It receives persisted sequence ranges and committed counts directly from the shared writer. It buffers display frames for 100 ms, tracks the latest frame/static data/source state/persisted sequence/committed count, and sends one renderer message containing all frames for `SessionIntelligence` plus the latest frame for React. It flushes immediately for holding, resume, replay, unavailable, completed, fatal, and final events.

Preload keeps the transferred port private, validates `sessionId` and `game`, validates the normalized payload, and invokes registered callbacks. A renderer-port closure is a group failure.

The recording API becomes:

```ts
startRecordingSession(config: {
  game: DesktopGame;
}): Promise<{
  sessionId: string;
  game: 'acc';
  filePath: string;
  startedAt: number;
  schemaVersion: 2;
}>;

stopRecordingSession(
  sessionId: string,
  reason: 'manual' | 'complete' | 'upload' | 'discard' | 'reset'
): Promise<{
  sessionId: string;
  game: 'acc';
  filePath: string;
  writtenSamples: number;
  reason: string;
}>;

onRecordingViewUpdate(callback): () => void;
onRecordingSessionEnded(callback): () => void;
```

For a recognized game without an implemented reader, `startRecordingSession` rejects with `{ code: 'READER_NOT_IMPLEMENTED', message: 'Telemetry reader has not been implemented for this game yet.' }`; an invalid ID uses `INVALID_RECORDING_GAME`. Neither case resolves a partial result. `LiveAnalysisSessionRecording` calls this API only when the active `DesktopGame` has an implemented reader; that reader's normalized frames always use the shared writing and saving path. In this phase, the only such game is ACC. It stores the returned session ID/path/game and awaits stop before upload, discard, or reset. It no longer owns Python shell IDs, Python listeners, or renderer write queues.

`LiveSessionContext` must:

- Replace `telemetryStatus: ACC_STATUS | null` with the shared source state plus the latest normalized frame.
- Track the active `RecordingSessionManager` session ID and immutable recording game.
- Tick `SessionIntelligence` for every normalized frame in a view batch and commit only the latest frame to React.
- Use the temporary ACC compatibility projection while ACC-only consumers are migrated.
- Update recorded count only from writer-committed summaries.
- Map `live → RECORDING`, non-recordable temporary states → `HOLDING`, and a matching-identity return to live → `RECORDING`.
- Map completion or worker failure to `UPLOAD_READY`, preserving the published partial file.
- Remove the renderer-owned writer session, append queue, pending acknowledgements, `appendTelemetrySample`, and `finalizeRecordingWrites`.
- Keep version-aware reading, validation, draft restoration, upload, and deletion.

`LiveSessionDetectionManager` uses the ACC live-telemetry availability probe and must not start a second probe while a recording session exists, including `HOLDING`. ACC may retain its generic Python checker. Non-ACC process detection may select the appropriate catalog entry and limited workspace, but it must not perform live telemetry reading checks or create the recording manager.

Generic Python IPC remains available for ACC detection, legacy file reading if still needed, analysis, and unrelated scripts. No live telemetry reader is implemented for iRacing or other simulators in this phase.

## Termination and Continuity Rules

The selected live telemetry reader, its telemetry runtime, the shared writer, and the view terminate together when:

- The user stops recording.
- Upload begins.
- The user discards or resets the live session.
- The ACC reader identifies a different ACC session after reconnect.
- Any reader/runtime/worker fails or exits unexpectedly.
- The renderer owning the session closes or is replaced.
- The application quits.

ACC pause/replay/`ACC_OFF` conditions are non-recordable holding conditions; none alone creates or destroys a worker. An ACC session resumes in the same file only when the ACC reader proves the same `sessionIdentity`. There are zero reader/writer/view workers at startup, exactly one ACC reader/shared-writer/shared-view group while the read-plus-save session is active or holding, and zero pipeline resources after final shutdown. Catalog-only simulator entries always have zero reader, writer, or view resources.

A later **Start Recording** always creates three new utilities, four new channels, one new game-specific reader instance, a new session ID, and a new JSONL file. Restored drafts never restart workers.

## Test Plan

- Manager tests:
  - constructor validation is side-effect free
  - startup/navigation/draft restoration creates no reader, writer, or view resource
  - ACC, mismatched detected-game, and malformed request validation
  - `iracing`, `ac`, and any other recognized catalog-only simulator throw `READER_NOT_IMPLEMENTED` with the expected message before manager construction
  - malformed or unknown simulator IDs return `INVALID_RECORDING_GAME` at the same boundary
  - non-ACC requests do not resolve reader config or create directories, files, processes, workers, or channels
  - a valid ACC request creates exactly three forks and four channels with the correct endpoints
  - reader config/game mismatch rolls back startup
  - path relay without manager path ownership
  - readiness, stale session/game rejection, conflicts, idempotent stop, worker failure, renderer destruction, and quit
- Shared reader-contract tests applied to the ACC reader and every future reader:
  - normalized frame validation and compatibility projection
  - normalized frame/state delivery to the shared writer and view without reader-side persisted sequencing or write batching
  - game-specific session continuity completes before another identity enters the pipeline
  - shared writer pause/resume commands are applied to the game-specific reading transport
  - reader completion/fatal propagation and stop-before-end-of-stream ordering
- ACC reader tests:
  - lazy Python spawn, split stdout lines, invalid JSON, and cleanup
  - field/unit normalization and legacy compatibility keys
  - pause/replay/off/unavailable classification
  - seven-field reconnect continuity and break detection
- Deferred simulator entry tests:
  - simulator catalog keeps iRacing, Assetto Corsa, and any currently configured future entries visible
  - ACC recording is enabled; every other entry is presented as coming soon
  - non-ACC workspaces disable recording and communicate that telemetry reading is not implemented
  - process detection of a non-ACC entry starts no telemetry probe, reader, manager, SDK, or Python process
- Writer tests:
  - status/non-recordable filtering, accepted identity, shared sequencing and batching, unique exclusive path, schema/game validation, contiguous JSONL, committed acknowledgements, final flush, and stream failures
  - shared pause/resume thresholds and direct committed-progress delivery to view
  - the same writer and saving behavior accepts every implemented game's normalized contract without a game-specific write path
  - unpublished empty-file startup rollback versus preservation after path publication
- Recorded-file reader/upload tests:
  - legacy ACC rows remain readable
  - version 2 ACC rows are readable/uploadable
  - mixed-game, mixed-identity, malformed, and metadata-game mismatch rejection
  - the Node reader reports bounded progress/chunks and honors cancellation
- View/preload/React tests:
  - 100 ms batching, every frame reaches intelligence, only latest commits to React
  - shared state/static/count propagation for ACC
  - holding/resume/completion/failure transitions with the same session ID
  - upload waits for writer finalization and restored drafts start no utilities
- Packaging gates:
  - Jest and React production build
  - Electron packaged smoke test includes `electron/recording/**/*`
  - ACC packaged test resolves bundled Python only when the ACC reader starts
  - packaged app contains no new iRacing SDK/native telemetry dependency from this work
