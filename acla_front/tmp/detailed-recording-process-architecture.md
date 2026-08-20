# Detailed Multi-Game Recording Process Architecture

## Scope and Decisions

The recording pipeline must support both ACC and iRacing without making the manager, writer, view worker, renderer API, or persisted-recording lifecycle game-specific.

The key runtime decision is:

- ACC uses a Python-backed source adapter because `pyaccsharedmemory` and the existing `ACCMemoryExtractor.py` are Python-based.
- iRacing uses a JavaScript/TypeScript source adapter running inside the collector utility process. It must not start Python or require the bundled Python environment during normal iRacing recording.
- The initial iRacing implementation should evaluate [`irsdk-node`](https://github.com/bengsfort/irsdk-node), which exposes the native iRacing SDK to Node with TypeScript types. Because it contains a native Node addon, a packaged Electron/Windows compatibility spike is a prerequisite; see [iRacing JavaScript Compatibility Gate](#iracing-javascript-compatibility-gate).
- If that package cannot be made compatible with the application's Electron version, retain the JavaScript adapter contract and replace only its SDK binding, preferably with an Electron-compatible Node-API addon. A Python iRacing adapter is a separately approved fallback, not the baseline architecture.
- Assetto Corsa (`ac`) is not included in this implementation. A start request for `ac` must return `UNSUPPORTED_RECORDING_GAME` and create no recording resources.

Every recording request carries an immutable `game: 'acc' | 'iracing'`. A worker may branch on that value only when selecting or configuring the telemetry source adapter. Downstream processes consume normalized events and do not branch on ACC versus iRacing transport details.

## Process and Module Location

Place the shared pipeline and game-specific source adapters at:

```text
electron/
└── recording/
    ├── recording-session-manager.js
    ├── recording-protocol.js
    ├── telemetry-contract.js
    ├── normalize-telemetry.js
    ├── sources/
    │   ├── source-adapter.js
    │   ├── create-source-adapter.js
    │   ├── acc-python-source.js
    │   ├── acc-normalizer.js
    │   ├── iracing-node-source.js
    │   └── iracing-normalizer.js
    └── workers/
        ├── collector-worker.js
        ├── recorded-file-reader-worker.js
        ├── writer-worker.js
        └── view-worker.js
```

`public/electron.js` remains the Electron bootstrap. After `app.whenReady()`, it registers the recording IPC handlers, but it does **not** create a recording manager, utility process, Python extractor, iRacing SDK instance, message channel, or recording file. The first valid `startRecordingSession` request from the active live-workspace renderer lazily creates the single `RecordingSessionManager` and starts the workers. Add `electron/**/*` to Electron Builder's `files` list.

The manager constructor must be side-effect free. Only `RecordingSessionManager.startSession()` may fork recording utilities, and only after confirming that the request belongs to an active, recordable ACC or iRacing live session. Application launch, window creation, route navigation, React provider mounting, game-process detection, and draft restoration must produce zero recording processes. The manager may remain as an idle main-process object after recording, but no collector, writer, view, ACC extractor, or iRacing SDK handle may remain.

```text
                         Electron main process
                    RecordingSessionManager
                    lifecycle and ownership only
                              │
             ┌────────────────┼────────────────┐
             │                │                │
        Collector          Writer             View
    selects one adapter   owns path/file    batches frames
             │                ▲                │
       ┌─────┴──────┐         │                │
       │            │         │                │
  ACC adapter   iRacing adapter               │
  Python child  Node SDK in-process           │
       │            │                         │
       └──── normalized source events ────────┘
             └──── MessagePort ───────────────► Writer
             └──── MessagePort ───────────────► View
                                                │
                                      direct MessagePort
                                                ▼
                                  Live workspace renderer
```

The collector creates exactly one source adapter for the immutable session game. ACC's Python child handle and iRacing's SDK handle are private adapter state. Telemetry never passes through the Electron main process.

## Shared Telemetry and Source Contract

The pipeline must not expose `ACC_STATUS` as its cross-process status type. Define recording-specific types in `telemetry-contract.js` and mirror them in renderer TypeScript declarations:

```ts
type RecordingGame = 'acc' | 'iracing';

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

`sourceData` preserves a versioned, serializable allowlist of game-specific fields needed for existing analysis, future features, and upload fidelity. It is not an unbounded dump of the SDK object: in particular, the iRacing normalizer must not repeat the complete session YAML or every all-car array at 60 Hz. It is namespaced so an iRacing field cannot accidentally masquerade as an ACC field. Shared live components and new analysis code consume `telemetry`; explicitly game-specific features consume `sourceData` only after checking `game`.

During migration, `normalize-telemetry.js` also creates a renderer-only compatibility projection for consumers that still expect ACC-style flat keys such as `Physics_speed_kmh`, `Graphics_completed_laps`, and `Graphics_normalized_car_position`. That projection must be covered by contract tests and must not become the persisted multi-game schema. New code uses the neutral names.

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

Only frames with `recordable: true`, `sourceState: 'live'`, and the session's accepted `sessionIdentity` are written. Status-only events are sent to the view but not persisted. The reader must accept both legacy flat ACC JSONL rows and version 2 rows. Draft restoration, validation, deletion, and upload continue to work for legacy files; upload metadata's `game_recorded_from` must match the row game for version 2 files. A mixed-game or mixed-identity file is invalid.

Each adapter conforms to this lifecycle interface:

```ts
interface TelemetrySourceAdapter {
  readonly game: RecordingGame;
  start(emit: (event: SourceEvent) => void): Promise<void>;
  pause(): void;
  resume(): void;
  stop(): Promise<void>;
}

type SourceEvent =
  | { type: 'frame'; frame: NormalizedSourceFrame }
  | { type: 'state'; state: TelemetrySourceState; reason?: string }
  | { type: 'complete'; reason: string }
  | { type: 'fatal'; error: string };
```

`pause()` and `resume()` are transport backpressure controls; they do not represent the game's pause state. `stop()` is idempotent, removes SDK/subprocess listeners, and resolves only after the underlying source can no longer emit.

## Manager Construction and State

`RecordingSessionManager` receives injected Electron dependencies and one lazy source-config resolver:

```js
new RecordingSessionManager({
  utilityProcess,
  MessageChannelMain,
  getMainWindow,
  getSourceLaunchConfig,
  recordingDirectory: path.join(app.getPath('userData'), 'acla-temp'),
});
```

All five properties are required. The constructor validates shapes and that `recordingDirectory` is a non-empty absolute path, then stores them without invoking callbacks, forking, creating channels/directories, resolving Python, or loading the iRacing native module.

`getSourceLaunchConfig(game)` is called only after a start request has passed renderer, live-session, and supported-game validation. It returns a discriminated object:

```ts
type SourceLaunchConfig =
  | {
      game: 'acc';
      runtime: 'python';
      pythonExecutable: string;
      scriptDirectory: string;
      scriptName: 'ACCMemoryExtractor.py';
    }
  | {
      game: 'iracing';
      runtime: 'node';
      sdkModule: 'irsdk-node';
    };
```

For ACC, the resolver lazily resolves the development/packaged Python executable and script directory. For iRacing, it returns only Node configuration: it must not inspect Python paths, run Python setup, or fall back silently to a Python package. The collector verifies that config `game` and `runtime` match the start request before creating the adapter.

The manager owns worker handles and lifecycle only. It does not create a source adapter, receive telemetry, or retain the writer's path.

```ts
type ManagedRecordingSession = {
  sessionId: string;
  game: 'acc' | 'iracing';
  status: 'starting' | 'running' | 'stopping' | 'terminated';
  ownerWebContentsId: number;
  collector: UtilityProcess;
  writer: UtilityProcess;
  view: UtilityProcess;
  readyWorkers: Set<'collector' | 'writer' | 'view'>;
  writtenSamples: number;
  stopPromise: Promise<StopResult> | null;
};
```

`game` is retained because it authorizes/scopes control and lifecycle replies; it is not inferred from process names or file paths. `filePath`, source handles, `sessionIdentity`, transferred ports, and file streams remain excluded. Starting while another session is starting, running, or stopping returns a conflict, regardless of game.

## Startup and Shutdown

Startup sequence:

1. Validate that the request came from the current main live-workspace renderer, `config.game` is `acc` or `iracing`, it matches the renderer's active detected game, and that game has a recordable live session. Invalid and `ac` requests return before manager creation.
2. Generate an opaque, globally unique `sessionId`; do not construct a filename.
3. Resolve the source launch config for only the selected game. A failure creates no worker or file.
4. Fork writer, view, and collector utilities with `stdio: 'pipe'` and separate service names. A synchronous fork failure terminates utilities already created by this attempt.
5. Attach lifecycle/error/exit listeners before initialization.
6. Construct three `MessageChannelMain` pairs: collector/writer, collector/view, and view/renderer.
7. Initialize writer with `sessionId`, `game`, `schemaVersion: 2`, and the recording directory. The writer chooses the filename and opens it exclusively.
8. Initialize view with `sessionId` and `game`, transfer its collector-input and renderer-output endpoints, and transfer the renderer endpoint through `webContents.postMessage`.
9. Wait for writer file-open readiness and the view/renderer direct-port handshake.
10. Initialize collector with `sessionId`, `game`, the matching source launch config, and the remaining writer/view endpoints.
11. Collector creates exactly one adapter, starts it, accepts its initial session identity, and reports ready only after the source proves that the selected game has a recordable live frame. For iRacing this occurs in Node; for ACC it occurs after the Python extractor emits a valid frame.
12. Resolve with `{ sessionId, game, filePath, startedAt, schemaVersion: 2 }`, relaying the writer-owned path without adding it to manager state.

If the source cannot become recordable within the startup deadline, the entire group is shut down and the exclusively created empty file is deleted by the writer before startup rejects. Once a path has been published, cleanup never silently deletes it.

Shutdown is idempotent and ordered:

1. Assign the shared `stopPromise` and mark the session `stopping`.
2. Ask collector to stop accepting source data and drain complete batches.
3. Collector awaits adapter `stop()`, sends end-of-stream directly to writer and view, and acknowledges.
4. Writer finishes queued writes, closes JSONL, and reports `{ filePath, writtenSamples }`.
5. View flushes its final display batch, closes ports, and acknowledges.
6. Manager waits for all three acknowledgements, with one five-second forced-cleanup deadline.
7. Kill remaining workers, remove listeners, set `terminated`, and clear the active reference.
8. Emit one ended result to the owning renderer.

An unexpected exit or fatal event from any worker uses the same group shutdown. On application quit, `public/electron.js` prevents the first quit only when the lazy manager has an active session, awaits shutdown, then quits through the existing guard.

## Collector and Source Adapters

### Collector utility

The collector is game-neutral. Its responsibilities are:

- Verify the selected game and source config agree, then create one adapter through `create-source-adapter.js`.
- Validate every source event against the shared contract.
- Accept the first recordable frame's non-empty `sessionIdentity` and reject frames from another identity.
- Assign a monotonically increasing writer sequence only to recordable frames.
- Send recordable samples directly to writer and all frames/state changes directly to view.
- Batch writer traffic at 100 ms or 30 samples, whichever occurs first, with its sequence range.
- Apply backpressure by calling adapter `pause()` above 300 unacknowledged samples and `resume()` below 120.
- Treat adapter completion and fatal events as lifecycle signals, never as telemetry rows.

Game-specific continuity rules belong to adapters. The collector only enforces that a recording file cannot contain more than one accepted `sessionIdentity`.

### ACC Python source

`acc-python-source.js` owns the Python subprocess because the ACC dependency is Python-based. It:

- Launches `ACCMemoryExtractor.py` in stream-only mode with the supplied executable/script directory.
- Parses stdout with a line reader so split or combined chunks are valid.
- Converts flattened ACC data through `acc-normalizer.js` into `NormalizedSourceFrame`.
- Maps ACC live/pause/replay/off/unavailable states to the shared source states.
- Pauses/resumes Python stdout only for collector backpressure.
- Keeps the extractor alive while ACC is paused, replaying, temporarily unavailable, or reports `ACC_OFF`.
- Applies the existing seven-field ACC continuity classifier after an unavailable frame. A continuity break emits `complete`; `ACC_OFF` alone does not.
- Terminates and awaits the Python child during `stop()`.

No other worker or manager knows that ACC uses Python.

### iRacing Node source

`iracing-node-source.js` loads the selected SDK package directly inside the collector utility process. It does not spawn a child process and does not invoke any Python callback. It:

- Opens the iRacing SDK/shared-memory connection on Windows and closes it in `stop()`.
- Runs SDK reads in a bounded, iterative loop that yields to the collector event loop between reads. A synchronous wait may use at most one telemetry-tick timeout; recursive or unbounded native waits must not starve stop messages, ports, timers, or writer acknowledgements.
- Copies a telemetry snapshot promptly when the SDK signals new data; parsing, normalization, batching, writing, and React work must not hold a view into a rotating shared-memory buffer.
- Reads session info only when its data version changes, not at every telemetry tick.
- Converts SDK telemetry/session fields through `iracing-normalizer.js` into the neutral contract, including unit conversion such as metres/second to kilometres/hour where required.
- Builds a stable `sessionIdentity` from validated iRacing session information, including a stable session/subsession identifier plus the active session number and player-car identity. The exact field set is locked by fixture tests before implementation.
- Marks live player-driving frames recordable. Replay, spectating, garage/out-of-car, disconnected, and invalid-buffer states are view/status events and are not written.
- Treats a short shared-memory disconnect as `unavailable`/`holding`, not immediate completion. On reconnect, it resumes the same file only if the rebuilt identity matches; a different identity emits `complete`.
- Uses the SDK tick/counter to suppress duplicate buffers and detect regressions. A regression after reconnect requires identity revalidation.
- Implements `pause()` by stopping delivery/SDK polling without discarding the last accepted identity and `resume()` by resuming from the next new tick.

iRacing does not have to mimic ACC enum values. The normalizer maps its own signals into shared states; renderer recording transitions depend on `sourceState` and `recordable`, not `ACC_STATUS`.

## iRacing JavaScript Compatibility Gate

The JavaScript path is technically viable because iRacing exposes live telemetry through Windows shared memory and Node SDK bindings exist. It is not complete until the exact Electron packaging path is proven.

Before building the full adapter, create a minimal spike using the application's pinned Electron version and target Windows architecture. It must prove all of the following:

1. `irsdk-node` installs with a locked version and can be required from an Electron `utilityProcess`.
2. Its native binary ABI matches Electron, either from a compatible prebuild or after an Electron-targeted rebuild.
3. The packaged application keeps the `.node` binary outside ASAR (for example with `asarUnpack`) and loads it without a development-only path.
4. With iRacing running, the utility reads changing telemetry, a source tick, and session information for at least ten minutes without blocking the main or renderer processes.
5. Stop/start cycles release all SDK/native handles, and application quit leaves no helper process or loaded recording worker.

Record the package version, Electron version, architecture, rebuild command, Builder configuration, and the successful packaged path in the implementation PR. CI can exercise fixtures/mocks, but a Windows packaged smoke test with iRacing is a release gate.

If the package fails the gate, first try an Electron-compatible rebuild or a small Node-API binding behind the same adapter. Do not move SDK access into the Electron main process or renderer. Do not silently switch iRacing to Python; that would change packaging and lifecycle requirements and needs an explicit architecture update.

## Writer Utility

The writer is identical for ACC and iRacing. It:

- Receives `game` and `schemaVersion` at initialization and rejects batches whose samples disagree.
- Chooses a unique JSONL path below the assigned recording directory and opens it with exclusive-create semantics.
- Retains the path as writer-owned state; the manager never mirrors it in the managed session.
- Validates schema, one game, one session identity, and contiguous collector sequences before serialization.
- Writes each batch with one stream operation and acknowledges only after the callback succeeds.
- Reports committed counts directly to collector and lifecycle summaries to manager.
- Ends the stream only after all earlier writes complete and reports the same path in `ready` and `finalized`.
- On failed startup before the path is published, closes and removes its empty file when instructed to roll back.
- Treats serialization, containment, stream, schema, identity, or sequence errors as fatal.

Filename generation may include the game for diagnostics, but correctness and game detection never depend on parsing the filename.

## Recorded-File Reading

Live iRacing acquisition and later reading of its saved recording both remain in Node. `recorded-file-reader-worker.js` is an independent, short-lived utility, not a fourth member of the active recording group. It starts only for an explicit read, validation, or upload operation after the writer has finalized, and it terminates after success, cancellation, or failure.

The reader uses Node `fs.createReadStream` plus a line reader, detects legacy ACC versus version 2 from parsed content rather than the filename, and emits bounded chunks with row and byte progress. It validates every version 2 row's schema, game, sequence, and session identity before returning it. It can project normalized rows into the temporary legacy flat shape for existing renderer consumers, but the upload path retains the canonical version 2 data and game metadata.

The current Python `read_telemetry_data.py` may remain temporarily for legacy ACC compatibility, but it must never be selected for a version 2 iRacing file. The target state is the Node reader for both formats, since JSONL parsing does not require a Python dependency. Reader cancellation, renderer destruction, and application quit must close the stream and terminate its utility without creating a `RecordingSessionManager`.

## View Utility and Renderer

The view worker receives normalized frames for either game. It buffers for 100 ms, tracks the latest frame/static data/source state/sequence range/committed count, and sends one renderer message containing all frames for `SessionIntelligence` plus the latest frame for React. It flushes immediately for holding, resume, replay, unavailable, completed, fatal, and final events.

Preload keeps the transferred port private, validates `sessionId` and `game`, validates the normalized payload, and invokes registered callbacks. A renderer-port closure is a group failure.

The recording API becomes:

```ts
startRecordingSession(config: {
  game: 'acc' | 'iracing';
}): Promise<{
  sessionId: string;
  game: 'acc' | 'iracing';
  filePath: string;
  startedAt: number;
  schemaVersion: 2;
}>;

stopRecordingSession(
  sessionId: string,
  reason: 'manual' | 'complete' | 'upload' | 'discard' | 'reset'
): Promise<{
  sessionId: string;
  game: 'acc' | 'iracing';
  filePath: string;
  writtenSamples: number;
  reason: string;
}>;

onRecordingViewUpdate(callback): () => void;
onRecordingSessionEnded(callback): () => void;
```

`LiveAnalysisSessionRecording` passes the active `DesktopGame`, starts only after game-specific recordability is established, stores the returned session ID/path/game, and awaits stop before upload, discard, or reset. It no longer owns Python shell IDs, Python listeners, or renderer write queues for either game.

`LiveSessionContext` must:

- Replace `telemetryStatus: ACC_STATUS | null` with the shared source state plus the latest normalized frame.
- Track the active recording utility session ID and immutable recording game.
- Tick `SessionIntelligence` for every normalized frame in a view batch and commit only the latest frame to React.
- Use the temporary ACC compatibility projection while ACC-only consumers are migrated.
- Update recorded count only from writer-committed summaries.
- Map `live → RECORDING`, non-recordable temporary states → `HOLDING`, and a matching-identity return to live → `RECORDING`.
- Map completion or worker failure to `UPLOAD_READY`, preserving the published partial file.
- Remove the renderer-owned writer session, append queue, pending acknowledgements, `appendTelemetrySample`, and `finalizeRecordingWrites`.
- Keep version-aware reading, validation, draft restoration, upload, and deletion.

`LiveSessionDetectionManager` selects a game-specific availability probe and must not start a second probe while a recording utility session exists, including `HOLDING`. ACC may retain its generic Python checker. iRacing detection/recordability should use a short-lived Node probe or existing desktop detection plus an SDK availability check; it must not create the recording manager or retain an iRacing SDK handle after the probe ends.

Generic Python IPC remains available for ACC detection, legacy file reading if still needed, analysis, and unrelated scripts. It is not part of iRacing live telemetry reading or recording.

## Termination and Continuity Rules

All three workers and the selected source terminate together when:

- The user stops recording.
- Upload begins.
- The user discards or resets the live session.
- The adapter identifies a different game session after reconnect.
- Any source/worker fails or exits unexpectedly.
- The renderer owning the session closes or is replaced.
- The application quits.

ACC pause/replay/`ACC_OFF` and iRacing replay/garage/short disconnect are non-recordable holding conditions; none alone creates or destroys a worker. A session resumes in the same file only when the adapter proves the same `sessionIdentity`. There are zero recording workers at startup, exactly one collector/writer/view group with one selected source while recording or holding, and zero recording resources after final shutdown.

A later **Start Recording** always creates three new utilities, three new channels, one new source instance, a new session ID, and a new JSONL file. Restored drafts never restart workers.

## Test Plan

- Manager tests:
  - constructor validation is side-effect free
  - startup/navigation/draft restoration creates no recording resource
  - `acc`, `iracing`, mismatched detected-game, and unsupported `ac` validation
  - only the selected game's source config resolves; iRacing never calls Python resolvers
  - exactly three forks/channels and correct endpoints for either supported game
  - source config/game mismatch rolls back startup
  - path relay without manager path ownership
  - readiness, stale session/game rejection, conflicts, idempotent stop, worker failure, renderer destruction, and quit
- Shared contract/collector tests:
  - normalized frame validation and compatibility projection
  - status-only frames never reach writer
  - one game and one session identity per recording
  - sequencing, batching, direct dual-port delivery, and adapter backpressure
  - adapter completion/fatal propagation and stop-before-end-of-stream ordering
- ACC adapter tests:
  - lazy Python spawn, split stdout lines, invalid JSON, and cleanup
  - field/unit normalization and legacy compatibility keys
  - pause/replay/off/unavailable classification
  - seven-field reconnect continuity and break detection
- iRacing adapter tests using captured SDK fixtures and a fake binding:
  - no Python process or resolver usage
  - telemetry/session-info normalization and unit conversion
  - duplicate tick suppression and prompt copy from SDK buffers
  - recordable driver frames versus replay/spectator/garage/disconnected frames
  - stable identity resume, changed identity completion, and reconnect regression handling
  - backpressure pause/resume and SDK handle cleanup across repeated start/stop
- Writer tests for both games:
  - unique exclusive path, schema/game/identity validation, contiguous JSONL, committed acknowledgements, final flush, and stream failures
  - unpublished empty-file startup rollback versus preservation after path publication
- Reader/upload tests:
  - legacy ACC rows remain readable
  - version 2 ACC and iRacing rows are readable/uploadable
  - mixed-game, mixed-identity, malformed, and metadata-game mismatch rejection
  - the Node reader reports bounded progress/chunks, honors cancellation, and is always used for iRacing
- View/preload/React tests:
  - 100 ms batching, every frame reaches intelligence, only latest commits to React
  - shared state/static/count propagation for ACC and iRacing
  - holding/resume/completion/failure transitions with the same session ID
  - upload waits for writer finalization and restored drafts start no utilities
- Packaging gates:
  - Jest and React production build
  - Electron packaged smoke test includes `electron/recording/**/*`
  - ACC packaged test resolves bundled Python only when ACC starts
  - iRacing packaged Windows test loads the native Node SDK from the packaged path, records valid version 2 JSONL, and starts no Python process
