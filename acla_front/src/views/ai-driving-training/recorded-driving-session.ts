import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { validateLiveTelemetryRow } from 'data/live-telemetry-dataset';
import { isDesktopGame, type DesktopGame } from 'contexts/DesktopGameContext';
import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import type { LiveTelemetrySnapshot } from 'views/live-session/live-telemetry-store';
import type { StandardTelemetrySample } from 'views/live-session/live-session-types';
import type { PolicyCameraImage } from 'views/live-session/track-vision/policy-camera';
import { DEFAULT_DRIVING_CONFIG, type DrivingModelConfig } from './driving-model-config';
import { readDrivingObservation, type ObservationResult } from './driving-observation';

type RecordedClock = 'Graphics_analyst_session_time' | 'Graphics_game_time';
export interface RecordedDrivingRow {
    sample: StandardTelemetrySample;
    index: number;
    /** Seconds after the first telemetry row. */
    timeSeconds: number;
    /** Changes when the car, circuit or simulator session changes. */
    segment: number;
}
export interface RecordedDrivingSession {
    id: string;
    game: DesktopGame;
    rows: RecordedDrivingRow[];
    startTimeSeconds: number;
    durationSeconds: number;
    timestampField: RecordedClock;
}

let nextSessionId = 0;
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const object = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === 'object' && !Array.isArray(value);

/** Imports the same flat telemetry JSONL written by the desktop recorder. */
export function parseRecordedDrivingSession(text: string, fallbackGame: DesktopGame): RecordedDrivingSession {
    if (!isDesktopGame(fallbackGame)) throw new Error('Choose the simulator used for this recording.');
    const rows: RecordedDrivingRow[] = [];
    let game: DesktopGame | undefined;
    let timestampField: RecordedClock | undefined;
    let startTimeSeconds = 0, previousTime = -Infinity, segment = 0, previousContext: string | undefined;
    text.replace(/^\uFEFF/, '').split(/\r?\n/).forEach((line, lineIndex) => {
        if (!line.trim()) return;
        const fail = (message: string): never => { throw new Error(`Telemetry JSONL line ${lineIndex + 1}: ${message}`); };
        let parsed: unknown;
        try { parsed = JSON.parse(line); } catch { fail('invalid JSON.'); }
        if (!object(parsed)) return fail('expected a telemetry object.');
        // Source-frame exports may wrap the canonical row in { game, sample }.
        const source = object(parsed.sample) ? parsed.sample : parsed;
        const { game: embeddedGame, ...flatSample } = source;
        const declaredGame = parsed.game ?? embeddedGame;
        if (declaredGame !== undefined && !isDesktopGame(declaredGame)) return fail('unknown simulator.');
        const rowGame = isDesktopGame(declaredGame) ? declaredGame : game ?? fallbackGame;
        if (game !== undefined && rowGame !== game) return fail('the simulator changed; import one simulator per recording.');
        game = rowGame;
        const validation = validateLiveTelemetryRow(flatSample);
        if (!validation.ok) return fail(validation.error!);
        const sample = flatSample as StandardTelemetrySample;
        if (!timestampField) timestampField = finite(sample.Graphics_analyst_session_time)
            ? 'Graphics_analyst_session_time' : 'Graphics_game_time';
        const time = sample[timestampField];
        if (!finite(time) || time < 0) return fail(`a non-negative ${timestampField} timestamp in seconds is required.`);
        if (time < previousTime) return fail('timestamps move backwards; split recordings at clock resets.');
        if (!rows.length) startTimeSeconds = time;
        const context = JSON.stringify([sample.Static_car_model, sample.Static_track, sample.Static_number_of_session]);
        if (previousContext !== undefined && context !== previousContext) segment++;
        rows.push({ sample, index: rows.length, timeSeconds: time - startTimeSeconds, segment });
        previousTime = time;
        previousContext = context;
    });
    if (!rows.length || !game || !timestampField) throw new Error('The telemetry JSONL file contains no samples.');
    return { id: `recorded-${++nextSessionId}`, game, rows, startTimeSeconds,
        durationSeconds: rows[rows.length - 1].timeSeconds, timestampField };
}

/** Positive offset means the first telemetry sample occurs later in the video. */
export function findRecordedTelemetry(session: RecordedDrivingSession, videoSeconds: number, offsetSeconds = 0,
    maxAgeMs = DEFAULT_DRIVING_CONFIG.runtime.telemetryMaxAgeMs): RecordedDrivingRow | null {
    if (!finite(videoSeconds) || videoSeconds < 0 || !finite(offsetSeconds) || !finite(maxAgeMs) || maxAgeMs < 0) return null;
    const time = videoSeconds - offsetSeconds;
    if (time < 0 || !session.rows.length) return null;
    let low = 0, high = session.rows.length;
    // Upper bound chooses the last sample at/before the image, never a future row.
    while (low < high) {
        const middle = Math.floor((low + high) / 2);
        if (session.rows[middle].timeSeconds <= time + 1e-9) low = middle + 1;
        else high = middle;
    }
    const row = session.rows[low - 1];
    return row && (time - row.timeSeconds) * 1000 <= maxAgeMs + 1e-6 ? row : null;
}

/** Uses the recording's clock and pixels without touching the live telemetry store. */
export function readRecordedDrivingObservation(session: RecordedDrivingSession, videoSeconds: number, camera: PolicyCameraImage,
    referenceLap: TopLapReferenceEntry | null, aggression = DEFAULT_DRIVING_CONFIG.aggression,
    config: DrivingModelConfig = DEFAULT_DRIVING_CONFIG, circuitMap: CircuitMapDto | null = null, offsetSeconds = 0): ObservationResult {
    const row = findRecordedTelemetry(session, videoSeconds, offsetSeconds, config.runtime.telemetryMaxAgeMs);
    if (!row) return { error: 'No telemetry close enough to this recorded frame. Check the video offset or choose another frame.' };
    const sample = row.sample;
    if (sample.Graphics_status !== ACC_STATUS.ACC_LIVE) {
        return { error: 'This recorded telemetry sample is not marked as running (Graphics_status = 2).' };
    }
    const measuredAt = (session.startTimeSeconds + row.timeSeconds) * 1000;
    // Clock subtraction can differ by a fraction of a microsecond at an exact match.
    const capturedAt = Math.max(measuredAt, (session.startTimeSeconds + videoSeconds - offsetSeconds) * 1000);
    const snapshot: LiveTelemetrySnapshot = {
        game: session.game, receivedAt: measuredAt,
        currentTelemetry: sample, graphicsTelemetry: sample, physicsTelemetry: sample,
        sampleIndex: row.index, committedSampleCount: session.rows.length,
        telemetryStatus: ACC_STATUS.ACC_LIVE, sessionGeneration: row.segment, streamGeneration: 0,
    };
    const result = readDrivingObservation(snapshot, { capturedAt, camera, telemetry: snapshot,
        width: camera.width, height: camera.height, detections: {}, reconstruction: null, geometry: null },
    referenceLap, aggression, capturedAt, config, circuitMap);
    if (!result.observation) return { error: result.error.replace(/Live Map/g, 'Reference map').replace(/in Live Session/g, 'for this recording') };
    return { ...result, observation: { ...result.observation,
        session: JSON.stringify([session.id, row.segment, session.game, sample.Static_car_model, sample.Static_track]) } };
}
