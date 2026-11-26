import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { validateLiveTelemetryRow } from 'data/live-telemetry-dataset';
import { isDesktopGame, type DesktopGame } from 'contexts/DesktopGameContext';
import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import type { LiveTelemetrySnapshot } from 'views/live-session/live-telemetry-store';
import type { StandardTelemetrySample } from 'views/live-session/live-session-types';
import type { PolicyCameraImage } from 'views/live-session/live-session-recording/policy-camera';
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
    /** Destination gear and largest telemetry gap in a shift through neutral. */
    neutralShift?: { gear: number; maxGapSeconds: number };
}
export interface RecordedDrivingSession {
    id: string;
    game: DesktopGame;
    rows: RecordedDrivingRow[];
    startTimeSeconds: number;
    durationSeconds: number;
    timestampField: RecordedClock;
}
export interface RecordedGearShift {
    direction: 'up' | 'down';
    before: number;
    after: number;
}
export type RecordedObservationResult = ObservationResult & { shiftGear?: number };

let nextSessionId = 0;
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const object = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === 'object' && !Array.isArray(value);

// Replay needs the observation inputs, reference-lap context and off-track reward data.
const REPLAY_TELEMETRY_FIELDS = [
    'Graphics_analyst_session_time', 'Graphics_game_time', 'Graphics_status', 'Graphics_normalized_car_position',
    'Graphics_track_grip_status', 'Graphics_player_track_surface', 'Static_car_model', 'Static_track', 'Static_number_of_session',
    'Physics_gas', 'Physics_brake', 'Physics_steer_angle', 'Physics_gear', 'Physics_rpm', 'Physics_speed_kmh',
    'Physics_local_velocity_x', 'Physics_local_velocity_y', 'Physics_local_velocity_z',
    'Physics_g_force_x', 'Physics_g_force_y', 'Physics_g_force_z',
    'Physics_local_angular_vel_y', 'Physics_local_angular_vel_z', 'Physics_autoshifter_on',
    'Physics_tyre_contact_point_front_left_x', 'Physics_tyre_contact_point_front_left_y',
    'Physics_tyre_contact_point_front_right_x', 'Physics_tyre_contact_point_front_right_y',
    'Physics_tyre_contact_point_rear_left_x', 'Physics_tyre_contact_point_rear_left_y',
    'Physics_tyre_contact_point_rear_right_x', 'Physics_tyre_contact_point_rear_right_y',
] as const satisfies readonly (keyof StandardTelemetrySample)[];

function selectReplayTelemetry(sample: StandardTelemetrySample): StandardTelemetrySample {
    return Object.fromEntries(REPLAY_TELEMETRY_FIELDS.filter((field) => sample[field] !== undefined)
        .map((field) => [field, sample[field]]));
}

function createRecordedDrivingParser(fallbackGame: DesktopGame, projectSample?: typeof selectReplayTelemetry) {
    if (!isDesktopGame(fallbackGame)) throw new Error('Choose the simulator used for this recording.');
    const rows: RecordedDrivingRow[] = [];
    let game: DesktopGame | undefined;
    let timestampField: RecordedClock | undefined;
    let startTimeSeconds = 0, previousTime = -Infinity, segment = 0, previousContext: string | undefined;
    let lineNumber = 0;

    const addLine = (line: string) => {
        if (++lineNumber === 1) line = line.replace(/^\uFEFF/, '');
        if (!line.trim()) return;
        const fail = (message: string): never => { throw new Error(`Telemetry JSONL line ${lineNumber}: ${message}`); };
        let parsed: unknown;
        try { parsed = JSON.parse(line); } catch { fail('invalid JSON.'); }
        if (!object(parsed)) return fail('expected a telemetry object.');
        // Source-frame exports may wrap the canonical row in { game, sample }.
        const source = object(parsed.sample) ? parsed.sample : parsed;
        let flatSample = source;
        let declaredGame: unknown = parsed.game ?? undefined;
        if (Object.prototype.hasOwnProperty.call(source, 'game')) {
            const { game: embeddedGame, ...fields } = source;
            declaredGame ??= embeddedGame;
            flatSample = fields;
        }
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
        rows.push({ sample: projectSample?.(sample) ?? sample, index: rows.length, timeSeconds: time - startTimeSeconds, segment });
        previousTime = time;
        previousContext = context;
    };
    const finish = (): RecordedDrivingSession => {
        if (!rows.length || !game || !timestampField) throw new Error('The telemetry JSONL file contains no samples.');
        labelNeutralTransitions(rows);
        return { id: `recorded-${++nextSessionId}`, game, rows, startTimeSeconds,
            durationSeconds: rows[rows.length - 1].timeSeconds, timestampField };
    };
    return { addLine, finish };
}

/** Imports flat telemetry JSONL already available as text. */
export function parseRecordedDrivingSession(text: string, fallbackGame: DesktopGame): RecordedDrivingSession {
    const parser = createRecordedDrivingParser(fallbackGame);
    text.split(/\r?\n/).forEach(parser.addLine);
    return parser.finish();
}

/** Imports buffered UTF-8 telemetry and retains the fields used for seeking and training. */
export async function parseRecordedDrivingSessionStream(stream: ReadableStream<Uint8Array>, fallbackGame: DesktopGame,
    signal?: AbortSignal): Promise<RecordedDrivingSession> {
    const checkAbort = () => {
        if (signal?.aborted) throw new DOMException('Recording operation cancelled.', 'AbortError');
    };
    checkAbort();
    const parser = createRecordedDrivingParser(fallbackGame, selectReplayTelemetry);
    // Keep BOM handling in the shared line parser.
    const decoder = new TextDecoder('utf-8', { ignoreBOM: true });
    const reader = stream.getReader();
    const cancel = () => { void reader.cancel().catch(() => undefined); };
    signal?.addEventListener('abort', cancel, { once: true });
    let pending = '';
    let exhausted = false;
    try {
        while (true) {
            const { done, value } = await reader.read();
            checkAbort();
            if (done) {
                exhausted = true;
                parser.addLine(pending + decoder.decode());
                return parser.finish();
            }
            const lines = (pending + decoder.decode(value, { stream: true })).split('\n');
            pending = lines.pop()!;
            lines.forEach(parser.addLine);
        }
    } finally {
        signal?.removeEventListener('abort', cancel);
        if (!exhausted && !signal?.aborted) cancel();
        reader.releaseLock();
    }
}

function labelNeutralTransitions(rows: RecordedDrivingRow[]) {
    let engaged: { index: number; segment: number; gear: number } | undefined;
    let maxGapSeconds = 0;
    for (const { sample, index, segment, timeSeconds } of rows) {
        if (!finite(sample.Physics_gear) || sample.Graphics_status !== ACC_STATUS.ACC_LIVE) {
            engaged = undefined;
            continue;
        }
        const gear = sample.Physics_gear - 1;
        if (engaged?.segment === segment) maxGapSeconds = Math.max(maxGapSeconds, timeSeconds - rows[index - 1].timeSeconds);
        if (gear === 0) continue;

        // Preserve neutral as a separate step between reverse and first in either direction.
        if (engaged?.segment === segment && index > engaged.index + 1 && !(Math.abs(gear) === 1 && engaged.gear === -gear)) {
            // Use the destination gear when neutral starts so the shift is learned at its onset.
            const shift = { gear, maxGapSeconds };
            for (let i = engaged.index + 1; i < index; i++) rows[i].neutralShift = shift;
        }
        engaged = { index, segment, gear };
        maxGapSeconds = 0;
    }
}

/** Finds single forward-gear changes bracketed by live samples on the recording clock. */
export function findRecordedGearShifts({ rows }: RecordedDrivingSession, maxSeconds: number): RecordedGearShift[] {
    const shifts: RecordedGearShift[] = [];
    let previous: { time: number; gear: number; segment: number } | undefined;
    for (let index = 0; index < rows.length; index++) {
        const { sample, segment, timeSeconds } = rows[index];
        // Match telemetry lookup, which takes the last sample at a shared timestamp.
        if (rows[index + 1]?.timeSeconds === timeSeconds) continue;
        if (previous?.segment !== segment) previous = undefined;
        const gear = sample.Physics_gear;
        if (sample.Graphics_status !== ACC_STATUS.ACC_LIVE || !finite(gear) || !Number.isInteger(gear) || gear < 1) {
            previous = undefined;
            continue;
        }
        if (gear === 1) continue; // Allow brief neutral between forward gears.
        if (previous) {
            const delta = gear - previous.gear;
            const elapsed = timeSeconds - previous.time;
            if (Math.abs(delta) === 1 && elapsed > 0 && elapsed <= maxSeconds + 1e-9) {
                shifts.push({ direction: delta > 0 ? 'up' : 'down', before: previous.time, after: timeSeconds });
            }
        }
        previous = { gear, segment, time: timeSeconds };
    }
    return shifts;
}

/** Finds the most recent sample at the frame's time on the telemetry clock. */
export function findRecordedTelemetry(session: RecordedDrivingSession, telemetrySeconds: number,
    maxAgeMs = DEFAULT_DRIVING_CONFIG.runtime.telemetryMaxAgeMs): RecordedDrivingRow | null {
    if (!finite(telemetrySeconds) || !finite(maxAgeMs) || maxAgeMs < 0) return null;
    const time = telemetrySeconds - session.startTimeSeconds;
    if (time < -1e-9 || !session.rows.length) return null;
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
export function readRecordedDrivingObservation(session: RecordedDrivingSession, telemetrySeconds: number, camera: PolicyCameraImage,
    referenceLap: TopLapReferenceEntry | null, aggression = DEFAULT_DRIVING_CONFIG.aggression,
    config: DrivingModelConfig = DEFAULT_DRIVING_CONFIG, circuitMap: CircuitMapDto | null = null): RecordedObservationResult {
    const row = findRecordedTelemetry(session, telemetrySeconds, config.runtime.telemetryMaxAgeMs);
    if (!row) return { error: 'No telemetry close enough to this recorded frame. Choose another frame.' };
    const sample = row.sample;
    if (sample.Graphics_status !== ACC_STATUS.ACC_LIVE) {
        return { error: 'This recorded telemetry sample is not marked as running (Graphics_status = 2).' };
    }
    const measuredAt = (session.startTimeSeconds + row.timeSeconds) * 1000;
    // Clock subtraction can differ by a fraction of a microsecond at an exact match.
    const capturedAt = Math.max(measuredAt, telemetrySeconds * 1000);
    const snapshot: LiveTelemetrySnapshot = {
        game: session.game, receivedAt: measuredAt,
        currentTelemetry: sample, graphicsTelemetry: sample, physicsTelemetry: sample,
        sampleIndex: row.index, committedSampleCount: session.rows.length,
        telemetryStatus: ACC_STATUS.ACC_LIVE, sessionGeneration: row.segment, streamGeneration: 0,
    };
    const result = readDrivingObservation(snapshot, { capturedAt, camera, telemetry: snapshot },
        referenceLap, aggression, capturedAt, config, circuitMap);
    if (!result.observation) return { error: result.error.replace(/Live Map/g, 'Reference map').replace(/in Live Session/g, 'for this recording') };
    const shift = row.neutralShift;
    const shiftGear = shift && shift.maxGapSeconds <= config.runtime.maxTransitionSeconds ? shift.gear : undefined;
    return { ...result, shiftGear,
        observation: { ...result.observation,
        session: JSON.stringify([session.id, row.segment, session.game, sample.Static_car_model, sample.Static_track]) } };
}
