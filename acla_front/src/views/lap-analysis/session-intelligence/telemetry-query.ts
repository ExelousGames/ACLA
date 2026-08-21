import {
    TelemetryQuery,
    TelemetrySample,
    QueryResult,
    FieldStats,
    QueryScope,
    ReduceOp,
    TelemetrySource,
    TelemetryValueByReduce,
} from './types';
import { getTelemetryLap } from './live-performance-analyst';

// ── Field groups ──────────────────────────────────────────────────────────────
// LLM uses group names; executor expands to raw Physics_* field names.

export const FIELD_GROUPS: Record<string, string[]> = {
    speed: ['Physics_speed_kmh'],
    throttle: ['Physics_gas'],
    brake: ['Physics_brake'],
    gear: ['Physics_gear'],
    steering: ['Physics_steer_angle'],
    rpm: ['Physics_rpm'],
    tyre_pressure: ['Physics_wheel_pressure_front_left', 'Physics_wheel_pressure_front_right', 'Physics_wheel_pressure_rear_left', 'Physics_wheel_pressure_rear_right'],
    tyre_temp: ['Physics_tyre_core_temp_front_left', 'Physics_tyre_core_temp_front_right', 'Physics_tyre_core_temp_rear_left', 'Physics_tyre_core_temp_rear_right'],
    brake_temp: ['Physics_brake_temp_front_left', 'Physics_brake_temp_front_right', 'Physics_brake_temp_rear_left', 'Physics_brake_temp_rear_right'],
    tyre_slip: ['Physics_wheel_slip_front_left', 'Physics_wheel_slip_front_right', 'Physics_wheel_slip_rear_left', 'Physics_wheel_slip_rear_right'],
    g_force: ['Physics_g_force_x', 'Physics_g_force_y', 'Physics_g_force_z'],
    suspension: ['Physics_suspension_travel_front_left', 'Physics_suspension_travel_front_right', 'Physics_suspension_travel_rear_left', 'Physics_suspension_travel_rear_right'],
    fuel: ['Physics_fuel'],
    lap_delta: ['Graphics_current_time_str', 'Graphics_last_time_str', 'Graphics_best_time_str'],
    position: ['Graphics_normalized_car_position'],
    race_position: ['Graphics_position'],
};

const FIELD_ALIASES: Record<string, string> = {
    fuel: 'Physics_fuel',
    fuellevel: 'Physics_fuel',
    fuel_level: 'Physics_fuel',
    tire_pressure: 'tyre_pressure',
    tirepressure: 'tyre_pressure',
    tyrepressure: 'tyre_pressure',
    tire_pressure_front_left: 'Physics_wheel_pressure_front_left',
    tire_pressure_front_right: 'Physics_wheel_pressure_front_right',
    tire_pressure_rear_left: 'Physics_wheel_pressure_rear_left',
    tire_pressure_rear_right: 'Physics_wheel_pressure_rear_right',
    tyre_pressure_front_left: 'Physics_wheel_pressure_front_left',
    tyre_pressure_front_right: 'Physics_wheel_pressure_front_right',
    tyre_pressure_rear_left: 'Physics_wheel_pressure_rear_left',
    tyre_pressure_rear_right: 'Physics_wheel_pressure_rear_right',
    tirepressurefl: 'Physics_wheel_pressure_front_left',
    tirepressurefr: 'Physics_wheel_pressure_front_right',
    tirepressurerl: 'Physics_wheel_pressure_rear_left',
    tirepressurerr: 'Physics_wheel_pressure_rear_right',
    tyrepressurefl: 'Physics_wheel_pressure_front_left',
    tyrepressurefr: 'Physics_wheel_pressure_front_right',
    tyrepressurerl: 'Physics_wheel_pressure_rear_left',
    tyrepressurerr: 'Physics_wheel_pressure_rear_right',
};

function normalizeFieldLookupKey(field: string): string {
    return field
        .trim()
        .toLowerCase()
        .replace(/[\s-]+/g, '_');
}

function resolveFieldAlias(field: string): string {
    const normalized = field.trim();
    return FIELD_ALIASES[normalizeFieldLookupKey(normalized)] ?? normalized;
}

// Expand group aliases to raw field names. Unknown names passed through as-is.
export function expandFields(fields: string[]): string[] {
    const expanded: string[] = [];
    for (const f of fields) {
        const field = resolveFieldAlias(f);
        const group = FIELD_GROUPS[field];
        if (group) {
            expanded.push(...group);
        } else {
            expanded.push(field);
        }
    }
    return Array.from(new Set(expanded));
}

export interface TelemetryScopeCollector {
    addRows(rows: TelemetrySample[]): void;
    getRows(): TelemetrySample[];
}

const getSampleTimestamp = (sample: TelemetrySample): number | null => {
    const value = sample.Physics_timestamp ?? sample.timestamp;
    const timestamp = Number(value);
    return value != null && Number.isFinite(timestamp) ? timestamp : null;
};

/**
 * Collects only the rows needed by a resolved scope while a JSONL file is
 * delivered in chunks. Row indexes are zero-based and match writer sequence
 * numbers minus one.
 */
export function createTelemetryScopeCollector(
    scope: Exclude<QueryScope, { type: 'event' }>,
    currentLap: number,
): TelemetryScopeCollector {
    let rowsSeen = 0;
    let selected: TelemetrySample[] = [];
    let selectedStart = 0;
    const fallbackCount = scope.type === 'last_seconds'
        ? Math.max(0, Math.ceil(scope.seconds * 1000 / 50))
        : 0;
    let fallbackRows: TelemetrySample[] = [];
    let lastRowHadTimestamp = false;

    const addRows = (rows: TelemetrySample[]) => {
        for (const sample of rows) {
            const rowIndex = rowsSeen;
            rowsSeen += 1;

            switch (scope.type) {
                case 'now':
                    selected = [sample];
                    selectedStart = 0;
                    break;

                case 'range':
                    if (rowIndex >= scope.start && rowIndex < scope.end) selected.push(sample);
                    break;

                case 'lap': {
                    const targetLap = scope.lap === 'current'
                        ? currentLap
                        : scope.lap === 'last'
                            ? currentLap - 1
                            : scope.lap;
                    if (getTelemetryLap(sample) === targetLap) selected.push(sample);
                    break;
                }

                case 'last_seconds': {
                    fallbackRows.push(sample);
                    if (fallbackRows.length > fallbackCount) {
                        fallbackRows.splice(0, fallbackRows.length - fallbackCount);
                    }

                    selected.push(sample);
                    const timestamp = getSampleTimestamp(sample);
                    lastRowHadTimestamp = timestamp !== null;
                    if (timestamp !== null) {
                        const cutoff = timestamp - scope.seconds * 1000;
                        while (selectedStart < selected.length) {
                            const candidateTimestamp = getSampleTimestamp(selected[selectedStart]);
                            if (candidateTimestamp !== null && candidateTimestamp >= cutoff) break;
                            selectedStart += 1;
                        }
                        if (selectedStart > 1024 && selectedStart * 2 > selected.length) {
                            selected = selected.slice(selectedStart);
                            selectedStart = 0;
                        }
                    }
                    break;
                }
            }
        }
    };

    return {
        addRows,
        getRows: () => (
            scope.type === 'last_seconds' && !lastRowHadTimestamp
                ? fallbackRows.slice()
                : selected.slice(selectedStart)
        ),
    };
}

// ── Scope resolver ────────────────────────────────────────────────────────────

export function resolveScope(
    scope: QueryScope,
    buffer: TelemetrySource,
    currentLap: number,
): TelemetrySample[] {
    switch (scope.type) {
        case 'now':
            return buffer.last(1);

        case 'last_seconds':
            return buffer.sliceByTime(scope.seconds * 1000);

        // Event ranges are owned and resolved by the mounted event-log
        // visualization before the live-session query is executed.
        case 'event':
            return [];

        case 'lap': {
            const targetLap = scope.lap === 'current'
                ? currentLap
                : scope.lap === 'last'
                    ? currentLap - 1
                    : scope.lap;
            const oldestAvailableIndex = buffer.length - buffer.size;
            return buffer
                .slice(oldestAvailableIndex, buffer.length)
                .filter((sample) => getTelemetryLap(sample) === targetLap);
        }

        case 'range':
            return buffer.slice(scope.start, scope.end);

        default:
            return [];
    }
}

// ── Reducer ───────────────────────────────────────────────────────────────────

function extractValues(samples: TelemetrySample[], field: string): number[] {
    return samples
        .map(s => s[field])
        .filter(v => typeof v === 'number') as number[];
}

function computeStats(values: number[]): FieldStats {
    if (values.length === 0) return { avg: 0, min: 0, max: 0, stddev: 0 };
    const avg = values.reduce((a, b) => a + b, 0) / values.length;
    const min = values.reduce((current, value) => Math.min(current, value), values[0]);
    const max = values.reduce((current, value) => Math.max(current, value), values[0]);
    const variance = values.reduce((a, b) => a + (b - avg) ** 2, 0) / values.length;
    return { avg, min, max, stddev: Math.sqrt(variance) };
}

const REDUCERS: {
    [TReduce in ReduceOp]: (values: number[]) => TelemetryValueByReduce[TReduce];
} = {
    raw: (values) => values,
    avg: (values) => values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0,
    min: (values) => values.length
        ? values.reduce((current, value) => Math.min(current, value), values[0])
        : 0,
    max: (values) => values.length
        ? values.reduce((current, value) => Math.max(current, value), values[0])
        : 0,
    stats: computeStats,
};

function reduceField<TReduce extends ReduceOp>(
    samples: TelemetrySample[],
    field: string,
    op: TReduce,
): TelemetryValueByReduce[TReduce] {
    return REDUCERS[op](extractValues(samples, field));
}

/** Reduces already-selected samples using the canonical aliases and groups. */
export function reduceTelemetrySamples<TReduce extends ReduceOp>(
    samples: TelemetrySample[],
    fields: string[],
    op: TReduce,
): QueryResult<TReduce> {
    const rawFields = expandFields(fields);
    const result: QueryResult<TReduce> = {};
    for (const field of rawFields) {
        result[field] = reduceField(samples, field, op);
    }
    return result;
}

// ── Public executor ───────────────────────────────────────────────────────────

export function executeQuery<TReduce extends ReduceOp>(
    query: TelemetryQuery<TReduce>,
    buffer: TelemetrySource,
    currentLap: number,
): QueryResult<TReduce> {
    const samples = resolveScope(query.scope, buffer, currentLap);
    return reduceTelemetrySamples(samples, query.fields, query.reduce);
}
