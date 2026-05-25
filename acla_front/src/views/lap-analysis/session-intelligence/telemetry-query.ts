import { TelemetryQuery, TelemetrySample, QueryResult, FieldStats, QueryScope, EventType } from './types';
import { TelemetryBuffer } from './TelemetryBuffer';
import { EventLog } from './EventLog';

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

// Expand group aliases to raw field names. Unknown names passed through as-is.
function expandFields(fields: string[]): string[] {
    const expanded: string[] = [];
    for (const f of fields) {
        const group = FIELD_GROUPS[f];
        if (group) {
            expanded.push(...group);
        } else {
            expanded.push(f);
        }
    }
    return Array.from(new Set(expanded));
}

// ── Scope resolver ────────────────────────────────────────────────────────────

export function resolveScope(
    scope: QueryScope,
    buffer: TelemetryBuffer,
    log: EventLog,
    currentLap: number,
): TelemetrySample[] {
    switch (scope.type) {
        case 'last_seconds':
            return buffer.sliceByTime(scope.seconds * 1000);

        case 'event': {
            const events = log.find({
                eventType: scope.eventType as EventType,
                scope: scope.which === 'last' ? 'last' : 'last',
                currentLap,
            });
            if (events.length === 0) return [];
            const ev = events[events.length - 1];
            return buffer.slice(ev.startSampleIdx, ev.endSampleIdx + 1);
        }

        case 'lap': {
            if (scope.lap === 'current') {
                const evs = log.byLap(currentLap);
                if (evs.length === 0) return buffer.last(200);
                const start = evs[0].startSampleIdx;
                return buffer.slice(start, buffer.length);
            }
            if (scope.lap === 'last') {
                const evs = log.byLap(currentLap - 1);
                if (evs.length === 0) return [];
                const start = evs[0].startSampleIdx;
                const end = evs[evs.length - 1].endSampleIdx;
                return buffer.slice(start, end + 1);
            }
            // Numeric lap
            const evs = log.byLap(scope.lap);
            if (evs.length === 0) return [];
            return buffer.slice(evs[0].startSampleIdx, evs[evs.length - 1].endSampleIdx + 1);
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
    const min = Math.min(...values);
    const max = Math.max(...values);
    const variance = values.reduce((a, b) => a + (b - avg) ** 2, 0) / values.length;
    return { avg, min, max, stddev: Math.sqrt(variance) };
}

function reduceField(
    samples: TelemetrySample[],
    field: string,
    op: TelemetryQuery['reduce'],
): number | number[] | FieldStats {
    const values = extractValues(samples, field);
    switch (op) {
        case 'raw': return values;
        case 'avg': return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
        case 'min': return values.length ? Math.min(...values) : 0;
        case 'max': return values.length ? Math.max(...values) : 0;
        case 'stats': return computeStats(values);
        default: return values;
    }
}

// ── Public executor ───────────────────────────────────────────────────────────

export function executeQuery(
    query: TelemetryQuery,
    buffer: TelemetryBuffer,
    log: EventLog,
    currentLap: number,
): QueryResult {
    const samples = resolveScope(query.scope, buffer, log, currentLap);
    const rawFields = expandFields(query.fields);
    const result: QueryResult = {};
    for (const field of rawFields) {
        result[field] = reduceField(samples, field, query.reduce);
    }
    return result;
}
