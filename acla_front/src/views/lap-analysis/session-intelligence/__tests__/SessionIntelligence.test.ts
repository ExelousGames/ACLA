import { SessionIntelligence } from '../SessionIntelligence';
import type { FieldStats, QueryResult, TelemetryQuery } from '../types';

// @ts-expect-error QueryResult requires an explicit reduction type.
type MissingQueryResultGeneric = QueryResult;

// @ts-expect-error TelemetryQuery requires an explicit reduction type.
type MissingTelemetryQueryGeneric = TelemetryQuery;

const sample = (lap: number, position: number, extra: Record<string, unknown> = {}) => ({
    Static_track: 'brands_hatch',
    Graphics_completed_laps: lap,
    Graphics_normalized_car_position: position,
    ...extra,
});

describe('SessionIntelligence normalized range windows', () => {
    it('returns rows for a non-wrapped normalized range', () => {
        const session = new SessionIntelligence();
        session.tick(sample(1, 0.1));
        session.tick(sample(1, 0.2));
        session.tick(sample(1, 0.3));

        const result = session.getTelemetryWindowForNormalizedRange({
            start_position: 0.15,
            end_position: 0.25,
            lap: 1,
        });

        expect(result).toMatchObject({
            status: 'ready',
            lap: 1,
            startSampleIdx: 1,
            endSampleIdx: 1,
        });
        expect(result.rows).toHaveLength(1);
        expect(result.rows[0].Graphics_normalized_car_position).toBe(0.2);
    });

    it('returns rows for a wrapped normalized range across start finish', () => {
        const session = new SessionIntelligence();
        session.tick(sample(0, 0.88));
        session.tick(sample(0, 0.92));
        session.tick(sample(0, 0.98));
        session.tick(sample(1, 0.02));
        session.tick(sample(1, 0.08));
        session.tick(sample(1, 0.14));

        const result = session.getTelemetryWindowForNormalizedRange({
            start_position: 0.9,
            end_position: 0.1,
            lap: 1,
        });

        expect(result).toMatchObject({
            status: 'ready',
            lap: 1,
            startSampleIdx: 1,
            endSampleIdx: 4,
        });
        expect(result.rows.map((row) => row.Graphics_normalized_car_position)).toEqual([
            0.92,
            0.98,
            0.02,
            0.08,
        ]);
    });

    it('returns empty when no telemetry rows match the range', () => {
        const session = new SessionIntelligence();
        session.tick(sample(0, 0.1));

        expect(session.getTelemetryWindowForNormalizedRange({
            start_position: 0.5,
            end_position: 0.6,
            lap: 0,
        })).toMatchObject({
            status: 'empty',
            rows: [],
        });
    });
});

describe('SessionIntelligence telemetry queries', () => {
    it('returns reduction-specific values with inferred result types', () => {
        const session = new SessionIntelligence();
        session.tick(sample(1, 0.1, {
            Physics_timestamp: 0,
            Physics_speed_kmh: 100,
        }));
        session.tick(sample(1, 0.2, {
            Physics_timestamp: 100,
            Physics_speed_kmh: 120,
        }));

        const raw: Record<string, number[]> = session.query({
            fields: ['speed'],
            scope: { type: 'last_seconds', seconds: 1 },
            reduce: 'raw',
        });
        const avg: Record<string, number> = session.query({
            fields: ['speed'],
            scope: { type: 'last_seconds', seconds: 1 },
            reduce: 'avg',
        });
        const min: Record<string, number> = session.query({
            fields: ['speed'],
            scope: { type: 'last_seconds', seconds: 1 },
            reduce: 'min',
        });
        const max: Record<string, number> = session.query({
            fields: ['speed'],
            scope: { type: 'last_seconds', seconds: 1 },
            reduce: 'max',
        });
        const stats: Record<string, FieldStats> = session.query({
            fields: ['speed'],
            scope: { type: 'last_seconds', seconds: 1 },
            reduce: 'stats',
        });

        // @ts-expect-error Raw query results cannot be assigned to scalar values.
        const incompatible: Record<string, number> = raw;
        expect(incompatible).toBe(raw);
        expect(raw).toEqual({ Physics_speed_kmh: [100, 120] });
        expect(avg).toEqual({ Physics_speed_kmh: 110 });
        expect(min).toEqual({ Physics_speed_kmh: 100 });
        expect(max).toEqual({ Physics_speed_kmh: 120 });
        expect(stats).toEqual({
            Physics_speed_kmh: { avg: 110, min: 100, max: 120, stddev: 10 },
        });
    });

    it('returns reducer defaults when the scope has no samples', () => {
        const session = new SessionIntelligence();

        expect(session.query({
            fields: ['speed'],
            scope: { type: 'now' },
            reduce: 'raw',
        })).toEqual({ Physics_speed_kmh: [] });
        expect(session.query({
            fields: ['speed'],
            scope: { type: 'now' },
            reduce: 'avg',
        })).toEqual({ Physics_speed_kmh: 0 });
        expect(session.query({
            fields: ['speed'],
            scope: { type: 'now' },
            reduce: 'stats',
        })).toEqual({
            Physics_speed_kmh: { avg: 0, min: 0, max: 0, stddev: 0 },
        });
    });

    it('preserves alias and group expansion while removing duplicate fields', () => {
        const session = new SessionIntelligence();
        session.tick(sample(1, 0.1, {
            Physics_fuel: 42,
            Physics_wheel_pressure_front_left: 27,
            Physics_wheel_pressure_front_right: 28,
            Physics_wheel_pressure_rear_left: 29,
            Physics_wheel_pressure_rear_right: 30,
        }));

        expect(session.query({
            fields: ['fuel_level'],
            scope: { type: 'now' },
            reduce: 'avg',
        })).toEqual({ Physics_fuel: 42 });

        expect(session.query({
            fields: ['tyre_pressure', 'tire_pressure', 'Physics_wheel_pressure_front_left'],
            scope: { type: 'now' },
            reduce: 'avg',
        })).toEqual({
            Physics_wheel_pressure_front_left: 27,
            Physics_wheel_pressure_front_right: 28,
            Physics_wheel_pressure_rear_left: 29,
            Physics_wheel_pressure_rear_right: 30,
        });
    });
});
