import {
    createTelemetryScopeCollector,
    reduceTelemetrySamples,
} from '../telemetry-query';
import type {
    FieldStats,
    QueryResult,
    ReduceOp,
    TelemetryQuery,
    TelemetrySample,
} from '../types';

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

const createTelemetryQuery = () => {
    const telemetry: TelemetrySample[] = [];
    return {
        query: <TReduce extends ReduceOp>(query: TelemetryQuery<TReduce>) => {
            const collector = createTelemetryScopeCollector(
                query.scope.type === 'event' ? { type: 'range', start: 0, end: 0 } : query.scope,
                1,
            );
            collector.addRows(telemetry);
            return reduceTelemetrySamples(collector.getRows(), query.fields, query.reduce);
        },
        tick: (nextSample: TelemetrySample) => {
            telemetry.push(nextSample);
        },
        telemetry,
    };
};

describe('telemetry query execution', () => {
    it.each([
        [{ type: 'now' } as const, [4]],
        [{ type: 'range', start: 1, end: 4 } as const, [1, 2, 3]],
        [{ type: 'lap', lap: 'current' } as const, [2, 3, 4]],
        [{ type: 'lap', lap: 'last' } as const, [0, 1]],
        [{ type: 'last_seconds', seconds: 0.15 } as const, [3, 4]],
    ])('collects a %o scope across recorded-file chunks', (scope, expectedSpeeds) => {
        const rows = Array.from({ length: 5 }, (_, index) => sample(
            index < 2 ? 1 : 2,
            index / 10,
            { Physics_timestamp: index * 100, Physics_speed_kmh: index },
        ));
        const collector = createTelemetryScopeCollector(scope, 2);

        collector.addRows(rows.slice(0, 2));
        collector.addRows(rows.slice(2, 4));
        collector.addRows(rows.slice(4));

        expect(collector.getRows().map((row) => row.Physics_speed_kmh)).toEqual(expectedSpeeds);
    });

    it('falls back to the 20 Hz window when the latest row has no timestamp', () => {
        const collector = createTelemetryScopeCollector({ type: 'last_seconds', seconds: 0.1 }, 1);
        collector.addRows([
            sample(1, 0.1, { Physics_speed_kmh: 1 }),
            sample(1, 0.2, { Physics_speed_kmh: 2 }),
            sample(1, 0.3, { Physics_speed_kmh: 3 }),
        ]);

        expect(collector.getRows().map((row) => row.Physics_speed_kmh)).toEqual([2, 3]);
    });

    it('reduces file-selected rows with canonical aliases and groups', () => {
        const rows = [sample(1, 0.1, {
            Physics_fuel: 40,
            Physics_wheel_pressure_front_left: 27,
            Physics_wheel_pressure_front_right: 28,
            Physics_wheel_pressure_rear_left: 29,
            Physics_wheel_pressure_rear_right: 30,
        }), sample(1, 0.2, {
            Physics_fuel: 42,
            Physics_wheel_pressure_front_left: 29,
            Physics_wheel_pressure_front_right: 30,
            Physics_wheel_pressure_rear_left: 31,
            Physics_wheel_pressure_rear_right: 32,
        })];

        expect(reduceTelemetrySamples(rows, ['fuel_level'], 'avg')).toEqual({ Physics_fuel: 41 });
        expect(reduceTelemetrySamples(rows, ['tire_pressure'], 'max')).toEqual({
            Physics_wheel_pressure_front_left: 29,
            Physics_wheel_pressure_front_right: 30,
            Physics_wheel_pressure_rear_left: 31,
            Physics_wheel_pressure_rear_right: 32,
        });
    });

    it('queries recorded telemetry without appending to it', () => {
        const { query, telemetry } = createTelemetryQuery();
        telemetry.push(sample(1, 0.2, { Physics_speed_kmh: 120 }));

        expect(telemetry).toHaveLength(1);
        expect(query({
            fields: ['speed'],
            scope: { type: 'now' },
            reduce: 'avg',
        })).toEqual({ Physics_speed_kmh: 120 });
        expect(telemetry).toHaveLength(1);
    });

    it('returns reduction-specific values with inferred result types', () => {
        const { query, tick } = createTelemetryQuery();
        tick(sample(1, 0.1, {
            Physics_timestamp: 0,
            Physics_speed_kmh: 100,
        }));
        tick(sample(1, 0.2, {
            Physics_timestamp: 100,
            Physics_speed_kmh: 120,
        }));

        const raw: Record<string, number[]> = query({
            fields: ['speed'],
            scope: { type: 'last_seconds', seconds: 1 },
            reduce: 'raw',
        });
        const avg: Record<string, number> = query({
            fields: ['speed'],
            scope: { type: 'last_seconds', seconds: 1 },
            reduce: 'avg',
        });
        const min: Record<string, number> = query({
            fields: ['speed'],
            scope: { type: 'last_seconds', seconds: 1 },
            reduce: 'min',
        });
        const max: Record<string, number> = query({
            fields: ['speed'],
            scope: { type: 'last_seconds', seconds: 1 },
            reduce: 'max',
        });
        const stats: Record<string, FieldStats> = query({
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
        const { query } = createTelemetryQuery();

        expect(query({
            fields: ['speed'],
            scope: { type: 'now' },
            reduce: 'raw',
        })).toEqual({ Physics_speed_kmh: [] });
        expect(query({
            fields: ['speed'],
            scope: { type: 'now' },
            reduce: 'avg',
        })).toEqual({ Physics_speed_kmh: 0 });
        expect(query({
            fields: ['speed'],
            scope: { type: 'now' },
            reduce: 'stats',
        })).toEqual({
            Physics_speed_kmh: { avg: 0, min: 0, max: 0, stddev: 0 },
        });
    });

    it('preserves alias and group expansion while removing duplicate fields', () => {
        const { query, tick } = createTelemetryQuery();
        tick(sample(1, 0.1, {
            Physics_fuel: 42,
            Physics_wheel_pressure_front_left: 27,
            Physics_wheel_pressure_front_right: 28,
            Physics_wheel_pressure_rear_left: 29,
            Physics_wheel_pressure_rear_right: 30,
        }));

        expect(query({
            fields: ['fuel_level'],
            scope: { type: 'now' },
            reduce: 'avg',
        })).toEqual({ Physics_fuel: 42 });

        expect(query({
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
