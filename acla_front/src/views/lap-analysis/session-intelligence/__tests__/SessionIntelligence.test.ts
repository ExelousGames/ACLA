import { SessionIntelligence } from '../SessionIntelligence';

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
