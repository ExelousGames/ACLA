import { getPlaybackFrameIndex, parseTelemetryFrame, parseTelemetryFrames, segmentVisiblePoints } from './mapTelemetry';

describe('MapVisualization telemetry parsing', () => {
    it('parses array coordinates and matches the player by car id', () => {
        const frame = parseTelemetryFrame({
            Graphics_current_time: 1.5,
            Graphics_player_car_id: 20,
            Graphics_car_id: [10, 20],
            Graphics_car_coordinates: [
                { x: 100, y: 10, z: 1 },
                { x: 120, y: 12, z: 2 }
            ]
        }, 0);

        expect(frame?.playerKey).toBe('id:20');
        expect(frame?.cars).toHaveLength(2);
        expect(frame?.cars[1].position).toEqual({ x: 120, y: 12, z: 2 });
    });

    it('parses JSON-string coordinates and car ids', () => {
        const frame = parseTelemetryFrame({
            Graphics_current_time: 2500,
            Graphics_player_car_id: 10,
            Graphics_car_id: JSON.stringify([10]),
            Graphics_car_coordinates: JSON.stringify([{ x: 50, y: -15, z: 0 }])
        }, 0);

        expect(frame?.time).toBe(2.5);
        expect(frame?.playerKey).toBe('id:10');
        expect(frame?.cars[0].position).toEqual({ x: 50, y: -15, z: 0 });
    });

    it('falls back to slot zero when player id is missing', () => {
        const frame = parseTelemetryFrame({
            Graphics_car_coordinates: [
                { x: 1, y: 2, z: 0 },
                { x: 3, y: 4, z: 0 }
            ]
        }, 3);

        expect(frame?.playerKey).toBe('slot:0');
        expect(frame?.time).toBeCloseTo(3 / 60);
    });

    it('filters invalid zero coordinates and empty rows', () => {
        const frames = parseTelemetryFrames([
            { Graphics_car_coordinates: [{ x: 0, y: 0, z: 0 }] },
            { Graphics_car_coordinates: [{ x: 8, y: 9, z: 0 }] }
        ]);

        expect(frames).toHaveLength(1);
        expect(frames[0].cars[0].position).toEqual({ x: 8, y: 9, z: 0 });
    });

    it('keeps session playback time monotonic when lap time resets', () => {
        const frames = parseTelemetryFrames([
            { Graphics_current_time: 0, Graphics_car_coordinates: [{ x: 1, y: 1, z: 0 }] },
            { Graphics_current_time: 1000, Graphics_car_coordinates: [{ x: 2, y: 2, z: 0 }] },
            { Graphics_current_time: 2000, Graphics_car_coordinates: [{ x: 3, y: 3, z: 0 }] },
            { Graphics_current_time: 200, Graphics_car_coordinates: [{ x: 4, y: 4, z: 0 }] },
            { Graphics_current_time: 1200, Graphics_car_coordinates: [{ x: 5, y: 5, z: 0 }] }
        ]);

        expect(frames.map((frame) => frame.time)).toEqual([0, 1, 2, 2.2, 3.2]);
    });

    it('advances playback time when a new lap starts at exactly zero', () => {
        const frames = parseTelemetryFrames([
            { Graphics_current_time: 0, Graphics_car_coordinates: [{ x: 1, y: 1, z: 0 }] },
            { Graphics_current_time: 1000, Graphics_car_coordinates: [{ x: 2, y: 2, z: 0 }] },
            { Graphics_current_time: 2000, Graphics_car_coordinates: [{ x: 3, y: 3, z: 0 }] },
            { Graphics_current_time: 0, Graphics_car_coordinates: [{ x: 4, y: 4, z: 0 }] },
            { Graphics_current_time: 1000, Graphics_car_coordinates: [{ x: 5, y: 5, z: 0 }] }
        ]);

        expect(frames[3].time).toBeGreaterThan(frames[2].time);
        expect(frames.map((frame) => frame.time)).toEqual([0, 1, 2, 2 + (1 / 60), 3]);
    });

    it('keeps millisecond lap times below 100ms near the start line', () => {
        const frames = parseTelemetryFrames([
            { Graphics_current_time: 0, Graphics_car_coordinates: [{ x: 1, y: 1, z: 0 }] },
            { Graphics_current_time: 16, Graphics_car_coordinates: [{ x: 2, y: 2, z: 0 }] },
            { Graphics_current_time: 33, Graphics_car_coordinates: [{ x: 3, y: 3, z: 0 }] },
            { Graphics_current_time: 100, Graphics_car_coordinates: [{ x: 4, y: 4, z: 0 }] },
            { Graphics_current_time: 116, Graphics_car_coordinates: [{ x: 5, y: 5, z: 0 }] }
        ]);

        expect(frames.map((frame) => frame.time)).toEqual([0, 0.016, 0.033, 0.1, 0.116]);
        expect(getPlaybackFrameIndex(frames, 0.02)).toBe(2);
    });
});

describe('MapVisualization trajectory clipping', () => {
    it('splits visible trajectory runs instead of connecting across hidden points', () => {
        const segments = segmentVisiblePoints([
            { position: { x: 0, y: 0, z: 0 }, visible: true },
            { position: { x: 1, y: 0, z: 0 }, visible: true },
            { position: { x: 2, y: 0, z: 0 }, visible: false },
            { position: { x: 3, y: 0, z: 0 }, visible: true },
            { position: { x: 4, y: 0, z: 0 }, visible: true }
        ]);

        expect(segments).toEqual([
            [
                { x: 0, y: 0, z: 0 },
                { x: 1, y: 0, z: 0 }
            ],
            [
                { x: 3, y: 0, z: 0 },
                { x: 4, y: 0, z: 0 }
            ]
        ]);
    });

    it('does not draw single visible points as line segments', () => {
        const segments = segmentVisiblePoints([
            { position: { x: 0, y: 0, z: 0 }, visible: true },
            { position: { x: 1, y: 0, z: 0 }, visible: false },
            { position: { x: 2, y: 0, z: 0 }, visible: true }
        ]);

        expect(segments).toEqual([]);
    });
});

describe('MapVisualization playback indexing', () => {
    const frames = parseTelemetryFrames([
        { Graphics_current_time: 0, Graphics_car_coordinates: [{ x: 1, y: 1, z: 0 }] },
        { Graphics_current_time: 1000, Graphics_car_coordinates: [{ x: 2, y: 2, z: 0 }] },
        { Graphics_current_time: 2500, Graphics_car_coordinates: [{ x: 3, y: 3, z: 0 }] }
    ]);

    it('finds the first telemetry frame at or after elapsed playback time', () => {
        expect(getPlaybackFrameIndex(frames, 0)).toBe(0);
        expect(getPlaybackFrameIndex(frames, 0.5)).toBe(1);
        expect(getPlaybackFrameIndex(frames, 1.5)).toBe(2);
    });

    it('signals completion after the final telemetry frame', () => {
        expect(getPlaybackFrameIndex(frames, 2.6)).toBe(-1);
    });
});
