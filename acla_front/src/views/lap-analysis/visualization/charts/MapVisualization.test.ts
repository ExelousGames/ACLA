import { getPlaybackFrameIndex, parseTelemetryFrame, parseTelemetryFrames, segmentVisiblePoints } from './mapTelemetry';
import { getSegmentMainLabelText, getSegmentSubLabelTexts } from './segmentClassificationDisplay';

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
        expect(frame?.sourceIndex).toBe(0);
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

    it('uses slot keys when ACC reports duplicate car ids for opponents', () => {
        const frame = parseTelemetryFrame({
            Graphics_current_time: 1000,
            Graphics_player_car_id: 0,
            Graphics_car_id: [0, 0, 0],
            Graphics_car_coordinates: [
                { x: 100, y: 10, z: 1 },
                { x: 120, y: 12, z: 2 },
                { x: 140, y: 14, z: 3 }
            ]
        }, 0);

        expect(frame?.playerKey).toBe('slot:0');
        expect(frame?.cars.map((car) => car.key)).toEqual(['slot:0', 'slot:1', 'slot:2']);
    });

    it('keeps duplicate-id opponent trajectories separate across frames', () => {
        const frames = parseTelemetryFrames([
            {
                Graphics_current_time: 1000,
                Graphics_player_car_id: 0,
                Graphics_car_id: [0, 0],
                Graphics_car_coordinates: [
                    { x: 100, y: 10, z: 1 },
                    { x: 200, y: 20, z: 2 }
                ]
            },
            {
                Graphics_current_time: 2000,
                Graphics_player_car_id: 0,
                Graphics_car_id: [0, 0],
                Graphics_car_coordinates: [
                    { x: 110, y: 10, z: 1 },
                    { x: 210, y: 20, z: 2 }
                ]
            }
        ]);

        expect(frames.map((frame) => frame.playerKey)).toEqual(['slot:0', 'slot:0']);
        expect(frames[0].cars.map((car) => car.key)).toEqual(['slot:0', 'slot:1']);
        expect(frames[1].cars.map((car) => car.key)).toEqual(['slot:0', 'slot:1']);
        expect(frames[1].cars[1].position.x - frames[0].cars[1].position.x).toBe(10);
    });

    it('compresses large multi-car telemetry gaps so opponent replays do not stall', () => {
        const frames = parseTelemetryFrames([
            {
                Graphics_current_time: 0,
                Graphics_car_coordinates: [
                    { x: 100, y: 10, z: 1 },
                    { x: 120, y: 12, z: 2 }
                ]
            },
            {
                Graphics_current_time: 1000,
                Graphics_car_coordinates: [
                    { x: 110, y: 10, z: 1 },
                    { x: 130, y: 12, z: 2 }
                ]
            },
            {
                Graphics_current_time: 9000,
                Graphics_car_coordinates: [
                    { x: 120, y: 10, z: 1 },
                    { x: 140, y: 12, z: 2 }
                ]
            },
            {
                Graphics_current_time: 10000,
                Graphics_car_coordinates: [
                    { x: 130, y: 10, z: 1 },
                    { x: 150, y: 12, z: 2 }
                ]
            }
        ]);

        expect(frames.map((frame) => frame.time)).toEqual([0, 1, 1.25, 1.5]);
        expect(getPlaybackFrameIndex(frames, 1.1)).toBe(2);
        expect(getPlaybackFrameIndex(frames, 1.4)).toBe(3);
    });

    it('filters invalid zero coordinates and empty rows', () => {
        const frames = parseTelemetryFrames([
            { Graphics_car_coordinates: [{ x: 0, y: 0, z: 0 }] },
            { Graphics_car_coordinates: [{ x: 8, y: 9, z: 0 }] }
        ]);

        expect(frames).toHaveLength(1);
        expect(frames[0].sourceIndex).toBe(1);
        expect(frames[0].cars[0].position).toEqual({ x: 8, y: 9, z: 0 });
    });

    it('filters impossible coordinate outliers before they affect map bounds', () => {
        const frame = parseTelemetryFrame({
            Graphics_current_time: 1000,
            Graphics_player_car_id: 0,
            Graphics_car_id: [1, 2],
            Graphics_car_coordinates: [
                { x: 3744661726298112, y: 0, z: 6.4337007530853315e35 },
                { x: 100, y: 12, z: -40 }
            ]
        }, 0);

        expect(frame?.cars).toHaveLength(1);
        expect(frame?.cars[0].position).toEqual({ x: 100, y: 12, z: -40 });
        expect(frame?.playerKey).toBe('id:2');
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
        expect(frames[3].sourceIndex).toBe(3);
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

describe('MapVisualization AI segment labels', () => {
    it('formats main-first segment labels with sub labels', () => {
        const segment = {
            labels: ['MSP', 'MSP1', 'ST3'],
            main_label_id: 'MSP',
            main_label_name: 'Mistake (Practice)',
            start_index: 0,
            end_index: 3,
            sub_labels: [
                { label_id: 'MSP1', label_name: 'Initiate brake too late' },
                { label_id: 'ST3', label_name: 'Approach to corner' }
            ],
            sub_segments: []
        };

        expect(getSegmentMainLabelText(segment)).toBe('Mistake (Practice)');
        expect(getSegmentSubLabelTexts(segment)).toEqual([
            'Initiate brake too late',
            'Approach to corner'
        ]);
    });

    it('falls back to flat labels for older segment responses', () => {
        const segment = {
            labels: ['EA', 'ST2'],
            start_index: 0,
            end_index: 3
        };

        expect(getSegmentMainLabelText(segment)).toBe('EA, ST2');
        expect(getSegmentSubLabelTexts(segment)).toEqual([]);
    });
});
