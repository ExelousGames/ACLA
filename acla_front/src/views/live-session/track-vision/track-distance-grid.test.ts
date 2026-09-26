import { createCameraProjection } from './camera-projection';
import { reconstructDistanceGrid, reconstructTrack } from './track-position-analysis';
import { vision } from './test-fixtures';
import { letterbox } from './yolo-segmentation';

const nearRoad = (width = 1600, height = 900) => {
    const frame = vision(0, { width, height, camera: { pitchDeg: 0 }, cars: [], road: () => true });
    const depth = frame.detections.depth!;
    if (depth.task !== 'depth') throw new Error('Expected depth');
    // The measured road is above the calibration plane, but retains its captured position.
    depth.values = depth.values.map((value) => value * 0.15);
    return frame;
};

it.each([[1600, 900], [3440, 1440], [900, 1600]])(
    'places near-distance guides on measured road depth without an edge fit at %s × %s', (width, height) => {
        const frame = nearRoad(width, height);
        expect(reconstructTrack(frame)?.geometry).toBeNull();
        const guide = reconstructDistanceGrid(frame).find(({ distanceM }) => distanceM === 3)!;
        expect(guide.segments.length).toBeGreaterThan(0);
        const projection = createCameraProjection(frame.calibration!);
        const elevation = frame.calibration!.heightM * 0.85;
        const expectedV = 0.5 + width / height / 2 * (frame.calibration!.heightM - elevation) / 3;
        for (const point of guide.segments.flat()) {
            expect(point.y).toBe(3);
            expect(point.z).toBeCloseTo(elevation, 2);
            expect(projection.localToImage(point)!.v).toBeCloseTo(expectedV, 3);
        }
    },
);

it('uses vehicle-forward distance with camera rotation and position offsets', () => {
    const frame = vision(0, { camera: { pitchDeg: 8, yawDeg: 15, forwardOffsetM: 1, lateralOffsetM: -0.4 }, cars: [] });
    const guide = reconstructDistanceGrid(frame).find(({ distanceM }) => distanceM === 20)!;
    expect(guide.segments.length).toBeGreaterThan(0);
    for (const point of guide.segments.flat()) {
        expect(point.y).toBe(20);
        expect(point.z).toBeCloseTo(0, 1);
    }
});

it('breaks guides at cars and excluded surfaces instead of using their depth as road', () => {
    const frame = nearRoad();
    const segment = frame.detections.segment!;
    if (segment.task !== 'segment') throw new Error('Expected segmentation');
    segment.classNames = ['track', 'car', 'grass'];
    const { padX, resizedWidth } = letterbox(frame.width, frame.height, 640);
    for (const [classId, left, right] of [[1, 0.4, 0.6], [2, 0.15, 0.3]]) {
        segment.instances.push({ classId, confidence: 0.9, box: [0, 0, 1, 1],
            mask: Uint8Array.from({ length: segment.width * segment.height }, (_, i) => {
                const u = ((i % segment.width + 0.5) / segment.width * 640 - padX) / resizedWidth;
                return Number(u >= left && u <= right);
            }) });
    }
    const guide = reconstructDistanceGrid(frame).find(({ distanceM }) => distanceM === 3)!;
    expect(guide.segments.length).toBeGreaterThan(0);
    const projection = createCameraProjection(frame.calibration!);
    for (const [a, b] of guide.segments) {
        for (const point of [a, b, { x: (a.x + b.x) / 2, y: 3, z: (a.z + b.z) / 2 }]) {
            const { u } = projection.localToImage(point)!;
            expect(u < 0.4 || u > 0.6).toBe(true);
            expect(u < 0.15 || u > 0.3).toBe(true);
        }
    }
});

it('clips the grid to the boundary cutoff', () => {
    const frame = nearRoad();
    frame.boundaryStartY = 0.55;
    const grid = reconstructDistanceGrid(frame);
    expect(grid.some(({ distanceM }) => distanceM === 3)).toBe(false);
    expect(grid.some(({ distanceM }) => distanceM === 4)).toBe(true);
    const projection = createCameraProjection(frame.calibration!);
    for (const point of grid.flatMap(({ segments }) => segments.flat())) {
        expect(projection.localToImage(point)!.v).toBeLessThanOrEqual(0.55);
    }
});

it.each(['calibration', 'depth', 'segment', 'road', 'invalid depth'] as const)('shows no invented grid without %s', (missing) => {
    const frame = nearRoad();
    if (missing === 'calibration') delete frame.calibration;
    if (missing === 'depth') delete frame.detections.depth;
    if (missing === 'segment') delete frame.detections.segment;
    if (missing === 'road' && frame.detections.segment?.task === 'segment') frame.detections.segment.instances = [];
    if (missing === 'invalid depth' && frame.detections.depth?.task === 'depth') frame.detections.depth.values.fill(NaN);
    expect(reconstructDistanceGrid(frame)).toEqual([]);
});
