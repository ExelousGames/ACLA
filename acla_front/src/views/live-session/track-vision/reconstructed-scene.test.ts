import { reconstructScene } from './reconstructed-scene';
import type { TrackVisionFrame } from './track-vision-types';

function fixture(interior: (x: number, y: number) => boolean = () => false) {
    const width = 80, height = 60;
    const mask = (active: (x: number, y: number) => boolean) => Uint8Array.from({ length: width * height },
        (_, i) => Number(active(i % width, Math.floor(i / width))));
    const frame: TrackVisionFrame = { capturedAt: 0, width: 800, height: 800, detections: {
        segment: { task: 'segment', width, height, inferenceMs: 0, classNames: ['track', '  CAR  Interior  '], instances: [
            { classId: 0, confidence: 0.9, box: [0, 0, 1, 1], mask: mask((x, y) => x >= 20 && x <= 60 && y >= 5 && y < 55) },
            { classId: 1, confidence: 0.5, box: [0, 0, 1, 1], mask: mask(interior) },
        ] },
    } };
    return frame;
}

it('reconstructs both sides without depth or calibration and leaves the track outline open', () => {
    const scene = reconstructScene(fixture())!;
    expect(scene.leftBoundary).toHaveLength(1);
    expect(scene.rightBoundary).toHaveLength(1);
    expect(scene.leftBoundary[0]).toHaveLength(50);
    scene.leftBoundary[0].forEach(({ x }) => expect(x).toBeCloseTo(205));
    scene.rightBoundary[0].forEach(({ x }) => expect(x).toBeCloseTo(605));
});

it.each(['left', 'right'].flatMap((side) => [-3, 0, 2].map((gap) => ({ side, gap }))))
('removes the $side cockpit outline with a $gap-pixel gap and preserves the other edge', ({ side, gap }) => {
    const before = reconstructScene(fixture())!;
    const frame = fixture((x, y) => y >= 20 && y <= 35 && (side === 'left' ? x <= 20 - gap : x >= 60 + gap));
    const segment = frame.detections.segment!;
    if (segment.task !== 'segment') throw new Error('segment');
    const originals = segment.instances.map(({ mask }) => mask.slice());
    const scene = reconstructScene(frame)!;
    const changed = side === 'left' ? scene.leftBoundary : scene.rightBoundary;
    expect(changed).toHaveLength(2);
    expect(changed.flat().every(({ y }) => y < 18 / 60 * 800 || y > 38 / 60 * 800)).toBe(true);
    expect(side === 'left' ? scene.rightBoundary : scene.leftBoundary).toEqual(side === 'left' ? before.rightBoundary : before.leftBoundary);
    expect(segment.instances.map(({ mask }) => mask)).toEqual(originals);
    segment.instances.reverse();
    expect(reconstructScene(frame)).toEqual(scene);
});

it('rejects overlapped dashboard edges without tracing the new cockpit cutout', () => {
    const scene = reconstructScene(fixture((_x, y) => y >= 40))!;
    for (const boundary of [scene.leftBoundary, scene.rightBoundary]) {
        expect(boundary).toHaveLength(1);
        expect(boundary[0].every(({ y }) => y < 38 / 60 * 800)).toBe(true);
    }
    expect(reconstructScene(fixture(() => true))).toMatchObject({ leftBoundary: [], rightBoundary: [] });
});

it('does not turn an interior hole into a track edge or erase distant edges', () => {
    expect(reconstructScene(fixture((x, y) => x >= 35 && x <= 45 && y >= 20 && y <= 35)))
        .toEqual(reconstructScene(fixture()));
});

it('maps non-square captures out of letterbox padding and rejects clipped side edges', () => {
    const frame = fixture();
    frame.width = 1600; frame.height = 900;
    const scene = reconstructScene(frame)!;
    const points = [...scene.leftBoundary.flat(), ...scene.rightBoundary.flat()];
    expect(points.length).toBeGreaterThan(0);
    expect(points.every(({ x, y }) => x >= 0 && x < 1600 && y >= 0 && y < 900)).toBe(true);
    const segment = frame.detections.segment!;
    if (segment.task !== 'segment') throw new Error('segment');
    segment.instances[0].mask.fill(1);
    expect(reconstructScene(frame)).toMatchObject({ leftBoundary: [], rightBoundary: [] });
});

it('clears missing and rejected track masks and keeps disconnected rows separate', () => {
    expect(reconstructScene(null)).toBeNull();
    expect(reconstructScene({ ...fixture(), detections: {} })).toBeNull();
    const frame = fixture(), segment = frame.detections.segment!;
    if (segment.task !== 'segment') throw new Error('segment');
    segment.instances[0].mask.fill(0, 25 * 80, 30 * 80);
    expect(reconstructScene(frame)!.leftBoundary).toHaveLength(2);
    segment.instances[0].confidence = 0.64;
    expect(reconstructScene(frame)).toMatchObject({ leftBoundary: [], rightBoundary: [] });
});

it('maps 768 model boxes back to captures with rounded letterbox dimensions', () => {
    const frame = fixture(), segment = frame.detections.segment!;
    if (segment.task !== 'segment') throw new Error('segment');
    frame.width = 1000; frame.height = 561;
    segment.classNames = ['car'];
    // At 768, this capture is resized to 768 x 431 with 168 pixels of top padding.
    segment.instances = [{ ...segment.instances[0], box: [192 / 768, (168 + 431 * 0.4) / 768,
        576 / 768, (168 + 431 * 0.6) / 768] }];
    const scene = reconstructScene(frame)!;
    expect(scene.cars).toHaveLength(1);
    scene.cars[0].box.forEach((value, index) => expect(value).toBeCloseTo([250, 224.4, 750, 336.6][index]));
});

it('includes every accepted car and car pack without track, depth or calibration', () => {
    const frame = fixture(), segment = frame.detections.segment!;
    if (segment.task !== 'segment') throw new Error('segment');
    segment.classNames = ['car', '  CAR   PACK  ', 'opponent car', 'car interior', 'grass'];
    segment.instances = segment.classNames.map((_, classId) => ({
        classId, confidence: 0.9, box: [0.1 + classId * 0.15, 0.4, 0.2 + classId * 0.15, 0.6],
        mask: new Uint8Array(segment.width * segment.height),
    }));
    const originals = segment.instances.map(({ box }) => [...box]);
    const scene = reconstructScene(frame)!;
    expect(scene).toMatchObject({ leftBoundary: [], rightBoundary: [], cars: [
        { classId: 0, confidence: 0.9, pack: false },
        { classId: 1, confidence: 0.9, pack: true },
        { classId: 2, confidence: 0.9, pack: false },
    ] });
    const boxes = [[80, 320, 160, 480], [200, 320, 280, 480], [320, 320, 400, 480]];
    scene.cars.forEach((car, i) => car.box.forEach((value, j) => expect(value).toBeCloseTo(boxes[i][j])));
    expect(segment.instances.map(({ box }) => box)).toEqual(originals);
});

it.each([
    { width: 1600, height: 900, expected: [0, 0, 1200, 450] },
    { width: 900, height: 1600, expected: [0, 320, 850, 800] },
])('maps traffic boxes out of letterbox padding for $width × $height captures', ({ width, height, expected }) => {
    const frame = fixture(), segment = frame.detections.segment!;
    if (segment.task !== 'segment') throw new Error('segment');
    frame.width = width; frame.height = height;
    segment.classNames = ['car'];
    segment.instances = [{ ...segment.instances[0], box: [-0.05, 0.2, 0.75, 0.5] }];
    const scene = reconstructScene(frame)!;
    expect(scene.cars).toHaveLength(1);
    scene.cars[0].box.forEach((value, index) => expect(value).toBeCloseTo(expected[index]));
});

it('rejects low-confidence, invalid and offscreen traffic boxes while retaining clipped cars', () => {
    const frame = fixture(), segment = frame.detections.segment!;
    if (segment.task !== 'segment') throw new Error('segment');
    segment.classNames = ['car'];
    const car = { ...segment.instances[0], box: [0.3, 0.4, 0.4, 0.6] as [number, number, number, number] };
    segment.instances = [
        { ...car, confidence: 0.64 },
        { ...car, box: [NaN, 0.4, 0.4, 0.6] },
        { ...car, box: [0.3, 0.4, Infinity, 0.6] },
        { ...car, box: [0.5, 0.4, 0.4, 0.6] },
        { ...car, box: [0.3, 0.6, 0.4, 0.6] },
        { ...car, box: [1.1, 0.4, 1.2, 0.6] },
        { ...car, confidence: 0.65, box: [0.9, 0.85, 1.1, 1.1] },
    ];
    expect(reconstructScene(frame)!.cars).toEqual([
        { classId: 0, confidence: 0.65, pack: false, box: [720, 680, 800, 800] },
    ]);
    frame.width = 1600; frame.height = 900;
    segment.instances = [{ ...car, box: [0.3, 0, 0.4, 0.2] }];
    expect(reconstructScene(frame)!.cars).toEqual([]);
});
