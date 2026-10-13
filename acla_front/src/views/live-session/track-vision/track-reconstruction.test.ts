import { createCameraProjection, DEFAULT_CAMERA } from './camera-projection';
import { cloudPoint, createDepthPointCloud, maskPointCloud } from './depth-point-cloud';
import { reconstructTrack } from './track-reconstruction';
import { createSemanticScene } from './semantic-scene';
import type { TrackVisionFrame } from './track-vision-types';

type Region = (u: number, v: number) => boolean;
const rectangle = (left: number, top: number, right: number, bottom: number): Region =>
    (u, v) => u >= left && u <= right && v >= top && v <= bottom;
function frame() {
    const source: TrackVisionFrame = { capturedAt: Date.now(), width: 100, height: 100,
        calibration: { ...DEFAULT_CAMERA, pitchDeg: 0, imageWidth: 100, imageHeight: 100 },
        detections: {
            depth: { task: 'depth', width: 80, height: 80, values: new Float32Array(6400).fill(10), inferenceMs: 1, classNames: [] },
            segment: { task: 'segment', width: 40, height: 40, instances: [], inferenceMs: 1, classNames: ['track', 'car', 'other', 'car pack'] },
        } };
    const add = (classId: number, bounds: [number, number, number, number], contains: Region = rectangle(...bounds)) => {
        const segment = source.detections.segment!;
        if (segment.task !== 'segment') throw new Error('segment');
        segment.instances.push({ classId, confidence: 0.9, box: bounds,
            mask: Uint8Array.from({ length: 1600 }, (_, i) => Number(contains((i % 40 + 0.5) / 40, (Math.floor(i / 40) + 0.5) / 40))) });
    };
    const setDepth = (region: Region, meters: number) => {
        const depth = source.detections.depth!;
        if (depth.task !== 'depth') throw new Error('depth');
        depth.values.forEach((_, i) => { if (region((i % 80 + 0.5) / 80, (Math.floor(i / 80) + 0.5) / 80)) depth.values[i] = meters; });
    };
    return { source, add, setDepth };
}

it('unprojects depth before segmentation and masks the resulting 3D points without changing their positions', () => {
    const { source } = frame();
    delete source.detections.segment;
    const cloud = createDepthPointCloud(source)!;
    const point = cloudPoint(cloud, 45, 50)!;
    expect(point).toMatchObject(createCameraProjection(source.calibration!).imageToLocal(point.u, point.v, 10)!);
    const selected = maskPointCloud(cloud, (u) => u > 0.5, undefined, 6400);
    expect(selected.points[50 * 80 + 45]).toEqual(point);
    expect(selected.points[50 * 80 + 35]).toBeNull();
    expect(cloudPoint(cloud, 35, 50)).not.toBeNull();
});

it.each([false, true])('keeps unlabelled point-cloud depth with interior detection %s', (interior) => {
    const { source, add } = frame();
    const segment = source.detections.segment!;
    if (segment.task !== 'segment') throw new Error('segment');
    segment.classNames.push('car interior');
    if (interior) add(4, [0.4, 0.4, 0.6, 0.6]);
    const cloud = createDepthPointCloud(source)!;
    expect(cloudPoint(cloud, 0, 0)?.depthM).toBe(10);
    expect(cloudPoint(cloud, 79, 79)?.depthM).toBe(10);
    expect(cloudPoint(cloud, 40, 40)?.depthM ?? null).toBe(interior ? null : 10);
    expect(Array.from(cloud.depths).filter((value) => value > 0)).toHaveLength(interior ? 6400 - 16 * 16 : 6400);
});

it('measures vehicle geometry without constructing 3D display assets', () => {
    const { source, add, setDepth } = frame();
    add(0, [0.15, 0.4, 0.85, 0.8]);
    add(1, [0.4, 0.5, 0.6, 0.7]);
    setDepth(rectangle(0.4, 0.5, 0.6, 0.7), 5);
    const segment = source.detections.segment!;
    if (segment.task !== 'segment') throw new Error('segment');
    const masks = segment.instances.map(({ mask }) => mask.slice());
    const scene = reconstructTrack(source)!;
    expect(scene).not.toHaveProperty('roadPoints');
    expect(scene).not.toHaveProperty('masks');
    expect(scene).not.toHaveProperty('pointCloud');
    expect(scene).not.toHaveProperty('roadMesh');
    expect(scene.cars[0]).not.toHaveProperty('mesh');
    expect(scene.cars[0].points.every((point) => point.y === 5)).toBe(true);
    expect(segment.instances.map(({ mask }) => mask)).toEqual(masks);
});

it('preserves holes in a partially occluded car mask and keeps its bounds and center measured', () => {
    const { source, add, setDepth } = frame();
    const rear = rectangle(0.3, 0.4, 0.7, 0.7), front = rectangle(0.45, 0.45, 0.55, 0.65);
    add(1, [0.3, 0.4, 0.7, 0.7], (u, v) => rear(u, v) && !front(u, v));
    add(1, [0.45, 0.45, 0.55, 0.65]);
    setDepth(front, 5);
    const scene = reconstructTrack(source)!;
    const car = scene.cars.find((item) => item.center.y === 10)!;
    expect(car.points.length).toBeGreaterThan(0);
    const camera = createCameraProjection(source.calibration!);
    expect(car.points.every((point) => {
        const pixel = camera.localToImage(point)!;
        return point.y === 10 && !front(pixel.u, pixel.v);
    })).toBe(true);
    expect(car.points.some((point) => 'estimated' in point)).toBe(false);
    expect(car.min.y).toBe(10);
    expect(car.max.y).toBe(10);
    const segment = source.detections.segment!;
    if (segment.task !== 'segment') throw new Error('segment');
    segment.instances.reverse();
    expect(reconstructTrack(source)!.cars.find((item) => item.center.y === 10)!.points).toEqual(car.points);
});

it('keeps a large foreground overlap from becoming the depth of the rear instance', () => {
    const { source, add, setDepth } = frame();
    add(1, [0.3, 0.4, 0.7, 0.7]);
    add(1, [0.325, 0.4, 0.675, 0.7]);
    setDepth(rectangle(0.325, 0.4, 0.675, 0.7), 5);
    const scene = reconstructTrack(source)!;
    expect(scene.cars).toHaveLength(2);
    expect(scene.cars.map((car) => car.center.y).sort((a, b) => a - b)).toEqual([5, 10]);
    for (const car of scene.cars) expect(car.points.every((point) => point.y === car.center.y)).toBe(true);
});

it.each(['unknown', 'excluded', 'missing depth', 'farther car'])(
    'keeps road mask gaps caused by %s empty', (reason) => {
        const { source, add, setDepth } = frame();
        const road = rectangle(0.15, 0.4, 0.85, 0.8), gap = rectangle(0.4, 0.5, 0.6, 0.7);
        add(0, [0.15, 0.4, 0.85, 0.8], (u, v) => road(u, v) && (reason === 'missing depth' || !gap(u, v)));
        if (reason === 'excluded') add(2, [0.4, 0.5, 0.6, 0.7]);
        if (reason === 'farther car') { add(1, [0.4, 0.5, 0.6, 0.7]); setDepth(gap, 20); }
        if (reason === 'missing depth') setDepth(gap, NaN);
        const points = maskPointCloud(createDepthPointCloud(source)!, createSemanticScene(source)!.visibleRoad).points.filter(Boolean);
        expect(points.length).toBeGreaterThan(0);
        const camera = createCameraProjection(source.calibration!);
        for (const point of points) {
            const pixel = camera.localToImage(point!)!;
            expect(rectangle(0.425, 0.525, 0.575, 0.675)(pixel.u, pixel.v)).toBe(false);
        }
    },
);

