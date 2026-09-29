import { createCameraProjection } from './camera-projection';
import { reconstructTrack } from './track-position-analysis';
import { fitRoadPolynomial } from './road-polynomial';
import { MODEL_LABELS, vision } from './test-fixtures';
import type { TrackVisionFrame } from './track-vision-types';
import { letterbox } from './yolo-segmentation';

function hideEdge(frame: TrackVisionFrame, side: 'left' | 'right' | 'both', label = 'car', near = 12, far = 21) {
    const segment = frame.detections.segment!, depth = frame.detections.depth!;
    if (segment.task !== 'segment' || depth.task !== 'depth') throw new Error('Expected segmentation and depth');
    const camera = createCameraProjection(frame.calibration!);
    const bottom = camera.localToImage({ x: 0, y: near, z: 0 })!.v;
    const top = camera.localToImage({ x: 0, y: far, z: 0 })!.v;
    const { padX, padY, resizedWidth, resizedHeight } = letterbox(frame.width, frame.height, 640);
    const covered = (i: number, width: number, height: number) => {
        const u = ((i % width + 0.5) / width * 640 - padX) / resizedWidth;
        const v = ((Math.floor(i / width) + 0.5) / height * 640 - padY) / resizedHeight;
        return v > top && v < bottom && (side === 'both' || (side === 'left' ? u < 0.5 : u >= 0.5));
    };
    const mask = Uint8Array.from(segment.instances[0].mask, (_, i) => Number(covered(i, segment.width, segment.height)));
    const traffic = { ...segment.instances[0], classId: MODEL_LABELS.indexOf(label), mask };
    segment.instances.push(traffic);
    // Segmentation need not retain track underneath the occluding vehicle.
    segment.instances[0].mask.forEach((_, i) => { if (mask[i]) segment.instances[0].mask[i] = 0; });
    return (value: number) => {
        depth.values = depth.values.map((old, i) => {
            const column = Math.floor((i % depth.width + 0.5) / depth.width * segment.width);
            const row = Math.floor((Math.floor(i / depth.width) + 0.5) / depth.height * segment.height);
            return mask[row * segment.width + column] ? value : old;
        });
    };
}

it.each((['left', 'right', 'both'] as const).flatMap((side) => ['car', 'car pack'].map((label) => ({ side, label }))))
('infers the $side boundary behind a $label from visible 3D points without using vehicle depth', ({ side, label }) => {
    const frame = vision(0, { cars: [], classNames: MODEL_LABELS, corner: 'straight', player: 'middle' });
    const before = reconstructTrack(frame)!;
    const changeCarDepth = hideEdge(frame, side, label);
    changeCarDepth(1.5);
    const segment = frame.detections.segment!;
    if (segment.task !== 'segment') throw new Error('Expected segmentation');
    const masks = segment.instances.map(({ mask }) => mask.slice());
    const after = reconstructTrack(frame)!;
    const boundaries = [['leftBoundary', -5], ['rightBoundary', 5]] as const;
    const unchanged = boundaries.filter(([name]) => side !== 'both' && name !== `${side}Boundary`);
    expect(unchanged.map(([name]) => after[name])).toEqual(unchanged.map(([name]) => before[name]));
    for (const [name, x] of boundaries.filter(([name]) => side === 'both' || name === `${side}Boundary`)) {
        const estimated = after[name].filter((point) => point.estimated);
        expect(estimated.length).toBeGreaterThan(2);
        for (const point of estimated) {
            expect(point.x).toBeCloseTo(x, 0);
            expect(point.y).toBeGreaterThan(11);
            expect(point.y).toBeLessThan(22);
            expect(point.z).toBeCloseTo(0, 1);
        }
        expect(after[name].every((point, i, points) => !i || point.y > points[i - 1].y)).toBe(true);
        expect(after[name].filter((point) => !point.estimated).length).toBeLessThan(before[name].length);
    }
    expect(after.geometry?.trackWidthM).toBeCloseTo(10, 0);
    expect(after.geometry?.left).toEqual(fitRoadPolynomial(after.leftBoundary.filter((point) => !point.estimated)));
    expect(after.geometry?.right).toEqual(fitRoadPolynomial(after.rightBoundary.filter((point) => !point.estimated)));
    expect(segment.instances.map(({ mask }) => mask)).toEqual(masks);
    for (const value of [NaN, 150]) {
        changeCarDepth(value);
        const changed = reconstructTrack(frame)!;
        expect(changed.leftBoundary).toEqual(after.leftBoundary);
        expect(changed.rightBoundary).toEqual(after.rightBoundary);
    }
    segment.instances.reverse();
    expect(reconstructTrack(frame)!.leftBoundary).toEqual(after.leftBoundary);
});

it('replaces estimates with measured points as soon as traffic clears', () => {
    const frame = vision(0, { cars: [], classNames: MODEL_LABELS, corner: 'straight', player: 'middle' });
    const before = reconstructTrack(frame)!;
    const segment = frame.detections.segment!;
    if (segment.task !== 'segment') throw new Error('Expected segmentation');
    const original = segment.instances[0].mask.slice();
    hideEdge(frame, 'both');
    expect(reconstructTrack(frame)!.leftBoundary.some((point) => point.estimated)).toBe(true);
    segment.instances.pop();
    segment.instances[0].mask = original;
    expect(reconstructTrack(frame)).toEqual(before);
});

it.each([[1600, 900, 320], [900, 1600, 160], [3440, 1440, 320]])
('preserves elevation and perspective on a curved road at %s × %s with a %s-pixel mask', (width, height, maskSize) => {
    const frame = vision(0, { width, height, maskSize, cars: [], classNames: MODEL_LABELS, player: 'middle',
        camera: { pitchDeg: 0, yawDeg: 10, heightM: 2, lateralOffsetM: -0.4, forwardOffsetM: 1 } });
    const depth = frame.detections.depth!;
    if (depth.task !== 'depth') throw new Error('Expected depth');
    depth.values = depth.values.map((value) => value * 0.8);
    const changeCarDepth = hideEdge(frame, 'left');
    changeCarDepth(1.5);
    const scene = reconstructTrack(frame)!;
    const estimates = scene.leftBoundary.filter((point) => point.estimated);
    expect(estimates.length).toBeGreaterThan(0);
    const camera = createCameraProjection(frame.calibration!);
    const { padY, resizedHeight } = letterbox(width, height, 640);
    for (const point of estimates) {
        expect(point.z).toBeCloseTo(0.4, 1);
        const originalY = 1 + (point.y - 1) / 0.8;
        const expectedX = -0.4 + (-5 - 0.003 * (originalY - 8) ** 2 + 0.4) * 0.8;
        expect(Math.abs(point.x - expectedX)).toBeLessThan(0.35);
        const maskY = (padY + camera.localToImage(point)!.v * resizedHeight) / 640 * maskSize;
        expect(maskY - Math.floor(maskY)).toBeCloseTo(0.5, 5);
    }
});

it('interpolates a road slope from visible depth instead of assuming flat ground', () => {
    const frame = vision(0, { cars: [], classNames: MODEL_LABELS, corner: 'straight', player: 'middle',
        camera: { pitchDeg: 0, heightM: 2 } });
    const depth = frame.detections.depth!;
    if (depth.task !== 'depth') throw new Error('Expected depth');
    depth.values = depth.values.map((value) => 2 / (2 / value + 0.03));
    hideEdge(frame, 'both')(1.5);
    const scene = reconstructTrack(frame)!;
    const estimated = [...scene.leftBoundary, ...scene.rightBoundary].filter((point) => point.estimated);
    expect(estimated.length).toBeGreaterThan(4);
    for (const point of estimated) expect(point.z).toBeCloseTo(0.03 * point.y, 2);
});

it.each(['grass', 'other', 'long gap', 'unbracketed', 'car interior'])
('leaves unsupported gaps unresolved: %s', (reason) => {
    const frame = vision(0, { cars: [], classNames: [...MODEL_LABELS, 'car interior'], corner: 'straight', player: 'middle' });
    const far = reason === 'long gap' ? 40 : 21;
    hideEdge(frame, 'left', ['grass', 'other'].includes(reason) ? reason : 'car', reason === 'unbracketed' ? 1 : 12, far)(1.5);
    if (reason === 'car interior') {
        const segment = frame.detections.segment!;
        if (segment.task !== 'segment') throw new Error('Expected segmentation');
        segment.instances.push({ ...segment.instances[1], classId: MODEL_LABELS.length });
    }
    const scene = reconstructTrack(frame)!;
    expect(scene.leftBoundary.some((point) => point.estimated)).toBe(false);
    expect(scene.rightBoundary.some((point) => point.estimated)).toBe(false);
});
