import { analyzeTrackPositions, reconstructTrack } from './track-position-analysis';
import { MODEL_LABELS, vision } from './test-fixtures';
import { evaluateRoad } from './road-polynomial';
import type { CornerPosition } from './track-vision-types';
import { createCameraProjection } from './camera-projection';
import { letterbox } from './yolo-segmentation';

describe('segmentation and depth reconstruction', () => {
    it.each([[1600, 900], [900, 1600], [1000, 1000], [3440, 1440]])
    ('excludes hood edges and depth below the source-image cutoff at %s × %s', (width, height) => {
        const frame = vision(0, { width, height, corner: 'straight', player: 'middle', cars: [] });
        frame.boundaryStartY = 0.57;
        const clean = reconstructTrack(frame)!;
        expect(clean.leftBoundary.length).toBeGreaterThan(8);
        expect(clean.rightBoundary.length).toBeGreaterThan(8);
        const projection = createCameraProjection(frame.calibration!);
        for (const point of [...clean.leftBoundary, ...clean.rightBoundary]) {
            expect(projection.localToImage(point)!.v).toBeLessThanOrEqual(frame.boundaryStartY);
        }
        const { padY, resizedHeight } = letterbox(width, height, 640);
        const segment = frame.detections.segment!, depth = frame.detections.depth!;
        if (segment.task !== 'segment' || depth.task !== 'depth') throw new Error('Expected segmentation and depth');
        const belowCutoff = (index: number, columns: number, rows: number) =>
            ((Math.floor(index / columns) + 0.5) / rows * 640 - padY) / resizedHeight > frame.boundaryStartY!;
        // A narrow false track region follows bodywork in the excluded lower image.
        segment.instances[0].mask = segment.instances[0].mask.map((value, i) => belowCutoff(i, segment.width, segment.height)
            ? Number(i % segment.width > segment.width * 0.4 && i % segment.width < segment.width * 0.6) : value);
        depth.values = depth.values.map((value, i) => belowCutoff(i, depth.width, depth.height) ? 2 : value);
        const mask = segment.instances[0].mask.slice(), values = depth.values.slice();
        expect(reconstructTrack(frame)).toEqual(clean);
        const unrestricted = reconstructTrack({ ...frame, boundaryStartY: 1 })!;
        expect(unrestricted.leftBoundary).not.toEqual(clean.leftBoundary);
        expect(segment.instances[0].mask).toEqual(mask);
        expect(depth.values).toEqual(values);
    });

    it('supports full-frame or empty boundary scans without clipping cars', () => {
        const frame = vision(0);
        const full = reconstructTrack(frame)!;
        expect(reconstructTrack({ ...frame, boundaryStartY: 1 })).toEqual(full);
        const excluded = reconstructTrack({ ...frame, boundaryStartY: 0 })!;
        expect(excluded).toEqual({ leftBoundary: [], rightBoundary: [], cars: full.cars, geometry: null });
        expect(analyzeTrackPositions({ ...frame, boundaryStartY: 0 })).toEqual({});
    });

    it.each(['segment', 'depth'] as const)('requires same-frame %s for reconstruction', (task) => {
        const frame = vision(0);
        delete frame.detections[task];
        expect(reconstructTrack(frame)).toBeNull();
        expect(analyzeTrackPositions(frame)).toEqual({});
    });

    it('changes local geometry and car distance when only depth changes', () => {
        const frame = vision(0);
        const before = reconstructTrack(frame)!;
        const depth = frame.detections.depth!;
        if (depth.task !== 'depth') throw new Error('Expected depth');
        depth.values = depth.values.map((value) => value * 1.2);
        const after = reconstructTrack(frame)!;
        expect(after.leftBoundary[0].y).toBeCloseTo(before.leftBoundary[0].y * 1.2, 3);
        expect(after.leftBoundary[0].z).toBeCloseTo(-0.24, 2);
        expect(after.cars[0].center.y).toBeCloseTo(before.cars[0].center.y * 1.2, 3);
        expect(after.cars[0].points.some(({ z }) => z > 0.3)).toBe(true);
    });

    it.each([NaN, Infinity, 0, -1, 201])('rejects invalid depth %s without inventing flat geometry', (value) => {
        const frame = vision(0);
        const depth = frame.detections.depth!;
        if (depth.task !== 'depth') throw new Error('Expected depth');
        depth.values.fill(value);
        expect(reconstructTrack(frame)).toMatchObject({ leftBoundary: [], rightBoundary: [], cars: [], geometry: null });
        expect(analyzeTrackPositions(frame)).toEqual({});
    });

    it.each((['left', 'right'] as const).flatMap((side) =>
        ['invalid depth', 'clipped', 'occluded'].map((reason) => ({ side, reason }))))
    ('preserves the opposite edge when the $side edge is $reason', ({ side, reason }) => {
        const frame = vision(0, { corner: 'straight', player: 'middle', cars: [] });
        const before = reconstructTrack(frame)!;
        const segment = frame.detections.segment!, depth = frame.detections.depth!;
        if (segment.task !== 'segment' || depth.task !== 'depth') throw new Error('Expected segmentation and depth');
        const affected = (column: number, width: number) => side === 'left' ? column < width / 2 : column >= width / 2;
        if (reason === 'invalid depth') {
            depth.values = depth.values.map((value, i) => affected(i % depth.width, depth.width) ? NaN : value);
        } else if (reason === 'clipped') {
            const mask = segment.instances[0].mask;
            for (let row = 0; row < segment.height; row++) {
                const offset = row * segment.width, pixels = mask.subarray(offset, offset + segment.width);
                const start = pixels.indexOf(1), end = pixels.lastIndexOf(1);
                if (start < 0) continue;
                if (side === 'left') mask.fill(1, offset, offset + start);
                else mask.fill(1, offset + end + 1, offset + segment.width);
            }
        } else {
            const track = segment.instances[0];
            segment.instances.push({ ...track, classId: 1,
                mask: track.mask.map((value, i) => affected(i % segment.width, segment.width) ? value : 0) });
        }
        const after = reconstructTrack(frame)!;
        const missing = side === 'left' ? 'leftBoundary' : 'rightBoundary';
        const supported = side === 'left' ? 'rightBoundary' : 'leftBoundary';
        expect(before[supported].length).toBeGreaterThan(8);
        expect(after[supported]).toEqual(before[supported]);
        expect(after[missing]).toEqual([]);
        expect(after.geometry).toBeNull();
    });

    it.each(['left', 'right'] as const)('resumes the %s edge after a depth gap without truncating the opposite edge', (side) => {
        const frame = vision(0, { corner: 'straight', player: 'middle', cars: [] });
        const before = reconstructTrack(frame)!;
        const depth = frame.detections.depth!;
        if (depth.task !== 'depth') throw new Error('Expected depth');
        depth.values = depth.values.map((value, i) => {
            const affected = side === 'left' ? i % depth.width < depth.width / 2 : i % depth.width >= depth.width / 2;
            return affected && value > 10 && value < 28 ? NaN : value;
        });
        const after = reconstructTrack(frame)!;
        const interrupted = side === 'left' ? 'leftBoundary' : 'rightBoundary';
        const supported = side === 'left' ? 'rightBoundary' : 'leftBoundary';
        expect(after[supported]).toEqual(before[supported]);
        expect(after[interrupted].some(({ y }) => y < 10)).toBe(true);
        expect(after[interrupted].some(({ y }) => y > 28)).toBe(true);
        expect(after[interrupted].length).toBeLessThan(after[supported].length);
        expect(after.geometry?.trackWidthM).toBeCloseTo(10, 0);
    });

    it('reconstructs cars independently when road geometry is unavailable', () => {
        const frame = vision(0);
        const segment = frame.detections.segment!;
        if (segment.task !== 'segment') throw new Error('Expected segmentation');
        segment.instances.shift();
        expect(reconstructTrack(frame)?.cars).toHaveLength(1);
        expect(reconstructTrack(frame)?.geometry).toBeNull();
    });

    it('retains a sloping road in 3D instead of flattening its edges', () => {
        const frame = vision(0, { cars: [], camera: { pitchDeg: 0, heightM: 2 } });
        const depth = frame.detections.depth!;
        if (depth.task !== 'depth') throw new Error('Expected depth');
        // For a level camera and road Z = 0.03 Y, depth = height / (ray-down + slope).
        depth.values = depth.values.map((value) => 2 / (2 / value + 0.03));
        const scene = reconstructTrack(frame)!;
        expect(scene.leftBoundary.length).toBeGreaterThan(8);
        for (const point of [...scene.leftBoundary, ...scene.rightBoundary]) {
            expect(point.z).toBeCloseTo(point.y * 0.03, 2);
        }
    });

    it('keeps traffic unknown when a detected car has no usable depth', () => {
        const frame = vision(0);
        const segment = frame.detections.segment!;
        if (segment.task !== 'segment') throw new Error('Expected segmentation');
        segment.instances[1].mask.fill(0);
        expect(reconstructTrack(frame)?.geometry).not.toBeNull();
        expect(reconstructTrack(frame)?.cars).toHaveLength(0);
        expect(analyzeTrackPositions(frame).carAhead).toBeUndefined();
        expect(analyzeTrackPositions(frame).opponentPosition).toBeUndefined();
    });

    it.each((['left', 'right'] as const).flatMap((corner) => (['inside', 'middle', 'outside'] as CornerPosition[]).flatMap((player) =>
        (['inside', 'middle', 'outside'] as CornerPosition[]).map((opponent) => ({ corner, player, opponent })))))
    ('locates both cars in a $corner corner: $player / $opponent', ({ corner, player, opponent }) => {
        expect(analyzeTrackPositions(vision(0, { corner, player, opponent }))).toEqual({
            cornerDirection: corner, playerPosition: player, carAhead: 1, opponentPosition: opponent,
        });
    });

    it.each([[1600, 900], [900, 1600], [1000, 1000], [3440, 1440]])('removes letterboxing and recovers metric geometry at %s × %s', (width, height) => {
        const frame = vision(0, { width, height, player: 'middle', camera: { heightM: 1.6, pitchDeg: 8, yawDeg: 3, lateralOffsetM: -0.4, forwardOffsetM: 1 } });
        const geometry = reconstructTrack(frame)!.geometry!;
        expect(geometry).not.toBeNull();
        expect(geometry.trackWidthM).toBeCloseTo(10, 0);
        expect(evaluateRoad(geometry.center, 20)).toBeCloseTo(-0.003 * 12 ** 2, 0);
        expect(geometry.leftBoundary.every(({ z }) => Math.abs(z) < 0.06)).toBe(true);
        expect(analyzeTrackPositions(frame)).toMatchObject({ cornerDirection: 'left', playerPosition: 'middle' });
    });

    it.each([160, 320])('fits the road from a %s-pixel segmentation mask', (maskSize) => {
        const result = reconstructTrack(vision(0, { maskSize, cars: [] }))!;
        expect(result.geometry?.trackWidthM).toBeCloseTo(10, 0);
        expect(result.geometry?.curvaturePerM).toBeLessThan(-0.003);
    });

    it.each([MODEL_LABELS, [...MODEL_LABELS].reverse()])('resolves semantic classes by normalized labels: %j', (...classNames) => {
        const frame = vision(0, { classNames });
        frame.detections.segment!.classNames = classNames.map((label) => ` ${label.toUpperCase().replace(/ /g, ' \t ')} `);
        expect(analyzeTrackPositions(frame)).toEqual({ cornerDirection: 'left', playerPosition: 'inside', carAhead: 1, opponentPosition: 'outside' });
    });

    it('recovers road geometry without requiring car labels', () => {
        const frame = vision(0, { classNames: ['track'], cars: [] });
        expect(reconstructTrack(frame)?.geometry).not.toBeNull();
        expect(analyzeTrackPositions(frame)).toEqual({ cornerDirection: 'left', playerPosition: 'inside' });
    });

    it.each(['missing calibration', 'bad height', 'wrong resolution', 'missing mask', 'low confidence', 'outfield', 'clipped edges'])
    ('withholds unsupported geometry: %s', (scenario) => {
        const frame = vision(0, { classNames: MODEL_LABELS });
        const segment = frame.detections.segment!;
        if (segment.task !== 'segment') throw new Error('Expected segmentation');
        if (scenario === 'missing calibration') frame.calibration = undefined;
        if (scenario === 'bad height') frame.calibration!.heightM = NaN;
        if (scenario === 'wrong resolution') frame.calibration!.imageWidth++;
        if (scenario === 'missing mask') segment.instances[0].mask = new Uint8Array();
        if (scenario === 'low confidence') segment.instances[0].confidence = 0.64;
        if (scenario === 'outfield') segment.instances[0].classId = MODEL_LABELS.indexOf('Outfield asphalt road');
        if (scenario === 'clipped edges') segment.instances[0].mask.fill(1);
        expect(reconstructTrack(frame)?.geometry ?? null).toBeNull();
        expect(analyzeTrackPositions(frame)).toEqual({});
    });

    it.each(['curb', 'grass', 'other', 'fence', 'sand', 'Outfield asphalt road'])('excludes %s even over a road mask', (label) => {
        const frame = vision(0, { classNames: MODEL_LABELS });
        const segment = frame.detections.segment!;
        if (segment.task !== 'segment') throw new Error('Expected segmentation');
        segment.instances.push({ ...segment.instances[0], classId: MODEL_LABELS.indexOf(label) });
        expect(reconstructTrack(frame)?.geometry).toBeNull();
        segment.instances.reverse();
        expect(reconstructTrack(frame)?.geometry).toBeNull();
    });

    it.each(['car', 'car pack'])('uses the %s mask to isolate depth from the road', (label) => {
        const frame = vision(0, { classNames: MODEL_LABELS, opponent: 'middle' });
        const segment = frame.detections.segment!;
        if (segment.task !== 'segment') throw new Error('Expected segmentation');
        segment.instances[1].classId = MODEL_LABELS.indexOf(label);
        const reconstruction = reconstructTrack(frame)!;
        expect(reconstruction.cars).toHaveLength(1);
        expect(reconstruction.cars[0].center.y).toBeCloseTo(18, 0);
        expect(reconstruction.cars[0].pack).toBe(label === 'car pack');
        expect(reconstruction.geometry!.trackWidthM).toBeCloseTo(10, 0);
        segment.instances.reverse();
        expect(reconstructTrack(frame)).toEqual(reconstruction);
    });

    it('bridges car occlusion using observed road on both sides and recognizes packs without an individual position', () => {
        const frame = vision(0, { classNames: MODEL_LABELS, opponent: 'middle' });
        const segment = frame.detections.segment!;
        if (segment.task !== 'segment') throw new Error('Expected segmentation');
        segment.instances[0].mask.forEach((_, i) => { if (segment.instances[1].mask[i]) segment.instances[0].mask[i] = 0; });
        expect(analyzeTrackPositions(frame)).toMatchObject({ carAhead: 1, opponentPosition: 'middle' });
        segment.instances[1].classId = MODEL_LABELS.indexOf('car pack');
        expect(analyzeTrackPositions(frame)).toEqual({ cornerDirection: 'left', playerPosition: 'inside', carAhead: 1 });
    });

    it('prefers an individual at the same contact distance as a pack', () => {
        const frame = vision(0, { classNames: MODEL_LABELS, opponent: 'middle' });
        const segment = frame.detections.segment!;
        if (segment.task !== 'segment') throw new Error('Expected segmentation');
        segment.instances.unshift({ ...segment.instances[1], classId: MODEL_LABELS.indexOf('car pack') });
        expect(analyzeTrackPositions(frame)).toMatchObject({ carAhead: 1, opponentPosition: 'middle' });
    });

    it('preserves road support beneath a car overlapping a car pack', () => {
        const frame = vision(0, { classNames: MODEL_LABELS, opponent: 'middle' });
        const segment = frame.detections.segment!;
        if (segment.task !== 'segment') throw new Error('Expected segmentation');
        const car = segment.instances[1];
        const trackMask = segment.instances[0].mask.slice();
        const box = [...car.box] as typeof car.box;
        box[3] += 0.02;
        const mask = Uint8Array.from(car.mask, (_, i) => Number(
            (i % segment.width + 0.5) / segment.width >= box[0]
            && (i % segment.width + 0.5) / segment.width <= box[2]
            && (Math.floor(i / segment.width) + 0.5) / segment.height >= box[1]
            && (Math.floor(i / segment.width) + 0.5) / segment.height <= box[3]));
        segment.instances.push({ ...car, classId: MODEL_LABELS.indexOf('car pack'), box, mask });
        for (const instances of [segment.instances.slice(), segment.instances.slice().reverse()]) {
            segment.instances = instances;
            const scene = reconstructTrack(frame)!;
            expect(scene.geometry).not.toBeNull();
            expect(scene.cars.find((item) => !item.pack)?.roadSupported).toBe(true);
            expect(instances.find((item) => item.classId === MODEL_LABELS.indexOf('track'))!.mask).toEqual(trackMask);
        }
    });

    it('does not invent a car pack from an empty mask', () => {
        const frame = vision(0, { classNames: MODEL_LABELS, opponent: 'middle' });
        const segment = frame.detections.segment!;
        if (segment.task !== 'segment') throw new Error('Expected segmentation');
        const car = segment.instances[1];
        segment.instances.push({ ...car, classId: MODEL_LABELS.indexOf('car pack'), mask: new Uint8Array(car.mask.length),
            box: [car.box[0], car.box[1], car.box[2], car.box[3] + 0.02] });
        expect(analyzeTrackPositions(frame)).toEqual({ cornerDirection: 'left', playerPosition: 'inside', carAhead: 1, opponentPosition: 'middle' });
    });

    it.each([
        [0.05, 0.05, 0.15, 0.15], // Mirror / HUD.
        [0.01, 0.45, 0.11, 0.65], // Edge of capture.
        [0.2, 0.65, 0.8, 1], // Own bodywork.
        [0.1, 0.4, 0.15, 0.5], // Outside the fitted road.
    ])('ignores unsupported opponent boxes: %s, %s, %s, %s', (...box) => {
        expect(analyzeTrackPositions(vision(0, { cars: [box as [number, number, number, number]] })).carAhead).toBe(0);
    });

    it.each([0, 0.15, -0.15])('does not confuse a straight road heading of %s with curvature', (slope) => {
        const frame = vision(0, { cars: [], road: (x, y) => y < 59 && Math.abs(x - (1 + slope * y)) < 5 });
        expect(reconstructTrack(frame)?.geometry).not.toBeNull();
        expect(analyzeTrackPositions(frame)).toEqual({ carAhead: 0 });
    });

    it('continues scanning beyond gaps in the road mask', () => {
        const road = (x: number, y: number) => Math.abs(x) < 5 && (y < 12 || y > 30) && y < 59;
        const scene = reconstructTrack(vision(0, { road, cars: [] }))!;
        for (const boundary of [scene.leftBoundary, scene.rightBoundary]) {
            expect(boundary.some(({ y }) => y < 12)).toBe(true);
            expect(boundary.some(({ y }) => y > 30)).toBe(true);
        }
        expect(scene.geometry?.trackWidthM).toBeCloseTo(10, 0);
    });

    it('rejects a poor polynomial fit', () => {
        const road = (x: number, y: number) => Math.abs(x - 3 * Math.sin(y / 5)) < 5;
        expect(reconstructTrack(vision(0, { road, cars: [] }))?.geometry).toBeNull();
    });

    it('uses the camera offset to locate the car independently of the image center', () => {
        const frame = vision(0, { player: 'middle', camera: { lateralOffsetM: -0.4 } });
        expect(analyzeTrackPositions(frame).playerPosition).toBe('middle');
        const geometry = reconstructTrack(frame)!.geometry!;
        const shifted = reconstructTrack({ ...frame, calibration: { ...frame.calibration!, lateralOffsetM: 0.6 } })!.geometry!;
        expect(shifted.lateralOffsetM - geometry.lateralOffsetM).toBeCloseTo(-1, 1);
    });
});
