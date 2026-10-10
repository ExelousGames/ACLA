import { projectBirdsEyeScene } from './birds-eye-scene';
import { createCameraProjection, DEFAULT_CAMERA } from './camera-projection';
import type { ImageCar, ReconstructedScene } from './reconstructed-scene';

const calibration = { ...DEFAULT_CAMERA, heightM: 2, pitchDeg: 0, imageWidth: 1600, imageHeight: 900 };
const scene = (patch: Partial<ReconstructedScene> = {}): ReconstructedScene => ({
    width: 1600, height: 900, ribbons: [], leftBoundary: [], rightBoundary: [], centerline: [], cars: [], ...patch,
});
const car = (box: ImageCar['box'], pack = false): ImageCar => ({ classId: pack ? 6 : 3, confidence: 0.85, pack, box });

it('corrects the tenfold ground scale for boundaries, centerline and traffic', () => {
    // The uncorrected level camera puts row 610 at 10 m; the calibrated distance is 1 m.
    const source = scene({ leftBoundary: [[{ x: 400, y: 610 }, { x: 600, y: 530 }]],
        rightBoundary: [[{ x: 1200, y: 610 }, { x: 1000, y: 530 }]],
        centerline: [[{ x: 800, y: 610 }, { x: 800, y: 530 }]],
        cars: [car([920, 550, 1000, 610])] });
    const original = JSON.stringify(source);
    const ground = projectBirdsEyeScene(source, calibration)!;
    expect(ground.cars[0].position.y).toBeCloseTo(1, 8);
    expect(ground.leftBoundary[0]).toEqual([
        { x: expect.closeTo(-0.5), y: expect.closeTo(1), z: 0 },
        { x: expect.closeTo(-0.5), y: expect.closeTo(2), z: 0 },
    ]);
    expect(ground.rightBoundary[0].map(({ x }) => x)).toEqual([expect.closeTo(0.5), expect.closeTo(0.5)]);
    expect(ground.centerline[0]).toEqual([{ x: 0, y: expect.closeTo(1), z: 0 }, { x: 0, y: expect.closeTo(2), z: 0 }]);
    expect(ground.cars[0]).toEqual({ classId: 3, confidence: 0.85, pack: false,
        position: { x: expect.closeTo(0.2), y: expect.closeTo(1), z: 0 } });
    expect(JSON.stringify(source)).toBe(original);
});

it('uses the applied yaw, pitch and seat offsets relative to the driver', () => {
    const camera = { ...calibration, yawDeg: 12, pitchDeg: 8, lateralOffsetM: -0.5, forwardOffsetM: 1.5 };
    const projection = createCameraProjection(camera);
    const points = [{ x: -4, y: 15, z: 0 }, { x: 2, y: 40, z: 0 }];
    const line = points.map((point) => {
        const image = projection.localToImage(point)!;
        return { x: image.u * 1600, y: image.v * 900 };
    });
    expect(projectBirdsEyeScene(scene({ leftBoundary: [line] }), camera)!.leftBoundary[0])
        .toEqual(points.map(({ x, y }) => ({
            x: expect.closeTo(camera.lateralOffsetM + (x - camera.lateralOffsetM) / 10),
            y: expect.closeTo(camera.forwardOffsetM + (y - camera.forwardOffsetM) / 10), z: 0,
        })));
});

it('preserves boundary gaps and splits at sky, invalid or out-of-range points', () => {
    const near = [{ x: 600, y: 610 }, { x: 700, y: 550 }];
    for (const invalid of [{ x: 800, y: 440 }, { x: NaN, y: 610 }, { x: 1600, y: 451 }]) {
        const ground = projectBirdsEyeScene(scene({ leftBoundary: [[...near, invalid, ...near], near] }), calibration)!;
        expect(ground.leftBoundary).toHaveLength(3);
        expect(ground.leftBoundary.every((line) => line.length === 2)).toBe(true);
    }
});

it('checks the range limit after converting the ground projection to meters', () => {
    const ground = projectBirdsEyeScene(scene({ cars: [car([760, 449, 840, 455])] }), calibration)!;
    expect(ground.cars).toHaveLength(1);
    expect(ground.cars[0].position).toEqual({ x: 0, y: expect.closeTo(32), z: 0 });
    expect(ground.unplacedCars).toBe(0);
});

it('carries cars and car packs without track edges and counts unprojectable cars', () => {
    const ground = projectBirdsEyeScene(scene({ cars: [car([920, 550, 1000, 610]), car([600, 500, 700, 550], true),
        car([700, 400, 800, 440]), car([700, 800, 800, 900]), car([700, 600, NaN, 650])] }), calibration)!;
    expect(ground.cars).toHaveLength(2);
    expect(ground.cars[1]).toMatchObject({ classId: 6, confidence: 0.85, pack: true });
    expect(ground.unplacedCars).toBe(3);
    expect(ground.leftBoundary).toEqual([]);
});

it('requires valid applied calibration for the exact scene dimensions', () => {
    expect(projectBirdsEyeScene(null, calibration)).toBeNull();
    expect(projectBirdsEyeScene(scene())).toBeNull();
    expect(projectBirdsEyeScene(scene(), { ...calibration, heightM: 0 })).toBeNull();
    expect(projectBirdsEyeScene(scene({ width: 1920 }), calibration)).toBeNull();
});
