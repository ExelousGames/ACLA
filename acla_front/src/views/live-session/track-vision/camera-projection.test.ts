import { createCameraProjection, DEFAULT_CAMERA, validCalibration } from './camera-projection';
import { evaluateRoad, fitRoadPolynomial } from './road-polynomial';

const camera = { ...DEFAULT_CAMERA, heightM: 2, pitchDeg: 0, imageWidth: 1600, imageHeight: 900 };
it('matches the analytic level-camera projection in meters', () => {
    const projection = createCameraProjection(camera);
    const point = projection.localToImage({ x: 2, y: 10, z: 0 })!;
    expect(point.u).toBeCloseTo(0.6, 8);
    expect(point.v).toBeCloseTo(0.5 + 160 / 900, 8);
    expect(projection.imageToGround(point.u, point.v)).toEqual({ x: expect.closeTo(2, 8), y: expect.closeTo(10, 8), z: 0 });
    expect(projection.imageToGround(0.5, 0.5)).toBeNull();
    expect(projection.imageToGround(0.5, 0.2)).toBeNull();
    expect(projection.localToImage({ x: 0, y: -1, z: 0 })).toBeNull();
});

it.each([-10, 0, 20])('round-trips ground points at pitch %s with seat offsets and yaw', (pitchDeg) => {
    const projection = createCameraProjection({ ...camera, pitchDeg, yawDeg: 12, lateralOffsetM: -0.4, forwardOffsetM: 1.2 });
    for (const point of [{ x: 2, y: 10, z: 0 }, { x: -3, y: 25, z: 0 }, { x: 6, y: 40, z: 0 }]) {
        const image = projection.localToImage(point)!;
        const actual = projection.imageToGround(image.u, image.v)!;
        expect(actual.x).toBeCloseTo(point.x, 8);
        expect(actual.y).toBeCloseTo(point.y, 8);
    }
});

it('scales metric distances with height and validates calibration limits', () => {
    const base = createCameraProjection(camera).imageToGround(0.6, 0.8)!;
    const higher = createCameraProjection({ ...camera, heightM: 4 }).imageToGround(0.6, 0.8)!;
    expect(higher.y).toBeCloseTo(base.y * 2, 8);
    expect(higher.x).toBeCloseTo(base.x * 2, 8);
    for (const patch of [{ heightM: 0 }, { pitchDeg: NaN }, { horizontalFovDeg: 180 }, { imageHeight: 0 }, { yawDeg: Infinity }]) {
        expect(validCalibration({ ...camera, ...patch })).toBe(false);
    }
    expect(validCalibration(camera)).toBe(true);
});

it('uses optical-axis depth and preserves height, pitch, yaw and camera offsets', () => {
    const projection = createCameraProjection({ ...camera, pitchDeg: 10, yawDeg: 12, lateralOffsetM: -0.4, forwardOffsetM: 1.2 });
    const point = projection.imageToLocal(0.6, 0.4, 20)!;
    expect(point.z).not.toBe(0);
    expect(projection.localToImage(point)).toEqual({ u: expect.closeTo(0.6, 8), v: expect.closeTo(0.4, 8) });
    const level = createCameraProjection(camera);
    expect(level.imageToLocal(0.6, 0.5, 10)).toEqual({ x: expect.closeTo(2, 8), y: 10, z: 2 });
    for (const depth of [0, -1, NaN, Infinity]) expect(level.imageToLocal(0.5, 0.5, depth)).toBeNull();
});

it('recovers a metric quadratic despite isolated segmentation outliers', () => {
    const points = Array.from({ length: 100 }, (_, i) => ({ x: 2 + 0.1 * i - 0.002 * i * i + (i % 23 === 0 ? 3 : 0), y: i, z: 0 }));
    const fit = fitRoadPolynomial(points)!;
    expect(evaluateRoad(fit, 10)).toBeCloseTo(2.8, 1);
    expect(evaluateRoad(fit, 50)).toBeCloseTo(2, 1);
    expect(fit.coefficients[2]).toBeCloseTo(-0.002, 4);
    expect(fitRoadPolynomial(points.slice(0, 5))).toBeNull();
    expect(fitRoadPolynomial(points.map((p) => ({ ...p, y: 10 })))).toBeNull();
});
