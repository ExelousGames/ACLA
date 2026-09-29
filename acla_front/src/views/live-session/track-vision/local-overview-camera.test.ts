import { createCameraProjection } from './camera-projection';
import { createLocalOverviewCamera } from './local-overview-camera';

it('keeps an elevated world in frame throughout a full orbit and at both tilt limits', () => {
    const points = [
        { x: 0, y: 0, z: 0 }, { x: -12, y: 10, z: -3 },
        { x: 9, y: 65, z: 7 }, { x: 24, y: 32, z: 2 },
    ];
    for (const pitchDeg of [-85, -40, 0, 40, 85]) {
        for (let yawDeg = -180; yawDeg <= 180; yawDeg += 15) {
            const projection = createCameraProjection(createLocalOverviewCamera(points, { pitchDeg, yawDeg }));
            for (const point of points) {
                const pixel = projection.localToImage(point);
                expect(pixel).not.toBeNull();
                expect(pixel!.u).toBeGreaterThan(0);
                expect(pixel!.u).toBeLessThan(1);
                expect(pixel!.v).toBeGreaterThan(0);
                expect(pixel!.v).toBeLessThan(1);
            }
        }
    }
});
