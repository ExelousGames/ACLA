import type { CameraCalibration, GroundPoint } from './track-vision-types';

/** An observer outside the reconstructed scene, independent of capture calibration. */
export function createLocalOverviewCamera(points: GroundPoint[]): CameraCalibration {
    const bounds = (axis: keyof GroundPoint) => {
        const values = points.map((point) => point[axis]);
        return (Math.min(...values) + Math.max(...values)) / 2;
    };
    const center = { x: bounds('x'), y: bounds('y'), z: bounds('z') };
    const pitchDeg = 40, yawDeg = -25, horizontalFovDeg = 60;
    const pitch = pitchDeg * Math.PI / 180, yaw = yawDeg * Math.PI / 180;
    const horizontalTan = Math.tan(horizontalFovDeg * Math.PI / 360), verticalTan = horizontalTan * 450 / 800;
    // Fit the projected extents so long, narrow roads still fill the viewport.
    const distance = Math.max(10, ...points.map((point) => {
        const x = point.x - center.x, y = point.y - center.y, z = point.z - center.z;
        const along = Math.sin(yaw) * x + Math.cos(yaw) * y;
        const right = Math.cos(yaw) * x - Math.sin(yaw) * y;
        const down = -Math.sin(pitch) * along - Math.cos(pitch) * z;
        const depth = Math.cos(pitch) * along - Math.sin(pitch) * z;
        return Math.max(Math.abs(right) / horizontalTan, Math.abs(down) / verticalTan) - depth;
    })) * 1.18;
    return {
        heightM: center.z + distance * Math.sin(pitch),
        lateralOffsetM: center.x - distance * Math.sin(yaw) * Math.cos(pitch),
        forwardOffsetM: center.y - distance * Math.cos(yaw) * Math.cos(pitch),
        pitchDeg, yawDeg, horizontalFovDeg, imageWidth: 800, imageHeight: 450,
    };
}
