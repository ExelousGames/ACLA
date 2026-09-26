import type { CameraCalibration, CameraParameters, GroundPoint } from './track-vision-types';

export const DEFAULT_CAMERA: CameraParameters = {
    heightM: 1.2, pitchDeg: 5, yawDeg: 0, horizontalFovDeg: 90, lateralOffsetM: 0, forwardOffsetM: 0,
};
const radians = (angle: number) => angle * Math.PI / 180;

export function validCalibration(camera: CameraCalibration | undefined): camera is CameraCalibration {
    return Boolean(camera && Object.values(camera).every(Number.isFinite)
        && camera.heightM >= 0.2 && camera.heightM <= 5
        && camera.pitchDeg >= -15 && camera.pitchDeg <= 45 && Math.abs(camera.yawDeg) <= 45
        && camera.horizontalFovDeg >= 20 && camera.horizontalFovDeg <= 150
        && Math.abs(camera.lateralOffsetM) <= 3 && Math.abs(camera.forwardOffsetM) <= 5
        && camera.imageWidth > 0 && camera.imageHeight > 0);
}

/** Pinhole intrinsics and camera pose for the local 3D scene. */
export function createCameraProjection(camera: CameraCalibration) {
    const pitch = radians(camera.pitchDeg);
    const yaw = radians(camera.yawDeg);
    const cp = Math.cos(pitch), sp = Math.sin(pitch), cy = Math.cos(yaw), sy = Math.sin(yaw);
    const fx = 1 / (2 * Math.tan(radians(camera.horizontalFovDeg) / 2));
    const fy = fx * camera.imageWidth / camera.imageHeight;
    return {
        localToImage({ x, y, z }: GroundPoint): { u: number; v: number } | null {
            const dx = x - camera.lateralOffsetM, dy = y - camera.forwardOffsetM;
            const along = sy * dx + cy * dy;
            const down = camera.heightM - z;
            const depth = cp * along + down * sp;
            if (!Number.isFinite(depth) || depth <= 0.001) return null;
            return { u: 0.5 + fx * (cy * dx - sy * dy) / depth,
                v: 0.5 + fy * (down * cp - sp * along) / depth };
        },
        imageToLocal(u: number, v: number, depthM: number): GroundPoint | null {
            if (![u, v, depthM].every(Number.isFinite) || u < 0 || u > 1 || v < 0 || v > 1 || depthM <= 0) return null;
            const right = (u - 0.5) / fx, down = (v - 0.5) / fy;
            const along = cp - down * sp;
            return { x: camera.lateralOffsetM + depthM * (cy * right + sy * along),
                y: camera.forwardOffsetM + depthM * (-sy * right + cy * along),
                z: camera.heightM - depthM * (sp + down * cp) };
        },
        /** Calibration grid and synthetic fixtures only; reconstruction uses measured depth. */
        imageToGround(u: number, v: number): GroundPoint | null {
            if (![u, v].every(Number.isFinite) || u < 0 || u > 1 || v < 0 || v > 1) return null;
            const right = (u - 0.5) / fx, down = (v - 0.5) / fy;
            const vertical = sp + down * cp;
            // Exclude sky and unstable near-horizon rays.
            if (vertical <= 0.001) return null;
            const scale = camera.heightM / vertical;
            const along = cp - down * sp;
            return { x: camera.lateralOffsetM + scale * (cy * right + sy * along),
                y: camera.forwardOffsetM + scale * (-sy * right + cy * along), z: 0 };
        },
    };
}
