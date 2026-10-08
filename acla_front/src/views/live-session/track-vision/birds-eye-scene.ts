import { createCameraProjection, validCalibration } from './camera-projection';
import type { ImageCar, ImagePoint, ReconstructedScene } from './reconstructed-scene';
import type { CameraCalibration, GroundPoint } from './track-vision-types';

export interface BirdsEyeScene {
    leftBoundary: GroundPoint[][];
    rightBoundary: GroundPoint[][];
    centerline: GroundPoint[][];
    cars: Array<Omit<ImageCar, 'box'> & { position: GroundPoint }>;
    unplacedCars: number;
}

/** Shared flat-road estimate for the BEV display and Live Phrases. */
export function projectBirdsEyeScene(scene: ReconstructedScene | null, calibration?: CameraCalibration): BirdsEyeScene | null {
    if (!scene || !validCalibration(calibration)
        || calibration.imageWidth !== scene.width || calibration.imageHeight !== scene.height) return null;
    const camera = createCameraProjection(calibration);
    const project = ({ x, y }: ImagePoint) => {
        const point = camera.imageToGround(x / scene.width, y / scene.height);
        return point && point.y > 0 && Math.hypot(point.x, point.y) <= 200 ? point : null;
    };
    const projectLines = (lines: ImagePoint[][]) => lines.flatMap((line) => {
        const segments: GroundPoint[][] = [];
        let segment: GroundPoint[] = [];
        for (const point of line) {
            const ground = project(point);
            if (ground) segment.push(ground);
            else {
                if (segment.length > 1) segments.push(segment);
                segment = [];
            }
        }
        if (segment.length > 1) segments.push(segment);
        return segments;
    });
    const cars = scene.cars.flatMap(({ box: [left, top, right, bottom], ...car }) => {
        // A clipped lower box has no visible road contact. Keep it in the unplaced count.
        if (![left, top, right, bottom].every(Number.isFinite) || right <= left || bottom <= top || bottom >= scene.height) return [];
        const position = project({ x: (left + right) / 2, y: bottom });
        return position ? [{ ...car, position }] : [];
    });
    return { leftBoundary: projectLines(scene.leftBoundary), rightBoundary: projectLines(scene.rightBoundary),
        centerline: projectLines(scene.centerline), cars, unplacedCars: scene.cars.length - cars.length };
}
