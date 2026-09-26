import type { CameraParameters, CornerPosition, SegmentResult, TrackVisionFrame } from './track-vision-types';
import { createCameraProjection, DEFAULT_CAMERA } from './camera-projection';
import { letterbox } from './yolo-segmentation';

type Box = SegmentResult['instances'][number]['box'];
export const MODEL_LABELS = ['track', 'curb', 'grass', 'car', 'other', 'fence', 'car pack', 'sand', 'Outfield asphalt road'];
/** Render known metric roads through a calibrated camera into letterboxed model masks. */
export function vision(capturedAt: number, options: {
    corner?: 'left' | 'right' | 'straight';
    player?: CornerPosition;
    opponent?: CornerPosition;
    camera?: Partial<CameraParameters>;
    cars?: Box[];
    width?: number;
    height?: number;
    maskSize?: number;
    road?: (x: number, y: number) => boolean;
    classNames?: string[];
} = {}): TrackVisionFrame {
    const { width = 1600, height = 900, corner = 'left', player = 'inside', opponent = 'outside', classNames = ['track', 'car'], maskSize: size = 320 } = options;
    const calibration = { ...DEFAULT_CAMERA, ...options.camera, imageWidth: width, imageHeight: height };
    const projection = createCameraProjection(calibration);
    const lane = { inside: 0.25, middle: 0.5, outside: 0.75 };
    const across = (position: CornerPosition) => corner === 'right' ? 1 - lane[position] : lane[position];
    const roadCenter = (y: number) => (0.5 - across(player)) * 10
        + (corner === 'straight' ? 0 : corner === 'left' ? -1 : 1) * 0.003 * (y - 8) ** 2;
    const opponentX = roadCenter(18) + (across(opponent) - 0.5) * 10;
    const contactLeft = projection.localToImage({ x: opponentX - 0.8, y: 18, z: 0 })!;
    const contactRight = projection.localToImage({ x: opponentX + 0.8, y: 18, z: 0 })!;
    const cars = options.cars ?? [[contactLeft.u, contactLeft.v - 0.07, contactRight.u, contactRight.v]];
    const { padX, padY, resizedWidth, resizedHeight } = letterbox(width, height, 640);
    const mask = (active: (x: number, y: number) => boolean) => Uint8Array.from({ length: size * size }, (_, index) => (
        Number(active(((index % size + 0.5) / size * 640 - padX) / resizedWidth,
            ((Math.floor(index / size) + 0.5) / size * 640 - padY) / resizedHeight))
    ));
    const road = options.road ?? ((x, y) => y >= 1 && y < 59 && Math.abs(x - roadCenter(y)) <= 5);
    return {
        capturedAt, width, height, calibration,
        detections: { depth: {
            task: 'depth', width: 640, height: 640, inferenceMs: 1, classNames: [],
            values: Float32Array.from({ length: 640 * 640 }, (_, i) => {
                const u = (i % 640 + 0.5 - padX) / resizedWidth;
                const v = (Math.floor(i / 640) + 0.5 - padY) / resizedHeight;
                const car = cars.find(([left, top, right, bottom]) => u >= left && u <= right && v >= top && v <= bottom);
                const ground = projection.imageToGround(u, car ? car[3] : v);
                if (!ground) return NaN;
                const pitch = calibration.pitchDeg * Math.PI / 180, yaw = calibration.yawDeg * Math.PI / 180;
                const along = Math.sin(yaw) * (ground.x - calibration.lateralOffsetM) + Math.cos(yaw) * (ground.y - calibration.forwardOffsetM);
                return Math.cos(pitch) * along + calibration.heightM * Math.sin(pitch);
            }),
        }, segment: {
            task: 'segment', width: size, height: size, inferenceMs: 1, classNames: [...classNames],
            instances: [
                { classId: classNames.indexOf('track'), confidence: 0.9, box: [0, 0, 1, 1], mask: mask((u, v) => {
                    const point = projection.imageToGround(u, v);
                    return Boolean(point && road(point.x, point.y));
                }) },
                ...cars.map(([left, top, right, bottom]) => ({
                    classId: classNames.indexOf('car'), confidence: 0.9,
                    box: [(padX + left * resizedWidth) / 640, (padY + top * resizedHeight) / 640,
                        (padX + right * resizedWidth) / 640, (padY + bottom * resizedHeight) / 640] as Box,
                    mask: mask((x, y) => x >= left && x <= right && y >= top && y <= bottom),
                })),
            ],
        } },
    };
}
