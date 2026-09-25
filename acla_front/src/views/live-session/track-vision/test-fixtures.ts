import type { CornerPosition, SegmentResult, TrackVisionFrame } from './track-vision-types';
import { letterbox } from './yolo-segmentation';

type Box = SegmentResult['instances'][number]['box'];
export const MODEL_LABELS = ['track', 'curb', 'grass', 'car', 'other', 'fence', 'car pack', 'sand', 'left_boundary', 'right_boundary', 'Outfield asphalt road'];
export function vision(capturedAt: number, options: {
    corner?: 'left' | 'right' | 'straight';
    player?: CornerPosition;
    opponent?: CornerPosition;
    playerCenterX?: number | null;
    cars?: Box[];
    width?: number;
    height?: number;
    road?: (x: number, y: number) => boolean;
    classNames?: string[];
} = {}): TrackVisionFrame {
    const { width = 1600, height = 900, corner = 'left', player = 'inside', opponent = 'outside', classNames = ['track', 'car'] } = options;
    const lane = { inside: 0.25, middle: 0.5, outside: 0.75 };
    const across = (position: CornerPosition) => corner === 'right' ? 1 - lane[position] : lane[position];
    const roadWidth = (y: number) => 0.28 + 0.8 * (y - 0.4);
    const roadCenter = (y: number) => 0.5 + (0.5 - across(player)) * 0.6
        + (corner === 'straight' ? 0 : corner === 'left' ? -1 : 1) * (0.8 - y) ** 2;
    const opponentX = roadCenter(0.625) + (across(opponent) - 0.5) * roadWidth(0.625);
    const cars = options.cars ?? [[opponentX - 0.035, 0.48, opponentX + 0.035, 0.6]];
    const size = 160;
    const { padX, padY, resizedWidth, resizedHeight } = letterbox(width, height, 640);
    const mask = (active: (x: number, y: number) => boolean) => Uint8Array.from({ length: size * size }, (_, index) => (
        Number(active(((index % size + 0.5) / size * 640 - padX) / resizedWidth,
            ((Math.floor(index / size) + 0.5) / size * 640 - padY) / resizedHeight))
    ));
    const road = options.road ?? ((x, y) => y >= 0.3 && y <= 0.95
        && Math.abs(x - roadCenter(y)) <= roadWidth(y) / 2);
    return {
        capturedAt, width, height,
        playerCenterX: options.playerCenterX === null ? undefined : options.playerCenterX ?? 0.5,
        detections: { segment: {
            task: 'segment', width: size, height: size, inferenceMs: 1, classNames: [...classNames],
            instances: [
                { classId: classNames.indexOf('track'), confidence: 0.9, box: [0, 0, 1, 1], mask: mask(road) },
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
