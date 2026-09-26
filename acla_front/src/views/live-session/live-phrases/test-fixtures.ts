import type { CornerPosition, TrackVisionDetection } from '../track-vision/track-vision-types';
import { DEFAULT_CAMERA } from '../track-vision/camera-projection';

/** Published positions only: phrase tests do not need screen masks or geometry. */
export function vision(capturedAt: number, options: {
    corner?: 'left' | 'right' | 'straight';
    player?: CornerPosition;
    opponent?: CornerPosition;
    cameraOffset?: number | null;
    carAhead?: boolean;
} = {}): TrackVisionDetection {
    const { corner = 'left', player = 'inside', opponent = 'outside', carAhead = true } = options;
    return {
        capturedAt, width: 1600, height: 900, detections: {},
        geometry: null, reconstruction: null,
        calibration: options.cameraOffset === null ? undefined : { ...DEFAULT_CAMERA, imageWidth: 1600, imageHeight: 900, lateralOffsetM: options.cameraOffset ?? 0 },
        analysis: corner === 'straight' || options.cameraOffset === null ? {} : {
            cornerDirection: corner, playerPosition: player,
            carAhead: carAhead ? 1 : 0, opponentPosition: carAhead ? opponent : undefined,
        },
    };
}
