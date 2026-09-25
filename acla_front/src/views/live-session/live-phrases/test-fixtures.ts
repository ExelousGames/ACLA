import type { CornerPosition, TrackVisionDetection } from '../track-vision/track-vision-types';

/** Published positions only: phrase tests do not need screen masks or geometry. */
export function vision(capturedAt: number, options: {
    corner?: 'left' | 'right' | 'straight';
    player?: CornerPosition;
    opponent?: CornerPosition;
    playerCenterX?: number | null;
    carAhead?: boolean;
} = {}): TrackVisionDetection {
    const { corner = 'left', player = 'inside', opponent = 'outside', carAhead = true } = options;
    return {
        capturedAt, width: 1600, height: 900, detections: {},
        playerCenterX: options.playerCenterX === null ? undefined : options.playerCenterX ?? 0.5,
        analysis: corner === 'straight' || options.playerCenterX === null ? {} : {
            cornerDirection: corner, playerPosition: player,
            carAhead: carAhead ? 1 : 0, opponentPosition: carAhead ? opponent : undefined,
        },
    };
}
