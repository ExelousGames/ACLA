export const DETECTION_TASKS = [
    { id: 'segment', label: 'Segmentation', description: 'Detect track features using the model and labels uploaded to the backend.' },
    { id: 'depth', label: 'Depth', description: 'Estimate distance throughout the scene.', file: 'yolo26n-depth.onnx' },
] as const;

export type DetectionTask = typeof DETECTION_TASKS[number]['id'];
export type EnabledDetections = Record<DetectionTask, boolean>;

export interface DepthRange {
    /** Estimated distances in meters at the warm and cool ends of the overlay. */
    near: number;
    far: number;
}

export const DEFAULT_DEPTH_RANGE: DepthRange = { near: 5, far: 50 };

export interface DepthResult {
    task: 'depth';
    width: number;
    height: number;
    values: Float32Array;
}

export interface SegmentResult {
    task: 'segment';
    width: number;
    height: number;
    instances: Array<{
        classId: number;
        confidence: number;
        /** Box coordinates in the square model input, normalized to 0–1. */
        box: [number, number, number, number];
        mask: Uint8Array;
    }>;
}

export type VisionResult = (SegmentResult | DepthResult) & { inferenceMs: number; classNames: string[] };
/** Image row at which the marked vehicle centerline is compared with track edges. */
export const PLAYER_TRACK_ROW = 0.8;
export const VISION_MAX_AGE_MS = 2000;
export type CornerDirection = 'left' | 'right';
export type CornerPosition = 'inside' | 'middle' | 'outside';
/** Scene interpretation produced by Track Vision, independent of phrase rules. */
export interface TrackVisionAnalysis {
    carAhead?: 0 | 1;
    cornerDirection?: CornerDirection;
    playerPosition?: CornerPosition;
    opponentPosition?: CornerPosition;
}

export interface TrackVisionFrame {
    capturedAt: number;
    width: number;
    height: number;
    /** Vehicle centerline marked on the car at PLAYER_TRACK_ROW, normalized to the capture. No implicit default. */
    playerCenterX?: number;
    detections: Partial<Record<DetectionTask, VisionResult>>;
}

export interface TrackVisionDetection extends TrackVisionFrame {
    /** Null without segmentation; unknown scene properties remain unset. */
    analysis: TrackVisionAnalysis | null;
}
