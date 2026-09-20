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
export interface TrackVisionDetection {
    capturedAt: number;
    width: number;
    height: number;
    detections: Partial<Record<DetectionTask, VisionResult>>;
}
