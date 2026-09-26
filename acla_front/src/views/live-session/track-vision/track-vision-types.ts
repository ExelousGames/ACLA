export const DETECTION_TASKS = [
    { id: 'segment', label: 'Segmentation', description: 'Detect track features using the model and labels uploaded to the backend.' },
    { id: 'depth', label: 'Depth', description: 'Estimate depth for local 3D track and car reconstruction.', file: 'yolo26n-depth.onnx' },
] as const;

export type DetectionTask = typeof DETECTION_TASKS[number]['id'];
export type EnabledDetections = Record<DetectionTask, boolean>;

export interface DepthResult {
    task: 'depth';
    width: number;
    height: number;
    /** Estimated optical-axis depth in meters in the letterboxed model input. */
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
export interface CameraParameters {
    heightM: number;
    /** Positive pitch looks down; positive yaw looks right. Roll is assumed zero. */
    pitchDeg: number;
    yawDeg: number;
    horizontalFovDeg: number;
    /** Camera position relative to the vehicle origin, in meters. */
    lateralOffsetM: number;
    forwardOffsetM: number;
}

export interface CameraCalibration extends CameraParameters {
    imageWidth: number;
    imageHeight: number;
}

/** Vehicle coordinates in meters: X right, Y forward, Z up. */
export interface GroundPoint { x: number; y: number; z: number }
export interface ReconstructedCar {
    classId: number;
    confidence: number;
    pack: boolean;
    points: GroundPoint[];
    center: GroundPoint;
    min: GroundPoint;
    max: GroundPoint;
    /** Road support just below this instance in the captured image. */
    roadSupported: boolean;
}
export interface LocalTrackScene {
    /** Independently observed edges; arrays need not have matching lengths or rows. */
    leftBoundary: GroundPoint[];
    rightBoundary: GroundPoint[];
    cars: ReconstructedCar[];
    geometry: TrackGeometry | null;
}
export interface RoadPolynomial {
    /** X(Y) = c0 + c1 Y + c2 Y², in meters. */
    coefficients: [number, number, number];
    minY: number;
    maxY: number;
    rmseM: number;
}
export interface TrackGeometry {
    leftBoundary: GroundPoint[];
    rightBoundary: GroundPoint[];
    left: RoadPolynomial;
    right: RoadPolynomial;
    center: RoadPolynomial;
    /** Position is evaluated at this observed distance, never extrapolated to the unseen car. */
    referenceY: number;
    trackWidthM: number;
    lateralOffsetM: number;
    headingDeg: number;
    curvaturePerM: number;
}
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
    /** Lowest row used for track edges, normalized to source-image height. Defaults to 1. */
    boundaryStartY?: number;
    /** Explicitly applied calibration for this capture resolution. No implicit default. */
    calibration?: CameraCalibration;
    detections: Partial<Record<DetectionTask, VisionResult>>;
}

export interface TrackVisionDetection extends TrackVisionFrame {
    reconstruction: LocalTrackScene | null;
    geometry: TrackGeometry | null;
    /** Null without segmentation; unknown scene properties remain unset. */
    analysis: TrackVisionAnalysis | null;
}
