export const DETECTION_TASKS = [
    { id: 'segment', label: 'Segmentation' },
    { id: 'depth', label: 'Depth', file: 'depth-anything-v2-small.onnx' },
] as const;

export type DetectionTask = typeof DETECTION_TASKS[number]['id'];

export interface DepthResult {
    task: 'depth';
    width: number;
    height: number;
    /** Relative depth is unitless. Omitted scale denotes legacy optical-axis meters. */
    scale?: 'metric' | 'relative';
    /** Near-to-far depth in the shared letterbox coordinates; excluded pixels are zero. */
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
export interface TrackBoundaryPoint extends GroundPoint {
    /** Inferred behind traffic from visible edge points; not a depth observation. */
    estimated?: boolean;
}
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
    /** Visible edges plus marked occlusion estimates; lengths and rows can differ. */
    leftBoundary: TrackBoundaryPoint[];
    rightBoundary: TrackBoundaryPoint[];
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
    leftBoundary: TrackBoundaryPoint[];
    rightBoundary: TrackBoundaryPoint[];
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
export interface TrackBoundaryPosition {
    /** Lateral distance from the vehicle center to each edge at the observed road slice. */
    leftBoundaryDistanceM: number;
    rightBoundaryDistanceM: number;
}
export interface DriverTrackPosition extends TrackBoundaryPosition {
    /** Distance ahead where both boundaries are observed; not extrapolated to the driver. */
    referenceDistanceM: number;
}
export interface OpponentTrackPosition {
    /** Vehicle coordinates relative to the driver: positive right and forward. */
    lateralOffsetM: number;
    longitudinalOffsetM: number;
}
/** Scene interpretation produced by Track Vision, independent of phrase rules. */
export interface TrackVisionAnalysis {
    carAhead?: 0 | 1;
    driverPosition?: DriverTrackPosition;
    /** Supported individual opponents, nearest first. Packs do not identify individuals. */
    opponents?: OpponentTrackPosition[];
}

export interface TrackVisionFrame {
    capturedAt: number;
    width: number;
    height: number;
    /** Explicitly applied calibration for this capture resolution. No implicit default. */
    calibration?: CameraCalibration;
    /** Minimum confidence for filtering, scene reconstruction and coaching. */
    filterConfidence?: number;
    detections: Partial<Record<DetectionTask, VisionResult>>;
}

export interface TrackVisionDetection extends TrackVisionFrame {
    reconstruction: LocalTrackScene | null;
    /** Track edges and all accepted car/car-pack boxes in the captured image, independent of depth and calibration. */
    reconstructedScene?: import('./reconstructed-scene').ReconstructedScene | null;
    geometry: TrackGeometry | null;
    /** Null without segmentation; unknown scene properties remain unset. */
    analysis: TrackVisionAnalysis | null;
}
