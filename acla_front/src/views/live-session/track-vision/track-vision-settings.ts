import { DEFAULT_CAMERA, validCalibration } from './camera-projection';
import { VISION_CONFIDENCE } from './semantic-scene';
import type { CameraCalibration, CameraParameters, DetectionTask } from './track-vision-types';
import { DEPTH_INPUT_SIZE, MODEL_INPUT_RESOLUTIONS, VISION_INPUT_SIZE } from './vision-config';

export const TRACK_VISION_SETTINGS_KEY = 'acla.track-vision-settings';

export interface TrackVisionSettings {
    inputSizes: Record<DetectionTask, number>;
    confidence: number;
    filterConfidence: number;
    displayLabel: string;
    cameraDraft: CameraParameters;
    calibration?: CameraCalibration;
    showCalibrationOnCapture: boolean;
}

const defaults: TrackVisionSettings = {
    inputSizes: { segment: VISION_INPUT_SIZE, depth: DEPTH_INPUT_SIZE },
    confidence: 0.5,
    filterConfidence: VISION_CONFIDENCE,
    displayLabel: '',
    cameraDraft: DEFAULT_CAMERA,
    showCalibrationOnCapture: false,
};
const validConfidence = (value: unknown): value is number => typeof value === 'number' && value >= 0.1 && value <= 0.95;

export function readTrackVisionSettings(): TrackVisionSettings {
    try {
        const saved: Partial<TrackVisionSettings> & { version?: number } = JSON.parse(window.localStorage.getItem(TRACK_VISION_SETTINGS_KEY) ?? '{}');
        if (saved?.version !== 1) return defaults;
        const camera = saved.cameraDraft && { ...saved.cameraDraft, imageWidth: 1, imageHeight: 1 };
        return {
            inputSizes: {
                segment: MODEL_INPUT_RESOLUTIONS.segment.some(({ size }) => size === saved.inputSizes?.segment)
                    ? saved.inputSizes!.segment : defaults.inputSizes.segment,
                depth: MODEL_INPUT_RESOLUTIONS.depth.some(({ size }) => size === saved.inputSizes?.depth)
                    ? saved.inputSizes!.depth : defaults.inputSizes.depth,
            },
            confidence: validConfidence(saved.confidence) ? saved.confidence : defaults.confidence,
            filterConfidence: validConfidence(saved.filterConfidence) ? saved.filterConfidence : defaults.filterConfidence,
            displayLabel: typeof saved.displayLabel === 'string' ? saved.displayLabel : defaults.displayLabel,
            cameraDraft: validCalibration(camera) ? saved.cameraDraft! : defaults.cameraDraft,
            calibration: validCalibration(saved.calibration) ? saved.calibration : undefined,
            showCalibrationOnCapture: saved.showCalibrationOnCapture === true,
        };
    } catch {
        return defaults;
    }
}

export function saveTrackVisionSettings(settings: TrackVisionSettings): void {
    try {
        window.localStorage.setItem(TRACK_VISION_SETTINGS_KEY, JSON.stringify({ version: 1, ...settings }));
    } catch (error) {
        console.warn('Unable to save Track Vision settings locally', error);
    }
}
