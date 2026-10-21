import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import type { LiveTelemetrySnapshot } from 'views/live-session/live-telemetry-store';
import type { TrackVisionDetection } from 'views/live-session/track-vision/track-vision-types';

export interface DrivingAction { throttle: number; brake: number; steering: number }
export const NEUTRAL: DrivingAction = { throttle: 0, brake: 0, steering: 0 };
export const clamp = (value: number, low = -1, high = 1) => Math.max(low, Math.min(high, value));
export interface DrivingObservation {
    capturedAt: number;
    session: string;
    context: { game: string; car: string; map: string; classes: string[] };
    features: number[];
    normalizedPosition: number;
    referenceDistance: number;
    speed: number;
    heading: number;
    pitch: number;
    roll: number;
    controls: DrivingAction;
}
export type ObservationResult = { observation: DrivingObservation; error?: never } | { error: string; observation?: never };
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

// A deterministic encoding preserves categorical inputs across saved policies.
function category(value: string): number[] {
    let hash = 2166136261;
    for (let i = 0; i < value.length; i++) hash = Math.imul(hash ^ value.charCodeAt(i), 16777619);
    return Array.from({ length: 16 }, (_, i) => (hash >>> i) & 1);
}

export function readDrivingObservation(live: LiveTelemetrySnapshot, vision: TrackVisionDetection | null,
    now = Date.now()): ObservationResult {
    if (live.telemetryStatus !== ACC_STATUS.ACC_LIVE || !finite(live.receivedAt)
        || now - live.receivedAt > 500 || now < live.receivedAt) return { error: 'Waiting for fresh, running live telemetry.' };
    if (!vision || !finite(vision.capturedAt) || now - vision.capturedAt > 1500 || now < vision.capturedAt) {
        return { error: 'Start Track Vision capture in Live Session. A fresh image is required.' };
    }
    const paired = vision.telemetry;
    if (!paired || paired.telemetryStatus !== ACC_STATUS.ACC_LIVE || !finite(paired.receivedAt)
        || vision.capturedAt - paired.receivedAt > 500 || vision.capturedAt < paired.receivedAt
        || paired.sessionGeneration !== live.sessionGeneration || paired.streamGeneration !== live.streamGeneration
        || paired.game !== live.game || paired.currentTelemetry.Static_track !== live.currentTelemetry.Static_track
        || paired.currentTelemetry.Static_car_model !== live.currentTelemetry.Static_car_model) {
        return { error: 'Waiting for Track Vision and telemetry from the same live session.' };
    }
    const sample = paired.currentTelemetry;
    const game = paired.game, car = sample.Static_car_model, map = sample.Static_track;
    if (!game || !car || !map) return { error: 'Game, car and map are required.' };
    const velocity = [sample.Physics_velocity_x, sample.Physics_velocity_y, sample.Physics_velocity_z];
    const rotation = [sample.Physics_heading, sample.Physics_pitch, sample.Physics_roll];
    const controls = [sample.Physics_gas, sample.Physics_brake, sample.Physics_steer_angle];
    const speed = sample.Physics_speed_kmh, position = sample.Graphics_normalized_car_position;
    if (![...velocity, ...rotation, ...controls, speed, position].every(finite)) {
        return { error: 'Velocity, speed, rotation, steering, throttle, brake and normalized position are required.' };
    }
    if (position! < 0 || position! > 1 || speed! < 0 || controls[0]! < 0 || controls[0]! > 1
        || controls[1]! < 0 || controls[1]! > 1 || Math.abs(controls[2]!) > 1) return { error: 'Telemetry inputs are outside their valid ranges.' };
    const prediction = vision.position;
    if (!prediction || prediction.referenceDelta.length !== 3 || !prediction.referenceDelta.every(finite)
        || !finite(prediction.normalizedPosition) || Math.abs(prediction.normalizedPosition - position!) > 0.0001) {
        return { error: 'Track Vision needs position-model output and a matching Live Map reference line.' };
    }
    const segment = vision.detections.segment;
    if (!segment || segment.task !== 'segment' || !Number.isInteger(segment.width) || !Number.isInteger(segment.height)
        || segment.width < 1 || segment.height < 1 || segment.width * segment.height > 4194304
        || !segment.classNames.length || segment.classNames.length > 64
        || segment.classNames.some((name) => typeof name !== 'string' || !name)
        || new Set(segment.classNames).size !== segment.classNames.length) return { error: 'Track Vision segmentation output is required.' };
    // Pool every class mask into a 4x4 spatial grid. Keep semantic channels separate.
    const pooled = new Array(segment.classNames.length * 16).fill(0);
    for (const instance of segment.instances) {
        if (!Number.isInteger(instance.classId) || instance.classId < 0 || instance.classId >= segment.classNames.length
            || !finite(instance.confidence) || instance.confidence < 0 || instance.confidence > 1
            || instance.mask.length !== segment.width * segment.height) return { error: 'Invalid segmentation mask.' };
        for (let y = 0; y < segment.height; y++) {
            for (let x = 0; x < segment.width; x++) {
                if (instance.mask[y * segment.width + x]) {
                    const cell = Math.floor(y * 4 / segment.height) * 4 + Math.floor(x * 4 / segment.width);
                    pooled[instance.classId * 16 + cell] += instance.confidence * 16 / instance.mask.length;
                }
            }
        }
    }
    const [heading, pitch, roll] = rotation as number[];
    const action = { throttle: controls[0]!, brake: controls[1]!, steering: controls[2]! };
    return { observation: {
        capturedAt: vision.capturedAt,
        session: JSON.stringify([paired.sessionGeneration, paired.streamGeneration, game, car, map]),
        context: { game, car, map, classes: segment.classNames },
        features: [
            ...(velocity as number[]).map((v) => clamp(v / 100)), clamp(speed! / 360, 0, 1),
            ...category(game), ...category(car), ...category(map),
            Math.sin(2 * Math.PI * position!), Math.cos(2 * Math.PI * position!),
            ...[heading, pitch, roll].flatMap((angle) => [Math.sin(angle), Math.cos(angle)]),
            ...controls as number[], ...prediction.referenceDelta.map((v) => clamp(v / 20)),
            ...pooled.map((v) => clamp(v, 0, 1)),
        ],
        normalizedPosition: position!, referenceDistance: Math.hypot(...prediction.referenceDelta),
        speed: speed!, heading, pitch, roll, controls: action,
    } };
}

export interface DrivingReward { line: number; convergence: number; instability: number; total: number }
export const angleDifference = (a: number, b: number) => Math.atan2(Math.sin(a - b), Math.cos(a - b));
export function drivingReward(previous: DrivingObservation, current: DrivingObservation,
    action: DrivingAction, previousAction: DrivingAction): DrivingReward {
    const dt = (current.capturedAt - previous.capturedAt) / 1000;
    if (dt <= 0 || dt > 1.5 || previous.session !== current.session) throw new Error('Training transition crossed a session boundary or data gap.');
    let progress = current.normalizedPosition - previous.normalizedPosition;
    if (progress < -0.5) progress += 1;
    if (progress > 0.5) progress -= 1;
    if (Math.abs(progress) > 0.1) throw new Error('Lap position jumped. Reposition the car and start a new episode.');
    const proximity = Math.exp(-current.referenceDistance / 3);
    // Reward forward progress on the reference; sitting still never earns line reward.
    const line = 100 * clamp(progress, -0.02, 0.02) * proximity
        - 0.15 * Math.min(current.referenceDistance / 3, 4) * dt - (current.speed < 1 ? 0.1 * dt : 0);
    const controlRate = (Math.abs(action.steering - previousAction.steering)
        + 0.5 * Math.abs(action.throttle - previousAction.throttle)
        + 0.5 * Math.abs(action.brake - previousAction.brake)) / dt;
    const yawRate = Math.abs(angleDifference(current.heading, previous.heading)) / dt;
    const attitude = Math.abs(angleDifference(current.roll, previous.roll)) / dt
        + Math.abs(angleDifference(current.pitch, previous.pitch)) / dt;
    const stability = 1 / (1 + controlRate + Math.max(0, yawRate - 0.8) + attitude);
    // Decreasing distance earns less when the correction is abrupt; divergence is always penalized.
    const improvement = clamp(previous.referenceDistance - current.referenceDistance, -3, 3);
    const convergence = 0.4 * improvement * (improvement > 0 ? stability : 1);
    const instability = -dt * (0.08 * controlRate ** 2 + 0.4 * Math.max(0, yawRate - 0.8) ** 2
        + 0.15 * attitude ** 2 + 0.5 * action.throttle * action.brake);
    return { line, convergence, instability, total: clamp(line + convergence + instability, -10, 10) };
}
