import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import type { LiveTelemetrySnapshot } from 'views/live-session/live-telemetry-store';
import type { TrackVisionDetection } from 'views/live-session/track-vision/track-vision-types';
import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import { selectLiveMapReference } from 'views/live-session/live-map-reference';
import { DrivingReference, inspectDrivingReference, normalizedAggression, Pedals, ReferencePedalTargets, referencePedalTargets } from './driving-reference';

export interface DrivingAction { throttle: number; brake: number; steering: number; shiftUp?: boolean; shiftDown?: boolean }
export const NEUTRAL: DrivingAction = { throttle: 0, brake: 0, steering: 0 };
export const clamp = (value: number, low = -1, high = 1) => Math.max(low, Math.min(high, value));
export interface DrivingObservation {
    capturedAt: number;
    measuredAt: number;
    session: string;
    context: { game: string; car: string; map: string; classes: string[] };
    features: number[];
    normalizedPosition: number;
    referenceDistance: number;
    speed: number;
    /** Forward gears start at 1; neutral is 0 and reverse is -1. */
    gear: number;
    rpm: number;
    heading: number;
    pitch: number;
    roll: number;
    controls: DrivingAction;
    reference: DrivingReference;
    aggression: number;
    pedals: ReferencePedalTargets;
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
    referenceLap: TopLapReferenceEntry | null, aggression = 50, now = Date.now()): ObservationResult {
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
    const gear = sample.Physics_gear, rpm = sample.Physics_rpm;
    // Both readers use ACC indexing: reverse = 0, neutral = 1, first = 2.
    if (!finite(gear) || !Number.isInteger(gear) || gear < 0 || !finite(rpm) || !Number.isInteger(rpm) || rpm < 0) {
        return { error: 'Valid current gear and engine RPM are required for learned shifting.' };
    }
    if (sample.Physics_autoshifter_on === true) return { error: 'Disable automatic gear shifting in the simulator so the model can learn gear changes.' };
    if (![...velocity, ...rotation, ...controls, speed, position].every(finite)) {
        return { error: 'Velocity, speed, rotation, steering, throttle, brake and normalized position are required.' };
    }
    if (position! < 0 || position! > 1 || speed! < 0 || controls[0]! < 0 || controls[0]! > 1
        || controls[1]! < 0 || controls[1]! > 1 || Math.abs(controls[2]!) > 1) return { error: 'Telemetry inputs are outside their valid ranges.' };
    const prediction = vision.position;
    if (!prediction || prediction.coordinateFrame !== 'car_local'
        || prediction.referenceDelta.length !== 3 || !prediction.referenceDelta.every(finite)
        || !finite(prediction.normalizedPosition) || Math.abs(prediction.normalizedPosition - position!) > 0.0001) {
        return { error: 'Track Vision needs car-local position-model output and a matching Live Map reference line.' };
    }
    if (referenceLap && !selectLiveMapReference([referenceLap], game, map, car, sample.Graphics_track_grip_status)) {
        return { error: 'Live Map reference lap does not match the current game, car or circuit. Load a matching reference lap in Live Session.' };
    }
    const prepared = inspectDrivingReference(referenceLap);
    if (!prepared.reference) return { error: prepared.error };
    const reference = prepared.reference;
    if (prediction.referenceLap !== referenceLap) return { error: 'Live Map reference changed. Waiting for matching Track Vision inputs.' };
    if (!finite(aggression) || aggression < 0 || aggression > 100) return { error: 'Driving aggression must be between 0 and 100.' };
    const pedals = referencePedalTargets(reference, position!, aggression);
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
        measuredAt: paired.receivedAt,
        session: JSON.stringify([paired.sessionGeneration, paired.streamGeneration, game, car, map]),
        context: { game, car, map, classes: segment.classNames },
        features: [
            ...(velocity as number[]).map((v) => clamp(v / 100)), clamp(speed! / 360, 0, 1),
            ...category(game), ...category(car), ...category(map),
            Math.sin(2 * Math.PI * position!), Math.cos(2 * Math.PI * position!),
            ...[heading, pitch, roll].flatMap((angle) => [Math.sin(angle), Math.cos(angle)]),
            // Signed right/forward/up offsets are already relative to the car.
            ...controls as number[], ...prediction.referenceDelta.map((v) => clamp(v / 20)),
            clamp((gear - 1) / 8), clamp(rpm / 20000, 0, 1),
            ...pooled.map((v) => clamp(v, 0, 1)),
            normalizedAggression(aggression), pedals.target.throttle, pedals.target.brake,
        ],
        // Rotation preserves distance, so the same proximity reward applies in local axes.
        normalizedPosition: position!, referenceDistance: Math.hypot(...prediction.referenceDelta),
        speed: speed!, gear: gear - 1, rpm, heading, pitch, roll, controls: action,
        reference, aggression, pedals,
    } };
}

export interface DrivingReward {
    speed: number; line: number; instability: number; referencePedal: number; total: number;
    pedalComparison: ReferencePedalTargets & { measured: Pedals; aggression: number };
}
export const angleDifference = (a: number, b: number) => Math.atan2(Math.sin(a - b), Math.cos(a - b));
export function drivingReward(previous: DrivingObservation, current: DrivingObservation,
    action: DrivingAction): DrivingReward {
    const dt = (current.capturedAt - previous.capturedAt) / 1000;
    if (dt <= 0 || dt > 1.5 || previous.session !== current.session) throw new Error('Training transition crossed a session boundary or data gap.');
    let progress = current.normalizedPosition - previous.normalizedPosition;
    if (progress < -0.5) progress += 1;
    if (progress > 0.5) progress -= 1;
    if (Math.abs(progress) > 0.1) throw new Error('Lap position jumped. Reposition the car and start a new episode.');
    if (previous.reference !== current.reference) throw new Error('Live Map reference changed. Start a new episode.');
    const proximity = Math.exp(-current.referenceDistance / 3);
    const forwardProgress = clamp(progress, 0, 0.02);
    // Weight forward lap progress by km/h so the same section pays more at higher speed.
    const speed = forwardProgress * current.speed * proximity;
    // Score measured pedals at the new position using the preceding action's settings.
    const pedals = referencePedalTargets(previous.reference, current.normalizedPosition, previous.aggression);
    const error = (Math.abs(current.controls.throttle - pedals.target.throttle)
        + Math.abs(current.controls.brake - pedals.target.brake)) / 2;
    const referencePedal = current.speed > 0 && current.gear >= 0
        ? 100 * forwardProgress * proximity * (1 - 2 * error) : 0;
    // Reward forward progress on the reference; sitting still never earns line reward.
    const line = 100 * clamp(progress, -0.02, 0.02) * proximity
        - 0.15 * Math.min(current.referenceDistance / 3, 4) * dt - (current.speed < 1 ? 0.1 * dt : 0);
    const yawRate = Math.abs(angleDifference(current.heading, previous.heading)) / dt;
    const attitude = Math.abs(angleDifference(current.roll, previous.roll)) / dt
        + Math.abs(angleDifference(current.pitch, previous.pitch)) / dt;
    const instability = -dt * (0.4 * Math.max(0, yawRate - 0.8) ** 2
        + 0.15 * attitude ** 2 + 0.5 * action.throttle * action.brake);
    return { speed, line, instability, referencePedal,
        pedalComparison: { ...pedals, measured: { throttle: current.controls.throttle, brake: current.controls.brake }, aggression: previous.aggression },
        total: clamp(speed + line + instability + referencePedal, -10, 10) };
}
