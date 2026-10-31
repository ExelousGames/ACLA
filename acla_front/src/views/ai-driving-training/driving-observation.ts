import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import type { LiveTelemetrySnapshot } from 'views/live-session/live-telemetry-store';
import type { TrackVisionDetection } from 'views/live-session/track-vision/track-vision-types';
import { DEFAULT_DRIVING_CONFIG, drivingFeatureEncoding } from './driving-model-config';
import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import { selectLiveMapReference } from 'views/live-session/live-map-reference';
import { DrivingReference, inspectDrivingReference, normalizedAggression, ReferenceSpeedTargets, referenceSpeedTargets } from './driving-reference';
import type { PolicyCameraImage } from 'views/live-session/track-vision/policy-camera';

export interface DrivingAction { throttle: number; brake: number; steering: number; shiftUp?: boolean; shiftDown?: boolean }
export const NEUTRAL: DrivingAction = { throttle: 0, brake: 0, steering: 0 };
export const clamp = (value: number, low = -1, high = 1) => Math.max(low, Math.min(high, value));
export interface DrivingObservation {
    capturedAt: number;
    measuredAt: number;
    session: string;
    context: { game: string; car: string; map: string };
    featureEncoding: string;
    features: number[];
    camera: PolicyCameraImage;
    normalizedPosition: number;
    /** Signed driver-to-reference offsets in car-local X (right) and Y (forward), in source units. */
    referenceError: [number, number];
    referenceDistance: number;
    speed: number;
    /** Adapter-local X/Y/Z velocity in m/s: ACC right/forward/up; iRacing right/up/forward. */
    localVelocity: [number, number, number];
    /** Signed yaw rate in rad/s: ACC adapter-local Z; iRacing adapter-local Y. */
    yawRate: number;
    /** Forward gears start at 1; neutral is 0 and reverse is -1. */
    gear: number;
    rpm: number;
    controls: DrivingAction;
    reference: DrivingReference;
    aggression: number;
    speedTargets: ReferenceSpeedTargets;
}
export type ObservationResult = { observation: DrivingObservation; error?: never } | { error: string; observation?: never };
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

// A deterministic encoding preserves categorical inputs across saved policies.
function category(value: string, bits: number): number[] {
    let hash = 2166136261;
    for (let i = 0; i < value.length; i++) hash = Math.imul(hash ^ value.charCodeAt(i), 16777619);
    return Array.from({ length: bits }, (_, i) => (hash >>> i) & 1);
}

export function readDrivingObservation(live: LiveTelemetrySnapshot, vision: TrackVisionDetection | null,
    referenceLap: TopLapReferenceEntry | null, aggression = DEFAULT_DRIVING_CONFIG.aggression, now = Date.now(), config = DEFAULT_DRIVING_CONFIG): ObservationResult {
    const o = config.observation;
    if (live.telemetryStatus !== ACC_STATUS.ACC_LIVE || !finite(live.receivedAt)
        || now - live.receivedAt > config.runtime.telemetryMaxAgeMs || now < live.receivedAt) return { error: 'Waiting for fresh, running live telemetry.' };
    if (!vision || !finite(vision.capturedAt) || now - vision.capturedAt > config.runtime.visionMaxAgeMs || now < vision.capturedAt) {
        return { error: 'Start Track Vision capture in Live Session. A fresh image is required.' };
    }
    const paired = vision.telemetry;
    if (!paired || paired.telemetryStatus !== ACC_STATUS.ACC_LIVE || !finite(paired.receivedAt)
        || vision.capturedAt - paired.receivedAt > config.runtime.telemetryMaxAgeMs || vision.capturedAt < paired.receivedAt
        || paired.sessionGeneration !== live.sessionGeneration || paired.streamGeneration !== live.streamGeneration
        || paired.game !== live.game || paired.currentTelemetry.Static_track !== live.currentTelemetry.Static_track
        || paired.currentTelemetry.Static_car_model !== live.currentTelemetry.Static_car_model) {
        return { error: 'Waiting for Track Vision and telemetry from the same live session.' };
    }
    const sample = paired.currentTelemetry;
    const game = paired.game, car = sample.Static_car_model, map = sample.Static_track;
    if (!game || !car || !map) return { error: 'Game, car and map are required.' };
    // Preserve adapter XYZ order and signs. ACC has already swapped Y/Z at capture.
    const velocity = [sample.Physics_local_velocity_x, sample.Physics_local_velocity_y, sample.Physics_local_velocity_z];
    const yawRate = game === 'iracing' ? sample.Physics_local_angular_vel_y : sample.Physics_local_angular_vel_z;
    const controls = [sample.Physics_gas, sample.Physics_brake, sample.Physics_steer_angle];
    const speed = sample.Physics_speed_kmh, position = sample.Graphics_normalized_car_position;
    // Both readers use ACC indexing: reverse = 0, neutral = 1, first = 2.
    // Gear/RPM must not block stalled-engine observations; keep missing readings finite for the policy.
    const gear = finite(sample.Physics_gear) ? sample.Physics_gear : 1;
    const rpm = finite(sample.Physics_rpm) ? sample.Physics_rpm : 0;
    if (sample.Physics_autoshifter_on === true) return { error: 'Disable automatic gear shifting in the simulator so the model can learn gear changes.' };
    if (![...velocity, yawRate, ...controls, speed, position].every(finite)) {
        return { error: 'Local velocity, speed, yaw rate, steering, throttle, brake and normalized position are required.' };
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
    const speedTargets = referenceSpeedTargets(reference, position!, aggression, config.reward.aggressionMaxRadius);
    const camera = vision.camera;
    if (!camera || camera.width !== o.cameraWidth || camera.height !== o.cameraHeight
        || !(camera.rgb instanceof Uint8Array) || camera.rgb.length !== camera.width * camera.height * 3) {
        return { error: `Track Vision camera pixels must be ${o.cameraWidth} × ${o.cameraHeight} RGB. Restart screen capture in Live Session.` };
    }
    const action = { throttle: controls[0]!, brake: controls[1]!, steering: controls[2]! };
    const groups: Record<string, number[]> = {
        localVelocity: (velocity as number[]).map((v) => clamp(v / o.velocityScale)), speed: [clamp(speed! / o.speedScale, 0, 1)],
        game: category(game, o.categoryBits), car: category(car, o.categoryBits), map: category(map, o.categoryBits),
        position: [Math.sin(2 * Math.PI * position!), Math.cos(2 * Math.PI * position!)], controls: controls as number[],
        referenceOffset: prediction.referenceDelta.map((v) => clamp(v / o.referenceOffsetScale)),
        gear: [clamp((gear - 1) / o.gearScale)], rpm: [clamp(rpm / o.rpmScale, 0, 1)], yawRate: [clamp(yawRate! / o.yawRateScale)],
        camera: [],
        aggression: [normalizedAggression(aggression)], referenceSpeed: [clamp(speedTargets.reference / o.speedScale, 0, 1)],
        targetSpeed: [clamp(speedTargets.target / o.speedScale, 0, 1)],
    };
    return { observation: {
        capturedAt: vision.capturedAt,
        measuredAt: paired.receivedAt,
        session: JSON.stringify([paired.sessionGeneration, paired.streamGeneration, game, car, map]),
        context: { game, car, map },
        featureEncoding: drivingFeatureEncoding(config),
        features: o.features.flatMap((name) => groups[name]),
        camera,
        // Rotation preserves distance, so the same proximity reward applies in local axes.
        normalizedPosition: position!, referenceDistance: Math.hypot(...prediction.referenceDelta),
        referenceError: [prediction.referenceDelta[0], prediction.referenceDelta[1]],
        speed: speed!, gear: gear - 1, rpm, controls: action,
        localVelocity: velocity as [number, number, number], yawRate: yawRate!,
        reference, aggression, speedTargets,
    } };
}

export interface DrivingReward {
    timing: number; line: number; progress: number; instability: number; total: number;
    speedComparison: ReferenceSpeedTargets & { measured: number; aggression: number;
        /** Speed error removed by the driver's speed change, against one fixed target, in km/h. */
        speedGoalProgress: number;
        /** Change in speed over this transition, in m/s². */
        measuredAcceleration: number; referenceAcceleration: number; targetAcceleration: number };
}

/** Reward-only history; ordinary turns and isolated direction changes are free. */
export class DrivingYawInstability {
    constructor(private config = DEFAULT_DRIVING_CONFIG) {}
    private direction = 0;
    private directionAt = -Infinity;
    private reversals: number[] = [];
    private last?: DrivingObservation;

    reset() {
        this.direction = 0;
        this.directionAt = -Infinity;
        this.reversals = [];
        this.last = undefined;
    }

    private observe(observation: DrivingObservation) {
        const { capturedAt, yawRate } = observation;
        this.reversals = this.reversals.filter((time) => capturedAt - time < this.config.reward.yawWindowMs);
        if (capturedAt - this.directionAt > this.config.reward.yawWindowMs) this.direction = 0;
        if (Math.abs(yawRate) <= this.config.reward.yawDeadband) return;
        const direction = Math.sign(yawRate);
        if (this.direction && direction !== this.direction) this.reversals.push(capturedAt);
        this.direction = direction;
        this.directionAt = capturedAt;
    }

    penalty(previous: DrivingObservation, current: DrivingObservation): number {
        if (!this.last || this.last.capturedAt !== previous.capturedAt || this.last.session !== previous.session
            || this.last.reference !== previous.reference || this.last.context.game !== previous.context.game) {
            this.reset();
            this.observe(previous);
        }
        this.observe(current);
        this.last = current;
        const excessReversals = Math.max(0, this.reversals.length - 1);
        return excessReversals ? -this.config.reward.yawPenalty * (current.capturedAt - previous.capturedAt) / 1000 * excessReversals ** 2 : 0;
    }
}

/** Smooth the estimated distance in source units, independently of capture frequency. */
export class DrivingLineProximity {
    constructor(private config = DEFAULT_DRIVING_CONFIG) {}
    private smoothedDistance = 0;
    private last?: DrivingObservation;

    reset() { this.last = undefined; }

    proximity(previous: DrivingObservation, current: DrivingObservation): number {
        if (!this.last || this.last.capturedAt !== previous.capturedAt || this.last.session !== previous.session
            || this.last.reference !== previous.reference || this.last.context.game !== previous.context.game) {
            this.smoothedDistance = previous.referenceDistance;
        }
        const alpha = -Math.expm1(-(current.capturedAt - previous.capturedAt) / this.config.reward.lineSmoothingMs);
        this.smoothedDistance += alpha * (current.referenceDistance - this.smoothedDistance);
        this.last = current;
        return 1 / (1 + (this.smoothedDistance / this.config.reward.lineDistanceTolerance) ** 2);
    }
}

export function drivingReward(previous: DrivingObservation, current: DrivingObservation,
    yawInstability = new DrivingYawInstability(), config = DEFAULT_DRIVING_CONFIG,
    lineProximity = new DrivingLineProximity(config)): DrivingReward {
    const r = config.reward;
    const dt = (current.capturedAt - previous.capturedAt) / 1000;
    if (dt <= 0 || dt > config.runtime.maxTransitionSeconds || previous.session !== current.session) throw new Error('Training transition crossed a session boundary or data gap.');
    const measuredSeconds = (current.measuredAt - previous.measuredAt) / 1000;
    if (!Number.isFinite(measuredSeconds) || measuredSeconds <= 0) throw new Error('Speed timing requires fresh, increasing telemetry timestamps.');
    let positionChange = current.normalizedPosition - previous.normalizedPosition;
    if (positionChange < -0.5) positionChange += 1;
    if (positionChange > 0.5) positionChange -= 1;
    if (Math.abs(positionChange) > config.runtime.maxPositionJump) throw new Error('Lap position jumped. Reposition the car and start a new episode.');
    if (previous.reference !== current.reference) throw new Error('Live Map reference changed. Start a new episode.');
    const proximity = Math.exp(-current.referenceDistance / r.proximityScale);
    const forwardProgress = current.speed > 0 && current.gear >= 0 ? clamp(positionChange, 0, r.progressLimit) : 0;
    const progressPerSecond = Math.abs(positionChange / measuredSeconds);
    // Half strength at 0.01 lap/second; faster progress approaches a quarter of ideal line reward.
    // Weight by signed distance so extra samples do not earn extra reward at the same rate.
    const signedProgress = positionChange < 0 ? clamp(positionChange, -r.progressLimit, 0) : forwardProgress;
    // No movement incurs a time-based penalty: -0.25 per second with default weights.
    const progress = current.speed === 0 || positionChange === 0
        ? -0.25 * r.lineWeight * 0.01 * measuredSeconds
        : 0.25 * r.lineWeight * signedProgress * progressPerSecond / (progressPerSecond + 0.01);
    // Keep both ends of the speed profile on the settings that produced the action.
    const start = referenceSpeedTargets(previous.reference, previous.normalizedPosition, previous.aggression, r.aggressionMaxRadius);
    const speeds = referenceSpeedTargets(previous.reference, current.normalizedPosition, previous.aggression, r.aggressionMaxRadius);
    const measuredAcceleration = (current.speed - previous.speed) / (3.6 * measuredSeconds);
    const referenceAcceleration = (speeds.reference - start.reference) / (3.6 * measuredSeconds);
    const targetAcceleration = (speeds.target - start.target) / (3.6 * measuredSeconds);
    // Compare both measured speeds with the same destination goal. A changing reference
    // or simply being close to it cannot earn credit without a useful speed change.
    // Absolute errors also charge for overshooting, instead of rewarding all acceleration.
    const speedGoalProgress = Math.abs(previous.speed - speeds.target) - Math.abs(current.speed - speeds.target);
    const correction = clamp(speedGoalProgress / (3.6 * measuredSeconds * r.accelerationTolerance));
    const timing = forwardProgress > 0
        ? r.timingWeight * forwardProgress * proximity * correction : 0;
    // Center the smooth gradient on the tolerance: close is positive, far is negative.
    const lineFactor = 2 * lineProximity.proximity(previous, current) - 1;
    const line = forwardProgress > 0 ? r.lineWeight * forwardProgress * lineFactor : 0;
    const instability = yawInstability.penalty(previous, current);
    return { timing, line, progress, instability,
        speedComparison: { ...speeds, measured: current.speed, aggression: previous.aggression, speedGoalProgress,
            measuredAcceleration, referenceAcceleration, targetAcceleration },
        total: clamp(timing + line + progress + instability, -r.totalLimit, r.totalLimit) };
}
