import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import type { LiveTelemetrySnapshot } from 'views/live-session/live-telemetry-store';
import type { TrackVisionDetection } from 'views/live-session/track-vision/track-vision-types';
import { DEFAULT_DRIVING_CONFIG, drivingFeatureEncoding } from './driving-model-config';
import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import { selectLiveMapReference } from 'views/live-session/live-map-reference';
import { DrivingReference, inspectDrivingReference, normalizedAggression, ReferenceSpeedTargets, referenceSpeedTargets } from './driving-reference';
import type { PolicyCameraImage } from 'views/live-session/track-vision/policy-camera';
import { detectDrivingImpact, DrivingImpact, ImpactSample, IMPACT_PENALTY_WEIGHT } from './driving-impact';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { detectDrivingOffTrack, OFF_TRACK_PENALTY_PER_SECOND } from './driving-off-track';

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
    speed: number;
    /** Adapter-local X/Y/Z velocity in m/s: ACC right/forward/up; iRacing right/up/forward. */
    localVelocity: [number, number, number];
    /** Adapter XYZ acceleration in g, preserving source signs and gravity contribution. */
    gForce: [number, number, number];
    /** Signed yaw rate in rad/s: ACC adapter-local Z; iRacing adapter-local Y. */
    yawRate: number;
    /** Forward gears start at 1; neutral is 0 and reverse is -1. */
    gear: number;
    rpm: number;
    controls: DrivingAction;
    reference: DrivingReference;
    aggression: number;
    speedTargets: ReferenceSpeedTargets;
    offTrack: boolean | null;
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
    referenceLap: TopLapReferenceEntry | null, aggression = DEFAULT_DRIVING_CONFIG.aggression, now = Date.now(), config = DEFAULT_DRIVING_CONFIG,
    circuitMap: CircuitMapDto | null = null): ObservationResult {
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
    const gForce = [sample.Physics_g_force_x, sample.Physics_g_force_y, sample.Physics_g_force_z];
    const yawRate = game === 'iracing' ? sample.Physics_local_angular_vel_y : sample.Physics_local_angular_vel_z;
    const controls = [sample.Physics_gas, sample.Physics_brake, sample.Physics_steer_angle];
    const speed = sample.Physics_speed_kmh, position = sample.Graphics_normalized_car_position;
    // Both readers use ACC indexing: reverse = 0, neutral = 1, first = 2.
    // Gear/RPM must not block stalled-engine observations; keep missing readings finite for the policy.
    const gear = finite(sample.Physics_gear) ? sample.Physics_gear : 1;
    const rpm = finite(sample.Physics_rpm) ? sample.Physics_rpm : 0;
    if (sample.Physics_autoshifter_on === true) return { error: 'Disable automatic gear shifting in the simulator so the model can learn gear changes.' };
    if (![...velocity, ...gForce, yawRate, ...controls, speed, position].every(finite)) {
        return { error: 'Local velocity, g-force XYZ, speed, yaw rate, steering, throttle, brake and normalized position are required.' };
    }
    if (position! < 0 || position! > 1 || speed! < 0 || controls[0]! < 0 || controls[0]! > 1
        || controls[1]! < 0 || controls[1]! > 1 || Math.abs(controls[2]!) > 1) return { error: 'Telemetry inputs are outside their valid ranges.' };
    if (referenceLap && !selectLiveMapReference([referenceLap], game, map, car, sample.Graphics_track_grip_status)) {
        return { error: 'Live Map reference lap does not match the current game, car or circuit. Load a matching reference lap in Live Session.' };
    }
    const prepared = inspectDrivingReference(referenceLap, circuitMap);
    if (!prepared.reference) return { error: prepared.error };
    const reference = prepared.reference;
    if (!finite(aggression) || aggression < 0 || aggression > 100) return { error: 'Driving aggression must be between 0 and 100.' };
    const speedTargets = referenceSpeedTargets(reference, position!, aggression);
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
        gear: [clamp((gear - 1) / o.gearScale)], rpm: [clamp(rpm / o.rpmScale, 0, 1)], yawRate: [clamp(yawRate! / o.yawRateScale)],
        gForce: (gForce as number[]).map((v) => clamp(v / o.gForceScale)),
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
        normalizedPosition: position!,
        speed: speed!, gear: gear - 1, rpm, controls: action,
        localVelocity: velocity as [number, number, number], yawRate: yawRate!,
        gForce: gForce as [number, number, number],
        reference, aggression, speedTargets,
        offTrack: detectDrivingOffTrack(game, sample, circuitMap),
    } };
}

export interface DrivingReward {
    timing: number; progress: number; impact: number; offTrack: number; total: number;
    offTrackDetected: boolean | null;
    impactEstimate: DrivingImpact;
    speedComparison: ReferenceSpeedTargets & { measured: number; aggression: number;
        /** Speed error removed by the driver's speed change, against one fixed target, in km/h. */
        speedGoalProgress: number;
        /** Change in speed over this transition, in m/s². */
        measuredAcceleration: number; referenceAcceleration: number; targetAcceleration: number };
}

export function drivingReward(previous: DrivingObservation, current: DrivingObservation,
    config = DEFAULT_DRIVING_CONFIG, telemetryImpact?: DrivingImpact): DrivingReward {
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
    const forwardProgress = current.speed > 0 && current.gear >= 0 ? clamp(positionChange, 0, r.progressLimit) : 0;
    const progressPerSecond = Math.abs(positionChange / measuredSeconds);
    // Half strength at 0.01 lap/second; faster progress approaches 0.25 * progressWeight per lap.
    // Weight by signed distance so extra samples do not earn extra reward at the same rate.
    const signedProgress = positionChange < 0 ? clamp(positionChange, -r.progressLimit, 0) : forwardProgress;
    // No movement incurs a time-based penalty: -0.25 per second with default weights.
    const progress = current.speed === 0 || positionChange === 0
        ? -0.25 * r.progressWeight * 0.01 * measuredSeconds
        : 0.25 * r.progressWeight * signedProgress * progressPerSecond / (progressPerSecond + 0.01);
    // Keep both ends of the speed profile on the settings that produced the action.
    const start = referenceSpeedTargets(previous.reference, previous.normalizedPosition, previous.aggression);
    const speeds = referenceSpeedTargets(previous.reference, current.normalizedPosition, previous.aggression);
    const measuredAcceleration = (current.speed - previous.speed) / (3.6 * measuredSeconds);
    const referenceAcceleration = (speeds.reference - start.reference) / (3.6 * measuredSeconds);
    const targetAcceleration = (speeds.target - start.target) / (3.6 * measuredSeconds);
    // Compare both measured speeds with the same destination goal. A changing reference
    // or simply being close to it cannot earn credit without a useful speed change.
    // Absolute errors also charge for overshooting, instead of rewarding all acceleration.
    const speedGoalProgress = Math.abs(previous.speed - speeds.target) - Math.abs(current.speed - speeds.target);
    const correction = clamp(speedGoalProgress / (3.6 * measuredSeconds * r.accelerationTolerance));
    // A held but incorrect speed must still cost reward, even when no correction occurs.
    const speedPenalty = clamp(Math.abs(current.speed - speeds.target) / r.speedTolerance, 0, 1);
    const timing = forwardProgress > 0
        ? r.timingWeight * forwardProgress * (correction - speedPenalty) : 0;
    const impactSample = (observation: DrivingObservation): ImpactSample => ({ measuredAt: observation.measuredAt,
        velocity: [observation.localVelocity[0], observation.localVelocity[observation.context.game === 'iracing' ? 2 : 1]],
        steering: observation.controls.steering, brake: observation.controls.brake });
    const impactEstimate = telemetryImpact ?? detectDrivingImpact(impactSample(previous), impactSample(current));
    const impact = impactEstimate.detected ? -IMPACT_PENALTY_WEIGHT * impactEstimate.confidence * impactEstimate.severity : 0;
    const offTrack = current.offTrack === true ? -OFF_TRACK_PENALTY_PER_SECOND * measuredSeconds : 0;
    return { timing, progress, impact, impactEstimate, offTrack, offTrackDetected: current.offTrack,
        speedComparison: { ...speeds, measured: current.speed, aggression: previous.aggression, speedGoalProgress,
            measuredAcceleration, referenceAcceleration, targetAcceleration },
        total: clamp(timing + progress + impact + offTrack, -r.totalLimit, r.totalLimit) };
}
