import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import type { LiveTelemetrySnapshot } from 'views/live-session/live-telemetry-store';
import { DEFAULT_DRIVING_CONFIG, drivingFeatureEncoding } from './driving-model-config';
import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import { selectLiveMapReference } from 'views/live-session/live-map-reference';
import { DrivingReference, inspectDrivingReference, normalizedAggression, ReferenceSpeedTargets, referenceSpeedTargets } from './driving-reference';
import type { PolicyCameraImage } from 'views/live-session/live-session-recording/policy-camera';
import { detectDrivingImpact, DrivingImpact, ImpactSample, IMPACT_PENALTY_WEIGHT } from './driving-impact';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { detectDrivingOffTrack, OFF_TRACK_PENALTY_PER_SECOND, ON_TRACK_REWARD_PER_SECOND } from './driving-off-track';
import { drivingCenterlineInput } from './driving-centerline';
import type { ControllerAction } from './virtual-controller';
import { ControlChanges, DrivingControlChanges } from './driving-control-changes';

export { PEDAL_CHANGE_REFERENCE_MS } from './driving-control-changes';

export interface DrivingAction {
    /** -1 is full brake, 0 is coast, and +1 is full throttle. */
    pedal: number;
    steering: number;
    shiftUp?: boolean;
    shiftDown?: boolean;
}

export const NEUTRAL: ControllerAction = { throttle: 0, brake: 0, steering: 0 };

export function drivingControllerAction({ pedal, ...controls }: DrivingAction): ControllerAction {
    return { ...controls, throttle: Math.max(0, pedal), brake: Math.max(0, -pedal) };
}

/** Encode human controls with brake priority when both pedals are pressed. */
export function encodeDrivingAction({ throttle, brake, ...controls }: ControllerAction): DrivingAction {
    return { ...controls, pedal: brake > 0 ? -brake : throttle };
}
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
    controls: ControllerAction;
    reference: DrivingReference;
    aggression: number;
    speedTargets: ReferenceSpeedTargets;
    offTrack: boolean | null;
}
export type DrivingTelemetrySample = Pick<DrivingObservation, 'measuredAt' | 'session' | 'context' | 'normalizedPosition'
    | 'speed' | 'localVelocity' | 'gForce' | 'yawRate' | 'gear' | 'rpm' | 'controls'> & { sampleIndex?: number };
export interface DrivingCameraFrame {
    capturedAt: number;
    telemetry?: LiveTelemetrySnapshot;
    camera?: PolicyCameraImage;
}
export type ObservationResult = ({
    observation: DrivingObservation; error?: never;
} | { error: string; observation?: never }) & {
    /** Current telemetry is collected independently of the camera's capture-time pair. */
    telemetry?: DrivingTelemetrySample;
    /** Camera interruptions may leave an otherwise valid demonstration running. */
    cameraUnavailable?: boolean;
    /** Startup may retry a delayed telemetry/camera delivery after loading model weights. */
    waitingForFreshInputs?: boolean;
};
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

// A deterministic encoding preserves categorical inputs across saved policies.
function category(value: string, bits: number): number[] {
    let hash = 2166136261;
    for (let i = 0; i < value.length; i++) hash = Math.imul(hash ^ value.charCodeAt(i), 16777619);
    return Array.from({ length: bits }, (_, i) => (hash >>> i) & 1);
}

export function readDrivingObservation(live: LiveTelemetrySnapshot, frame: DrivingCameraFrame | null,
    referenceLap: TopLapReferenceEntry | null, aggression = DEFAULT_DRIVING_CONFIG.aggression, now = Date.now(), config = DEFAULT_DRIVING_CONFIG,
    circuitMap: CircuitMapDto | null = null): ObservationResult {
    const telemetry = readDrivingTelemetry(live, now, config);
    return { ...readCameraObservation(live, frame, referenceLap, aggression, now, config, circuitMap),
        ...(telemetry && { telemetry }) };
}

export function readDrivingTelemetry(live: LiveTelemetrySnapshot, now = Date.now(), config = DEFAULT_DRIVING_CONFIG): DrivingTelemetrySample | undefined {
    if (live.telemetryStatus !== ACC_STATUS.ACC_LIVE || !finite(live.receivedAt)
        || now - live.receivedAt > config.runtime.telemetryMaxAgeMs || now < live.receivedAt) return;
    const sample = live.currentTelemetry;
    const game = live.game, car = sample.Static_car_model, map = sample.Static_track;
    const localVelocity = [sample.Physics_local_velocity_x, sample.Physics_local_velocity_y, sample.Physics_local_velocity_z];
    const gForce = [sample.Physics_g_force_x, sample.Physics_g_force_y, sample.Physics_g_force_z];
    const yawRate = game === 'iracing' ? sample.Physics_local_angular_vel_y : sample.Physics_local_angular_vel_z;
    const throttle = sample.Physics_gas, brake = sample.Physics_brake, steering = sample.Physics_steer_angle;
    const speed = sample.Physics_speed_kmh, normalizedPosition = sample.Graphics_normalized_car_position;
    if (!game || !car || !map || sample.Physics_autoshifter_on === true
        || ![...localVelocity, ...gForce, yawRate, throttle, brake, steering, speed, normalizedPosition].every(finite)
        || normalizedPosition! < 0 || normalizedPosition! > 1 || speed! < 0
        || throttle! < 0 || throttle! > 1 || brake! < 0 || brake! > 1 || Math.abs(steering!) > 1) return;
    return { measuredAt: live.receivedAt, sampleIndex: live.sampleIndex,
        session: JSON.stringify([live.sessionGeneration, live.streamGeneration, game, car, map]), context: { game, car, map },
        normalizedPosition: normalizedPosition!, speed: speed!, localVelocity: localVelocity as [number, number, number],
        gForce: gForce as [number, number, number], yawRate: yawRate!,
        gear: finite(sample.Physics_gear) ? sample.Physics_gear - 1 : 0, rpm: finite(sample.Physics_rpm) ? sample.Physics_rpm : 0,
        controls: { throttle: throttle!, brake: brake!, steering: steering! } };
}

function readCameraObservation(live: LiveTelemetrySnapshot, frame: DrivingCameraFrame | null,
    referenceLap: TopLapReferenceEntry | null, aggression: number, now: number, config: typeof DEFAULT_DRIVING_CONFIG,
    circuitMap: CircuitMapDto | null): ObservationResult {
    const o = config.observation;
    if (live.telemetryStatus !== ACC_STATUS.ACC_LIVE) return { error: 'Waiting for fresh, running live telemetry.' };
    if (!finite(live.receivedAt) || now - live.receivedAt > config.runtime.telemetryMaxAgeMs || now < live.receivedAt) {
        return { error: 'Waiting for fresh, running live telemetry.', waitingForFreshInputs: true };
    }
    if (!frame || !finite(frame.capturedAt) || now - frame.capturedAt > config.runtime.visionMaxAgeMs || now < frame.capturedAt) {
        return { error: 'Share the game screen from the Live Session recording bar. A fresh image is required.', cameraUnavailable: true, waitingForFreshInputs: true };
    }
    const paired = frame.telemetry;
    if (!paired || paired.telemetryStatus !== ACC_STATUS.ACC_LIVE || !finite(paired.receivedAt)
        || frame.capturedAt - paired.receivedAt > config.runtime.telemetryMaxAgeMs || frame.capturedAt < paired.receivedAt
        || paired.sessionGeneration !== live.sessionGeneration || paired.streamGeneration !== live.streamGeneration
        || paired.game !== live.game || paired.currentTelemetry.Static_track !== live.currentTelemetry.Static_track
        || paired.currentTelemetry.Static_car_model !== live.currentTelemetry.Static_car_model) {
        return { error: 'Waiting for screen capture and telemetry from the same live session.', cameraUnavailable: true, waitingForFreshInputs: true };
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
    const camera = frame.camera;
    if (!camera || camera.width !== o.cameraWidth || camera.height !== o.cameraHeight
        || !(camera.rgb instanceof Uint8Array) || camera.rgb.length !== camera.width * camera.height * 3) {
        return { error: `Screen capture camera pixels must be ${o.cameraWidth} × ${o.cameraHeight} RGB.`, cameraUnavailable: true };
    }
    const centerline = drivingCenterlineInput(circuitMap, game, map, position!, config);
    if (!centerline) return { error: 'Live Map needs a valid centerline for the current game and circuit. Load a matching circuit map in Live Session.' };
    const action = { throttle: controls[0]!, brake: controls[1]!, steering: controls[2]! };
    const groups: Record<string, number[]> = {
        localVelocity: (velocity as number[]).map((v) => clamp(v / o.velocityScale)), speed: [clamp(speed! / o.speedScale, 0, 1)],
        game: category(game, o.categoryBits), car: category(car, o.categoryBits), map: category(map, o.categoryBits),
        position: [Math.sin(2 * Math.PI * position!), Math.cos(2 * Math.PI * position!)], controls: controls as number[],
        gear: [clamp((gear - 1) / o.gearScale)], rpm: [clamp(rpm / o.rpmScale, 0, 1)], yawRate: [clamp(yawRate! / o.yawRateScale)],
        gForce: (gForce as number[]).map((v) => clamp(v / o.gForceScale)),
        camera: [],
        centerline,
        aggression: [normalizedAggression(aggression)], referenceSpeed: [clamp(speedTargets.reference / o.speedScale, 0, 1)],
        targetSpeed: [clamp(speedTargets.target / o.speedScale, 0, 1)],
    };
    return { observation: {
        capturedAt: frame.capturedAt,
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
    timing: number; progress: number; impact: number; blockedMotion: number; onTrack: number; offTrack: number;
    steeringChange: number; throttleChange: number; brakeChange: number; total: number;
    offTrackDetected: boolean | null;
    impactEstimate: DrivingImpact;
    speedComparison: ReferenceSpeedTargets & { aggression: number;
        /** Signed speed in km/h, with reverse motion scored as negative. */
        measured: number;
        /** Integrated speed gap used for scoring, in (km/h)·s. */
        speedGapTime: number;
        /** Change in signed speed over this transition, in m/s². */
        measuredAcceleration: number; referenceAcceleration: number; targetAcceleration: number };
}

function longitudinalVelocity({ context, localVelocity }: DrivingObservation): number {
    return localVelocity[context.game === 'iracing' ? 2 : 1];
}

function signedSpeed(observation: DrivingObservation): number {
    const { speed, gear } = observation;
    return speed > 0 && (gear < 0 || longitudinalVelocity(observation) < 0) ? -speed : speed;
}

export function drivingReward(previous: DrivingObservation, current: DrivingObservation,
    config = DEFAULT_DRIVING_CONFIG, telemetryImpact?: DrivingImpact,
    controlChanges?: ControlChanges, blockedMotion = 0): DrivingReward {
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
    const previousSpeed = signedSpeed(previous), currentSpeed = signedSpeed(current);
    const movingForward = currentSpeed > 0 && positionChange > 0;
    const forwardProgress = movingForward ? Math.min(positionChange, r.progressLimit) : 0;
    const progressPerSecond = Math.abs(positionChange / measuredSeconds);
    // Half strength at 0.01 lap/second; faster progress approaches 0.25 * progressWeight per lap.
    // Weight by signed distance so extra samples do not earn extra reward at the same rate.
    const signedProgress = positionChange < 0 ? clamp(positionChange, -r.progressLimit, 0) : forwardProgress;
    const distanceReward = 0.25 * r.progressWeight * signedProgress * progressPerSecond / (progressPerSecond + 0.01);
    const stationaryPenalty = -0.25 * r.progressWeight * 0.01 * measuredSeconds;
    // Reversing incurs at least the stationary penalty, even when lap position increases.
    const progress = current.speed < 1 ? stationaryPenalty
        : currentSpeed < 0 ? Math.min(distanceReward, stationaryPenalty) : distanceReward;
    // Keep both ends of the speed profile on the settings that produced the action.
    const start = referenceSpeedTargets(previous.reference, previous.normalizedPosition, previous.aggression);
    const speeds = referenceSpeedTargets(previous.reference, current.normalizedPosition, previous.aggression);
    const measuredAcceleration = (currentSpeed - previousSpeed) / (3.6 * measuredSeconds);
    const referenceAcceleration = (speeds.reference - start.reference) / (3.6 * measuredSeconds);
    const targetAcceleration = (speeds.target - start.target) / (3.6 * measuredSeconds);
    const startError = previousSpeed - start.target, endError = currentSpeed - speeds.target;
    const startGap = Math.abs(startError), endGap = Math.abs(endError);
    // Integrate the linearly interpolated error. A target crossing splits the area into two triangles.
    const averageGap = startError * endError < 0
        ? (startGap ** 2 + endGap ** 2) / (2 * (startGap + endGap))
        : (startGap + endGap) / 2;
    // A speed match only earns relief below the standstill cost with forward lap progress.
    const minimumGap = positionChange > 0 ? 0 : (start.target + speeds.target) / 2;
    const speedGapTime = Math.max(averageGap, minimumGap) * measuredSeconds;
    const timing = speedGapTime > 0 && r.timingWeight > 0 ? -r.timingWeight * speedGapTime / r.speedTolerance : 0;
    const impactSample = (observation: DrivingObservation): ImpactSample => ({ measuredAt: observation.measuredAt,
        velocity: [observation.localVelocity[0], longitudinalVelocity(observation)],
        steering: observation.controls.steering, brake: observation.controls.brake });
    const impactEstimate = telemetryImpact ?? detectDrivingImpact(impactSample(previous), impactSample(current));
    const impact = impactEstimate.detected ? -IMPACT_PENALTY_WEIGHT * impactEstimate.confidence * impactEstimate.severity : 0;
    const offTrack = current.offTrack === true ? -OFF_TRACK_PENALTY_PER_SECOND * measuredSeconds : 0;
    // Reward confirmed on-track driving without paying the policy to park or reverse.
    const onTrack = current.offTrack === false && forwardProgress > 0 ? ON_TRACK_REWARD_PER_SECOND * measuredSeconds : 0;
    const changes = controlChanges ?? new DrivingControlChanges().record(previous.controls, current.controls, measuredSeconds * 1000);
    const penalty = (weight: number, change: number) => weight > 0 && change > 0 ? -weight * change : 0;
    const steeringChange = penalty(r.steeringChangeWeight, changes.steering);
    const throttleChange = penalty(r.throttleChangeWeight, changes.throttle);
    const brakeChange = penalty(r.brakeChangeWeight, changes.brake);
    return { timing, progress, impact, impactEstimate, blockedMotion, onTrack, offTrack, steeringChange, throttleChange, brakeChange,
        offTrackDetected: current.offTrack,
        speedComparison: { ...speeds, measured: currentSpeed, aggression: previous.aggression, speedGapTime,
            measuredAcceleration, referenceAcceleration, targetAcceleration },
        total: clamp(timing + progress + impact + blockedMotion + onTrack + offTrack + steeringChange + throttleChange + brakeChange,
            -r.totalLimit, r.totalLimit) };
}
