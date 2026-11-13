import type { TrackVisionDetection } from 'views/live-session/track-vision/track-vision-types';
import { reconstructScene } from 'views/live-session/track-vision/reconstructed-scene';
import { POLICY_CAMERA_HEIGHT, POLICY_CAMERA_INPUTS, POLICY_CAMERA_WIDTH } from 'views/live-session/track-vision/policy-camera';
import { createLiveTelemetryStore } from 'views/live-session/live-telemetry-store';
import { readDrivingObservation } from './driving-observation';
import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';

export function referenceLapFixture(): TopLapReferenceEntry {
    // A 400-coordinate-unit square, with a slow corner between deceleration and acceleration.
    const x = [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1];
    return { game: 'acc', car: 'GT3', track: 'monza', avg_grip_int: 2, x,
        target_features: ['expert_optimal_player_pos_x', 'expert_optimal_player_pos_y', 'expert_optimal_player_pos_z',
            'expert_optimal_throttle', 'expert_optimal_brake', 'expert_optimal_speed'],
        y: x.map((p) => {
            const brake = p <= 0.2 || p >= 0.6 ? 0 : p < 0.3 ? (p - 0.2) / 0.1 : p <= 0.5 ? 1 : (0.6 - p) / 0.1;
            return [p <= 0.25 ? p * 400 : p <= 0.5 ? 100 : p <= 0.75 ? (0.75 - p) * 400 : 0,
                p <= 0.25 ? 0 : p <= 0.5 ? (p - 0.25) * 400 : p <= 0.75 ? 100 : (1 - p) * 400,
                0, 1 - brake, brake, 120 - 40 * brake];
        }) };
}

export function trainingFixture(now = 1000, aggression = 50, referenceLap = referenceLapFixture()) {
    const store = createLiveTelemetryStore();
    store.publishFrame({ type: 'frame', game: 'acc', sequence: 1, committedSequence: 1, committedCount: 1, sample: {
        Graphics_status: 2, Static_car_model: 'GT3', Static_track: 'monza',
        Physics_local_velocity_x: 1, Physics_local_velocity_y: 30, Physics_local_velocity_z: 0, Physics_speed_kmh: 108,
        Physics_g_force_x: 0.5, Physics_g_force_y: -1, Physics_g_force_z: 1,
        Physics_heading: 0, Physics_pitch: 0, Physics_roll: 0, Physics_steer_angle: 0,
        Physics_local_angular_vel_x: 0.1, Physics_local_angular_vel_y: 0.3, Physics_local_angular_vel_z: -0.2,
        Physics_gas: 0.3, Physics_brake: 0, Graphics_normalized_car_position: 0.25,
        Physics_gear: 4, Physics_rpm: 6000,
    } });
    const live = { ...store.getSnapshot(), receivedAt: now, currentTelemetry: { ...store.getSnapshot().currentTelemetry } };
    const vision: TrackVisionDetection = { capturedAt: now, width: 4, height: 4, telemetry: { ...live },
        camera: { width: POLICY_CAMERA_WIDTH, height: POLICY_CAMERA_HEIGHT, rgb: new Uint8Array(POLICY_CAMERA_INPUTS).fill(128) },
        position: { coordinateFrame: 'car_local', referenceDelta: [2, 0, 0], normalizedPosition: 0.25, references: { reference: { x: 3, y: 0, z: 0 } }, inferenceMs: 1, referenceLap },
        detections: { segment: { task: 'segment', width: 4, height: 4, classNames: ['road', 'car'], inferenceMs: 1,
            instances: [{ classId: 0, confidence: 1, box: [0, 0, 1, 1], mask: Uint8Array.from({ length: 16 }, (_, i) => Number(i % 4 === 1 || i % 4 === 2)) }] },
            depth: { task: 'depth', width: 8, height: 8, scale: 'relative', classNames: [], inferenceMs: 1,
                values: Float32Array.from({ length: 64 }, (_, i) => (i + 1) / 64) } },
        reconstruction: null, geometry: null };
    vision.reconstructedScene = reconstructScene(vision);
    return { live, vision, referenceLap, observation: readDrivingObservation(live, vision, referenceLap, aggression, now).observation! };
}

/** Known braking work: 50 m at 6 m/s², then 50 m at 1 m/s²; turn-in is at 240 m. */
export function brakingReferenceFixture() {
    const x = Array.from({ length: 100 }, (_, i) => i / 100);
    const lap: TopLapReferenceEntry = { game: 'acc', track: 'monza', car: 'GT3', avg_grip_int: 2, x,
        target_features: ['expert_optimal_player_pos_x', 'expert_optimal_player_pos_y', 'expert_optimal_player_pos_z',
            'expert_optimal_brake', 'expert_optimal_throttle', 'expert_optimal_speed'],
        y: x.map((_, i) => {
            const d = i * 10;
            const xy = d < 240 ? [d, 0] : d < 500 ? [240, d - 240] : d < 740 ? [740 - d, 260] : [0, 1000 - d];
            const speed = i < 20 || i >= 70 ? 144 : i <= 25 ? Math.sqrt(1600 - 120 * (i - 20)) * 3.6
                : i < 30 ? Math.sqrt(1000 - 20 * (i - 25)) * 3.6 : i <= 60 ? 108 : 108 + (i - 60) * 3.6;
            return [...xy, 0, i >= 20 && i < 25 ? 1 : i >= 25 && i < 30 ? 1 / 6 : 0, i >= 60 ? 1 : 0, speed];
        }) };
    const map: CircuitMapDto = {
        id: 'braking', game: 'acc', source_track_key: 'monza', circuit_name: 'Monza', resolution: 1000, samples: {},
        centerline_tags: [{ id: 'turn', label: 'corner', start_position: 0.24, end_position: 0.5 }],
    };
    return { lap, map };
}

/** Analytic speed from the unchanged 30 m/s minimum and the two braking rates above. */
export function brakingFixtureSpeed(position: number, aggression: number): number {
    const caution = Math.max(0, 1 - aggression / 50);
    const start = 200 - 25 * caution / 3, transition = 250 - 10 * caution;
    const distance = position * 1000;
    const work = 6 * Math.max(0, transition - Math.max(start, distance))
        + Math.max(0, 300 - Math.max(transition, distance));
    const speed = position <= 0.6 ? Math.sqrt(900 + 2 * work) * 3.6
        : position < 0.7 ? 108 + (position - 0.6) * 360 : 144;
    return speed * (1 + Math.max(0, aggression - 50) / 100);
}
