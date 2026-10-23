import type { TrackVisionDetection } from 'views/live-session/track-vision/track-vision-types';
import { createLiveTelemetryStore } from 'views/live-session/live-telemetry-store';
import { readDrivingObservation } from './driving-observation';
import type { TopLapReferenceEntry } from 'services/topLapReferenceService';

export function referenceLapFixture(): TopLapReferenceEntry {
    // A 400-coordinate-unit square, with a braking plateau and complementary throttle.
    const x = [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1];
    return { game: 'acc', car: 'GT3', track: 'monza', avg_grip_int: 2, x,
        target_features: ['expert_optimal_player_pos_x', 'expert_optimal_player_pos_y', 'expert_optimal_player_pos_z',
            'expert_optimal_throttle', 'expert_optimal_brake'],
        y: x.map((p) => {
            const brake = p <= 0.2 || p >= 0.6 ? 0 : p < 0.3 ? (p - 0.2) / 0.1 : p <= 0.5 ? 1 : (0.6 - p) / 0.1;
            return [p <= 0.25 ? p * 400 : p <= 0.5 ? 100 : p <= 0.75 ? (0.75 - p) * 400 : 0,
                p <= 0.25 ? 0 : p <= 0.5 ? (p - 0.25) * 400 : p <= 0.75 ? 100 : (1 - p) * 400,
                0, 1 - brake, brake];
        }) };
}

export function trainingFixture(now = 1000, aggression = 50, referenceLap = referenceLapFixture()) {
    const store = createLiveTelemetryStore();
    store.publishFrame({ type: 'frame', game: 'acc', sequence: 1, committedSequence: 1, committedCount: 1, sample: {
        Graphics_status: 2, Static_car_model: 'GT3', Static_track: 'monza',
        Physics_velocity_x: 1, Physics_velocity_y: 0, Physics_velocity_z: 30, Physics_speed_kmh: 108,
        Physics_heading: 0, Physics_pitch: 0, Physics_roll: 0, Physics_steer_angle: 0,
        Physics_gas: 0.3, Physics_brake: 0, Graphics_normalized_car_position: 0.25,
        Physics_gear: 4, Physics_rpm: 6000,
    } });
    const live = { ...store.getSnapshot(), receivedAt: now, currentTelemetry: { ...store.getSnapshot().currentTelemetry } };
    const vision: TrackVisionDetection = { capturedAt: now, width: 4, height: 4, telemetry: { ...live },
        position: { coordinateFrame: 'car_local', referenceDelta: [2, 0, 0], normalizedPosition: 0.25, references: { reference: { x: 3, y: 0, z: 0 } }, inferenceMs: 1, referenceLap },
        detections: { segment: { task: 'segment', width: 4, height: 4, classNames: ['road', 'car'], inferenceMs: 1,
            instances: [{ classId: 0, confidence: 1, box: [0, 0, 1, 1], mask: new Uint8Array(16).fill(1) }] } },
        reconstruction: null, geometry: null };
    return { live, vision, referenceLap, observation: readDrivingObservation(live, vision, referenceLap, aggression, now).observation! };
}
