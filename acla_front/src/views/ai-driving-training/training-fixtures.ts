import type { TrackVisionDetection } from 'views/live-session/track-vision/track-vision-types';
import { createLiveTelemetryStore } from 'views/live-session/live-telemetry-store';
import { readDrivingObservation } from './driving-observation';

export function trainingFixture(now = 1000) {
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
        position: { referenceDelta: [2, 0, 0], normalizedPosition: 0.25, references: { reference: { x: 3, y: 0, z: 0 } }, inferenceMs: 1 },
        detections: { segment: { task: 'segment', width: 4, height: 4, classNames: ['road', 'car'], inferenceMs: 1,
            instances: [{ classId: 0, confidence: 1, box: [0, 0, 1, 1], mask: new Uint8Array(16).fill(1) }] } },
        reconstruction: null, geometry: null };
    return { live, vision, observation: readDrivingObservation(live, vision, now).observation! };
}
