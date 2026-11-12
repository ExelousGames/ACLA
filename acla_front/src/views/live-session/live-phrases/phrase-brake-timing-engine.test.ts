import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { LiveTelemetryFrameEvent } from '../live-telemetry-store';
import type { StandardTelemetrySample } from '../live-session-types';
import { PhraseEngine } from './phrase-engine';
import { getPhraseSpeechInstances } from './phrase-speech';

// A 1000 m square, with constant deceleration from 200 to 300 m and a 10 m/s exit.
function brakingReference(deceleration = 4): TopLapReferenceEntry {
    const entryEnergy = 100 + 200 * deceleration;
    return {
        game: 'acc', track: 'test', car: 'car', avg_grip_int: 2,
        target_features: ['expert_optimal_player_pos_x', 'expert_optimal_player_pos_y', 'expert_optimal_player_pos_z',
            'expert_optimal_brake', 'expert_optimal_throttle', 'expert_optimal_speed'],
        x: Array.from({ length: 100 }, (_, index) => index / 100),
        y: Array.from({ length: 100 }, (_, index) => {
            const distance = index * 10;
            const xy = distance < 250 ? [distance, 0] : distance < 500 ? [250, distance - 250]
                : distance < 750 ? [750 - distance, 250] : [0, 1000 - distance];
            const energy = distance < 200 ? entryEnergy : distance <= 300 ? entryEnergy - 2 * deceleration * (distance - 200)
                : distance < 400 ? 100 + (entryEnergy - 100) * (distance - 300) / 100 : entryEnergy;
            return [...xy, 0, index > 20 && index < 30 ? 0.8 : 0,
                index > 30 && index < 90 ? 0.8 : 0, Math.sqrt(energy) * 3.6];
        }),
    };
}

function frame(distance: number, speed: number, brake = 0): LiveTelemetryFrameEvent {
    const sample: StandardTelemetrySample = { Static_track: 'test', Static_car_model: 'car',
        Graphics_normalized_car_position: distance / 1000, Physics_speed_kmh: speed * 3.6,
        Physics_brake: brake, Physics_gas: 0 };
    return { type: 'frame', sample, sampleIndex: 0, telemetryStatus: 2, sessionGeneration: 0, streamGeneration: 0,
        committedSampleCount: 0,
        update: { type: 'frame', game: 'acc', sample, sequence: 1, committedCount: 0, committedSequence: 0 } };
}

function engineWithReference(reference = brakingReference(), brakeSpeechMs = 0) {
    const engine = new PhraseEngine();
    engine.setSpeechDurations(new Map(getPhraseSpeechInstances().map((instance) => [instance.cacheKey,
        instance.id.startsWith('reference-brake:') ? brakeSpeechMs : 0])));
    engine.receiveMap(null, 0, reference);
    return engine;
}

it.each([20, 30, 40])('uses the arriving speed of %s m/s to extend Brake before its reference zone', (speed) => {
    const snapshot = engineWithReference().receiveTelemetry(frame(120, speed), 1000);
    // At 40 m/s, slowing to 10 m/s at 4 m/s² takes 7.5 s and 187.5 m: start at 112.5 m.
    expect(snapshot.events.map(({ sentence }) => sentence)).toEqual(speed === 40 ? ['Brake.'] : []);
});

it.each([112.49, 112.51])('starts the extended cue at the calculated braking boundary near %s m', (distance) => {
    const snapshot = engineWithReference().receiveTelemetry(frame(distance, 40), 1000);
    expect(snapshot.events.map(({ sentence }) => sentence)).toEqual(distance > 112.5 ? ['Brake.'] : []);
});

it('adds the prepared Brake speech duration once to the calculated onset', () => {
    // 500 ms at 40 m/s moves speech onset 20 m before the 112.5 m braking boundary.
    expect(engineWithReference(brakingReference(), 500).receiveTelemetry(frame(92.49, 40), 1000).events).toEqual([]);
    const snapshot = engineWithReference(brakingReference(), 500).receiveTelemetry(frame(92.51, 40), 1000);
    expect(snapshot.events).toEqual([expect.objectContaining({ sentence: 'Brake.',
        speech: expect.objectContaining({ preparationTimeMs: 500 }) })]);
});

it('keeps the captured reference braking capability after source mutation and aggression changes', () => {
    const reference = brakingReference(), engine = engineWithReference(reference);
    const changed = brakingReference(8);
    reference.y.forEach((row, index) => { row[5] = changed.y[index][5]; });
    engine.setDrivingAggression(100, 100);
    expect(engine.receiveTelemetry(frame(120, 40), 1000).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
});

it('refreshes braking capability when a different reference is loaded', () => {
    const engine = engineWithReference();
    const first = engine.receiveTelemetry(frame(120, 40), 1000).events[0];
    expect(first.sentence).toBe('Brake.');
    engine.speechFinished(first.id, 1100);
    engine.receiveMap(null, 2000, brakingReference(8));
    // With 8 m/s² available, the same speed needs only 93.75 m to the target at 300 m.
    expect(engine.receiveTelemetry(frame(120, 40), 3000).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
});

it('speaks once for the extended approach and rearms after measured reference-zone exit', () => {
    const engine = engineWithReference();
    const first = engine.receiveTelemetry(frame(120, 40), 1000).events[0];
    expect(first.sentence).toBe('Brake.');
    engine.speechFinished(first.id, 1100);
    expect(engine.receiveTelemetry(frame(150, 40), 4000).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    expect(engine.receiveTelemetry(frame(210, 40), 5000).events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    engine.receiveTelemetry(frame(320, 10, 0.8), 6000);
    expect(engine.receiveTelemetry(frame(120, 40), 9000).events.filter(({ ruleId }) => ruleId === 'reference-brake'))
        .toHaveLength(2);
});

it('keeps throttle guidance silent when early braking extends into a reference throttle zone', () => {
    const reference = brakingReference();
    reference.y.forEach((row, index) => { if (index > 5 && index < 20) row[4] = 0.8; });
    const snapshot = engineWithReference(reference).receiveTelemetry(frame(120, 40), 1000);
    expect(snapshot.events.map(({ sentence }) => sentence)).toEqual(['Brake.']);
    const throttle = snapshot.closures.find(({ id }) => id === 'reference-throttle')!;
    expect(throttle.conditions.some(({ conditionFit }) => conditionFit === false)).toBe(true);
});
