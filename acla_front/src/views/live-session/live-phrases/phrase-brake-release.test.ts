import type { LiveTelemetryFrameEvent } from '../live-telemetry-store';
import { brakeReleaseSettings, SLOWLY_RELEASE_BRAKE } from './phrase-brake-release';
import { PHRASE_DEFINITIONS, PhraseEngine } from './phrase-engine';
import { getPhraseSpeechInstances } from './phrase-speech';
import { steeringCornerFixture } from './test-fixtures';

it.each([[500, 0], [710, 1 / 3], [790, 2 / 3], [1500, 1], [750, 1 / 3]])(
    'matches a %s ms easing window to the nearest measured clip', (duration, timing) => {
        expect(brakeReleaseSettings(duration, (settings) => 600 + settings.inputs!.timing! * 300))
            .toEqual({ inputs: { timing } });
    });

it('selects only prepared, audible durations and falls back to neutral before preparation', () => {
    expect(brakeReleaseSettings(800, (settings) => settings.inputs!.timing === 1 ? 900 : undefined))
        .toEqual({ inputs: { timing: 1 } });
    for (const duration of [undefined, 0, NaN, Infinity]) {
        expect(brakeReleaseSettings(800, () => duration)).toEqual({ inputs: { timing: 0 } });
        expect(brakeReleaseSettings(duration, () => 900)).toEqual({ inputs: { timing: 0 } });
    }
});

function releaseEvent(endIndex: number, drop = 0.2) {
    const { map, reference } = steeringCornerFixture();
    map.centerline_segments![0].start_position = 0.25;
    let energy = 40 ** 2;
    reference.y.forEach((row, index) => {
        if (index >= 25 && index <= endIndex) energy -= 20 * (index <= 26 ? 4 : 4 - drop);
        row[6] = Math.sqrt(energy) * 3.6;
        row[4] = 0.8;
    });
    const phrases = PHRASE_DEFINITIONS.filter(({ id }) => id === 'reference-steering-release');
    const engine = new PhraseEngine(phrases);
    const instances = getPhraseSpeechInstances(phrases);
    const durations = new Map(instances.map((instance) => [instance.cacheKey, 600 + instance.inputs.timing * 300]));
    engine.setSpeechDurations(durations);
    engine.receiveMap(map, 0, reference);
    // The 900 ms prepared-audio horizon places the decision at 261 m, inside the easing interval.
    const sample = { Static_track: 'test', Static_car_model: 'car', Graphics_normalized_car_position: 0.252,
        Physics_speed_kmh: 36, Physics_brake: 0.8, Physics_gas: 0 };
    const frame: LiveTelemetryFrameEvent = { type: 'frame', sample, sampleIndex: 0, telemetryStatus: 2,
        sessionGeneration: 0, streamGeneration: 0, committedSampleCount: 0,
        update: { type: 'frame', game: 'acc', sample, sequence: 1, committedCount: 0, committedSequence: 0 } };
    const snapshot = engine.receiveTelemetry(frame, 1000);
    expect(snapshot.events).toHaveLength(1);
    const event = snapshot.events[0];
    expect(event.sentence).toBe(SLOWLY_RELEASE_BRAKE);
    expect(durations.has(event.instance!.cacheKey)).toBe(true);
    expect(event.speech!.durationMs).toBe(durations.get(event.instance!.cacheKey));
    return { event, snapshot };
}

it('stretches longer remaining easing even when the initial deceleration easing rate is identical', () => {
    const short = releaseEvent(28), long = releaseEvent(32);
    expect(short.snapshot.referenceBrakeReleaseRate).toBeCloseTo(long.snapshot.referenceBrakeReleaseRate!, 8);
    expect(short.snapshot.referenceBrakeReleaseDurationMs).toBeLessThan(long.snapshot.referenceBrakeReleaseDurationMs!);
    expect(short.event.instance!.inputs.timing).toBe(0);
    expect(long.event.instance!.inputs.timing).toBe(1);
});

it('keeps the same duration-matched delivery for different instantaneous easing rates', () => {
    const gentle = releaseEvent(29, 0.1), steep = releaseEvent(29, 0.8);
    expect(gentle.snapshot.referenceBrakeReleaseRate).toBeLessThan(steep.snapshot.referenceBrakeReleaseRate!);
    expect(gentle.event.instance!.inputs.timing).toBe(2 / 3);
    expect(steep.event.instance!.inputs.timing).toBe(2 / 3);
});
