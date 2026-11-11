import type { StandardTelemetrySample } from '../live-session-types';
import type { LiveTelemetryFrameEvent } from '../live-telemetry-store';
import { PHRASE_DEFINITIONS, PhraseEngine, type PhraseSnapshot } from './phrase-engine';
import { readReferenceGuidance } from './phrase-reference-guidance';
import { steeringCornerFixture } from './test-fixtures';

const frame = (sample: StandardTelemetrySample): LiveTelemetryFrameEvent => ({
    type: 'frame', sample, sampleIndex: 0, telemetryStatus: 2, sessionGeneration: 0, streamGeneration: 0,
    committedSampleCount: 0, update: { type: 'frame', game: 'acc', sample, sequence: 1, committedCount: 0, committedSequence: 0 },
});
const spoken = (snapshot: PhraseSnapshot) => snapshot.events.filter(({ sentence }) => sentence === 'More brake.');

function setup() {
    const { reference } = steeringCornerFixture();
    reference.y.forEach((row, i) => {
        row[4] = i >= 5 && i < 23 ? 0.8 : 0;
        // Constant 3 m/s² reference braking through the tested straight.
        row[6] = Math.sqrt(Math.max(100, 900 - 6 * (i * 10 - 100))) * 3.6;
    });
    const engine = new PhraseEngine();
    engine.setDrivingAggression(0, 0);
    engine.receiveMap(null, 0, reference);
    const sample = (elapsed: number, patch: StandardTelemetrySample = {}): StandardTelemetrySample => ({
        Static_track: 'test', Static_car_model: 'car', Graphics_normalized_car_position: 0.1 + elapsed * 0.00003,
        Physics_speed_kmh: 108 - 3.6 * elapsed / 1000, Physics_brake: 0.8, Physics_gas: 0, ...patch,
    });
    const at = (elapsed: number, patch: StandardTelemetrySample = {}) => engine.receiveTelemetry(frame(sample(elapsed, patch)), 1000 + elapsed);
    return { engine, reference, sample, at };
}

it('requires 500 continuous ms after the first measured deficit, with no timer-only speech', () => {
    const { engine, at } = setup();
    expect(PHRASE_DEFINITIONS.find(({ referenceCue }) => referenceCue === 'moreBrake')!.holdMs).toBe(500);
    for (const elapsed of [0, 100, 200, 400, 599]) expect(spoken(at(elapsed))).toEqual([]);
    expect(spoken(engine.evaluate(1600))).toEqual([]);
    expect(spoken(at(600))).toEqual([expect.objectContaining({ timestamp: 1600 })]);
    expect(spoken(at(700))).toHaveLength(1);
});

it('resets confirmation as soon as current deceleration recovers, even between fallback measurements', () => {
    const { at } = setup();
    [0, 100, 200].forEach((elapsed) => at(elapsed));
    // A 100 ms burst of 5 m/s² braking interrupts the deficit.
    expect(spoken(at(300, { Physics_speed_kmh: 105.48 }))).toEqual([]);
    for (const elapsed of [400, 600, 899]) {
        expect(spoken(at(elapsed, { Physics_speed_kmh: 105.48 - 3.6 * (elapsed - 300) / 1000 }))).toEqual([]);
    }
    expect(spoken(at(900, { Physics_speed_kmh: 103.32 }))).toEqual([expect.objectContaining({ timestamp: 1900 })]);
});

it.each(['pause', 'gap', 'missing speed', 'invalid brake', 'session', 'stream', 'repeated time', 'backward time'])(
    'starts a fresh confirmation after %s', (kind) => {
        const { engine, sample, at } = setup();
        [0, 100, 300].forEach((elapsed) => at(elapsed));
        let now = 1400;
        const interrupted = frame(sample(400));
        if (kind === 'pause') interrupted.telemetryStatus = 1;
        if (kind === 'gap') now = 2901;
        if (kind === 'missing speed') interrupted.sample = { ...interrupted.sample, Physics_speed_kmh: undefined };
        if (kind === 'invalid brake') interrupted.sample = { ...interrupted.sample, Physics_brake: NaN };
        if (kind === 'session') interrupted.sessionGeneration++;
        if (kind === 'stream') interrupted.streamGeneration++;
        if (kind === 'repeated time') now = 1300;
        if (kind === 'backward time') now = 1299;
        expect(spoken(engine.receiveTelemetry(interrupted, now))).toEqual([]);
        const resume = (elapsed: number) => engine.receiveTelemetry({ ...interrupted, telemetryStatus: 2,
            sample: sample(500 + elapsed, { Graphics_normalized_car_position: 0.12 }) }, now + 100 + elapsed);
        for (const elapsed of [0, 100, 499]) expect(spoken(resume(elapsed))).toEqual([]);
        expect(spoken(resume(600))).toHaveLength(1);
    });

it.each([0.09, 1, undefined, NaN])('does not confirm with current brake pressure %s', (brake) => {
    const { at } = setup();
    for (const elapsed of [0, 100, 300, 600, 900]) {
        expect(spoken(at(elapsed, { Physics_brake: brake }))).toEqual([]);
    }
});

it('uses current pressure even when rising pressure projects to full brake', () => {
    const { at } = setup();
    for (const elapsed of [0, 100, 300, 599]) expect(spoken(at(elapsed, { Physics_brake: 0.6 + elapsed / 2000 }))).toEqual([]);
    expect(spoken(at(600, { Physics_brake: 0.9 }))).toHaveLength(1);
});

it('resets the hold on current pedal release and requires a new stable 500 ms', () => {
    const { at } = setup();
    [0, 100, 300].forEach((elapsed) => at(elapsed));
    expect(spoken(at(400, { Physics_brake: 0.6 }))).toEqual([]);
    for (const elapsed of [500, 700, 999]) expect(spoken(at(elapsed, { Physics_brake: 0.6 }))).toEqual([]);
    expect(spoken(at(1000, { Physics_brake: 0.6 }))).toHaveLength(1);
});

it.each([0.1, 0.01])('accepts a strict reference deficit of %s m/s² without the old tolerance band', (deficit) => {
    const { at } = setup();
    for (const elapsed of [0, 100, 300, 599]) {
        expect(spoken(at(elapsed, { Physics_speed_kmh: 108 - (3 - deficit) * 3.6 * elapsed / 1000 }))).toEqual([]);
    }
    expect(spoken(at(600, { Physics_speed_kmh: 108 - (3 - deficit) * 3.6 * 0.6 }))).toHaveLength(1);
});

it.each([3, 4])('does not request More brake from excess speed with sufficient current deceleration (%s m/s²)', (deceleration) => {
    const { at } = setup();
    for (const elapsed of [0, 100, 300, 600, 900]) {
        expect(spoken(at(elapsed, { Physics_speed_kmh: 130 - deceleration * 3.6 * elapsed / 1000 }))).toEqual([]);
    }
});

it('uses current position, pressure and deceleration even when every preview value disagrees', () => {
    const { reference, sample } = setup();
    const current = sample(0), intended = { ...current, Physics_speed_kmh: 80, Physics_brake: 1, Graphics_normalized_car_position: 0.4 };
    expect(readReferenceGuidance(current, null, reference, 'acc', 1000, reference, 20,
        undefined, undefined, true, 0.4, intended, { deceleration: 1, releasing: false }).moreBrake).toBe(2500);
    expect(readReferenceGuidance(current, null, reference, 'acc', 1000, reference, 0,
        undefined, undefined, false, 0.1, current, { deceleration: 4, releasing: false }).moreBrake).toBeUndefined();
    expect(readReferenceGuidance({ ...current, Graphics_normalized_car_position: 0.03 }, null, reference, 'acc', 1000, reference, 0,
        undefined, undefined, false, 0.1, current, { deceleration: 1, releasing: false }).moreBrake).toBeUndefined();
});
