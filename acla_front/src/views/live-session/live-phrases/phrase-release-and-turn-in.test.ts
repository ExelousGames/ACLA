import type { StandardTelemetrySample } from '../live-session-types';
import type { LiveTelemetryFrameEvent } from '../live-telemetry-store';
import { PHRASE_DEFINITIONS, PhraseEngine } from './phrase-engine';
import { getPhraseSpeechInstances } from './phrase-speech';
import { steeringCornerFixture } from './test-fixtures';

const TURN = 'reference-steering';
const RELEASE = 'reference-releaseBrake';

const frame = (sample: StandardTelemetrySample): LiveTelemetryFrameEvent => ({
    type: 'frame', sample, sampleIndex: 0, telemetryStatus: 2, sessionGeneration: 0, streamGeneration: 0,
    committedSampleCount: 0,
    update: { type: 'frame', game: 'acc', sample, sequence: 1, committedCount: 0, committedSequence: 0 },
});

function setup(options: { aggression?: number; turnDuration?: number; standaloneRelease?: boolean } = {}) {
    const { map, reference } = steeringCornerFixture();
    const ids = [TURN, ...(options.standaloneRelease === false ? [] : [RELEASE])];
    const phrases = PHRASE_DEFINITIONS.filter(({ id }) => ids.includes(id));
    const engine = new PhraseEngine(phrases);
    engine.setDrivingAggression(options.aggression ?? 100, 0);
    engine.setSpeechDurations(new Map(getPhraseSpeechInstances(phrases).map((instance) => [instance.cacheKey,
        instance.speech.text === 'Turn in' ? options.turnDuration ?? 200 : 200])));
    engine.receiveMap(map, 0, reference);
    const at = (position: number, now: number, patch: StandardTelemetrySample = {}) => engine.receiveTelemetry(frame({
        Static_track: 'test', Static_car_model: 'car', Graphics_normalized_car_position: position,
        Physics_speed_kmh: 36, Physics_brake: 0.6, Physics_gas: 0, ...patch,
    }), now);
    return { engine, at };
}

it.each([0.1, 0.6, 1])('prioritizes brake release over turn-in with %s brake pressure', (brake) => {
    const { at } = setup();
    const entered = at(0.27, 1000, { Physics_brake: brake });
    expect(entered.events).toEqual([expect.objectContaining({ ruleId: RELEASE, sentence: 'Off brake.' })]);
    expect(entered.state.current).toBe('say phrase');
});

it.each([0, 0.0999, undefined, NaN, -0.1, 1.1])('keeps ordinary turn-in for released or invalid brake %s', (brake) => {
    const { at } = setup();
    expect(at(0.27, 1000, { Physics_brake: brake }).events)
        .toEqual([expect.objectContaining({ ruleId: TURN, sentence: 'Turn in' })]);
});

it('does not add a full brake release while the forecast speed is above the reference turn-in speed', () => {
    const { at } = setup();
    expect(at(0.27, 1000, { Physics_speed_kmh: 50 }).events)
        .toEqual([expect.objectContaining({ ruleId: TURN, sentence: 'Turn in' })]);
});

it('speaks turn-in separately after brake release without repeating either cue', () => {
    const { engine, at } = setup();
    const first = at(0.27, 1000);
    expect(first.events.map(({ sentence }) => sentence)).toEqual(['Off brake.']);
    engine.speechFinished(first.events[0].id, 1200);
    const turn = at(0.28, 1300);
    expect(turn.events.map(({ sentence }) => sentence)).toEqual(['Off brake.', 'Turn in']);
    engine.speechFinished(turn.events[1].id, 1500);
    expect(at(0.29, 1700, { Physics_brake: 0 }).events.map(({ sentence }) => sentence)).toEqual(['Off brake.', 'Turn in']);
});

it('can release the brake after turn-in has already spoken', () => {
    const { engine, at } = setup();
    const first = at(0.27, 1000, { Physics_brake: 0 });
    expect(first.events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
    engine.speechFinished(first.events[0].id, 1200);
    const braking = at(0.28, 1300);
    expect(braking.events.map(({ sentence }) => sentence)).toEqual(['Turn in', 'Off brake.']);
});

it('rearms turn-in only after measured progress leaves the mapped corner', () => {
    const { engine, at } = setup();
    const first = at(0.27, 1000, { Physics_brake: 0 });
    expect(first.events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
    engine.speechFinished(first.events[0].id, 1200);
    at(0.43, 1300, { Physics_brake: 0 });
    expect(at(0.27, 1400, { Physics_brake: 0 }).events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
    at(0.46, 1500, { Physics_brake: 0 });
    expect(at(0.27, 1600, { Physics_brake: 0 }).events.map(({ sentence }) => sentence)).toEqual(['Turn in', 'Turn in']);
});

it.each([0, 50, 99, 99.9])('keeps only standalone release below Direction guide at %s%% aggression', (aggression) => {
    const { at } = setup({ aggression });
    expect(at(0.27, 1000).events)
        .toEqual([expect.objectContaining({ ruleId: RELEASE, sentence: 'Off brake.' })]);
});

it.each([400, 2000])('uses the turn-in clip\'s own %s ms forecast', (durationMs) => {
    const { at } = setup({ turnDuration: durationMs, standaloneRelease: false });
    const entryPosition = 0.27 - durationMs / 100000;
    expect(at(entryPosition - 0.0001, 1000).events).toEqual([]);
    expect(at(entryPosition + 0.0001, 1001).events).toEqual([expect.objectContaining({
        ruleId: TURN, sentence: 'Turn in', timestamp: 1001,
        speech: { durationMs, preparationTimeMs: durationMs, reactionAllowanceMs: 0 },
    })]);
});
