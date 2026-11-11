import type { StandardTelemetrySample } from '../live-session-types';
import type { LiveTelemetryFrameEvent } from '../live-telemetry-store';
import { PHRASE_DEFINITIONS, PhraseEngine } from './phrase-engine';
import { getPhraseSpeechInstances } from './phrase-speech';
import { steeringCornerFixture } from './test-fixtures';

type Side = 'left' | 'middle' | 'right';
const instruction = (side: Side) => `Set up ${side === 'middle' ? 'center' : side}`;
const wrap = (position: number) => ((position % 1) + 1) % 1;

function setup({ game = 'acc', opposite = false, shift = 0, labeled = true, members = 3,
    aggression = 100, missingSteering = false, missingBoundaries = false, duration = 200, fullCatalog = false,
    sides = ['left', 'left', 'left'] as const }:
    { game?: 'acc' | 'iracing'; opposite?: boolean; shift?: number; labeled?: boolean; members?: number;
        aggression?: number; missingSteering?: boolean; missingBoundaries?: boolean; duration?: number; fullCatalog?: boolean;
        sides?: readonly [Side, Side, Side] } = {}) {
    const { map, reference } = steeringCornerFixture(game);
    const starts = [25, 50, 75];
    const steering = [0.1, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.35, 0.25, 0.15, 0.08, 0.02];
    let x = 0, ground = 0;
    reference.y.forEach((row, index) => {
        row[0] = x;
        row[game === 'acc' ? 1 : 2] = ground;
        row[3] = starts.reduce((value, start, turn) => value
            + (steering[index - start - 2] ?? 0) * (opposite && turn === 1 ? -1 : 1), 0);
        const angle = starts.reduce((value, start, turn) => value + 12
            * (Math.max(0, Math.min(15, index - start)) + (index === start ? 0.5 : 0))
            * (opposite && turn === 1 ? -1 : 1), 0) * Math.PI / 180;
        x += 10 * Math.cos(angle);
        ground += 10 * Math.sin(angle);
    });
    if (missingSteering) reference.target_features[3] = 'missing-steering';
    const groundAxis = game === 'acc' ? 1 : 2;
    const points = reference.x.map((position, index) => {
        const before = reference.y[Math.max(0, index - 1)], after = reference.y[Math.min(reference.y.length - 1, index + 1)];
        const dx = after[0] - before[0], dy = after[groundAxis] - before[groundAxis], length = Math.hypot(dx, dy);
        const sign = game === 'acc' ? 1 : -1;
        const right = [-dy / length * sign, dx / length * sign];
        const offsets = sides.map((side) => side === 'left' ? 6 : side === 'right' ? -6 : 0);
        const segment = index < 52 ? 0 : 1;
        const fraction = Math.max(0, Math.min(1, (index - (starts[segment] + 2)) / 25));
        const offset = offsets[segment] + (offsets[segment + 1] - offsets[segment]) * fraction;
        return { position: wrap(position + shift), row: reference.y[index], right, offset };
    })
        .sort((a, b) => a.position - b.position);
    reference.x = points.map(({ position }) => position);
    reference.y = points.map(({ row }) => row);
    const line = (boundary: number) => points.map(({ position, row, right, offset }, index) => ({
        bin: index, normalized_position: position, x: row[0] + right[0] * (offset + boundary),
        y: row[1] + (game === 'acc' ? right[1] * (offset + boundary) : 0),
        z: row[2] + (game === 'iracing' ? right[1] * (offset + boundary) : 0), sample_count: 1, updated_at: '',
    }));
    map.samples.middle_line = line(0);
    map.samples.left_boundary = missingBoundaries ? [] : line(-10);
    map.samples.right_boundary = missingBoundaries ? [] : line(10);
    map.centerline_segments = starts.slice(0, members).map((start, index) => ({
        id: `turn-${index + 1}`, tags: ['corner'],
        start_position: wrap((start + 2) / 100 + shift), end_position: wrap((start + 20) / 100 + shift),
    }));
    if (labeled) map.centerline_segments.push({ id: 'sequence', tags: ['consecutive corners'],
        start_position: wrap(0.27 + shift), end_position: wrap((starts[members - 1] + 20) / 100 + shift) });
    const phrases = fullCatalog ? PHRASE_DEFINITIONS : PHRASE_DEFINITIONS.filter(({ id }) =>
        id === 'reference-consecutive-corners' || id === 'reference-cornerExit');
    const engine = new PhraseEngine(phrases);
    engine.setDrivingAggression(aggression, 0);
    engine.setSpeechDurations(new Map(getPhraseSpeechInstances(phrases).map((instance) => [instance.cacheKey, duration])));
    engine.receiveMap(map, 0, reference);
    const at = (position: number, now: number, patch: StandardTelemetrySample = {}) => {
        const sample: StandardTelemetrySample = { Static_track: 'test', Static_car_model: 'car',
            Graphics_normalized_car_position: wrap(position + shift), Physics_speed_kmh: 36,
            Physics_brake: 0, Physics_gas: 0, ...patch };
        const frame: LiveTelemetryFrameEvent = { type: 'frame', sample, sampleIndex: 0, telemetryStatus: 2,
            sessionGeneration: 0, streamGeneration: 0, committedSampleCount: 0,
            update: { type: 'frame', game, sample, sequence: 1, committedCount: 0, committedSequence: 0 } };
        return engine.receiveTelemetry(frame, now);
    };
    return { engine, at, map, reference };
}

describe.each(['acc', 'iracing'] as const)('consecutive corner coaching in %s', (game) => {
    it.each([false, true])('sets up the next corner once through each linked arc (opposite: %s)', (opposite) => {
        const { engine, at } = setup({ game, opposite });
        const first = instruction('left'), second = instruction('left');
        expect(at(0.31, 1000).events.map(({ sentence }) => sentence)).toEqual([first]);
        engine.speechFinished(1, 1200);
        expect(at(0.32, 1300).events.map(({ sentence }) => sentence)).toEqual([first]);
        expect(at(0.36, 1400).events.map(({ sentence }) => sentence)).toEqual([first]);
        at(0.47, 1500);
        expect(at(0.56, 1600).events.map(({ sentence }) => sentence)).toEqual([first, second]);
        engine.speechFinished(2, 1800);
        expect(at(0.61, 1900).events.map(({ sentence }) => sentence)).toEqual([first, second]);
        expect(at(0.86, 2000).events.map(({ sentence }) => sentence)).toEqual([first, second]);
    });

    it.each(['left', 'middle', 'right'] as const)('uses the reference %s position instead of assuming the outside', (side) => {
        const { at } = setup({ game, sides: [side, side, side] });
        expect(at(0.31, 1000).events.map(({ sentence }) => sentence))
            .toEqual([instruction(side)]);
    });

    it('keeps Unwind for an isolated corner', () => {
        const { at } = setup({ game, labeled: false, members: 1 });
        expect(at(0.36, 1000).events.map(({ sentence }) => sentence)).toEqual(['Unwind']);
    });

    it.each([1, 3])('does not invent a next corner for the final member of a %s-corner sequence', (members) => {
        const { at } = setup({ game, members });
        const lastHold = members === 1 ? 0.31 : 0.81;
        expect(at(lastHold, 1000).events).toEqual([]);
        expect(at(lastHold + 0.05, 1100).events).toEqual([]);
    });
});

it('keeps sequence membership and once-per-turn behavior across start/finish', () => {
    const { engine, at } = setup({ shift: 0.7, opposite: true });
    const first = instruction('left'), second = instruction('left');
    expect(at(0.31, 1000).events.map(({ sentence }) => sentence)).toEqual([first]);
    engine.speechFinished(1, 1200);
    expect(at(0.36, 1300).events.map(({ sentence }) => sentence)).toEqual([first]);
    at(0.47, 1400);
    expect(at(0.56, 1500).events.map(({ sentence }) => sentence)).toEqual([first, second]);
});

it.each([0, 50, 99])('withholds sequence direction advice at %s%% aggression', (aggression) => {
    const { at } = setup({ aggression });
    expect(at(0.31, 1000).events).toEqual([]);
    expect(at(0.36, 1100).events).toEqual([]);
});

it('requires reference steering evidence for the setup cue', () => {
    const { at } = setup({ missingSteering: true });
    expect(at(0.31, 1000).events).toEqual([]);
    expect(at(0.36, 1100).events).toEqual([]);
});

it('withholds positioning without usable track boundaries instead of guessing from corner direction', () => {
    const { at } = setup({ missingBoundaries: true });
    expect(at(0.31, 1000).events).toEqual([]);
    expect(at(0.36, 1100).events).toEqual([]);
});

it('withholds sequence setup while stationary and when telemetry has expired', () => {
    const { engine, at } = setup();
    expect(at(0.31, 1000, { Physics_speed_kmh: 0 }).events).toEqual([]);
    expect(engine.evaluate(2600, true).events).toEqual([]);
});

it('uses the longest prepared positioning clip to enter a sequence before measured membership', () => {
    const { engine, at, map, reference } = setup();
    engine.setSpeechDurations(new Map(getPhraseSpeechInstances().map((instance) => [instance.cacheKey,
        instance.speech.text === instruction('right') ? 3000 : 200])));
    map.centerline_segments!.push({ id: 'preceding-corner', tags: ['corner'], start_position: 0.24, end_position: 0.269 });
    engine.receiveMap({ ...map }, 0, reference);
    const snapshot = at(0.265, 1000);
    expect(snapshot.mapContext.sectionId).toBe('preceding-corner');
    expect(snapshot.mapContext.sequenceId).toBeUndefined();
    expect(snapshot.events.map(({ sentence }) => sentence)).toEqual([instruction('left')]);
    expect(snapshot.events[0].speech).toMatchObject({ durationMs: 200, preparationTimeMs: 3000 });
});

it('integrates the new cue after Turn in without issuing Unwind on the same exit', () => {
    const { engine, at } = setup({ fullCatalog: true });
    expect(at(0.27, 1000).events.map(({ sentence }) => sentence)).toEqual(['Turn in']);
    engine.speechFinished(1, 1200);
    expect(at(0.31, 1300).events.map(({ sentence }) => sentence)).toEqual(['Turn in', instruction('left')]);
    engine.speechFinished(2, 1500);
    expect(at(0.36, 1600).events.map(({ sentence }) => sentence)).toEqual(['Turn in', instruction('left')]);
});

it('selects each next entry position deterministically and preloads three unique positioning commands', () => {
    const random = jest.spyOn(Math, 'random').mockReturnValue(0.99);
    try {
        const definition = PHRASE_DEFINITIONS.find(({ id }) => id === 'reference-consecutive-corners')!;
        const clips = getPhraseSpeechInstances([definition]);
        expect(clips.map(({ speech }) => speech.text).sort()).toEqual(['Set up center', 'Set up left', 'Set up right']);
        const { engine, at } = setup({ opposite: true, sides: ['left', 'right', 'middle'] });
        const first = instruction('right'), second = instruction('middle');
        expect(at(0.31, 1000).events.map(({ sentence }) => sentence)).toEqual([first]);
        engine.speechFinished(1, 1200);
        at(0.47, 1300);
        expect(at(0.56, 1400).events.map(({ sentence }) => sentence)).toEqual([first, second]);
        expect(random).not.toHaveBeenCalled();
    } finally {
        random.mockRestore();
    }
});
