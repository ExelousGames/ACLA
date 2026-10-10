import type { LiveTelemetryFrameEvent } from '../live-telemetry-store';
import type { StandardTelemetrySample } from '../live-session-types';
import { VISION_MAX_AGE_MS } from '../track-vision/track-vision-types';
import { PHRASE_DEFINITIONS, PhraseEngine } from './phrase-engine';
import { getPhrasePositions, type CornerDirection, type TrackPosition } from './phrase-positions';
import { circuitMap, vision } from './test-fixtures';

const rule = PHRASE_DEFINITIONS.find((phrase) => phrase.id === 'one-tight-slow-corner')!;
const slipstream = 'Opponent didnt defend, overtake from inside is possible here';
const defending = 'Opponent is defending, Pressure is on';
const positions = ['left', 'middle', 'right'] as const;
const frame = (sample: StandardTelemetrySample, status = 2): LiveTelemetryFrameEvent => ({
    type: 'frame', sample, sampleIndex: 0, telemetryStatus: status,
    committedSampleCount: 0, sessionGeneration: 0, streamGeneration: 0,
    update: { type: 'frame', game: 'acc', sample, sequence: 1, committedSequence: 0, committedCount: 0 },
});
const sample = (eta: number): StandardTelemetrySample => ({
    Physics_speed_kmh: 100, Graphics_player_car_id: 1,
    Graphics_normalized_car_position: 0.125 - eta / 64 - 0.001,
    Graphics_normalized_positions: { 1: 0.125 - eta / 64 - 0.001, 2: 0.125 - eta / 64 },
});
const slowCorner = (corner: CornerDirection = 'right') => {
    const map = circuitMap();
    map.samples.middle_line!.forEach((point) => {
        point.normalized_position += 0.025;
        if (corner === 'left') point.z *= -1;
    });
    map.centerline_segments = [{ id: 'turn', tags: ['corner', 'slow'], start_position: 0.125, end_position: 0.225 }];
    return map;
};
const closeVision = (now: number, player: TrackPosition = 'left', opponent: TrackPosition = 'left', y = 1) => {
    const relative = { left: 'inside', middle: 'middle', right: 'outside' } as const;
    const detection = vision(now, { corner: 'straight', player: relative[player], opponent: relative[opponent] });
    const offset = { left: 2.25, middle: 3, right: 3.75 };
    const edges = { leftBoundary: 0, rightBoundary: 6, centerline: 3 };
    // Keep every tested side combination within 2 m, with visible boundaries down to the player.
    for (const side of ['leftBoundary', 'rightBoundary', 'centerline'] as const) {
        detection.birdsEyeScene![side] = detection.birdsEyeScene![side].map((points) =>
            points.map((point) => ({ ...point, x: edges[side] - offset[player], y: point.y - 8 })));
    }
    detection.birdsEyeScene!.cars[0].position = { x: offset[opponent] - offset[player], y, z: 0 };
    return detection;
};

describe('One Tight Slow Corner', () => {
    it.each([
        { corner: 'left', opponent: 'left', expected: [null, null, null] },
        { corner: 'left', opponent: 'middle', expected: [defending, null, null] },
        { corner: 'left', opponent: 'right', expected: [slipstream, slipstream, slipstream] },
        { corner: 'right', opponent: 'left', expected: [slipstream, slipstream, slipstream] },
        { corner: 'right', opponent: 'middle', expected: [null, null, defending] },
        { corner: 'right', opponent: 'right', expected: [null, null, null] },
    ].flatMap(({ corner, opponent, expected }) => positions.map((player, index) => ({
        corner: corner as CornerDirection, opponent: opponent as TrackPosition, player, sentence: expected[index],
    }))))('selects the phrase for a $corner turn with opponent $opponent and driver $player', ({ corner, opponent, player, sentence }) => {
        const engine = new PhraseEngine();
        engine.receiveMap(slowCorner(corner), 0);
        engine.receiveVision(closeVision(0, player, opponent, 3), 0);
        expect(engine.receiveTelemetry(frame(sample(3)), 0).events).toEqual([]);
        const detection = closeVision(1000, player, opponent);
        expect(getPhrasePositions(detection.birdsEyeScene)).toMatchObject({
            playerPosition: player, opponentPosition: opponent, playerCorner: undefined, opponentCorner: undefined,
        });
        expect(engine.receiveVision(detection, 1000).events).toEqual([]);
        const snapshot = engine.receiveTelemetry(frame(sample(2)), 1000);
        expect(snapshot.state.path).toEqual(['root', rule.name, ...(sentence ? ['say phrase'] : [])]);
        expect(snapshot.events).toEqual(sentence ? [{ id: 1, ruleId: rule.id, sentence, timestamp: 1000 }] : []);
        if (sentence) {
            expect(engine.evaluate(1000).state.path).toEqual(['root']);
            expect(engine.receiveTelemetry(frame(sample(1.9)), 1100).events).toEqual(snapshot.events);
        }
    });

    it('includes the 2 m entry boundary without an additional hold or motion rate', () => {
        const engine = new PhraseEngine([rule]);
        engine.receiveMap(slowCorner(), 0);
        engine.receiveVision(closeVision(1000, 'left', 'left', 2), 1000);
        const snapshot = engine.receiveTelemetry(frame(sample(3)), 1000);
        expect(snapshot.closures[0].conditions.map((condition) => ({ description: 'description' in condition ? condition.description : '',
            connector: condition.connector, conditionFit: condition.conditionFit }))).toEqual([
            { description: 'Estimated opponent distance (m) <= 2', connector: 'and', conditionFit: true },
            { description: 'Corner is not inside a consecutive corners label of the Live Map', connector: 'and', conditionFit: true },
        ]);
        expect(snapshot.events).toEqual([
            { id: 1, ruleId: rule.id, sentence: slipstream, timestamp: 1000 },
        ]);
    });

    it.each(['too far', 'missing map', 'missing corner', 'missing opponent positions', 'missing vision', 'stale vision', 'paused'])(
        'does not enter with %s', (reason) => {
            const engine = new PhraseEngine([rule]);
            const map = slowCorner();
            if (reason === 'missing corner') map.centerline_segments![0].tags = ['long straight'];
            engine.receiveMap(reason === 'missing map' ? null : map, 0);
            engine.receiveVision(closeVision(0), 0);
            const now = reason === 'stale vision' ? VISION_MAX_AGE_MS + 1 : 1000;
            if (reason !== 'stale vision') engine.receiveVision(reason === 'missing vision' ? null
                : closeVision(now, 'left', 'left', reason === 'too far' ? 2.01 : 1), now);
            const next = reason === 'missing opponent positions'
                ? { ...sample(2), Graphics_normalized_positions: undefined } : sample(2);
            const snapshot = engine.receiveTelemetry(frame(next, reason === 'paused' ? 1 : 2), now);
            expect(snapshot.events).toEqual([]);
            expect(snapshot.state.path).toEqual(['root']);
        },
    );

    it.each(['slow', 'fast', 'unlabeled'])('enters a standalone %s corner with more than 2 s until entry', (speed) => {
        const engine = new PhraseEngine([rule]);
        const map = slowCorner();
        map.centerline_segments![0].tags = speed === 'unlabeled' ? ['corner'] : ['corner', speed];
        // A label elsewhere on the lap must not exclude this corner.
        map.centerline_segments!.push({ id: 'other-sequence', tags: ['consecutive corners'], start_position: 0.4, end_position: 0.6 });
        engine.receiveMap(map, 0);
        engine.receiveTelemetry(frame(sample(4)), 0);
        engine.receiveVision(closeVision(1000), 1000);
        expect(engine.receiveTelemetry(frame(sample(3)), 1000).events).toEqual([
            { id: 1, ruleId: rule.id, sentence: slipstream, timestamp: 1000 },
        ]);
    });

    it.each(['first', 'last', 'only', 'wraparound'])('excludes the %s corner inside a consecutive corners label', (member) => {
        const engine = new PhraseEngine([rule]);
        const map = slowCorner();
        if (member === 'first') map.centerline_segments!.push({ id: 'next', tags: ['corner'], start_position: 0.25, end_position: 0.3 });
        if (member === 'last') map.centerline_segments!.push({ id: 'previous', tags: ['corner'], start_position: 0.025, end_position: 0.05 });
        map.centerline_segments!.push({ id: 'sequence', tags: ['consecutive corners'], start_position: member === 'wraparound' ? 0.9 : 0, end_position: 0.35 });
        engine.receiveMap(map, 0);
        engine.receiveTelemetry(frame(sample(3)), 0);
        engine.receiveVision(closeVision(1000), 1000);
        const snapshot = engine.receiveTelemetry(frame(sample(2)), 1000);
        expect(snapshot.closures[0].conditions.map((condition) => condition.conditionFit)).toEqual([true, false]);
        expect(snapshot.closures[0].status).toBe('Not matched');
        expect(snapshot.events).toEqual([]);
        expect(snapshot.state.path).toEqual(['root']);
    });

    it('waits for a matching action, speaks on telemetry, and exits before another action can speak', () => {
        const engine = new PhraseEngine([rule]);
        engine.receiveMap(slowCorner(), 0);
        engine.receiveTelemetry(frame(sample(3)), 0);
        engine.receiveVision(closeVision(1000, 'left', 'middle'), 1000);
        const waiting = engine.receiveTelemetry(frame(sample(2)), 1000);
        expect(waiting.state.path).toEqual(['root', rule.name]);
        expect(waiting.closures[0].status).toBe('Waiting for action');
        expect(waiting.events).toEqual([]);
        const detection = closeVision(1100, 'right', 'middle');
        expect(engine.receiveVision(detection, 1100).events).toEqual([]);
        const spoken = engine.receiveTelemetry(frame(sample(1.9)), 1100);
        expect(spoken.events).toEqual([{ id: 1, ruleId: rule.id, sentence: defending, timestamp: 1100 }]);
        // Make the first action eligible before the next step without evaluating an exit in between.
        detection.birdsEyeScene!.cars = closeVision(1110, 'right', 'left').birdsEyeScene!.cars;
        const exited = engine.receiveTelemetry(frame(sample(1.89)), 1110);
        expect(exited.state.path).toEqual(['root']);
        expect(exited.events).toEqual(spoken.events);
    });

    it('waits without guessing a turn direction when map geometry has no turn', () => {
        const engine = new PhraseEngine([rule]);
        const map = slowCorner();
        map.samples.middle_line!.forEach((point) => { point.z = 0; });
        engine.receiveMap(map, 0);
        engine.receiveTelemetry(frame(sample(3)), 0);
        engine.receiveVision(closeVision(1000), 1000);
        const snapshot = engine.receiveTelemetry(frame(sample(2)), 1000);
        expect(snapshot.events).toEqual([]);
        expect(snapshot.state.path).toEqual(['root', rule.name]);
    });
});
