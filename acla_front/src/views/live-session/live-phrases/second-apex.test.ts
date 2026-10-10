import { createLiveTelemetryStore, type LiveTelemetryFrameEvent } from '../live-telemetry-store';
import type { StandardTelemetrySample } from '../live-session-types';
import { PHRASE_DEFINITIONS, PhraseEngine, TELEMETRY_MAX_AGE_MS } from './phrase-engine';
import { getPhrasePositions } from './phrase-positions';
import { circuitMap, vision } from './test-fixtures';

const rule = PHRASE_DEFINITIONS.find((phrase) => phrase.id === 'second-apex')!;
const sentence = 'Go wide in the first turn. then take second apex if possible';
const holdInsideSentence = 'brake early, Hold inside';
const normalize = (position: number) => (position + 1) % 1;
const frame = (sample: StandardTelemetrySample, status = 2): LiveTelemetryFrameEvent => ({
    type: 'frame', sample, sampleIndex: 0, telemetryStatus: status,
    committedSampleCount: 0, sessionGeneration: 0, streamGeneration: 0,
    update: { type: 'frame', game: 'acc', sample, sequence: 1, committedSequence: 0, committedCount: 0 },
});
const sample = (eta: number, start = 0.125, id = '2'): StandardTelemetrySample => ({
    Physics_speed_kmh: 100, Graphics_player_car_id: 1,
    Graphics_normalized_car_position: normalize(start - eta / 64 - 0.001),
    Graphics_normalized_positions: { 1: normalize(start - eta / 64 - 0.001), [id]: normalize(start - eta / 64) },
});
const linkedMap = (wrap = false, corner: 'left' | 'right' = 'right') => {
    const map = circuitMap('slow', true);
    map.samples.middle_line!.forEach((point) => { point.normalized_position = normalize(point.normalized_position + 0.025 + (wrap ? 0.90625 : 0)); });
    if (corner === 'left') map.samples.middle_line!.forEach((point) => { point.z *= -1; });
    map.centerline_segments = [
        { id: 'first', tags: ['corner', 'slow'], start_position: 0.125, end_position: 0.225 },
        { id: 'second', tags: ['corner', 'fast'], start_position: 0.235, end_position: 0.335 },
        { id: 'sequence', tags: ['consecutive corners'], start_position: 0.125, end_position: 0.335 },
    ];
    if (wrap) map.centerline_segments.forEach((segment) => {
        segment.start_position = normalize(segment.start_position + 0.90625);
        segment.end_position = normalize(segment.end_position + 0.90625);
    });
    return map;
};
const closeVision = (now: number, corner: 'left' | 'right' = 'right', distance = 1.5,
    player: 'inside' | 'middle' | 'outside' = 'inside', opponent: 'inside' | 'middle' | 'outside' = 'inside') => {
    const detection = vision(now, { corner, player, opponent });
    // Straight boundaries extend to the player so opponents within 2 m have visible track support.
    for (const side of ['leftBoundary', 'rightBoundary', 'centerline'] as const) {
        detection.birdsEyeScene![side] = detection.birdsEyeScene![side].map((points) =>
            points.map((point) => ({ ...point, x: points[0].x, y: point.y - 8 })));
    }
    const offset = { inside: 2.5, middle: 5, outside: 7.5 };
    detection.birdsEyeScene!.cars[0].position = {
        x: (offset[opponent] - offset[player]) * (corner === 'right' ? -1 : 1), y: distance, z: 0,
    };
    return detection;
};
const update = (engine: PhraseEngine, now: number, eta: number, corner: 'left' | 'right' = 'right', start = 0.125) => {
    engine.receiveVision(closeVision(now, corner), now);
    return engine.receiveTelemetry(frame(sample(eta, start)), now);
};

describe('second-apex closure', () => {
    it.each(['left', 'right'] as const)('waits inside root child, speaks once before a %s turn, then exits', (corner) => {
        const engine = new PhraseEngine();
        engine.receiveMap(linkedMap(false, corner), 0);
        expect(update(engine, 0, 3, corner).state.path).toEqual(['root']);
        const entered = update(engine, 1000, 2, corner);
        expect(entered.state.path).toEqual(['root', rule.name]);
        expect(entered.closures[0].status).toBe('Waiting for action');
        expect(entered.events).toEqual([]);
        expect(update(engine, 2000, 1, corner).events).toEqual([]);
        expect(update(engine, 2500, 0.5, corner).events).toEqual([]);
        // A timer or vision update may evaluate a match, but only a telemetry frame speaks.
        const detection = closeVision(2750, corner);
        const positions = getPhrasePositions(detection.birdsEyeScene);
        expect(positions.playerCorner).toBeUndefined();
        expect(positions.opponentCorner).toBeUndefined();
        expect(positions.opponentPosition).toBe(corner);
        expect(engine.receiveVision(detection, 2750).events).toEqual([]);
        const spoken = engine.receiveTelemetry(frame(sample(0.25)), 2750);
        expect(spoken.events).toEqual([{ id: 1, ruleId: rule.id, sentence, timestamp: 2750 }]);
        expect(spoken.state.path).toEqual(['root', rule.name, 'say phrase']);
        expect(spoken.closures[0].status).toBe('Active');
        expect(engine.evaluate(2750).state.path).toEqual(['root']);
        expect(update(engine, 2800, 0.2, corner).events).toHaveLength(1);
    });

    it('enters at the inclusive 2 m boundary without the ordinary 0.8 s hold', () => {
        const engine = new PhraseEngine([rule]);
        engine.receiveMap(linkedMap(), 0);
        update(engine, 0, 3);
        engine.receiveVision(closeVision(1000, 'right', 2), 1000);
        expect(engine.receiveTelemetry(frame(sample(2)), 1000).state.path).toEqual(['root', rule.name]);
    });

    it.each(['too far', 'too early', 'fast', 'unlabeled sequence', 'same direction', 'unknown geometry', 'no opponent positions', 'no player ID', 'stationary', 'reversing'])(
        'does not enter for %s', (reason) => {
            const map = linkedMap();
            if (reason === 'fast') map.centerline_segments![0].tags = ['corner', 'fast'];
            if (reason === 'unlabeled sequence') map.centerline_segments!.pop();
            if (reason === 'same direction') map.samples.middle_line![6].x = 1100;
            if (reason === 'unknown geometry') map.samples.middle_line = [];
            const engine = new PhraseEngine([rule]);
            engine.receiveMap(map, 0);
            update(engine, 0, reason === 'too early' ? 4 : 3);
            let next = sample(reason === 'too early' ? 3 : reason === 'stationary' ? 3 : reason === 'reversing' ? 4 : 2);
            if (reason === 'no opponent positions') next = { ...next, Graphics_normalized_positions: undefined };
            if (reason === 'no player ID') next = { ...next, Graphics_player_car_id: undefined };
            engine.receiveVision(closeVision(1000, 'right', reason === 'too far' ? 2.01 : 1.5), 1000);
            const snapshot = engine.receiveTelemetry(frame(next), 1000);
            expect(snapshot.state.path).toEqual(['root']);
            expect(snapshot.events).toEqual([]);
        },
    );

    it.each((['left', 'right'] as const).flatMap((corner) =>
        (['inside', 'middle', 'outside', 'unknown'] as const).map((player) => ({ corner, player }))))(
        'speaks before a $corner turn with the opponent inside and the player $player', ({ corner, player }) => {
            const engine = new PhraseEngine([rule]);
            engine.receiveMap(linkedMap(false, corner), 0);
            update(engine, 0, 3, corner);
            update(engine, 1000, 2, corner);
            update(engine, 2000, 1, corner);
            const detection = closeVision(2750, corner, 9, player === 'unknown' ? 'inside' : player);
            if (player === 'unknown') {
                const scene = detection.birdsEyeScene!;
                for (const side of ['leftBoundary', 'rightBoundary'] as const) {
                    scene[side] = scene[side].map((points) => points.map((point) => ({ ...point, x: point.x + 20 })));
                }
                scene.cars[0].position.x += 20;
                expect(getPhrasePositions(scene).playerPosition).toBeUndefined();
            }
            expect(getPhrasePositions(detection.birdsEyeScene).opponentPosition).toBe(corner);
            engine.receiveVision(detection, 2750);
            expect(engine.receiveTelemetry(frame(sample(0.25)), 2750).events).toEqual([
                { id: 1, ruleId: rule.id, sentence, timestamp: 2750 },
            ]);
        },
    );

    it.each((['left', 'right'] as const).flatMap((corner) =>
        (['inside', 'middle', 'outside', 'unknown'] as const).map((player) => ({ corner, player }))))(
        'brakes early before a $corner turn with the opponent outside and the player $player', ({ corner, player }) => {
            const engine = new PhraseEngine([rule]);
            engine.receiveMap(linkedMap(false, corner), 0);
            const advance = (now: number, eta: number) => {
                const detection = closeVision(now, corner, 9, player === 'unknown' ? 'inside' : player, 'outside');
                if (player === 'unknown') {
                    const scene = detection.birdsEyeScene!;
                    for (const side of ['leftBoundary', 'rightBoundary'] as const) {
                        scene[side] = scene[side].map((points) => points.map((point) => ({ ...point, x: point.x + 20 })));
                    }
                    scene.cars[0].position.x += 20;
                    expect(getPhrasePositions(scene).playerPosition).toBeUndefined();
                }
                expect(getPhrasePositions(detection.birdsEyeScene).opponentPosition).toBe(corner === 'right' ? 'left' : 'right');
                expect(engine.receiveVision(detection, now).events).toEqual([]);
                return engine.receiveTelemetry(frame(sample(eta)), now);
            };
            update(engine, 0, 3, corner);
            expect(update(engine, 1000, 2, corner).closures[0].status).toBe('Waiting for action');
            expect(advance(2000, 1).events).toEqual([]);
            expect(advance(2500, 0.5).events).toEqual([]);
            const spoken = advance(2750, 0.25);
            expect(spoken.events).toEqual([{ id: 1, ruleId: rule.id, sentence: holdInsideSentence, timestamp: 2750 }]);
            expect(spoken.closures[0].status).toBe('Active');
            expect(spoken.root.children[0].children[0].execution?.status).toBe('idle');
            expect(spoken.root.children[0].children[1]).toMatchObject({
                current: true, description: holdInsideSentence, execution: { status: 'completed' },
            });
            expect(engine.evaluate(2750).state.path).toEqual(['root']);
            expect(update(engine, 2800, 0.2, corner).events).toHaveLength(1);
        },
    );

    it.each(['inside', 'outside'] as const)('exits after speaking with the opponent %s even when the other action becomes eligible', (opponent) => {
        const engine = new PhraseEngine([rule]);
        engine.receiveMap(linkedMap(), 0);
        update(engine, 0, 3);
        update(engine, 1000, 2);
        update(engine, 2000, 1);
        const first = closeVision(2750, 'right', 9, 'inside', opponent);
        engine.receiveVision(first, 2750);
        const spoken = engine.receiveTelemetry(frame(sample(0.25)), 2750);
        expect(spoken.events[0].sentence).toBe(opponent === 'inside' ? sentence : holdInsideSentence);
        // Change the published scene before the next telemetry step, with no intervening exit step.
        const other = closeVision(2760, 'right', 9, 'inside', opponent === 'inside' ? 'outside' : 'inside');
        first.birdsEyeScene!.cars = other.birdsEyeScene!.cars;
        const next = engine.receiveTelemetry(frame(sample(0.24)), 2760);
        expect(next.state.path).toEqual(['root']);
        expect(next.events).toEqual(spoken.events);
    });

    it.each((['left', 'right'] as const).flatMap((corner) =>
        (['inside', 'outside'] as const).flatMap((opponent) =>
            ['middle', 'missing opponent', 'stale vision', 'paused', 'already entered'].map((reason) => ({ corner, opponent, reason })))))(
        'does not speak or exit prematurely before a $corner turn with the opponent $opponent and $reason', ({ corner, opponent, reason }) => {
            const engine = new PhraseEngine([rule]);
            engine.receiveMap(linkedMap(false, corner), 0);
            update(engine, 0, 3, corner);
            update(engine, 1000, 2, corner);
            engine.receiveTelemetry(frame(sample(1)), 2000);
            const detection = closeVision(2750, corner, 9, 'outside', opponent);
            if (reason === 'middle') {
                const middle = vision(2750, { corner, player: 'middle', opponent: 'middle' });
                middle.birdsEyeScene!.cars[0].position = { x: 0, y: 9, z: 0 };
                engine.receiveVision(middle, 2750);
            } else if (reason === 'missing opponent') {
                detection.birdsEyeScene!.cars = [];
                engine.receiveVision(detection, 2750);
            } else if (reason !== 'stale vision') engine.receiveVision(detection, 2750);
            const snapshot = engine.receiveTelemetry(frame(sample(reason === 'already entered' ? -0.1 : 0.25), reason === 'paused' ? 1 : 2), reason === 'stale vision' ? 3250 : 2750);
            expect(snapshot.events).toEqual([]);
            expect(snapshot.state.path).toEqual(['root', rule.name]);
        },
    );

    it('keeps the entered scope selected if entry conditions stop matching', () => {
        const engine = new PhraseEngine([rule]);
        engine.receiveMap(linkedMap(), 0);
        update(engine, 0, 3);
        update(engine, 1000, 2);
        engine.receiveVision(closeVision(1500, 'right', 20), 1500);
        expect(engine.receiveTelemetry(frame(sample(1.5)), 1500).state.path).toEqual(['root', rule.name]);
        expect(engine.evaluate(1500).events).toEqual([]);
    });

    it('handles an opponent crossing start/finish while approaching a linked slow corner', () => {
        const engine = new PhraseEngine([rule]);
        engine.receiveMap(linkedMap(true), 0);
        update(engine, 0, 3, 'right', 0.03125);
        expect(update(engine, 1000, 2, 'right', 0.03125).state.path).toEqual(['root', rule.name]);
        update(engine, 2000, 1, 'right', 0.03125);
        expect(update(engine, 2750, 0.25, 'right', 0.03125).events[0].sentence).toBe(sentence);
    });

    it('retains the rate across repeated packets but discards stale or changed opponents', () => {
        const engine = new PhraseEngine([rule]);
        engine.receiveMap(linkedMap(), 0);
        update(engine, 0, 4);
        update(engine, 1000, 3);
        expect(engine.receiveTelemetry(frame(sample(3)), 1010).closures[0].conditions[1].inputMissing).toBe(false);
        expect(engine.receiveTelemetry(frame(sample(2, 0.125, '3')), 1100).state.path).toEqual(['root']);
        expect(engine.evaluate(1100).closures[0].conditions[1].inputMissing).toBe(true);
        expect(engine.receiveTelemetry(frame(sample(1)), 1100 + TELEMETRY_MAX_AGE_MS + 1).state.path).toEqual(['root']);
    });

    it.each(['session-reset', 'stream-reset'] as const)('clears a waiting closure and motion history on %s', (type) => {
        const engine = new PhraseEngine([rule]);
        engine.receiveMap(linkedMap(), 0);
        update(engine, 0, 3);
        update(engine, 1000, 2);
        expect(engine.receiveTelemetry({ type, snapshot: createLiveTelemetryStore().getSnapshot() }, 1100).state.path).toEqual(['root']);
        expect(update(engine, 1200, 0.25).state.path).toEqual(['root']);
        expect(engine.evaluate(1200).events).toEqual([]);
    });
});
