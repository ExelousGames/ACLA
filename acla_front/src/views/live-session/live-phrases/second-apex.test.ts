import { createLiveTelemetryStore, type LiveTelemetryFrameEvent } from '../live-telemetry-store';
import type { StandardTelemetrySample } from '../live-session-types';
import { PHRASE_DEFINITIONS, PhraseEngine, TELEMETRY_MAX_AGE_MS } from './phrase-engine';
import { getPhrasePositions } from './phrase-positions';
import { circuitMap, vision } from './test-fixtures';
import type { DriverCornerSummary } from '../event-log/corner-summary';

const rule = PHRASE_DEFINITIONS.find((phrase) => phrase.id === 'second-apex')!;
const sentence = 'Go wide in the first turn. then take second apex if possible';
const holdInsideSentence = 'brake early, Hold inside';
const exitAdvice = PHRASE_DEFINITIONS.find((phrase) => phrase.id === 'one-tight-slow-corner')!.additionalActions![1].sentence;
const exitHistory = (playerSpeed = 0.03): DriverCornerSummary[] => [playerSpeed, 0.02].map((averageExitSpeed, index) => ({
    carId: String(index + 1), isPlayer: index === 0, cornerId: 'first', cornerName: 'First corner',
    sampleCount: 0, includedCount: 0, averageTimeMs: null,
    averageExitSpeed, exitSpeedSampleCount: 3, exitSpeedIncludedCount: 1,
}));
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
    describe('Inside the Long Straight', () => {
        const path = ['root', rule.name, 'Inside the Long Straight'];
        const setup = () => {
            const engine = new PhraseEngine();
            const map = linkedMap();
            map.centerline_segments!.push(
                { id: 'approach-straight', tags: ['long straight'], start_position: 0.04, end_position: 0.115 },
                { id: 'exit-straight', tags: ['long straight'], start_position: 0.225, end_position: 0.235 },
            );
            return { engine, map };
        };

        it('speaks once per straight visit, returns to Chicane overtake, then allows its timed prompt', () => {
            const { engine, map } = setup();
            engine.receiveMap(map, 0);
            engine.receiveVision(closeVision(0, 'right', 2), 0);
            const spoken = engine.receiveTelemetry(frame(sample(3)), 0, exitHistory());
            expect(spoken.state.path).toEqual([...path, 'say phrase']);
            expect(spoken.events).toEqual([{ id: 1, ruleId: rule.id, sentence: exitAdvice, timestamp: 0 }]);
            expect(spoken.root.children[0].children[0]).toMatchObject({
                id: 'second-apex:inside-long-straight', kind: 'closure', name: 'Inside the Long Straight',
                children: [{ name: 'exit to parent' }, { name: 'say phrase', description: exitAdvice,
                    conditions: [{ conditionFit: true }, { conditionFit: true }, { conditionFit: true }] }],
            });
            expect(update(engine, 1000, 2).state.path).toEqual([...path, 'say phrase']);
            expect(update(engine, 2000, 1).events).toEqual(spoken.events);
            expect(update(engine, 2500, 0.5).state.path).toEqual(['root', rule.name]);
            const parentSpoken = update(engine, 2750, 0.25);
            expect(parentSpoken.events).toEqual([...spoken.events, { id: 2, ruleId: rule.id, sentence, timestamp: 2750 }]);
            expect(engine.evaluate(2750).state.path).toEqual(['root']);
        });

        it.each(['missing history', 'equal exit speeds', 'distant opponent', 'missing vision', 'no exit straight',
            'ordinary exit straight', 'wrong corner history', 'outside approach straight'])(
            'withholds exit advice with %s', (reason) => {
                const { engine, map } = setup();
                if (reason === 'no exit straight') map.centerline_segments!.pop();
                if (reason === 'ordinary exit straight') map.centerline_segments![4].tags = ['straight'];
                if (reason === 'outside approach straight') map.centerline_segments![3].tags = ['straight'];
                engine.receiveMap(map, 0);
                if (reason !== 'missing vision') engine.receiveVision(closeVision(0, 'right', reason === 'distant opponent' ? 2.01 : 2), 0);
                let history = reason === 'missing history' ? [] : exitHistory(reason === 'equal exit speeds' ? 0.02 : 0.03);
                if (reason === 'wrong corner history') history = history.map((summary) => ({ ...summary, cornerId: 'second' }));
                const waiting = engine.receiveTelemetry(frame(sample(3)), 0, history);
                expect(waiting.state.path).toEqual(reason === 'outside approach straight' ? ['root', rule.name] : path);
                expect(waiting.events).toEqual([]);
            },
        );

        it('waits for history and fresh telemetry before speaking, retaining the child when position is missing', () => {
            const { engine, map } = setup();
            engine.receiveMap(map, 0);
            engine.receiveVision(closeVision(0), 0);
            expect(engine.receiveTelemetry(frame(sample(3)), 0).state.path).toEqual(path);
            expect(engine.evaluate(100).events).toEqual([]);
            const missing = engine.receiveTelemetry(frame({ ...sample(3), Graphics_normalized_car_position: undefined }), 100, exitHistory());
            expect(missing.state.path).toEqual(path);
            expect(missing.events).toEqual([]);
            const spoken = engine.receiveTelemetry(frame(sample(3)), 200, exitHistory());
            expect(spoken.events).toEqual([{ id: 1, ruleId: rule.id, sentence: exitAdvice, timestamp: 200 }]);
        });
    });

    it.each(['left', 'right'] as const)('waits inside root child, speaks once before a %s turn, then exits', (corner) => {
        const engine = new PhraseEngine();
        engine.receiveMap(linkedMap(false, corner), 0);
        expect(update(engine, 0, 3, corner).state.path).toEqual(['root', rule.name]);
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

    it('enters immediately from the mapped sequence without vision or an opponent motion rate', () => {
        const engine = new PhraseEngine([rule]);
        engine.receiveMap(linkedMap(), 0);
        const entered = engine.receiveTelemetry(frame(sample(3)), 0);
        expect(entered.state.path).toEqual(['root', rule.name]);
        expect(entered.closures[0].status).toBe('Waiting for action');
        expect(entered.events).toEqual([]);
    });

    it.each(['distant opponent', 'early approach', 'stationary', 'reversing'])(
        'enters and waits for speech with %s', (reason) => {
            const engine = new PhraseEngine([rule]);
            engine.receiveMap(linkedMap(), 0);
            engine.receiveVision(closeVision(0, 'right', 20), 0);
            expect(engine.receiveTelemetry(frame(sample(4)), 0).state.path).toEqual(['root', rule.name]);
            engine.receiveVision(closeVision(1000, 'right', reason === 'distant opponent' ? 20 : 1.5), 1000);
            const eta = reason === 'stationary' ? 4 : reason === 'reversing' ? 5 : 3;
            const snapshot = engine.receiveTelemetry(frame(sample(eta)), 1000);
            expect(snapshot.state.path).toEqual(['root', rule.name]);
            expect(snapshot.closures[0].status).toBe('Waiting for action');
            expect(snapshot.events).toEqual([]);
        },
    );

    it.each(['fast', 'unlabeled sequence', 'same direction', 'unknown geometry', 'no opponent positions', 'no player ID'])(
        'does not enter for %s', (reason) => {
            const map = linkedMap();
            if (reason === 'fast') map.centerline_segments![0].tags = ['corner', 'fast'];
            if (reason === 'unlabeled sequence') map.centerline_segments!.pop();
            if (reason === 'same direction') map.samples.middle_line![6].x = 1100;
            if (reason === 'unknown geometry') map.samples.middle_line = [];
            const engine = new PhraseEngine([rule]);
            engine.receiveMap(map, 0);
            let next = sample(3);
            if (reason === 'no opponent positions') next = { ...next, Graphics_normalized_positions: undefined };
            if (reason === 'no player ID') next = { ...next, Graphics_player_car_id: undefined };
            engine.receiveVision(closeVision(0), 0);
            const snapshot = engine.receiveTelemetry(frame(next), 0);
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
            expect(spoken.root.children[0].children[1].execution?.status).toBe('idle');
            expect(spoken.root.children[0].children[2]).toMatchObject({
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
        const noSequence = linkedMap();
        noSequence.centerline_segments!.pop();
        engine.receiveMap(noSequence, 1500);
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
        expect(engine.receiveTelemetry(frame(sample(3)), 1010).closures[0].actionConditions![0].inputMissing).toBe(false);
        expect(engine.receiveTelemetry(frame(sample(2, 0.125, '3')), 1100).state.path).toEqual(['root', rule.name]);
        expect(engine.evaluate(1100).closures[0].actionConditions![0].inputMissing).toBe(true);
        const stale = engine.receiveTelemetry(frame(sample(1)), 1100 + TELEMETRY_MAX_AGE_MS + 1);
        expect(stale.state.path).toEqual(['root', rule.name]);
        expect(stale.closures[0].actionConditions![0].inputMissing).toBe(true);
        expect(stale.events).toEqual([]);
    });

    it.each(['session-reset', 'stream-reset'] as const)('clears a waiting closure and motion history on %s', (type) => {
        const engine = new PhraseEngine([rule]);
        engine.receiveMap(linkedMap(), 0);
        update(engine, 0, 3);
        update(engine, 1000, 2);
        expect(engine.receiveTelemetry({ type, snapshot: createLiveTelemetryStore().getSnapshot() }, 1100).state.path).toEqual(['root']);
        expect(update(engine, 1200, 0.25).state.path).toEqual(['root', rule.name]);
        expect(engine.evaluate(1200).events).toEqual([]);
    });
});
