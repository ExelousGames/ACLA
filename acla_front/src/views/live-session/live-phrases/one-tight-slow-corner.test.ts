import type { LiveTelemetryFrameEvent } from '../live-telemetry-store';
import type { StandardTelemetrySample } from '../live-session-types';
import { VISION_MAX_AGE_MS } from '../track-vision/track-vision-types';
import { PHRASE_DEFINITIONS, PhraseEngine, TELEMETRY_MAX_AGE_MS } from './phrase-engine';
import { getPhrasePositions, type CornerDirection, type TrackPosition } from './phrase-positions';
import { circuitMap, vision } from './test-fixtures';
import { summarizeDriverCorners } from '../event-log/corner-summary';
import type { DriverCornerRecord } from '../event-log/CornerRecorder';

const rule = PHRASE_DEFINITIONS.find((phrase) => phrase.id === 'one-tight-slow-corner')!;
const slipstream = 'Opponent didnt defend, overtake from inside is possible here';
const defending = 'Opponent is defending, Pressure is on';
const exitAdvice = rule.additionalActions![1].sentence;
const exitHistory = (playerSpeeds = [0.02, 0.03, 0.04], opponentSpeeds = [0.01, 0.02, 0.03]) =>
    summarizeDriverCorners([playerSpeeds, opponentSpeeds].flatMap((speeds, carIndex) => speeds.map((exitSpeed, index): DriverCornerRecord => ({
        id: `${carIndex}:${index}`, carId: String(carIndex + 1), isPlayer: carIndex === 0,
        cornerId: 'turn', cornerName: 'Corner 1', mapId: 'test-map',
        entry: null, exit: { timeMs: index * 1000, position: 0.225, sampleIndex: index },
        decelerationStart: null, accelerationStart: null, cornerTimeMs: null, decelerationToExitMs: null, exitSpeed,
    }))));
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
const slowCornerWithLongStraight = (corner: CornerDirection = 'right') => {
    const map = slowCorner(corner);
    map.centerline_segments!.push({ id: 'long-straight', tags: ['long straight'], start_position: 0.04, end_position: 0.115 });
    return map;
};
const closeVision = (now: number, player: TrackPosition = 'left', opponent: TrackPosition = 'left', y = 1) => {
    const relative = { left: 'inside', middle: 'middle', right: 'outside' } as const;
    const detection = vision(now, { corner: 'straight', player: relative[player], opponent: relative[opponent] });
    const offset = { left: 2.25, middle: 3, right: 3.75 };
    const edges = { leftBoundary: 0, rightBoundary: 6, centerline: 3 };
    // Provide visible boundaries down to the player for every tested side combination.
    for (const side of ['leftBoundary', 'rightBoundary', 'centerline'] as const) {
        detection.birdsEyeScene![side] = detection.birdsEyeScene![side].map((points) =>
            points.map((point) => ({ ...point, x: edges[side] - offset[player], y: point.y - 8 })));
    }
    detection.birdsEyeScene!.cars[0].position = { x: offset[opponent] - offset[player], y, z: 0 };
    return detection;
};

describe('One Tight Slow Corner', () => {
    describe('Inside the Corner', () => {
        const path = ['root', rule.name, 'Inside the Corner'];
        const parent = ['root', rule.name];
        const drivingAt = (position: number, opponent = position + 0.001): StandardTelemetrySample => ({
            ...sample(3), Graphics_normalized_car_position: position,
            Graphics_normalized_positions: { 1: position, 2: opponent },
        });
        const setup = () => {
            const engine = new PhraseEngine([rule]);
            const map = slowCorner();
            engine.receiveMap(map, 0);
            expect(engine.receiveTelemetry(frame(sample(3)), 0).state.path).toEqual(parent);
            return { engine, map };
        };

        it('enters at the corner boundary, stays through the corner, and exits to the parent', () => {
            const { engine } = setup();
            const approaching = engine.receiveTelemetry(frame(drivingAt(0.124, 0.15)), 100);
            expect(approaching.mapContext.phase).toBe('entry');
            expect(approaching.state.path).toEqual(parent);
            const entered = engine.receiveTelemetry(frame(drivingAt(0.125)), 200);
            expect(entered.state.path).toEqual(path);
            expect(entered.closures[0].status).toBe('Active');
            expect(entered.root.children[0].children[1]).toMatchObject({
                id: 'one-tight-slow-corner:inside-corner', kind: 'closure', current: true,
                conditions: [{ description: 'Driver is inside a corner', conditionFit: true, inputMissing: false }],
                children: [{ name: 'exit to parent', conditions: [{
                    description: 'Driver is no longer inside that corner', conditionFit: false,
                }] }],
            });
            expect(engine.receiveTelemetry(frame(drivingAt(0.18)), 300).state.path).toEqual(path);
            expect(engine.receiveTelemetry(frame(drivingAt(0.225)), 400).state.path).toEqual(path);
            const exited = engine.receiveTelemetry(frame(drivingAt(0.226)), 500);
            expect(exited.state.path).toEqual(parent);
            expect(exited.events).toEqual([]);
            engine.receiveVision(closeVision(600, 'right', 'middle'), 600);
            expect(engine.receiveTelemetry(frame(sample(3)), 600).events).toEqual([
                { id: 1, ruleId: rule.id, sentence: defending, timestamp: 600 },
            ]);
        });

        it('exits the entered corner even when another overlaps it, then supports repeated visits', () => {
            const { engine, map } = setup();
            map.centerline_segments!.push({ id: 'next-corner', tags: ['corner'], start_position: 0.2, end_position: 0.3 });
            engine.receiveMap({ ...map }, 0);
            expect(engine.receiveTelemetry(frame(drivingAt(0.15)), 100).state.path).toEqual(path);
            expect(engine.receiveTelemetry(frame(drivingAt(0.21)), 200).state.path).toEqual(path);
            expect(engine.receiveTelemetry(frame(drivingAt(0.226)), 300).state.path).toEqual(parent);
            expect(engine.receiveTelemetry(frame(drivingAt(0.226)), 400).state.path).toEqual(path);
            expect(engine.receiveTelemetry(frame(drivingAt(0.301)), 500).state.path).toEqual(parent);
            expect(engine.receiveTelemetry(frame(drivingAt(0.15)), 600).state.path).toEqual(path);
            expect(engine.receiveTelemetry(frame(drivingAt(0.124)), 700).state.path).toEqual(parent);
        });

        it('stays in a corner across the start/finish line', () => {
            const { engine, map } = setup();
            map.centerline_segments = [{ id: 'wrap-corner', tags: ['slow corner'], start_position: 0.95, end_position: 0.05 }];
            engine.receiveMap({ ...map }, 0);
            expect(engine.receiveTelemetry(frame(drivingAt(0.95)), 100).state.path).toEqual(path);
            expect(engine.receiveTelemetry(frame(drivingAt(0)), 200).state.path).toEqual(path);
            expect(engine.receiveTelemetry(frame(drivingAt(0.05)), 300).state.path).toEqual(path);
            expect(engine.receiveTelemetry(frame(drivingAt(0.051)), 400).state.path).toEqual(parent);
        });

        it.each(['stale telemetry', 'missing position', 'missing map', 'wrong map', 'paused'])(
            'waits for known position instead of treating %s as an exit', (missing) => {
                const { engine, map } = setup();
                engine.receiveTelemetry(frame(drivingAt(0.15)), 0);
                const now = TELEMETRY_MAX_AGE_MS + 1;
                const unavailable = missing === 'missing map' ? engine.receiveMap(null, 100)
                    : missing === 'wrong map' ? engine.receiveMap({ ...map, game: 'iracing' }, 100)
                        : missing === 'missing position' ? engine.receiveTelemetry(frame({ ...drivingAt(0.15), Graphics_normalized_car_position: undefined }), 100)
                            : missing === 'paused' ? engine.receiveTelemetry(frame(drivingAt(0.15), 1), 100)
                                : engine.evaluate(now);
                expect(unavailable.state.path).toEqual(path);
                expect(unavailable.root.children[0].children[1].children[0].conditions[0]).toMatchObject({
                    conditionFit: false, inputMissing: true,
                });
                engine.receiveMap(map, now);
                expect(engine.receiveTelemetry(frame(drivingAt(0.226)), now).state.path).toEqual(parent);
            },
        );

        it('lets a completed speech action exit to root when the driver enters the corner', () => {
            const { engine } = setup();
            engine.receiveVision(closeVision(100, 'right', 'middle'), 100);
            const spoken = engine.receiveTelemetry(frame(sample(3)), 100);
            expect(spoken.state.path).toEqual([...parent, 'say phrase']);
            const exited = engine.receiveTelemetry(frame(drivingAt(0.15)), 200);
            expect(exited.state.path).toEqual(['root']);
            expect(exited.events).toEqual(spoken.events);
        });
    });

    describe('Inside the Long Straight', () => {
        const path = ['root', rule.name, 'Inside the Long Straight'];
        const drivingAt = (position: number): StandardTelemetrySample => ({
            ...sample(3), Graphics_normalized_car_position: position,
            Graphics_normalized_positions: { 1: position, 2: 0.12 },
        });
        const setup = (tag = 'long straight') => {
            const engine = new PhraseEngine([rule]);
            const map = slowCorner();
            map.centerline_segments!.push({ id: 'long-straight', tags: [tag], start_position: 0.04, end_position: 0.115 });
            engine.receiveMap(map, 0);
            return { engine, map };
        };

        const exitSetup = () => {
            const { engine, map } = setup();
            map.centerline_segments!.push({ id: 'exit-straight', tags: ['long straight'], start_position: 0.225, end_position: 0.6 });
            engine.receiveMap({ ...map }, 0);
            return { engine, map };
        };

        it('prioritizes exit-speed advice when all three conditions match and speaks once per straight visit', () => {
            const { engine } = exitSetup();
            const history = exitHistory();
            // The existing inside-overtake action also matches; exit preparation takes priority.
            engine.receiveVision(closeVision(0, 'left', 'left', 2), 0);
            const spoken = engine.receiveTelemetry(frame(drivingAt(0.11)), 0, history);
            expect(spoken.events).toEqual([{ id: 1, ruleId: rule.id, sentence: exitAdvice, timestamp: 0 }]);
            expect(spoken.state.path).toEqual([...path, 'say phrase']);
            expect(spoken.root.children[0].children[0].children[1].conditions.map((condition) => condition.conditionFit))
                .toEqual([true, true, true]);
            expect(engine.receiveTelemetry(frame(drivingAt(0.112)), 100, history).events).toEqual(spoken.events);
            expect(engine.receiveTelemetry(frame(drivingAt(0.116)), 200, history).state.path).toEqual(['root', rule.name]);
        });

        it.each([
            { reason: 'player slower', player: [0.01, 0.015, 0.02], opponent: [0.02, 0.03, 0.04] },
            { reason: 'equal speeds', player: [0.01, 0.02, 0.03], opponent: [0.01, 0.02, 0.03] },
            { reason: 'too few player exits', player: [0.03, 0.04], opponent: [0.01, 0.02, 0.03] },
            { reason: 'too few opponent exits', player: [0.02, 0.03, 0.04], opponent: [0.01, 0.02] },
        ])('withholds exit advice for $reason', ({ player, opponent }) => {
            const { engine } = exitSetup();
            engine.receiveVision(closeVision(0, 'middle', 'middle'), 0);
            expect(engine.receiveTelemetry(frame(drivingAt(0.11)), 0, exitHistory(player, opponent)).events).toEqual([]);
        });

        it('uses trimmed speeds instead of being fooled by a fast outlier', () => {
            const { engine } = exitSetup();
            engine.receiveVision(closeVision(0, 'middle', 'middle'), 0);
            expect(engine.receiveTelemetry(frame(drivingAt(0.11)), 0, exitHistory([0.01, 0.015, 0.3])).events).toEqual([]);
        });

        it.each(['distant', 'missing vision', 'stale vision', 'no exit straight', 'ordinary exit straight',
            'wrong corner', 'different opponent', 'outside approach straight', 'paused', 'wrong map'])(
            'withholds exit advice with %s', (reason) => {
                const { engine, map } = exitSetup();
                let history = exitHistory();
                let telemetry = drivingAt(0.11);
                if (reason !== 'missing vision') engine.receiveVision(closeVision(0, 'middle', 'middle', reason === 'distant' ? 2.01 : 1), 0);
                if (reason === 'no exit straight') map.centerline_segments!.pop();
                if (reason === 'ordinary exit straight') map.centerline_segments![2].tags = ['straight'];
                if (reason === 'wrong map') map.game = 'iracing';
                if (reason === 'wrong corner') history = history.map((summary) => ({ ...summary, cornerId: 'other-corner' }));
                if (reason === 'different opponent') telemetry = { ...telemetry, Graphics_normalized_positions: { 1: 0.11, 3: 0.12, 2: 0.13 } };
                if (reason === 'outside approach straight') telemetry = drivingAt(0.03);
                engine.receiveMap({ ...map }, 0);
                const now = reason === 'stale vision' ? VISION_MAX_AGE_MS + 1 : 0;
                expect(engine.receiveTelemetry(frame(telemetry, reason === 'paused' ? 1 : 2), now, history).events).toEqual([]);
            },
        );

        it('waits for event-log history and only emits on a fresh telemetry frame', () => {
            const { engine } = exitSetup();
            engine.receiveVision(closeVision(0, 'middle', 'middle'), 0);
            const waiting = engine.receiveTelemetry(frame(drivingAt(0.11)), 0);
            expect(waiting.events).toEqual([]);
            expect(waiting.root.children[0].children[0].children[1].conditions[0].inputMissing).toBe(true);
            expect(engine.evaluate(100).events).toEqual([]);
            expect(engine.receiveTelemetry(frame(drivingAt(0.11)), 100, exitHistory()).events[0].sentence).toBe(exitAdvice);
            engine.reset(200);
            engine.receiveVision(closeVision(200, 'middle', 'middle'), 200);
            expect(engine.receiveTelemetry(frame(drivingAt(0.11)), 200).events).toEqual([]);
        });

        it('speaks inside the straight once, stays through corner approach, and exits to the parent', () => {
            const { engine } = setup();
            const entered = engine.receiveTelemetry(frame(drivingAt(0.05)), 0);
            expect(entered.state.path).toEqual(path);
            expect(entered.events).toEqual([]);
            expect(entered.closures[0].status).toBe('Active');
            expect(entered.root.children[0].children[0]).toMatchObject({
                kind: 'closure', name: 'Inside the Long Straight', current: true,
                conditions: [{ description: 'Driver is inside a long straight', conditionFit: true, inputMissing: false }],
                children: [{ name: 'exit to parent', conditions: [{
                    description: 'Driver is no longer inside that long straight', conditionFit: false,
                }] }, { name: 'say phrase', description: exitAdvice }, { name: 'say phrase', description: slipstream }],
            });
            expect(entered.root.children[0].children.some((child) => child.description === slipstream)).toBe(false);
            expect(engine.receiveVision(closeVision(100), 100).events).toEqual([]);
            const approaching = engine.receiveTelemetry(frame(drivingAt(0.11)), 100);
            expect(approaching.mapContext.phase).toBe('entry');
            expect(approaching.state.path).toEqual([...path, 'say phrase']);
            expect(approaching.events).toEqual([{ id: 1, ruleId: rule.id, sentence: slipstream, timestamp: 100 }]);
            const staying = engine.receiveTelemetry(frame(drivingAt(0.115)), 150);
            expect(staying.state.path).toEqual([...path, 'say phrase']);
            expect(staying.events).toEqual(approaching.events);
            const exited = engine.receiveTelemetry(frame(drivingAt(0.116)), 200);
            expect(exited.state.path).toEqual(['root', rule.name]);
            expect(exited.events).toEqual(approaching.events);
            engine.receiveVision(closeVision(300, 'right', 'middle'), 300);
            const spoken = engine.receiveTelemetry(frame(drivingAt(0.117)), 300);
            expect(spoken.events).toEqual([...approaching.events, { id: 2, ruleId: rule.id, sentence: defending, timestamp: 300 }]);
            expect(engine.evaluate(300).state.path).toEqual(['root']);
        });

        it.each(['left', 'right'] as const)('keeps the position conditions for a %s turn inside the straight', (corner) => {
            for (const player of positions) {
                for (const opponent of positions) {
                    const engine = new PhraseEngine([rule]);
                    engine.receiveMap(slowCornerWithLongStraight(corner), 0);
                    engine.receiveVision(closeVision(0, player, opponent), 0);
                    const snapshot = engine.receiveTelemetry(frame(drivingAt(0.05)), 0);
                    const matches = opponent !== 'middle' && opponent !== corner;
                    expect(snapshot.state.path).toEqual([...path, ...(matches ? ['say phrase'] : [])]);
                    expect(snapshot.events).toEqual(matches ? [{ id: 1, ruleId: rule.id, sentence: slipstream, timestamp: 0 }] : []);
                }
            }
        });

        it('exits before speaking if the position conditions first match after leaving the straight', () => {
            const { engine } = setup();
            engine.receiveTelemetry(frame(drivingAt(0.05)), 0);
            engine.receiveVision(closeVision(100), 100);
            const exited = engine.receiveTelemetry(frame(drivingAt(0.116)), 100);
            expect(exited.state.path).toEqual(['root', rule.name]);
            expect(exited.events).toEqual([]);
            expect(engine.receiveTelemetry(frame(drivingAt(0.117)), 200).events).toEqual([]);
        });

        it('uses the driver position and requires the long straight tag', () => {
            const { engine } = setup();
            const outside = { ...drivingAt(0.03), Graphics_normalized_positions: { 1: 0.03, 2: 0.08 } };
            engine.receiveVision(closeVision(0), 0);
            expect(engine.receiveTelemetry(frame(outside), 0).state.path).toEqual(['root', rule.name]);
            expect(engine.receiveTelemetry(frame(drivingAt(0.05)), 100).state.path).toEqual([...path, 'say phrase']);
            const ordinary = setup('straight').engine;
            ordinary.receiveVision(closeVision(0), 0);
            expect(ordinary.receiveTelemetry(frame(drivingAt(0.05)), 0).state.path).toEqual(['root', rule.name]);
        });

        it('exits the entered straight even when another overlaps it, then supports repeated visits', () => {
            const { engine, map } = setup();
            map.centerline_segments!.push({ id: 'next-straight', tags: ['long straight'], start_position: 0.1, end_position: 0.12 });
            engine.receiveMap({ ...map }, 0);
            expect(engine.receiveTelemetry(frame(drivingAt(0.05)), 0).state.path).toEqual(path);
            expect(engine.receiveTelemetry(frame(drivingAt(0.11)), 100).state.path).toEqual(path);
            expect(engine.receiveTelemetry(frame(drivingAt(0.116)), 200).state.path).toEqual(['root', rule.name]);
            expect(engine.receiveTelemetry(frame(drivingAt(0.116)), 300).state.path).toEqual(path);
            expect(engine.receiveTelemetry(frame(drivingAt(0.121)), 400).state.path).toEqual(['root', rule.name]);
            expect(engine.receiveTelemetry(frame(drivingAt(0.05)), 500).state.path).toEqual(path);
            expect(engine.receiveTelemetry(frame(drivingAt(0.03)), 600).state.path).toEqual(['root', rule.name]);
        });

        it.each(['stale telemetry', 'missing position', 'missing map', 'wrong map', 'paused'])(
            'waits for known position instead of treating %s as an exit', (missing) => {
                const { engine, map } = setup();
                engine.receiveTelemetry(frame(drivingAt(0.05)), 0);
                const now = TELEMETRY_MAX_AGE_MS + 1;
                engine.receiveVision(closeVision(missing === 'stale telemetry' ? now : 100), missing === 'stale telemetry' ? now : 100);
                const unavailable = missing === 'missing map' ? engine.receiveMap(null, 100)
                    : missing === 'wrong map' ? engine.receiveMap({ ...map, game: 'iracing' }, 100)
                        : missing === 'missing position' ? engine.receiveTelemetry(frame({ ...drivingAt(0.05), Graphics_normalized_car_position: undefined }), 100)
                            : missing === 'paused' ? engine.receiveTelemetry(frame(drivingAt(0.05), 1), 100)
                                : engine.evaluate(now);
                expect(unavailable.state.path).toEqual(path);
                expect(unavailable.events).toEqual([]);
                expect(unavailable.root.children[0].children[0].children[0].conditions[0]).toMatchObject({
                    conditionFit: false, inputMissing: true,
                });
                engine.receiveMap(map, now);
                expect(engine.receiveTelemetry(frame(drivingAt(0.116)), now).state.path).toEqual(['root', rule.name]);
            },
        );
    });

    it.each([
        { corner: 'left', opponent: 'left', expected: [null, null, null] },
        { corner: 'left', opponent: 'middle', expected: [defending, null, null] },
        { corner: 'left', opponent: 'right', expected: [null, null, null] },
        { corner: 'right', opponent: 'left', expected: [null, null, null] },
        { corner: 'right', opponent: 'middle', expected: [null, null, defending] },
        { corner: 'right', opponent: 'right', expected: [null, null, null] },
    ].flatMap(({ corner, opponent, expected }) => positions.map((player, index) => ({
        corner: corner as CornerDirection, opponent: opponent as TrackPosition, player, sentence: expected[index],
    }))))('selects the parent phrase outside the long straight for a $corner turn with opponent $opponent and driver $player', ({ corner, opponent, player, sentence }) => {
        const engine = new PhraseEngine();
        engine.receiveMap(slowCorner(corner), 0);
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

    it.each((['left', 'right'] as const).flatMap((corner) => [1.99, 2, 2.01, 10].map((distance) => ({ corner, distance }))))(
        'requires an opponent within 2 m before speaking for a $corner turn ($distance m)', ({ corner, distance }) => {
            const engine = new PhraseEngine([rule]);
            const opponent = corner === 'left' ? 'right' : 'left';
            engine.receiveMap(slowCornerWithLongStraight(corner), 0);
            engine.receiveVision(closeVision(1000, opponent, opponent, distance), 1000);
            const snapshot = engine.receiveTelemetry(frame(sample(3)), 1000);
            expect(snapshot.closures[0].conditions.map((condition) => ({ description: 'description' in condition ? condition.description : '',
                connector: condition.connector, conditionFit: condition.conditionFit }))).toEqual([
                { description: 'Upcoming mapped corner speed = slow', connector: 'and', conditionFit: true },
                { description: 'Corner is not inside a consecutive corners label of the Live Map', connector: 'and', conditionFit: true },
            ]);
            expect(snapshot.state.path).toEqual(['root', rule.name, 'Inside the Long Straight', ...(distance <= 2 ? ['say phrase'] : [])]);
            expect(snapshot.events).toEqual(distance <= 2 ? [
                { id: 1, ruleId: rule.id, sentence: slipstream, timestamp: 1000 },
            ] : []);
        },
    );

    it.each(['missing map', 'missing corner', 'missing opponent positions', 'paused'])(
        'does not enter with %s', (reason) => {
            const engine = new PhraseEngine([rule]);
            const map = slowCorner();
            if (reason === 'missing corner') map.centerline_segments![0].tags = ['long straight'];
            engine.receiveMap(reason === 'missing map' ? null : map, 0);
            engine.receiveVision(closeVision(0), 0);
            const now = 1000;
            engine.receiveVision(closeVision(now), now);
            const next = reason === 'missing opponent positions'
                ? { ...sample(2), Graphics_normalized_positions: undefined } : sample(2);
            const snapshot = engine.receiveTelemetry(frame(next, reason === 'paused' ? 1 : 2), now);
            expect(snapshot.events).toEqual([]);
            expect(snapshot.state.path).toEqual(['root']);
        },
    );

    it.each(['missing', 'stale'])('enters with %s vision but waits for fresh positions before speaking', (visionStatus) => {
        const engine = new PhraseEngine([rule]);
        engine.receiveMap(slowCornerWithLongStraight(), 0);
        if (visionStatus === 'stale') engine.receiveVision(closeVision(0), 0);
        const now = VISION_MAX_AGE_MS + 1;
        const waiting = engine.receiveTelemetry(frame(sample(3)), now);
        expect(waiting.state.path).toEqual(['root', rule.name, 'Inside the Long Straight']);
        expect(waiting.closures[0].status).toBe('Active');
        expect(waiting.events).toEqual([]);
        expect(engine.receiveVision(closeVision(now), now).events).toEqual([]);
        expect(engine.receiveTelemetry(frame(sample(3)), now).events).toEqual([
            { id: 1, ruleId: rule.id, sentence: slipstream, timestamp: now },
        ]);
    });

    it.each(['slow', 'slow corner'])('enters a standalone corner labeled %s with more than 2 s until entry', (label) => {
        const engine = new PhraseEngine([rule]);
        const map = slowCornerWithLongStraight();
        map.centerline_segments![0].tags = label === 'slow' ? ['corner', 'slow'] : ['slow corner'];
        // A label elsewhere on the lap must not exclude this corner.
        map.centerline_segments!.push({ id: 'other-sequence', tags: ['consecutive corners'], start_position: 0.4, end_position: 0.6 });
        engine.receiveMap(map, 0);
        engine.receiveTelemetry(frame(sample(4)), 0);
        engine.receiveVision(closeVision(1000), 1000);
        expect(engine.receiveTelemetry(frame(sample(3)), 1000).events).toEqual([
            { id: 1, ruleId: rule.id, sentence: slipstream, timestamp: 1000 },
        ]);
    });

    it.each([
        { label: 'fast', tags: ['corner', 'fast'] },
        { label: 'fast corner', tags: ['fast corner'] },
        { label: 'unlabeled', tags: ['corner'] },
        { label: 'conflicting', tags: ['corner', 'slow', 'fast'] },
    ])('does not enter when the next corner has $label speed tags, even with a later slow corner', ({ tags }) => {
        const engine = new PhraseEngine([rule]);
        const map = slowCorner();
        map.centerline_segments![0].tags = tags;
        map.centerline_segments!.push({ id: 'later-slow', tags: ['slow corner'], start_position: 0.4, end_position: 0.5 });
        engine.receiveMap(map, 0);
        engine.receiveTelemetry(frame(sample(3)), 0);
        engine.receiveVision(closeVision(1000), 1000);
        const snapshot = engine.receiveTelemetry(frame(sample(2)), 1000);
        expect(snapshot.events).toEqual([]);
        expect(snapshot.state.path).toEqual(['root']);
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
