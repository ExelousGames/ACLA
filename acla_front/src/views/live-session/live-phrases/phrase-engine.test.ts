import { createLiveTelemetryStore, LiveTelemetryFrameEvent } from '../live-telemetry-store';
import type { StandardTelemetrySample } from '../live-session-types';
import { COOLDOWN_MS, PHRASE_RULES, PhraseCondition, PhraseConditionGroup, PhraseEngine, TELEMETRY_MAX_AGE_MS, conditionGroup, type PhraseConditionConnector, type PhraseConditionNode } from './phrase-engine';
import { VISION_MAX_AGE_MS } from '../track-vision/track-vision-types';
import { circuitMap, shapedCornerMap, vision } from './test-fixtures';

const frame = (sample: StandardTelemetrySample, status = 2): LiveTelemetryFrameEvent => ({
    type: 'frame', sample, sampleIndex: 0, telemetryStatus: status,
    committedSampleCount: 0, sessionGeneration: 0, streamGeneration: 0,
    update: { type: 'frame', game: 'acc', sample, sequence: 1, committedSequence: 0, committedCount: 0 },
});
const driving = { Physics_speed_kmh: 100, Graphics_normalized_car_position: 0.11 };
const map = circuitMap();
const defaultRule = 'inside-outbraking';
const ids = (engine: PhraseEngine, now: number) => engine.evaluate(now).events.map((event) => event.ruleId);
const update = (engine: PhraseEngine, now: number, sample: StandardTelemetrySample = driving) => {
    engine.receiveVision(vision(now), now);
    return engine.receiveTelemetry(frame(sample), now);
};
const straightDriving = { ...driving, Graphics_normalized_car_position: 0.5 };
const slipstreamVision = (capturedAt: number, x = 2, y = 9) => {
    const detection = vision(capturedAt, { corner: 'straight' });
    detection.birdsEyeScene!.cars[0].position = { x, y, z: 0 };
    return detection;
};

describe('condition connectors', () => {
    const ruleWith = (conditions: readonly PhraseCondition[]) => ({ ...PHRASE_RULES[0], conditions });

    describe.each([
        { expression: 'A AND B AND C', connectors: ['and', 'and'], matches: (a: boolean, b: boolean, c: boolean) => a && b && c },
        { expression: 'A OR B OR C', connectors: ['or', 'or'], matches: (a: boolean, b: boolean, c: boolean) => a || b || c },
        { expression: 'A AND B OR C', connectors: ['and', 'or'], matches: (a: boolean, b: boolean, c: boolean) => (a && b) || c },
        { expression: 'A OR B AND C', connectors: ['or', 'and'], matches: (a: boolean, b: boolean, c: boolean) => a || (b && c) },
    ] as const)('$expression', ({ connectors, matches }) => {
        it.each([
            [false, false, false], [false, false, true], [false, true, false], [false, true, true],
            [true, false, false], [true, false, true], [true, true, false], [true, true, true],
        ])('evaluates A=%s B=%s C=%s with the configured hold', (a, b, c) => {
            const conditions = [a, b, c].map((met, index) => new PhraseCondition(
                'speed', '>=', met ? 30 : 120, undefined, index === 0 ? 'and' : connectors[index - 1],
            ));
            const rule = ruleWith(conditions);
            const engine = new PhraseEngine([rule]);
            const started = engine.receiveTelemetry(frame(driving), 0);
            expect(started.rules[0].status).toBe(matches(a, b, c) ? 'Confirming' : 'Not matched');
            expect(engine.receiveTelemetry(frame(driving), 799).events).toEqual([]);
            expect(engine.receiveTelemetry(frame(driving), 800).events.map((event) => event.ruleId))
                .toEqual(matches(a, b, c) ? [rule.id] : []);
            expect(started.rules[0].conditions.map((condition) => condition.connector)).toEqual(['and', ...connectors]);
        });
    });

    it.each(['and', 'or'] as const)('handles a missing alternative joined by %s', (connector: PhraseConditionConnector) => {
        const engine = new PhraseEngine([ruleWith([
            new PhraseCondition('carAhead', '=', 1),
            new PhraseCondition('speed', '>=', 30, undefined, connector),
        ])]);
        const before = engine.receiveTelemetry(frame(driving), 0);
        const after = engine.receiveTelemetry(frame(driving), 800);
        expect(after.rules[0].status).toBe(connector === 'or' ? 'Active' : 'Missing input');
        expect(after.rules[0].missing).toEqual(connector === 'or' ? [] : ['Opponent ahead on visible track']);
        expect(after.rules[0].conditions[0]).toMatchObject({ inputMissing: true, conditionFit: false });
        expect(before.rules[0].conditions[1].connector).toBe(connector);
        const unmatched = engine.receiveTelemetry(frame({ Physics_speed_kmh: 0 }), 900);
        expect(unmatched.rules[0].status).toBe('Missing input');
        expect(before.rules[0].conditions[1].conditionFit).toBe(true);
    });

    it('keeps a later missing OR alternative visible without blocking a matching rule', () => {
        const engine = new PhraseEngine([ruleWith([
            new PhraseCondition('speed', '>=', 30),
            new PhraseCondition('carAhead', '=', 1, undefined, 'or'),
        ])]);
        engine.receiveTelemetry(frame(driving), 0);
        const snapshot = engine.receiveTelemetry(frame(driving), 800);
        expect(snapshot.rules[0]).toMatchObject({ status: 'Active', missing: [] });
        expect(snapshot.rules[0].conditions[1]).toMatchObject({ inputMissing: true, conditionFit: false, connector: 'or' });
    });

    it.each(['and', 'or'] as const)('ignores the first connector (%s) and requires live telemetry', (connector) => {
        const engine = new PhraseEngine([ruleWith([new PhraseCondition('carAhead', '=', 1, undefined, connector)])]);
        engine.receiveVision(vision(0), 0);
        expect(engine.evaluate(0, true).events).toEqual([]);
        engine.receiveTelemetry(frame(driving), 0);
        expect(engine.receiveTelemetry(frame(driving), 800).rules[0].status).toBe('Active');
        const expired = engine.receiveVision(vision(2400), 2400);
        expect(expired.rules[0].conditions[0].conditionFit).toBe(true);
        expect(expired.rules[0].status).toBe('Missing input');
        expect(engine.receiveTelemetry(frame(driving, 1), 2500).rules[0].status).toBe('Missing input');
    });

    it('requires at least one condition to match', () => {
        const engine = new PhraseEngine([ruleWith([])]);
        engine.receiveTelemetry(frame(driving), 0);
        expect(engine.receiveTelemetry(frame(driving), 800).events).toEqual([]);
    });
});

describe('nested condition groups', () => {
    const leaf = (met: boolean, connector: PhraseConditionConnector = 'and') => new PhraseCondition('speed', '>=', met ? 30 : 120, undefined, connector);
    const ruleWith = (conditions: readonly PhraseConditionNode[]) => ({ ...PHRASE_RULES[0], conditions });

    describe.each([
        {
            expression: 'A OR (B AND C)',
            conditions: (a: boolean, b: boolean, c: boolean) => [leaf(a), conditionGroup([leaf(b), leaf(c)], 'or')],
            matches: (a: boolean, b: boolean, c: boolean) => a || (b && c),
        },
        {
            expression: '(A OR B) AND C',
            conditions: (a: boolean, b: boolean, c: boolean) => [conditionGroup([leaf(a), leaf(b, 'or')]), leaf(c)],
            matches: (a: boolean, b: boolean, c: boolean) => (a || b) && c,
        },
        {
            expression: 'A AND (B OR C)',
            conditions: (a: boolean, b: boolean, c: boolean) => [leaf(a), conditionGroup([leaf(b), leaf(c, 'or')])],
            matches: (a: boolean, b: boolean, c: boolean) => a && (b || c),
        },
        {
            expression: '(A OR (B AND C))',
            conditions: (a: boolean, b: boolean, c: boolean) => [conditionGroup([leaf(a), conditionGroup([leaf(b), leaf(c)], 'or')])],
            matches: (a: boolean, b: boolean, c: boolean) => a || (b && c),
        },
        {
            expression: 'A AND (B OR (C AND true))',
            conditions: (a: boolean, b: boolean, c: boolean) => [leaf(a), conditionGroup([leaf(b), conditionGroup([leaf(c), leaf(true)], 'or')])],
            matches: (a: boolean, b: boolean, c: boolean) => a && (b || c),
        },
        {
            expression: '(A OR B AND C)',
            conditions: (a: boolean, b: boolean, c: boolean) => [conditionGroup([leaf(a), leaf(b, 'or'), leaf(c)])],
            matches: (a: boolean, b: boolean, c: boolean) => a || (b && c),
        },
    ])('$expression', ({ conditions, matches }) => {
        it.each([
            [false, false, false], [false, false, true], [false, true, false], [false, true, true],
            [true, false, false], [true, false, true], [true, true, false], [true, true, true],
        ])('evaluates A=%s B=%s C=%s before applying hold and emitting', (a, b, c) => {
            const rule = ruleWith(conditions(a, b, c));
            const engine = new PhraseEngine([rule]);
            expect(engine.receiveTelemetry(frame(driving), 0).rules[0].status).toBe(matches(a, b, c) ? 'Confirming' : 'Not matched');
            expect(engine.receiveTelemetry(frame(driving), 799).events).toEqual([]);
            expect(engine.receiveTelemetry(frame(driving), 800).events.map((event) => event.ruleId)).toEqual(matches(a, b, c) ? [rule.id] : []);
        });
    });

    it('ignores the first connector independently at every nesting level', () => {
        const engine = new PhraseEngine([ruleWith([
            conditionGroup([conditionGroup([leaf(true, 'or')], 'or'), leaf(false)], 'or'),
        ])]);
        expect(engine.receiveTelemetry(frame(driving), 0).rules[0].status).toBe('Not matched');
        expect(engine.receiveTelemetry(frame(driving), 800).events).toEqual([]);
    });

    it('never treats empty groups as matching', () => {
        const engine = new PhraseEngine([ruleWith([conditionGroup([conditionGroup([])])])]);
        expect(engine.receiveTelemetry(frame(driving), 0).rules[0].status).toBe('Not matched');
        expect(engine.receiveTelemetry(frame(driving), 800).events).toEqual([]);
    });

    it.each(['and', 'or'] as const)('handles nested missing inputs joined by %s', (connector) => {
        const engine = new PhraseEngine([ruleWith([conditionGroup([
            new PhraseCondition('carAhead', '=', 1),
            conditionGroup([leaf(true)], connector),
        ])])]);
        engine.receiveTelemetry(frame(driving), 0);
        const snapshot = engine.receiveTelemetry(frame(driving), 800);
        expect(snapshot.rules[0]).toMatchObject({
            status: connector === 'or' ? 'Active' : 'Missing input',
            missing: connector === 'or' ? [] : ['Opponent ahead on visible track'],
            conditions: [{ conditionFit: connector === 'or', inputMissing: connector === 'and', conditions: [
                { conditionFit: false, inputMissing: true },
                { connector, conditionFit: true, inputMissing: false },
            ] }],
        });
    });

    it('does not report missing inputs from a satisfied group when a sibling fails', () => {
        const engine = new PhraseEngine([ruleWith([
            conditionGroup([new PhraseCondition('carAhead', '=', 1), leaf(true, 'or')]),
            leaf(false),
        ])]);
        expect(engine.receiveTelemetry(frame(driving), 0).rules[0]).toMatchObject({ status: 'Not matched', missing: [] });
    });

    it('evaluates every descendant and preserves past snapshots and the catalog', () => {
        const group = conditionGroup([leaf(true), conditionGroup([leaf(true), leaf(false)], 'or')]);
        const rule = ruleWith([group]);
        const engine = new PhraseEngine([rule]);
        const before = engine.receiveTelemetry(frame(driving), 0);
        const after = engine.receiveTelemetry(frame({ Physics_speed_kmh: 0 }), 100);
        const evaluated = before.rules[0].conditions[0] as PhraseConditionGroup;
        expect(evaluated).toBeInstanceOf(PhraseConditionGroup);
        expect(evaluated).not.toBe(group);
        expect(evaluated.conditions[0]).toBeInstanceOf(PhraseCondition);
        expect(evaluated.conditions[0]).not.toBe(group.conditions[0]);
        expect(evaluated.conditions[1]).toBeInstanceOf(PhraseConditionGroup);
        expect(evaluated).toMatchObject({ conditionFit: true, conditions: [
            { conditionFit: true },
            { connector: 'or', conditionFit: false, conditions: [{ conditionFit: true }, { conditionFit: false }] },
        ] });
        expect(after.rules[0].conditions[0]).toMatchObject({ conditionFit: false });
        expect(new PhraseEngine([rule]).evaluate(100).rules[0].conditions[0]).toMatchObject({ conditionFit: false, inputMissing: true });
        expect(group).toMatchObject({ conditionFit: false, inputMissing: true, conditions: [
            { conditionFit: false }, { conditionFit: false, conditions: [{ conditionFit: false }, { conditionFit: false }] },
        ] });
    });

    it('requires live telemetry even when nested vision conditions remain met', () => {
        const engine = new PhraseEngine([ruleWith([conditionGroup([conditionGroup([new PhraseCondition('carAhead', '=', 1)])])])]);
        engine.receiveVision(vision(0), 0);
        expect(engine.evaluate(0, true).events).toEqual([]);
        engine.receiveTelemetry(frame(driving), 0);
        expect(engine.receiveTelemetry(frame(driving), 800).rules[0].status).toBe('Active');
        const expired = engine.receiveVision(vision(2400), 2400);
        expect(expired.rules[0].conditions[0].conditionFit).toBe(true);
        expect(expired.rules[0].status).toBe('Missing input');
    });
});

describe('slipstream distance, motion and alignment', () => {
    it.each([
        { name: 'steady gap within 10 m', x: 2, distances: [9, 9, 9], emits: true },
        { name: 'exactly 10 m away', x: 6, distances: [8, 8, 8], emits: true },
        { name: 'offset to the left', x: -2, distances: [9, 9, 9], emits: true },
        { name: 'increasing gap', x: 2, distances: [8, 8.5, 9], emits: true },
        { name: 'more than 10 m away', x: 6, distances: [8.01, 8.01, 8.01], emits: false },
        { name: 'closing gap', x: 2, distances: [9.6, 9.4, 9.2], emits: false },
        { name: 'directly behind', x: 0, distances: [9, 9, 9], emits: false },
        { name: 'right alignment boundary', x: 1, distances: [9, 9, 9], emits: false },
        { name: 'left alignment boundary', x: -1, distances: [9, 9, 9], emits: false },
    ])('$name', ({ x, distances, emits }) => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        [0, 100, 900].forEach((now, index) => {
            engine.receiveVision(slipstreamVision(now, x, distances[index]), now);
            engine.receiveTelemetry(frame(straightDriving), now);
            expect(ids(engine, now)).toEqual(emits && now === 900 ? ['slipstream'] : []);
        });
    });

    it('requires two distinct captures before starting the hold, and only telemetry emits', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        const first = slipstreamVision(0);
        engine.receiveVision(first, 0);
        engine.receiveTelemetry(frame(straightDriving), 0);
        engine.receiveVision(first, 800);
        const waiting = engine.receiveTelemetry(frame(straightDriving), 800);
        expect(waiting.events).toEqual([]);
        expect(waiting.rules.find((rule) => rule.id === 'slipstream')!.conditions
            .find((condition): condition is PhraseCondition => condition instanceof PhraseCondition && condition.input === 'closingOnOpponent')).toMatchObject({ inputMissing: true, conditionFit: false });
        engine.receiveVision(slipstreamVision(900), 900);
        engine.receiveTelemetry(frame(straightDriving), 900);
        engine.receiveVision(slipstreamVision(1699), 1699);
        engine.receiveTelemetry(frame(straightDriving), 1699);
        expect(ids(engine, 1699)).toEqual([]);
        engine.receiveVision(slipstreamVision(1700), 1700);
        expect(ids(engine, 1700)).toEqual([]);
        engine.receiveTelemetry(frame(straightDriving), 1700);
        expect(ids(engine, 1700)).toEqual(['slipstream']);
    });

    it('does not turn duplicate or out-of-order captures into evidence of a steady gap', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        engine.receiveVision(slipstreamVision(0, 2, 9.5), 0);
        engine.receiveTelemetry(frame(straightDriving), 0);
        engine.receiveVision(slipstreamVision(100, 2, 9), 100);
        engine.receiveTelemetry(frame(straightDriving), 100);
        engine.receiveVision(slipstreamVision(100, 2, 9), 200);
        engine.receiveVision(slipstreamVision(50, 2, 8.5), 300);
        const snapshot = engine.receiveTelemetry(frame(straightDriving), 1000);
        expect(snapshot.events).toEqual([]);
        expect(snapshot.rules.find((rule) => rule.id === 'slipstream')!.conditions
            .find((condition): condition is PhraseCondition => condition instanceof PhraseCondition && condition.input === 'closingOnOpponent')).toMatchObject({ inputMissing: false, conditionFit: false });
    });

    it.each(['stopped', 'missing scene', 'missing opponent', 'pack', 'calibration', 'expired', 'future timestamp', 'reset'])
    ('requires new motion evidence after %s', (change) => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        engine.receiveVision(slipstreamVision(0), 0);
        engine.receiveTelemetry(frame(straightDriving), 0);
        engine.receiveVision(slipstreamVision(100), 100);
        engine.receiveTelemetry(frame(straightDriving), 100);
        const interrupted = slipstreamVision(200);
        if (change === 'missing scene') interrupted.birdsEyeScene = null;
        if (change === 'missing opponent') interrupted.birdsEyeScene!.cars = [];
        if (change === 'pack') interrupted.birdsEyeScene!.cars[0].pack = true;
        if (change === 'calibration') interrupted.calibration = { ...interrupted.calibration!, lateralOffsetM: 0.5 };
        if (change === 'future timestamp') interrupted.capturedAt = 5000;
        if (change === 'reset') engine.reset(200);
        else if (change !== 'expired') engine.receiveVision(change === 'stopped' ? null : interrupted, 200);
        const now = change === 'expired' ? VISION_MAX_AGE_MS + 200 : 300;
        engine.receiveVision(slipstreamVision(now), now);
        const waiting = engine.receiveTelemetry(frame(straightDriving), now);
        expect(waiting.rules.find((rule) => rule.id === 'slipstream')!.conditions
            .find((condition): condition is PhraseCondition => condition instanceof PhraseCondition && condition.input === 'closingOnOpponent')).toMatchObject({ inputMissing: true });
        expect(ids(engine, now)).toEqual([]);
        engine.receiveVision(slipstreamVision(now + 100), now + 100);
        engine.receiveTelemetry(frame(straightDriving), now + 100);
        engine.receiveVision(slipstreamVision(now + 899), now + 899);
        engine.receiveTelemetry(frame(straightDriving), now + 899);
        expect(ids(engine, now + 899)).toEqual([]);
        engine.receiveVision(slipstreamVision(now + 900), now + 900);
        engine.receiveTelemetry(frame(straightDriving), now + 900);
        expect(ids(engine, now + 900)).toEqual(['slipstream']);
    });
});

describe('vision and Live Map overtaking guides', () => {
    it('uses the published BEV when depth-based geometry is unavailable', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        const detection = { ...vision(0), geometry: null };
        engine.receiveVision(detection, 0);
        const snapshot = engine.receiveTelemetry(frame(driving), 0);
        expect(snapshot.visionReady).toBe(true);
        const conditions = snapshot.rules.find((rule) => rule.id === defaultRule)!.conditions;
        for (const input of ['carAhead', 'playerCorner', 'playerPosition', 'opponentCorner', 'opponentPosition']) {
            expect(conditions.find((condition): condition is PhraseCondition => condition instanceof PhraseCondition && condition.input === input)).toMatchObject({ conditionFit: true, inputMissing: false });
        }
        engine.receiveTelemetry(frame(driving), 800);
        expect(ids(engine, 800)).toEqual([defaultRule]);
    });

    it('reports individual fits even when another condition fails or has missing input', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        const belowThreshold = update(engine, 0, { ...driving, Physics_speed_kmh: 29 });
        const conditions = belowThreshold.rules.find((rule) => rule.id === defaultRule)!.conditions;
        expect(conditions.find((condition): condition is PhraseCondition => condition instanceof PhraseCondition && condition.input === 'speed')).toMatchObject({ conditionFit: false, inputMissing: false });
        expect(conditions.filter((condition) => condition instanceof PhraseCondition && condition.input !== 'speed').every((condition) => condition.conditionFit)).toBe(true);

        engine.receiveVision(null, 100);
        const missingVision = engine.receiveTelemetry(frame({ ...driving, Physics_speed_kmh: 30 }), 100);
        const current = missingVision.rules.find((rule) => rule.id === defaultRule)!;
        expect(current.status).toBe('Missing input');
        expect(current.conditions.find((condition): condition is PhraseCondition => condition instanceof PhraseCondition && condition.input === 'speed')).toMatchObject({ conditionFit: true, inputMissing: false });
        expect(current.conditions.find((condition): condition is PhraseCondition => condition instanceof PhraseCondition && condition.input === 'carAhead')).toMatchObject({ conditionFit: false, inputMissing: true });
        expect(current.conditions.find((condition): condition is PhraseCondition => condition instanceof PhraseCondition && condition.input === 'phase')).toMatchObject({ conditionFit: true, inputMissing: false });
    });

    it('reports known track positions separately from missing turn directions', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        engine.receiveVision(vision(0, { corner: 'straight' }), 0);
        engine.receiveTelemetry(frame(driving), 0);
        const snapshot = engine.receiveTelemetry(frame(driving), 800);
        expect(snapshot.events).toEqual([]);
        const rule = snapshot.rules.find((rule) => rule.id === defaultRule)!;
        expect(rule.missing).toEqual(['Player turn corner', 'Opponent turn corner', 'Player inside, opponent off the inside']);
        for (const input of ['playerPosition', 'opponentPosition']) {
            expect(rule.conditions.find((condition): condition is PhraseCondition => condition instanceof PhraseCondition && condition.input === input)).toMatchObject({ conditionFit: true, inputMissing: false });
        }
        for (const input of ['playerCorner', 'opponentCorner']) {
            expect(rule.conditions.find((condition): condition is PhraseCondition => condition instanceof PhraseCondition && condition.input === input)).toMatchObject({ conditionFit: false, inputMissing: true });
        }
    });

    it('evaluates all conditions in guides suppressed by a higher-priority match', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(circuitMap('slow', true), 0);
        engine.receiveVision(vision(0, { player: 'outside', opponent: 'inside' }), 0);
        const snapshot = engine.receiveTelemetry(frame(driving), 0);
        expect(snapshot.rules.find((rule) => rule.id === 'next-corner')?.status).toBe('Confirming');
        const switchback = snapshot.rules.find((rule) => rule.id === 'switchback')!;
        expect(switchback.status).toBe('Not matched');
        expect(switchback.conditions.every((condition) => condition.conditionFit)).toBe(true);
    });

    it('keeps evaluated condition instances independent of past snapshots, other engines and the catalog', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        const before = update(engine, 0);
        const after = update(engine, 100, { ...driving, Physics_speed_kmh: 0 });
        const other = new PhraseEngine().evaluate(100);
        expect(before.rules.find((rule) => rule.id === defaultRule)!.conditions.every((condition) => condition.conditionFit)).toBe(true);
        expect(after.rules.find((rule) => rule.id === defaultRule)!.conditions.find((condition): condition is PhraseCondition => condition instanceof PhraseCondition && condition.input === 'speed')?.conditionFit).toBe(false);
        expect(other.rules.every((rule) => rule.conditions.every((condition) => !condition.conditionFit))).toBe(true);
        for (const rule of [...PHRASE_RULES, ...before.rules, ...after.rules]) {
            rule.conditions.forEach((condition) => expect(condition).toBeInstanceOf(PhraseCondition));
        }
        expect(PHRASE_RULES.every((rule) => rule.conditions.every((condition) => !condition.conditionFit))).toBe(true);
    });

    it('uses only the published BEV without accessing raw detections or depth-based measurements', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        const result = vision(0, { corner: 'right' });
        for (const property of ['detections', 'analysis', 'geometry', 'reconstructedScene']) {
            Object.defineProperty(result, property, { get: () => { throw new Error('Phrase rules must only read the BEV'); } });
        }
        engine.receiveVision(result, 0);
        engine.receiveTelemetry(frame(driving), 0);
        engine.receiveTelemetry(frame(driving), 800);
        expect(ids(engine, 800)).toEqual([defaultRule]);
    });

    it('lists shape-specific guides before general overtaking guides', () => {
        expect(PHRASE_RULES.map((rule) => rule.id)).toEqual([
            'next-corner', 'same-direction', 'sequence-exit', 's-bend',
            'tightening-corner', 'opening-corner', 'hairpin-exit',
            'inside-outbraking', 'around-outside', 'switchback',
            'better-exit', 'slipstream', 'pressure-feint',
        ]);
        PHRASE_RULES.forEach((rule) => {
            expect(rule.sentence).not.toMatch(/you are .+; the opponent ahead is/);
            expect(rule.conditions.find((condition): condition is PhraseCondition => condition instanceof PhraseCondition && condition.input === 'phase')?.description).toContain('Live Map section');
        });
    });

    it.each([
        { shape: 's-bend', position: 0.12, tactic: 's-bend' },
        { shape: 'tightening', position: 0.12, tactic: 'tightening-corner' },
        { shape: 'opening', position: 0.2, tactic: 'opening-corner' },
        { shape: 'hairpin', position: 0.2, tactic: 'hairpin-exit' },
    ] as const)('selects $tactic from centerline geometry', ({ shape, position, tactic }) => {
        const engine = new PhraseEngine();
        engine.receiveMap(shapedCornerMap(shape), 0);
        const sample = { ...driving, Graphics_normalized_car_position: position };
        update(engine, 0, sample);
        update(engine, 800, sample);
        expect(ids(engine, 800)).toEqual([tactic]);
    });

    it.each(['opposite', 'same'] as const)('uses the %s direction sequence guide and reserves exit guidance for the last corner', (direction) => {
        const sequence = circuitMap('slow', true);
        if (direction === 'same') sequence.samples.middle_line!.find((point) => point.normalized_position === 0.31)!.x = 1100;
        sequence.centerline_tags!.push({ id: 'sequence', label: 'consecutive corners', start_position: 0.1, end_position: 0.31 });
        for (const [position, tactic] of [
            [0.11, direction === 'opposite' ? 'next-corner' : 'same-direction'],
            [0.18, 'sequence-exit'],
            [0.29, direction === 'same' ? 'hairpin-exit' : 'better-exit'],
        ] as const) {
            const engine = new PhraseEngine();
            engine.receiveMap(sequence, 0);
            const sample = { ...driving, Graphics_normalized_car_position: position };
            engine.receiveVision(vision(0, { player: 'outside', opponent: 'inside' }), 0);
            engine.receiveTelemetry(frame(sample), 0);
            engine.receiveTelemetry(frame(sample), 800);
            expect(ids(engine, 800)).toEqual([tactic]);
        }
    });

    it('does not inherit a partly confirmed phrase when moving to another corner in the same area', () => {
        const sequence = shapedCornerMap('hairpin');
        sequence.centerline_segments = [
            { id: 'area', tags: ['consecutive corners'], start_position: 0.1, end_position: 0.3 },
            { id: 'first', tags: ['corner', 'slow'], start_position: 0.1, end_position: 0.2 },
            { id: 'second', tags: ['corner', 'slow'], start_position: 0.2, end_position: 0.3 },
        ];
        const engine = new PhraseEngine();
        engine.receiveMap(sequence, 0);
        const first = { ...driving, Graphics_normalized_car_position: 0.11 };
        const second = { ...driving, Graphics_normalized_car_position: 0.21 };
        update(engine, 0, first);
        update(engine, 700, second);
        update(engine, 800, second);
        expect(ids(engine, 800)).toEqual([]);
        update(engine, 1500, second);
        expect(ids(engine, 1500)).toEqual(['inside-outbraking']);
    });

    it.each([
        { tactic: 'inside-outbraking', speed: 'slow', position: 0.11, player: 'inside', opponent: 'outside' },
        { tactic: 'around-outside', speed: 'fast', position: 0.11, player: 'outside', opponent: 'inside' },
        { tactic: 'switchback', speed: 'slow', position: 0.15, player: 'outside', opponent: 'inside' },
        { tactic: 'better-exit', speed: 'slow', position: 0.18, player: 'middle', opponent: 'inside' },
        { tactic: 'better-exit', speed: 'fast', position: 0.18, player: 'middle', opponent: 'inside' },
        { tactic: 'next-corner', speed: 'fast', position: 0.11, player: 'outside', opponent: 'inside', linked: true },
        { tactic: 'next-corner', speed: 'slow', position: 0.15, player: 'outside', opponent: 'inside', linked: true },
        { tactic: 'pressure-feint', speed: 'slow', position: 0.11, player: 'middle', opponent: 'outside' },
    ] as const)('selects only $tactic for $speed at $position', (scenario) => {
        for (const corner of ['left', 'right'] as const) {
            const engine = new PhraseEngine();
            engine.receiveMap(circuitMap(scenario.speed, 'linked' in scenario), 0);
            const sample = { ...driving, Graphics_normalized_car_position: scenario.position };
            engine.receiveVision(vision(0, { corner, player: scenario.player, opponent: scenario.opponent }), 0);
            engine.receiveTelemetry(frame(sample), 0);
            engine.receiveTelemetry(frame(sample), 800);
            expect(ids(engine, 800)).toEqual([scenario.tactic]);
        }
    });

    it.each([null, { ...map, centerline_tags: [] }, circuitMap('fast')])('withholds inside outbraking without a mapped slow corner', (input) => {
        const engine = new PhraseEngine();
        engine.receiveMap(input, 0);
        update(engine, 0);
        update(engine, 800);
        expect(ids(engine, 800)).toEqual([]);
    });

    it.each([undefined, NaN, -0.1, 1.1])('withholds guidance for invalid lap position %s', (position) => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        const sample = { ...driving, Graphics_normalized_car_position: position };
        update(engine, 0, sample);
        update(engine, 800, sample);
        expect(ids(engine, 800)).toEqual([]);
    });

    it('restarts confirmation when the map is removed or replaced', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        update(engine, 0);
        engine.receiveMap(null, 500);
        update(engine, 800);
        expect(ids(engine, 800)).toEqual([]);
        engine.receiveMap(circuitMap(), 900);
        update(engine, 1000);
        update(engine, 1700);
        expect(ids(engine, 1700)).toEqual([defaultRule]);
    });

    it('emits once after a sustained match, only on a fresh telemetry frame', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        update(engine, 0);
        update(engine, 799);
        engine.receiveVision(vision(800), 800);
        expect(ids(engine, 800)).toEqual([]);
        engine.receiveTelemetry(frame(driving), 800);
        update(engine, 1000);
        expect(ids(engine, 1000)).toEqual([defaultRule]);
    });

    it.each(['other-track', 'other-game'])('withholds an old map after switching to %s', (change) => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        update(engine, 0);
        const event = frame({ ...driving, Static_track: change === 'other-track' ? 'spa' : 'test' });
        if (change === 'other-game') event.update.game = 'iracing';
        engine.receiveTelemetry(event, 800);
        const result = engine.receiveTelemetry(event, 1600);
        expect(result.mapReady).toBe(false);
        expect(result.events).toEqual([]);
    });

    it('normalizes ACC display names when matching the current map', () => {
        const engine = new PhraseEngine();
        engine.receiveMap({ ...map, source_track_key: 'brands_hatch' }, 0);
        update(engine, 0, { ...driving, Static_track: 'Brands Hatch Circuit' });
        update(engine, 800, { ...driving, Static_track: 'Brands Hatch Circuit' });
        expect(ids(engine, 800)).toEqual([defaultRule]);
    });

    it('preserves the cooldown across map reloads', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        update(engine, 0);
        update(engine, 800);
        engine.receiveMap(null, 900);
        engine.receiveMap(circuitMap(), 1500);
        update(engine, 1600);
        update(engine, 2400);
        expect(ids(engine, 2400)).toEqual([defaultRule]);
        expect(engine.evaluate(2400).rules.find((rule) => rule.id === defaultRule)?.status).toBe('Cooldown');
    });

    it('restarts the hold when moving straight into another tagged section', () => {
        const engine = new PhraseEngine();
        const split = circuitMap();
        split.centerline_tags = [
            { id: 'first', label: 'straight', start_position: 0.4, end_position: 0.5 },
            { id: 'second', label: 'straight', start_position: 0.5, end_position: 0.8 },
        ];
        engine.receiveMap(split, 0);
        engine.receiveVision(slipstreamVision(0), 0);
        for (const [now, position] of [[100, 0.49], [700, 0.51], [800, 0.52], [1500, 0.53]]) {
            engine.receiveVision(slipstreamVision(now), now);
            engine.receiveTelemetry(frame({ ...driving, Graphics_normalized_car_position: position }), now);
            expect(ids(engine, now)).toEqual(now < 1500 ? [] : ['slipstream']);
        }
    });

    it.each([
        driving,
        { ...driving, Physics_g_force_x: -20, Physics_g_force_y: 50, Physics_g_force_z: 100 },
        { ...driving, Physics_g_force_x: NaN, Physics_g_force_y: Infinity },
        { ...driving, Physics_brake: 1 },
    ])('uses the same visual positions regardless of G-forces or brake data: %s', (sample) => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        update(engine, 0, sample);
        update(engine, 800, sample);
        expect(ids(engine, 800)).toEqual([defaultRule]);
    });

    it('requires a clear period and cooldown before repeating a suggestion', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        update(engine, 0);
        update(engine, 800);
        update(engine, 900, { ...driving, Physics_speed_kmh: 0 });
        update(engine, 1000);
        update(engine, 1800);
        expect(ids(engine, 1800)).toEqual([defaultRule]);
        update(engine, 1900, { ...driving, Physics_speed_kmh: 0 });
        update(engine, 2500);
        const snapshot = update(engine, 3300);
        expect(snapshot.rules.find((rule) => rule.id === defaultRule)?.status).toBe('Cooldown');
        for (let now = 3800; now <= 8800; now += 500) update(engine, now);
        expect(ids(engine, 8800)).toEqual([defaultRule, defaultRule]);
    });

    it.each([
        {},
        { Physics_speed_kmh: undefined },
        { Physics_speed_kmh: NaN },
        { Physics_speed_kmh: -1 },
        { ...driving, Physics_speed_kmh: 29 },
    ])('does not fill absent or invalid driving inputs with zero: %s', (sample) => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        engine.receiveVision(vision(0), 0);
        engine.receiveTelemetry(frame(sample), 0);
        engine.receiveTelemetry(frame(sample), 1000);
        expect(ids(engine, 1000)).toEqual([]);
    });

    it.each([
        ['missing vision', null],
        ['missing published BEV', { ...vision(0), birdsEyeScene: null }],
        ['car pack without an individual position', { ...vision(0), birdsEyeScene: {
            ...vision(0).birdsEyeScene!, cars: vision(0).birdsEyeScene!.cars.map((car) => ({ ...car, pack: true })),
        } }],
        ['missing camera calibration', vision(0, { cameraOffset: null })],
        ['no car ahead', vision(0, { carAhead: false })],
        ['straight road', vision(0, { corner: 'straight' })],
        ['future timestamp', vision(5000)],
        ['depth only', { capturedAt: 0, width: 100, height: 100, detections: { depth: {
            task: 'depth' as const, width: 1, height: 1, values: new Float32Array([10]), inferenceMs: 1, classNames: [],
        } } }],
    ])('withholds every suggestion with %s', (_label, detection) => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        engine.receiveVision(detection, 0);
        engine.receiveTelemetry(frame(driving), 0);
        engine.receiveTelemetry(frame(driving), 1000);
        expect(ids(engine, 1000)).toEqual([]);
    });

    it('expires vision while telemetry remains live', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        engine.receiveVision(vision(0), 0);
        engine.receiveTelemetry(frame(driving), 1300);
        const snapshot = engine.receiveTelemetry(frame(driving), VISION_MAX_AGE_MS + 1);
        expect(snapshot.telemetryReady).toBe(true);
        expect(snapshot.visionReady).toBe(false);
        expect(snapshot.events).toEqual([]);
        const conditions = snapshot.rules.find((rule) => rule.id === defaultRule)!.conditions;
        expect(conditions.find((condition): condition is PhraseCondition => condition instanceof PhraseCondition && condition.input === 'speed')).toMatchObject({ conditionFit: true, inputMissing: false });
        expect(conditions.find((condition): condition is PhraseCondition => condition instanceof PhraseCondition && condition.input === 'carAhead')).toMatchObject({ conditionFit: false, inputMissing: true });
    });

    it('restarts the hold when vision stops or capture resumes after a gap', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        update(engine, 0);
        engine.receiveVision(null, 500);
        update(engine, 700);
        update(engine, 1499);
        expect(ids(engine, 1499)).toEqual([]);
        update(engine, 1500);
        expect(ids(engine, 1500)).toEqual([defaultRule]);

        const gap = new PhraseEngine();
        gap.receiveMap(map, 0);
        gap.receiveVision(vision(0), 0);
        gap.receiveTelemetry(frame(driving), 1400);
        update(gap, 2100);
        expect(ids(gap, 2100)).toEqual([]);
        update(gap, 2900);
        expect(ids(gap, 2900)).toEqual([defaultRule]);
    });

    it('restarts the hold when visual positions change', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        update(engine, 0);
        engine.receiveVision(vision(500, { player: 'middle', opponent: 'outside' }), 500);
        engine.receiveTelemetry(frame(driving), 800);
        expect(ids(engine, 800)).toEqual([]);
        engine.receiveTelemetry(frame(driving), 1300);
        expect(ids(engine, 1300)).toEqual(['pressure-feint']);
    });

    it('restarts the hold when the camera alignment changes even within the same position band', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        update(engine, 0);
        engine.receiveVision(vision(500, { cameraOffset: -0.35 }), 500);
        engine.receiveTelemetry(frame(driving), 800);
        expect(ids(engine, 800)).toEqual([]);
        engine.receiveTelemetry(frame(driving), 1300);
        expect(ids(engine, 1300)).toEqual([defaultRule]);
    });

    it('uses all velocity components as a speed fallback without position data', () => {
        const sample = { Graphics_normalized_car_position: 0.11, Physics_velocity_x: 6, Physics_velocity_y: 0, Physics_velocity_z: 8 };
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        engine.receiveVision(vision(0), 0);
        engine.receiveTelemetry(frame(sample), 0);
        engine.receiveTelemetry(frame(sample), 800);
        expect(ids(engine, 800)).toEqual([defaultRule]);
        const other = new PhraseEngine();
        other.receiveMap(map, 0);
        other.receiveVision(vision(0), 0);
        other.receiveTelemetry(frame({ ...sample, Physics_velocity_y: undefined }), 0);
        other.receiveTelemetry(frame({ ...sample, Physics_velocity_y: undefined }), 800);
        expect(ids(other, 800)).toEqual([]);
    });

    it.each([0, 1, 3])('suppresses output for simulator status %s', (status) => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        engine.receiveVision(vision(0), 0);
        engine.receiveTelemetry(frame(driving, status), 0);
        engine.receiveTelemetry(frame(driving, status), 1000);
        expect(ids(engine, 1000)).toEqual([]);
        expect(engine.evaluate(1000).telemetryReady).toBe(false);
    });

    it('expires telemetry and does not count a silent gap toward the hold', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        update(engine, 0);
        update(engine, TELEMETRY_MAX_AGE_MS + 1);
        expect(ids(engine, TELEMETRY_MAX_AGE_MS + 1)).toEqual([]);
        const expired = engine.evaluate(TELEMETRY_MAX_AGE_MS * 2 + 2);
        expect(expired.telemetryReady).toBe(false);
        expect(expired.events).toEqual([]);
        const conditions = expired.rules.find((rule) => rule.id === defaultRule)!.conditions;
        expect(conditions.find((condition): condition is PhraseCondition => condition instanceof PhraseCondition && condition.input === 'speed')).toMatchObject({ conditionFit: false, inputMissing: true });
        expect(conditions.find((condition): condition is PhraseCondition => condition instanceof PhraseCondition && condition.input === 'phase')).toMatchObject({ conditionFit: false, inputMissing: true });
        expect(conditions.find((condition): condition is PhraseCondition => condition instanceof PhraseCondition && condition.input === 'carAhead')).toMatchObject({ conditionFit: true, inputMissing: false });
    });

    it.each(['session-reset', 'stream-reset'] as const)('clears history, holds and old vision on %s', (type) => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        update(engine, 0);
        update(engine, 800);
        const reset = engine.receiveTelemetry({ type, snapshot: createLiveTelemetryStore().getSnapshot() }, 900);
        expect(reset.rules.every((rule) => rule.conditions.every((condition) => !condition.conditionFit && condition.inputMissing))).toBe(true);
        engine.receiveVision(vision(800), 900);
        engine.receiveTelemetry(frame(driving), 1000);
        engine.receiveTelemetry(frame(driving), 1800);
        expect(ids(engine, 1800)).toEqual([]);
        update(engine, 1900);
        update(engine, 2700);
        expect(ids(engine, 2700)).toEqual([defaultRule]);
    });

    it('bounds session history', () => {
        const engine = new PhraseEngine();
        engine.receiveMap(map, 0);
        for (let index = 0; index < 60; index++) {
            const now = index * (COOLDOWN_MS + 1000);
            update(engine, now);
            update(engine, now + 800);
            engine.receiveVision(null, now + 900);
        }
        expect(engine.evaluate(60 * (COOLDOWN_MS + 1000)).events).toHaveLength(50);
    });
});
