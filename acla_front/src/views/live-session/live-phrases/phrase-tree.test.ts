import { Action, Closure, describeCondition } from './closure';
import { createPhraseRoot, PHRASE_DEFINITIONS, PhraseEngine, type PhraseContext } from './phrase-engine';
import { circuitMap, vision } from './test-fixtures';

it('snapshots arbitrary root actions and nested catalog closures without relying on catalog position', () => {
    const run = jest.fn();
    const rule = PHRASE_DEFINITIONS.find((phrase) => phrase.id === 'inside-outbraking')!;
    const engine = new PhraseEngine([rule], (phrases) => {
        const catalog = createPhraseRoot(phrases);
        return new Closure<PhraseContext>('custom root', 'Custom root.', () => true, [
            new Action('record frame', 'Optional recording.', run, describeCondition(() => false, 'Recording enabled')),
            new Closure('coaching', 'Nested coaching scope.', () => true, catalog.children),
        ]);
    });
    engine.receiveMap(circuitMap(), 0);
    engine.receiveVision(vision(0), 0);
    const frame = (now: number) => {
        const sample = { Physics_speed_kmh: 100, Graphics_normalized_car_position: 0.11 };
        return engine.receiveTelemetry({
            type: 'frame', sample, sampleIndex: 0, telemetryStatus: 2,
            committedSampleCount: 0, sessionGeneration: 0, streamGeneration: 0,
            update: { type: 'frame', game: 'acc', sample, sequence: 1, committedSequence: 0, committedCount: 0 },
        }, now);
    };
    frame(0);
    const snapshot = frame(800);
    expect(snapshot.root.children[0]).toMatchObject({ name: 'record frame', kind: 'action', conditions: [{ description: 'Recording enabled', conditionFit: false }] });
    expect(snapshot.root.children[1].children[0]).toMatchObject({
        id: rule.id, name: rule.name, status: 'Active', onPath: true,
        fields: { 'Hold for': '0.8 s', Cooldown: '8 s' },
        children: [
            { kind: 'action', name: 'say phrase', current: true, execution: { status: 'completed' } },
            { kind: 'action', name: 'exit to root', execution: { status: 'idle' } },
        ],
    });
    expect(snapshot.closures[0]).toMatchObject({ id: rule.id, status: 'Active' });
    expect(snapshot.events.map((event) => event.ruleId)).toEqual([rule.id]);
    expect(run).not.toHaveBeenCalled();
});

it.each([false, true])('publishes custom action failures for the UI (async: %s)', async (async) => {
    const error = new Error('Recorder failed');
    const run = jest.fn(() => { if (async) return Promise.reject(error); throw error; });
    const engine = new PhraseEngine([], () => new Closure<PhraseContext>('root', 'Root.', () => true, [
        new Action('record frame', 'Record diagnostics.', run),
    ]));
    const first = engine.evaluate(0);
    if (async) {
        expect(first.root.children[0].execution?.status).toBe('running');
        await Promise.resolve();
        await Promise.resolve();
        await Promise.resolve();
    }
    expect(engine.evaluate(1).root.children[0]).toMatchObject({
        status: 'Failed', execution: { status: 'failed', error: 'Recorder failed' },
    });
    expect(run).toHaveBeenCalledTimes(1);
});

it('does not hide condition errors after a custom action failed', () => {
    const engine = new PhraseEngine([], () => new Closure<PhraseContext>('root', 'Root.', () => true, [
        new Action('fail', 'Fail an action.', () => { throw new Error('Action failed'); }),
        new Action('later', 'Check a broken condition.', jest.fn(), () => { throw new Error('Condition failed'); }),
    ]));
    expect(engine.evaluate(0).root.children[0].execution?.status).toBe('failed');
    expect(() => engine.evaluate(1)).toThrow('Condition failed');
});
