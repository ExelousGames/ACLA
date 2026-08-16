import {
    LiveRangeTodoListRunner,
    calculateForwardCircularDistance,
    calculateLiveRangeEta,
    crossedLiveRangeTodoPosition,
} from 'components/ai-engineering-tools/LiveRangeTodoList';
import type {
    LiveRangeTodoEventInput,
    LiveRangeTodoEventUpdate,
} from 'components/ai-engineering-tools';

const event = (
    id: string,
    normalizedPosition: number,
    taskStart: LiveRangeTodoEventInput['taskStart'] = jest.fn(),
): LiveRangeTodoEventInput => ({
    id,
    normalized_position: normalizedPosition,
    lead_time_seconds: 0,
    content: { title: id },
    taskStart,
});

const makeDue = (runner: LiveRangeTodoListRunner, endPosition = 0.5) => {
    runner.acceptTelemetry({ Graphics_normalized_car_position: 0, Graphics_completed_laps: 1 });
    runner.acceptTelemetry({ Graphics_normalized_car_position: endPosition, Graphics_completed_laps: 1 });
};

const flushPromises = async () => {
    await Promise.resolve();
    await Promise.resolve();
};

describe('live range helpers', () => {
    it('calculates circular distance, ETA, and rollover crossings', () => {
        expect(calculateForwardCircularDistance(0.9, 0.1)).toBeCloseTo(0.2);
        expect(calculateLiveRangeEta(0.9, 0.1, 0.1)).toBeCloseTo(2);
        expect(crossedLiveRangeTodoPosition(
            { position: 0.95, receivedAt: 0, lap: 2 },
            { position: 0.05, receivedAt: 1000, lap: 3 },
            0.99,
        )).toBe(true);
    });
});

describe('LiveRangeTodoListRunner executable events', () => {
    it('waits for telemetry, invokes taskStart, and omits functions from snapshots', async () => {
        const taskStart = jest.fn();
        const runner = new LiveRangeTodoListRunner('live-range');

        const result = runner.addEvent(event('one', 0.2, taskStart));

        expect(result.todo_list?.events[0]).not.toHaveProperty('taskStart');
        expect(taskStart).not.toHaveBeenCalled();
        makeDue(runner, 0.25);
        expect(taskStart).toHaveBeenCalledWith(expect.any(AbortSignal));
        await flushPromises();
        expect(runner.get()).toMatchObject({ status: 'empty' });
    });

    it('keeps asynchronous work visible as running and executes due tasks sequentially', async () => {
        let resolveFirst!: () => void;
        let resolveSecond!: () => void;
        const first = jest.fn(() => new Promise<void>((resolve) => { resolveFirst = resolve; }));
        const second = jest.fn(() => new Promise<void>((resolve) => { resolveSecond = resolve; }));
        const runner = new LiveRangeTodoListRunner('live-range');
        runner.replaceEvents([event('one', 0.2, first), event('two', 0.4, second)]);

        makeDue(runner);

        expect(first).toHaveBeenCalledTimes(1);
        expect(second).not.toHaveBeenCalled();
        expect(runner.get().todo_list?.events).toEqual([
            expect.objectContaining({ id: 'one', status: 'running' }),
            expect.objectContaining({ id: 'two', status: 'pending' }),
        ]);

        resolveFirst();
        await flushPromises();
        expect(second).toHaveBeenCalledTimes(1);
        expect(runner.get().todo_list?.events).toEqual([
            expect.objectContaining({ id: 'two', status: 'running' }),
        ]);

        resolveSecond();
        await flushPromises();
        expect(runner.get().todo_list?.events).toHaveLength(0);
    });

    it('logs synchronous throws and promise rejections without stalling the queue', async () => {
        const failure = new Error('task failed');
        const finalTask = jest.fn();
        const consoleError = jest.spyOn(console, 'error').mockImplementation(() => undefined);
        const runner = new LiveRangeTodoListRunner('live-range');
        runner.replaceEvents([
            event('throws', 0.1, () => { throw failure; }),
            event('rejects', 0.2, () => Promise.reject(failure)),
            event('continues', 0.3, finalTask),
        ]);

        makeDue(runner, 0.4);
        await flushPromises();
        await flushPromises();

        expect(finalTask).toHaveBeenCalledTimes(1);
        expect(consoleError).toHaveBeenCalledWith(
            "Live range to-do event 'throws' task failed.",
            failure,
        );
        expect(consoleError).toHaveBeenCalledWith(
            "Live range to-do event 'rejects' task failed.",
            failure,
        );
        expect(runner.get().todo_list?.events).toHaveLength(0);
        consoleError.mockRestore();
    });

    it.each([
        ['update', (runner: LiveRangeTodoListRunner) => runner.updateEvents([{ id: 'one', content: { description: 'updated' } }])],
        ['reset', (runner: LiveRangeTodoListRunner) => runner.resetEvents(['one'])],
        ['remove', (runner: LiveRangeTodoListRunner) => runner.removeEvents(['one'])],
        ['replacement', (runner: LiveRangeTodoListRunner) => runner.replaceEvents([event('replacement', 0.8)])],
        ['clear', (runner: LiveRangeTodoListRunner) => runner.clear()],
        ['session reset', (runner: LiveRangeTodoListRunner) => runner.reset()],
        ['disposal', (runner: LiveRangeTodoListRunner) => runner.dispose()],
    ])('aborts running work on %s', (_name, mutate) => {
        let receivedSignal: AbortSignal | undefined;
        const runner = new LiveRangeTodoListRunner('live-range');
        runner.addEvent(event('one', 0.2, (signal) => {
            receivedSignal = signal;
            return new Promise(() => undefined);
        }));
        makeDue(runner, 0.25);

        mutate(runner);

        expect(receivedSignal?.aborted).toBe(true);
        runner.dispose();
    });

    it('preserves taskStart on ordinary updates and replaces it when explicitly supplied', async () => {
        const preserved = jest.fn();
        const runner = new LiveRangeTodoListRunner('live-range');
        runner.addEvent(event('one', 0.2, preserved));
        runner.updateEvents([{ id: 'one', content: { description: 'Current description' } }]);
        makeDue(runner, 0.25);
        expect(preserved).toHaveBeenCalledTimes(1);
        await flushPromises();

        const original = jest.fn();
        const replacement = jest.fn();
        const secondRunner = new LiveRangeTodoListRunner('live-range-two');
        secondRunner.addEvent(event('two', 0.2, original));
        secondRunner.updateEvents([{ id: 'two', taskStart: replacement }]);
        makeDue(secondRunner, 0.25);
        expect(original).not.toHaveBeenCalled();
        expect(replacement).toHaveBeenCalledTimes(1);
    });

    it('validates functions, content, legacy fields, and duplicate ids atomically', () => {
        const runner = new LiveRangeTodoListRunner('live-range');
        runner.addEvent(event('existing', 0.8));

        expect(() => runner.addEvent({
            id: 'missing-function',
            normalized_position: 0.2,
            content: { title: 'Missing function' },
        } as LiveRangeTodoEventInput)).toThrow(/taskStart function/);
        expect(() => runner.addEvent({
            ...event('legacy-data', 0.2),
            data: {},
        } as LiveRangeTodoEventInput)).toThrow(/property 'data' is not supported/);
        expect(() => runner.addEvent({
            ...event('legacy-detail', 0.2),
            content: { title: 'Legacy', detail: 'No longer supported' },
        } as LiveRangeTodoEventInput)).toThrow(/property 'detail' is not supported/);
        expect(() => runner.replaceEvents([
            event('duplicate', 0.2),
            event('duplicate', 0.4),
        ])).toThrow(/Duplicate/);
        expect(runner.get().todo_list?.events).toEqual([
            expect.objectContaining({ id: 'existing' }),
        ]);
        expect(() => runner.updateEvents([{
            id: 'existing',
            data: {},
        } as LiveRangeTodoEventUpdate])).toThrow(/property 'data' is not supported/);
    });
});
