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
import {
    createAiToolOperationFrom,
    createControlledAiToolOperation,
    resolvedAiToolOperation,
} from 'components/ai-engineering-tools';
import {
    AI_TOOL_COMPONENT_NAMES,
    createAiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';

const event = (
    id: string,
    normalizedPosition: number,
    taskStart: LiveRangeTodoEventInput['taskStart'] = jest.fn(() => (
        resolvedAiToolOperation({}, 'complete')
    )),
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
    for (let index = 0; index < 8; index += 1) await Promise.resolve();
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
    it('removes its registration completely after the final task finishes', async () => {
        const directory = createAiToolComponentRefDirectory();
        const runner = new LiveRangeTodoListRunner(
            AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
        );
        runner.addComponentRef(directory);
        runner.addEvent(event('one', 0.2));

        makeDue(runner, 0.25);
        await flushPromises();

        expect(directory.findComponentRef(
            AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
        )).toBeNull();
    });

    it('preserves a tool-provided ETA until rolling telemetry can refine it', () => {
        const runner = new LiveRangeTodoListRunner('live-range');

        runner.addEvent({
            ...event('estimated', 0.4),
            eta_seconds: 30,
        });
        runner.acceptTelemetry({
            Graphics_normalized_car_position: 0.1,
            Graphics_completed_laps: 1,
        });

        expect(runner.get().todo_list?.events[0]).toMatchObject({
            id: 'estimated',
        });
        expect(runner.get().todo_list?.events[0].eta_seconds).toBeCloseTo(30, 1);
    });

    it('uses a tool-provided ETA for lead-time scheduling without a rolling rate', () => {
        const now = jest.spyOn(Date, 'now').mockReturnValue(1_000);
        const taskStart = jest.fn(() => resolvedAiToolOperation({}, 'complete'));
        const runner = new LiveRangeTodoListRunner('live-range');
        runner.addEvent({
            ...event('estimated', 0.4, taskStart),
            lead_time_seconds: 2,
            eta_seconds: 5,
        });

        now.mockReturnValue(5_000);
        runner.acceptTelemetry({
            Graphics_normalized_car_position: 0.1,
            Graphics_completed_laps: 1,
        });

        expect(taskStart).toHaveBeenCalledTimes(1);
        now.mockRestore();
    });

    it('waits for telemetry, invokes taskStart, and omits functions from snapshots', async () => {
        const taskStart = jest.fn(() => resolvedAiToolOperation({}, 'complete'));
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
        const firstController = createControlledAiToolOperation<Record<string, never>, never, 'complete'>();
        const secondController = createControlledAiToolOperation<Record<string, never>, never, 'complete'>();
        const first = jest.fn(() => firstController.operation);
        const second = jest.fn(() => secondController.operation);
        const runner = new LiveRangeTodoListRunner('live-range');
        runner.replaceEvents([event('one', 0.2, first), event('two', 0.4, second)]);

        makeDue(runner);

        expect(first).toHaveBeenCalledTimes(1);
        expect(second).not.toHaveBeenCalled();
        expect(runner.get().todo_list?.events).toEqual([
            expect.objectContaining({ id: 'one', status: 'running' }),
            expect.objectContaining({ id: 'two', status: 'pending' }),
        ]);

        firstController.resolve('complete', {});
        await flushPromises();
        expect(second).toHaveBeenCalledTimes(1);
        expect(runner.get().todo_list?.events).toEqual([
            expect.objectContaining({ id: 'two', status: 'running' }),
        ]);

        secondController.resolve('complete', {});
        await flushPromises();
        expect(runner.get().todo_list?.events).toHaveLength(0);
    });

    it('advances only after notifyTerminated, not when operation.result settles', async () => {
        let notifyFirstTerminated!: (termination: {
            status: string;
            result: Record<string, never> | Error;
        }) => void;
        const firstOperation = {
            result: Promise.resolve({}),
            statuses: [],
            notifyTerminated: (listener: typeof notifyFirstTerminated) => {
                notifyFirstTerminated = listener;
                return () => undefined;
            },
        };
        const first = jest.fn(() => firstOperation);
        const second = jest.fn(() => resolvedAiToolOperation({}, 'complete'));
        const runner = new LiveRangeTodoListRunner('live-range');
        runner.replaceEvents([event('one', 0.2, first), event('two', 0.4, second)]);

        makeDue(runner);
        await flushPromises();

        expect(second).not.toHaveBeenCalled();
        expect(runner.get().todo_list?.events[0]).toMatchObject({
            id: 'one',
            status: 'running',
        });

        notifyFirstTerminated({ status: 'complete', result: {} });
        await flushPromises();

        expect(second).toHaveBeenCalledTimes(1);
    });

    it('unsubscribes from termination before aborting a replaced task', () => {
        const lifecycle: string[] = [];
        let staleTermination!: (termination: { status: string; result: Record<string, never> }) => void;
        const operation = {
            result: new Promise<Record<string, never>>(() => undefined),
            statuses: [],
            notifyTerminated: (listener: typeof staleTermination) => {
                staleTermination = listener;
                return () => lifecycle.push('unsubscribe');
            },
        };
        const runner = new LiveRangeTodoListRunner('live-range');
        runner.addEvent(event('one', 0.2, (signal) => {
            signal.addEventListener('abort', () => lifecycle.push('abort'));
            return operation;
        }));
        makeDue(runner, 0.25);

        runner.replaceEvents([event('replacement', 0.8)]);

        expect(lifecycle).toEqual(['unsubscribe', 'abort']);
        staleTermination({ status: 'complete', result: {} });
        expect(runner.get().todo_list?.events).toEqual([
            expect.objectContaining({ id: 'replacement', status: 'pending' }),
        ]);
    });

    it('logs synchronous throws and promise rejections without stalling the queue', async () => {
        const failure = new Error('task failed');
        const finalTask = jest.fn();
        const consoleError = jest.spyOn(console, 'error').mockImplementation(() => undefined);
        const runner = new LiveRangeTodoListRunner('live-range');
        runner.replaceEvents([
            event('throws', 0.1, () => { throw failure; }),
            event('rejects', 0.2, () => createAiToolOperationFrom(() => {
                throw failure;
            }, 'failed')),
            event('continues', 0.3, () => {
                finalTask();
                return resolvedAiToolOperation({}, 'complete');
            }),
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
            return createControlledAiToolOperation<Record<string, never>, never, 'complete'>()
                .operation;
        }));
        makeDue(runner, 0.25);

        mutate(runner);

        expect(receivedSignal?.aborted).toBe(true);
        runner.dispose();
    });

    it('preserves taskStart on ordinary updates and replaces it when explicitly supplied', async () => {
        const preserved = jest.fn(() => resolvedAiToolOperation({}, 'complete'));
        const runner = new LiveRangeTodoListRunner('live-range');
        runner.addEvent(event('one', 0.2, preserved));
        runner.updateEvents([{ id: 'one', content: { description: 'Current description' } }]);
        makeDue(runner, 0.25);
        expect(preserved).toHaveBeenCalledTimes(1);
        await flushPromises();

        const original = jest.fn(() => resolvedAiToolOperation({}, 'complete'));
        const replacement = jest.fn(() => resolvedAiToolOperation({}, 'complete'));
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
