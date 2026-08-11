import React, { useContext } from 'react';
import { act, render, screen, waitFor } from '@testing-library/react';
import LiveRangeTodoList, {
    calculateForwardCircularDistance,
    calculateLiveRangeEta,
    calculateRollingForwardRate,
    crossedLiveRangeTodoPosition,
} from 'components/ai-engineering-tools/LiveRangeTodoList';
import { LiveSessionContext, LiveSessionProvider } from '../LiveSessionContext';
import type { LiveSessionRuntime } from '../live-session-types';
import {
    AI_TOOL_COMPONENT_NAMES,
    AiToolComponentRefProvider,
    resolveNamedComponentHandle,
    useAiToolComponentRefDirectory,
    type AiToolComponentRefDirectory,
} from 'contexts/AiToolComponentRefContext';
import type {
    LiveRangeTodoEventInput,
    LiveRangeTodoListHandle,
    LiveRangeTodoListSnapshot,
    LiveRangeTodoListToolResult,
    TaskStartFunction,
} from 'components/ai-engineering-tools';

const telemetry = (lap: number, position: number) => ({
    Graphics_completed_laps: lap,
    Graphics_normalized_car_position: position,
});

let runtime: LiveSessionRuntime;
let directory: AiToolComponentRefDirectory;
let latestSnapshot: LiveRangeTodoListSnapshot | null = null;
const handleSnapshotChange = (snapshot: LiveRangeTodoListSnapshot | null) => {
    latestSnapshot = snapshot;
};
const Harness = ({ show = true }: { show?: boolean }) => {
    runtime = useContext(LiveSessionContext);
    directory = useAiToolComponentRefDirectory();
    return show ? (
        <LiveRangeTodoList
            name={AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST}
            onSnapshotChange={handleSnapshotChange}
            surface="chat"
        />
    ) : null;
};

const getHandle = (): LiveRangeTodoListHandle => resolveNamedComponentHandle<LiveRangeTodoListHandle>(
    directory,
    AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
);

const publishTelemetry = (lap: number, position: number, elapsedMs = 100) => {
    act(() => {
        runtime.setCurrentTelemetry(telemetry(lap, position));
        jest.advanceTimersByTime(elapsedMs);
    });
};

const callHandle = (call: () => LiveRangeTodoListToolResult) => {
    let result!: LiveRangeTodoListToolResult;
    act(() => {
        result = call();
    });
    return result;
};

const makeEvent = (
    id: string,
    taskStart: TaskStartFunction = jest.fn(),
    overrides: Partial<LiveRangeTodoEventInput> = {},
): LiveRangeTodoEventInput => ({
    id,
    normalized_position: 0.25,
    lead_time_seconds: 0,
    content: { title: id },
    data: { producer: id },
    taskStart,
    ...overrides,
});

const flushCallbacks = async () => {
    await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
    });
};

describe('live range arrival prediction', () => {
    it('calculates rolling forward rate, lap wrap, ETA, and circular distance', () => {
        expect(calculateRollingForwardRate([
            { position: 0.8, lap: 0, receivedAt: 0 },
            { position: 0.9, lap: 0, receivedAt: 1000 },
            { position: 0.1, lap: 1, receivedAt: 2000 },
        ])).toBeCloseTo(0.15);
        expect(calculateForwardCircularDistance(0.9, 0.1)).toBeCloseTo(0.2);
        expect(calculateLiveRangeEta(0.9, 0.1, 0.1)).toBeCloseTo(2);
    });

    it('treats stopped, reverse, and non-rollover noise as unavailable progress', () => {
        expect(calculateRollingForwardRate([
            { position: 0.5, receivedAt: 0 },
            { position: 0.5, receivedAt: 1000 },
        ])).toBeNull();
        expect(calculateRollingForwardRate([
            { position: 0.6, receivedAt: 0 },
            { position: 0.4, receivedAt: 1000 },
        ])).toBeNull();
        expect(crossedLiveRangeTodoPosition(
            { position: 0.6, receivedAt: 0 },
            { position: 0.4, receivedAt: 1000 },
            0.5,
        )).toBe(false);
    });

    it('infers a rollover only for a high-to-low transition without lap data', () => {
        expect(crossedLiveRangeTodoPosition(
            { position: 0.95, receivedAt: 0 },
            { position: 0.05, receivedAt: 1000 },
            1,
        )).toBe(true);
        expect(crossedLiveRangeTodoPosition(
            { position: 0.7, receivedAt: 0 },
            { position: 0.1, receivedAt: 1000 },
            0.9,
        )).toBe(false);
    });
});

describe('LiveRangeTodoList', () => {
    beforeEach(() => {
        jest.useFakeTimers();
        jest.setSystemTime(new Date('2026-01-01T00:00:00Z'));
    });

    afterEach(() => {
        jest.restoreAllMocks();
        jest.useRealTimers();
    });

    const renderQueue = async () => {
        latestSnapshot = null;
        const view = render(
            <AiToolComponentRefProvider>
                <LiveSessionProvider><Harness /></LiveSessionProvider>
            </AiToolComponentRefProvider>,
        );
        await waitFor(() => expect(directory
            .findComponentRef(AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST)?.current).not.toBeNull());
        return view;
    };

    it('supports the typed event operations and keeps task functions out of snapshots', async () => {
        await renderQueue();
        expect(screen.queryByTestId('live-range-todo-list-empty')).not.toBeInTheDocument();
        const firstCallback = jest.fn();
        const secondCallback = jest.fn();

        expect(callHandle(() => getHandle().replaceEvents([
            makeEvent('brake', firstCallback),
        ])).status).toBe('ready');
        callHandle(() => getHandle().addEvent(makeEvent('exit', secondCallback, {
            normalized_position: 0.5,
        })));
        callHandle(() => getHandle().updateEvents([{
            id: 'brake',
            content: { detail: 'Use the 100 board' },
            data: { producer: 'component-a', priority: 1 },
        }]));

        const events = getHandle().get().todo_list!.events;
        expect(events).toHaveLength(2);
        expect(events[0]).toMatchObject({
            id: 'brake',
            content: { title: 'brake', detail: 'Use the 100 board' },
            data: { producer: 'component-a', priority: 1 },
        });
        expect(events[0]).not.toHaveProperty('taskStart');
        expect(latestSnapshot?.events).toHaveLength(2);
        expect(screen.getByText('brake')).toBeInTheDocument();

        callHandle(() => getHandle().resetEvents(['brake']));
        callHandle(() => getHandle().removeEvents(['exit']));
        expect(getHandle().get().todo_list!.events).toHaveLength(1);
        expect(callHandle(() => getHandle().clear()).status).toBe('empty');
        expect(latestSnapshot).toBeNull();
        expect(screen.queryByLabelText('Live range to-do list')).not.toBeInTheDocument();
    });

    it('rejects function-free events, invalid fields, and duplicate ids atomically', async () => {
        await renderQueue();
        const callback = jest.fn();
        const callbackFree = makeEvent('missing', callback) as Partial<LiveRangeTodoEventInput>;
        delete callbackFree.taskStart;

        const results = [
            callHandle(() => getHandle().addEvent(callbackFree as LiveRangeTodoEventInput)),
            callHandle(() => getHandle().replaceEvents([callbackFree as LiveRangeTodoEventInput])),
            callHandle(() => getHandle().replaceEvents([
                makeEvent('bad-position', callback, { normalized_position: 1.1 }),
            ])),
            callHandle(() => getHandle().replaceEvents([
                makeEvent('bad-lead', callback, { lead_time_seconds: -1 }),
            ])),
            callHandle(() => getHandle().replaceEvents([
                makeEvent('same', callback),
                makeEvent('same', callback, { normalized_position: 0.5 }),
            ])),
        ];

        results.forEach((result) => {
            expect(result).toMatchObject({ status: 'error', error: 'invalid_live_range_todo_list' });
        });
        expect(getHandle().get().todo_list!.events).toHaveLength(0);
    });

    it('lets multiple producers keep distinct task functions and preserves them through updates', async () => {
        await renderQueue();
        const componentACallback = jest.fn();
        const componentBCallback = jest.fn();
        callHandle(() => getHandle().addEvent(makeEvent('component-a', componentACallback)));
        callHandle(() => getHandle().addEvent(makeEvent('component-b', componentBCallback)));
        callHandle(() => getHandle().updateEvents([
            { id: 'component-a', content: { detail: 'Updated by AI-compatible data' } },
            { id: 'component-b', data: { producer: 'component-b', updated: true } },
        ]));

        publishTelemetry(3, 0.1);
        publishTelemetry(3, 0.3);
        await flushCallbacks();

        expect(componentACallback).toHaveBeenCalledTimes(1);
        expect(componentBCallback).toHaveBeenCalledTimes(1);
        expect(getHandle().get().todo_list!.events).toHaveLength(0);
    });

    it('waits for the running callback to settle before starting the next due event', async () => {
        await renderQueue();
        let finishFirst!: () => void;
        const firstCallback = jest.fn(() => new Promise<void>((resolve) => {
            finishFirst = resolve;
        }));
        const secondCallback = jest.fn(() => new Promise<void>(() => undefined));
        callHandle(() => getHandle().replaceEvents([
            makeEvent('first', firstCallback),
            makeEvent('second', secondCallback),
        ]));

        publishTelemetry(2, 0.1);
        publishTelemetry(2, 0.3);

        expect(firstCallback).toHaveBeenCalledTimes(1);
        expect(secondCallback).not.toHaveBeenCalled();
        expect(getHandle().get().todo_list!.events).toMatchObject([
            { id: 'first', status: 'running' },
            { id: 'second', status: 'pending' },
        ]);
        expect(getHandle().get().todo_list!.events[1]).not.toHaveProperty('due');

        await act(async () => {
            finishFirst();
            await Promise.resolve();
            await Promise.resolve();
        });

        expect(secondCallback).toHaveBeenCalledTimes(1);
        expect(getHandle().get().todo_list!.events).toMatchObject([
            { id: 'second', status: 'running' },
        ]);
    });

    it('queues an event that becomes due while another callback is running', async () => {
        await renderQueue();
        let finishFirst!: () => void;
        const firstCallback = jest.fn(() => new Promise<void>((resolve) => {
            finishFirst = resolve;
        }));
        const secondCallback = jest.fn(() => new Promise<void>(() => undefined));
        callHandle(() => getHandle().replaceEvents([
            makeEvent('first', firstCallback),
            makeEvent('second', secondCallback, { normalized_position: 0.5 }),
        ]));

        publishTelemetry(2, 0.1);
        publishTelemetry(2, 0.3);
        expect(firstCallback).toHaveBeenCalledTimes(1);
        expect(secondCallback).not.toHaveBeenCalled();

        publishTelemetry(2, 0.6);
        expect(secondCallback).not.toHaveBeenCalled();
        expect(getHandle().get().todo_list!.events).toMatchObject([
            { id: 'first', status: 'running' },
            { id: 'second', status: 'pending' },
        ]);
        expect(getHandle().get().todo_list!.events[1]).not.toHaveProperty('due');

        await act(async () => {
            finishFirst();
            await Promise.resolve();
            await Promise.resolve();
        });

        expect(secondCallback).toHaveBeenCalledWith(expect.any(AbortSignal));
    });

    it('passes only an abort signal, removes settled events, and logs rejection', async () => {
        await renderQueue();
        const success = jest.fn(async () => undefined);
        const failure = jest.fn(async () => {
            throw new Error('producer failed');
        });
        const errorSpy = jest.spyOn(console, 'error').mockImplementation(() => undefined);
        callHandle(() => getHandle().replaceEvents([
            makeEvent('success', success, { data: { instruction: 'notify' } }),
            makeEvent('failure', failure),
        ]));

        publishTelemetry(4, 0.1);
        publishTelemetry(4, 0.3);
        await flushCallbacks();

        expect(success).toHaveBeenCalledWith(expect.any(AbortSignal));
        expect(failure).toHaveBeenCalledTimes(1);
        expect(getHandle().get().todo_list!.events).toHaveLength(0);
        expect(errorSpy).toHaveBeenCalledWith(
            "Live range to-do event 'failure' task failed.",
            expect.objectContaining({ message: 'producer failed' }),
        );
    });

    it('releases the queue when a callback throws synchronously', async () => {
        await renderQueue();
        const thrownError = new Error('synchronous producer failure');
        const failure = jest.fn(() => {
            throw thrownError;
        });
        const next = jest.fn(() => new Promise<void>(() => undefined));
        const errorSpy = jest.spyOn(console, 'error').mockImplementation(() => undefined);
        callHandle(() => getHandle().replaceEvents([
            makeEvent('failure', failure),
            makeEvent('next', next),
        ]));

        publishTelemetry(4, 0.1);
        publishTelemetry(4, 0.3);

        expect(failure).toHaveBeenCalledTimes(1);
        expect(next).toHaveBeenCalledTimes(1);
        expect(getHandle().get().todo_list!.events).toMatchObject([
            { id: 'next', status: 'running' },
        ]);
        expect(errorSpy).toHaveBeenCalledWith(
            "Live range to-do event 'failure' task failed.",
            thrownError,
        );
    });

    it.each([
        ['update', (id: string) => getHandle().updateEvents([{ id, content: { detail: 'changed' } }])],
        ['remove', (id: string) => getHandle().removeEvents([id])],
        ['reset', (id: string) => getHandle().resetEvents([id])],
        ['clear', () => getHandle().clear()],
        ['replace', () => getHandle().replaceEvents([makeEvent('replacement')])],
    ])('aborts running callbacks during %s', async (_name, mutate) => {
        await renderQueue();
        let capturedSignal: AbortSignal | undefined;
        const pending = jest.fn((signal: AbortSignal) => {
            capturedSignal = signal;
            return new Promise<void>(() => undefined);
        });
        callHandle(() => getHandle().addEvent(makeEvent('pending', pending)));
        publishTelemetry(0, 0.1);
        publishTelemetry(0, 0.3);
        expect(capturedSignal?.aborted).toBe(false);

        callHandle(() => mutate('pending'));
        expect(capturedSignal?.aborted).toBe(true);
    });

    it('allows an update to explicitly replace an event task function', async () => {
        await renderQueue();
        const original = jest.fn();
        const replacement = jest.fn();
        callHandle(() => getHandle().addEvent(makeEvent('replace-callback', original)));
        callHandle(() => getHandle().updateEvents([{
            id: 'replace-callback',
            taskStart: replacement,
        }]));

        publishTelemetry(1, 0.1);
        publishTelemetry(1, 0.3);
        await flushCallbacks();
        expect(original).not.toHaveBeenCalled();
        expect(replacement).toHaveBeenCalledTimes(1);
    });

    it('survives persistent-parent rerenders and clears tasks plus telemetry history for a new session', async () => {
        const view = await renderQueue();
        const staleCrossingCallback = jest.fn();
        act(() => runtime.startLiveSession('acc'));
        publishTelemetry(1, 0.8);
        publishTelemetry(1, 0.9);
        callHandle(() => getHandle().addEvent(makeEvent('previous-session', jest.fn(), {
            normalized_position: 0.5,
        })));

        view.rerender(
            <AiToolComponentRefProvider>
                <LiveSessionProvider><Harness /></LiveSessionProvider>
            </AiToolComponentRefProvider>,
        );
        expect(getHandle().get().todo_list!.events).toHaveLength(1);
        expect(latestSnapshot?.events).toHaveLength(1);

        act(() => runtime.endLiveSession());
        expect(getHandle().get().todo_list!.events).toHaveLength(1);
        act(() => runtime.startLiveSession('acc'));
        expect(getHandle().get().todo_list!.events).toHaveLength(0);
        expect(latestSnapshot).toBeNull();

        callHandle(() => getHandle().addEvent(makeEvent('new-session', staleCrossingCallback, {
            normalized_position: 0.1,
        })));
        publishTelemetry(0, 0.2);
        await flushCallbacks();
        expect(staleCrossingCallback).not.toHaveBeenCalled();
    });

    it('clears the component ref and snapshot callback and aborts running work when the runtime unmounts', async () => {
        let capturedSignal: AbortSignal | undefined;
        latestSnapshot = null;
        const app = (show: boolean) => (
            <AiToolComponentRefProvider>
                <LiveSessionProvider><Harness show={show} /></LiveSessionProvider>
            </AiToolComponentRefProvider>
        );
        const view = render(
            app(true),
        );
        await waitFor(() => expect(directory
            .findComponentRef(AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST)?.current).not.toBeNull());
        callHandle(() => getHandle().addEvent(makeEvent('pending', (signal) => {
            capturedSignal = signal;
            return new Promise<void>(() => undefined);
        })));
        publishTelemetry(0, 0.1);
        publishTelemetry(0, 0.3);
        expect(capturedSignal?.aborted).toBe(false);

        view.rerender(app(false));
        await waitFor(() => expect(directory
            .findComponentRef(AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST)).toBeNull());
        expect(latestSnapshot).toBeNull();
        expect(capturedSignal?.aborted).toBe(true);
    });
});
