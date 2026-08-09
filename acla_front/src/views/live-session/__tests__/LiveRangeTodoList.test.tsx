import React, { useContext } from 'react';
import { act, render, screen, waitFor } from '@testing-library/react';
import LiveRangeTodoList, {
    calculateForwardCircularDistance,
    calculateLiveRangeEta,
    calculateRollingForwardRate,
    crossedLiveRangeTodoPosition,
} from '../LiveRangeTodoList';
import { LiveSessionContext, LiveSessionProvider } from '../LiveSessionContext';
import type { LiveSessionRuntime } from '../live-session-types';
import type {
    LiveRangeTodoEventCallback,
    LiveRangeTodoEventInput,
    LiveRangeTodoListToolResult,
} from '../live-range-todo-list-types';

const telemetry = (lap: number, position: number) => ({
    Graphics_completed_laps: lap,
    Graphics_normalized_car_position: position,
});

let runtime: LiveSessionRuntime;
const Harness = () => {
    runtime = useContext(LiveSessionContext);
    return <LiveRangeTodoList name="live-range-todo-list" />;
};

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
    callback: LiveRangeTodoEventCallback = jest.fn(),
    overrides: Partial<LiveRangeTodoEventInput> = {},
): LiveRangeTodoEventInput => ({
    id,
    normalized_position: 0.25,
    lead_time_seconds: 0,
    content: { title: id },
    data: { producer: id },
    callback,
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
        const view = render(<LiveSessionProvider><Harness /></LiveSessionProvider>);
        await waitFor(() => expect(runtime.liveRangeTodoListHandle).not.toBeNull());
        return view;
    };

    it('supports the typed event operations and keeps callbacks out of snapshots', async () => {
        await renderQueue();
        expect(screen.getByTestId('live-range-todo-list-empty')).toBeInTheDocument();
        const firstCallback = jest.fn();
        const secondCallback = jest.fn();

        expect(callHandle(() => runtime.liveRangeTodoListHandle!.replaceEvents([
            makeEvent('brake', firstCallback),
        ])).status).toBe('ready');
        callHandle(() => runtime.liveRangeTodoListHandle!.addEvent(makeEvent('exit', secondCallback, {
            normalized_position: 0.5,
        })));
        callHandle(() => runtime.liveRangeTodoListHandle!.updateEvents([{
            id: 'brake',
            content: { detail: 'Use the 100 board' },
            data: { producer: 'component-a', priority: 1 },
        }]));

        const events = runtime.liveRangeTodoListHandle!.get().todo_list!.events;
        expect(events).toHaveLength(2);
        expect(events[0]).toMatchObject({
            id: 'brake',
            content: { title: 'brake', detail: 'Use the 100 board' },
            data: { producer: 'component-a', priority: 1 },
        });
        expect(events[0]).not.toHaveProperty('callback');

        callHandle(() => runtime.liveRangeTodoListHandle!.resetEvents(['brake']));
        callHandle(() => runtime.liveRangeTodoListHandle!.removeEvents(['exit']));
        expect(runtime.liveRangeTodoListHandle!.get().todo_list!.events).toHaveLength(1);
        expect(callHandle(() => runtime.liveRangeTodoListHandle!.clear()).status).toBe('empty');
    });

    it('rejects callback-free events, invalid fields, and duplicate ids atomically', async () => {
        await renderQueue();
        const callback = jest.fn();
        const callbackFree = makeEvent('missing', callback) as Partial<LiveRangeTodoEventInput>;
        delete callbackFree.callback;

        const results = [
            callHandle(() => runtime.liveRangeTodoListHandle!.addEvent(callbackFree as LiveRangeTodoEventInput)),
            callHandle(() => runtime.liveRangeTodoListHandle!.replaceEvents([callbackFree as LiveRangeTodoEventInput])),
            callHandle(() => runtime.liveRangeTodoListHandle!.replaceEvents([
                makeEvent('bad-position', callback, { normalized_position: 1.1 }),
            ])),
            callHandle(() => runtime.liveRangeTodoListHandle!.replaceEvents([
                makeEvent('bad-lead', callback, { lead_time_seconds: -1 }),
            ])),
            callHandle(() => runtime.liveRangeTodoListHandle!.replaceEvents([
                makeEvent('same', callback),
                makeEvent('same', callback, { normalized_position: 0.5 }),
            ])),
        ];

        results.forEach((result) => {
            expect(result).toMatchObject({ status: 'error', error: 'invalid_live_range_todo_list' });
        });
        expect(runtime.liveRangeTodoListHandle!.get().todo_list!.events).toHaveLength(0);
    });

    it('lets multiple producers keep distinct callbacks and preserves them through updates', async () => {
        await renderQueue();
        const componentACallback = jest.fn();
        const componentBCallback = jest.fn();
        callHandle(() => runtime.liveRangeTodoListHandle!.addEvent(makeEvent('component-a', componentACallback)));
        callHandle(() => runtime.liveRangeTodoListHandle!.addEvent(makeEvent('component-b', componentBCallback)));
        callHandle(() => runtime.liveRangeTodoListHandle!.updateEvents([
            { id: 'component-a', content: { detail: 'Updated by AI-compatible data' } },
            { id: 'component-b', data: { producer: 'component-b', updated: true } },
        ]));

        publishTelemetry(3, 0.1);
        publishTelemetry(3, 0.3);
        await flushCallbacks();

        expect(componentACallback).toHaveBeenCalledTimes(1);
        expect(componentBCallback).toHaveBeenCalledTimes(1);
        expect(runtime.liveRangeTodoListHandle!.get().todo_list!.events).toHaveLength(0);
    });

    it('passes event context, ignores callback results, removes settled events, and logs rejection', async () => {
        await renderQueue();
        const success = jest.fn(async () => ({ ignored: true }));
        const failure = jest.fn(async () => {
            throw new Error('producer failed');
        });
        const errorSpy = jest.spyOn(console, 'error').mockImplementation(() => undefined);
        callHandle(() => runtime.liveRangeTodoListHandle!.replaceEvents([
            makeEvent('success', success, { data: { instruction: 'notify' } }),
            makeEvent('failure', failure),
        ]));

        publishTelemetry(4, 0.1);
        publishTelemetry(4, 0.3);
        await flushCallbacks();

        expect(success).toHaveBeenCalledWith(expect.objectContaining({
            event: expect.objectContaining({ id: 'success', data: { instruction: 'notify' } }),
            data: { instruction: 'notify' },
            telemetry: telemetry(4, 0.3),
            lap: 4,
            eta_seconds: expect.any(Number),
            signal: expect.any(AbortSignal),
            sessionIntelligence: runtime.sessionIntelligence,
        }));
        expect(failure).toHaveBeenCalledTimes(1);
        expect(runtime.liveRangeTodoListHandle!.get().todo_list!.events).toHaveLength(0);
        expect(errorSpy).toHaveBeenCalledWith(
            "Live range to-do event 'failure' callback failed.",
            expect.objectContaining({ message: 'producer failed' }),
        );
    });

    it.each([
        ['update', (id: string) => runtime.liveRangeTodoListHandle!.updateEvents([{ id, content: { detail: 'changed' } }])],
        ['remove', (id: string) => runtime.liveRangeTodoListHandle!.removeEvents([id])],
        ['reset', (id: string) => runtime.liveRangeTodoListHandle!.resetEvents([id])],
        ['clear', () => runtime.liveRangeTodoListHandle!.clear()],
        ['replace', () => runtime.liveRangeTodoListHandle!.replaceEvents([makeEvent('replacement')])],
    ])('aborts running callbacks during %s', async (_name, mutate) => {
        await renderQueue();
        let capturedSignal: AbortSignal | undefined;
        const pending = jest.fn(({ signal }) => {
            capturedSignal = signal;
            return new Promise<void>(() => undefined);
        });
        callHandle(() => runtime.liveRangeTodoListHandle!.addEvent(makeEvent('pending', pending)));
        publishTelemetry(0, 0.1);
        publishTelemetry(0, 0.3);
        expect(capturedSignal?.aborted).toBe(false);

        callHandle(() => mutate('pending'));
        expect(capturedSignal?.aborted).toBe(true);
    });

    it('allows an update to explicitly replace an event callback', async () => {
        await renderQueue();
        const original = jest.fn();
        const replacement = jest.fn();
        callHandle(() => runtime.liveRangeTodoListHandle!.addEvent(makeEvent('replace-callback', original)));
        callHandle(() => runtime.liveRangeTodoListHandle!.updateEvents([{
            id: 'replace-callback',
            callback: replacement,
        }]));

        publishTelemetry(1, 0.1);
        publishTelemetry(1, 0.3);
        await flushCallbacks();
        expect(original).not.toHaveBeenCalled();
        expect(replacement).toHaveBeenCalledTimes(1);
    });

    it('clears the handle and snapshot and aborts running work when the panel unmounts', async () => {
        let capturedSignal: AbortSignal | undefined;
        const Wrapper = ({ show }: { show: boolean }) => {
            runtime = useContext(LiveSessionContext);
            return show ? <LiveRangeTodoList name="live-range-todo-list" /> : null;
        };
        const view = render(
            <LiveSessionProvider><Wrapper show /></LiveSessionProvider>,
        );
        await waitFor(() => expect(runtime.liveRangeTodoListHandle).not.toBeNull());
        callHandle(() => runtime.liveRangeTodoListHandle!.addEvent(makeEvent('pending', ({ signal }) => {
            capturedSignal = signal;
            return new Promise<void>(() => undefined);
        })));
        publishTelemetry(0, 0.1);
        publishTelemetry(0, 0.3);
        expect(capturedSignal?.aborted).toBe(false);

        view.rerender(<LiveSessionProvider><Wrapper show={false} /></LiveSessionProvider>);
        await waitFor(() => expect(runtime.liveRangeTodoListHandle).toBeNull());
        expect(runtime.liveRangeTodoListSnapshot).toBeNull();
        expect(capturedSignal?.aborted).toBe(true);
    });
});
