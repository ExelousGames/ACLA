import React, { createRef } from 'react';
import { act, cleanup, render, screen } from '@testing-library/react';
import { createOperationComponentRefDirectory, type OperationComponentRefDirectory, type OperationComponentRef } from 'contexts/OperationComponentRefContext';
import { createAiCommandRegistry, createWorkflowDispatcher } from 'views/lap-analysis/ai-chat/ai-command-registry';
import WorkflowPanel, { type WorkflowPanelHandle } from '../WorkflowPanel';
import { WorkflowComponentBase } from '../WorkflowComponentBase';
import { LiveRangeTodoListRunner } from '../LiveRangeTodoList';
import { createControlledOperation, createOperation } from '../operation';
import { asTool } from '../tool';
import { liveTelemetryStore } from 'views/live-session/live-telemetry-store';

let mockDirectory: OperationComponentRefDirectory;
jest.mock('views/floating-chat/AiOverlayManager', () => ({ __esModule: true, default: () => null }));
jest.mock('contexts/OperationComponentRefContext', () => ({
    ...jest.requireActual('contexts/OperationComponentRefContext'),
    useOperationComponentRefs: () => ({ directory: mockDirectory, revision: 0 }),
    useRegisterOperationComponentRef: function useRegisterOperationComponentRef(ref: OperationComponentRef) {
        const { useLayoutEffect } = jest.requireActual('react');
        useLayoutEffect(() => {
            mockDirectory.registerComponentRef(ref);
            return () => { mockDirectory.unregisterComponentRef(ref); };
        }, [ref]);
    },
}));

const names = { procedure: 'set_procedure_plan', repeatable: 'create_repeatable_plan', live: 'create_live_range_todo_list' } as const;
type Kind = keyof typeof names;
const componentNames = { procedure: 'procedure-plan', repeatable: 'repeatable-plan', live: 'live-range-todo-list' };
const step = (kind: Kind, id: string, name = 'query_analysis_result', args: object = { query: id }) => ({ operation: {
    name, arguments: args,
    ...(kind === 'live' ? { event: { id, normalized_position: 0.2, lead_time_seconds: 0, content: { title: id } } }
        : { title: id, ...(kind === 'repeatable' ? { id } : {}) }),
} });
const input = (kind: Kind, operations = [step(kind, 'wait')]): any => ({ workflow: { name: names[kind], operations,
    ...(kind === 'live' ? {} : { goal: kind }),
    ...(kind === 'repeatable' ? { stop_when: { tool: { name: 'query_analysis_result', arguments: { query: 'stop' } }, operator: 'eq', target: 1 } } : {}),
} });
const runner = (kind: Kind) => mockDirectory.findComponentRef(componentNames[kind])?.current as WorkflowComponentBase<any>;
const due = () => {
    const live = runner('live') as LiveRangeTodoListRunner;
    live.acceptTelemetry({ Graphics_normalized_car_position: 0 });
    live.acceptTelemetry({ Graphics_normalized_car_position: 0.3 });
};
const setup = () => {
    const waiting = createControlledOperation<Record<string, unknown>>();
    const query = jest.fn(({ query }: { query: string }) => query === 'stop'
        ? asTool(createOperation({ status: 'ready', data: 1 }, 'complete')) : asTool(waiting.operation));
    mockDirectory.registerComponentRef({ current: { getComponentName: () => 'visualization:analysis-results', queryAnalysisResult: query } });
    const context = { componentRefs: mockDirectory, sessionMode: 'live' as const };
    const registry = createAiCommandRegistry(context);
    const ref = createRef<WorkflowPanelHandle>();
    const view = render(<WorkflowPanel ref={ref} dispatchOperation={createWorkflowDispatcher(context)} live sessionGame="acc" />);
    const create = (kind: Kind, value = input(kind), caller?: WorkflowComponentBase<any>) => (
        (createAiCommandRegistry({ ...context, workflowCaller: caller })[names[kind]] as any)(value)
    );
    const append = (kind: Kind, id = 'added', caller?: WorkflowComponentBase<any>, operations = [step(kind, id)]) => {
        const name = kind === 'live' ? 'add_event_to_live_range_todo_list' : `append_${kind}_plan`;
        return (createAiCommandRegistry({ ...context, workflowCaller: caller }) as any)[name]({ workflow: { name, operations } });
    };
    return { ...view, waiting, query, registry, ref, create, append };
};
beforeEach(() => { mockDirectory = createOperationComponentRefDirectory(); });
afterEach(() => { cleanup(); jest.useRealTimers(); jest.restoreAllMocks(); });

describe('nested workflow ownership and lifetime', () => {
    it.each<[Kind, Kind]>([
        ['procedure', 'repeatable'], ['procedure', 'live'], ['repeatable', 'procedure'],
        ['repeatable', 'live'], ['live', 'procedure'], ['live', 'repeatable'],
    ])('%s waits for %s, including a pending grandchild tool', async (parent, child) => {
        const test = setup();
        let operation: any;
        await act(async () => {
            operation = test.create(parent, input(parent, [step(parent, 'child', names[child], input(child))]));
            if (parent === 'live' || child === 'live') due();
        });
        const terminated = jest.fn();
        operation.notifyTerminated(terminated);
        expect(test.query).toHaveBeenCalledWith({ query: 'wait' });
        expect(terminated).not.toHaveBeenCalled();
        expect(runner(parent)).toBeDefined();
        expect(runner(child)).toBeDefined();
        await act(async () => { test.waiting.resolve('complete', {}); await operation.result; });
        expect(terminated).toHaveBeenCalledTimes(1);
        expect(runner(parent)).toBeUndefined();
        expect(runner(child)).toBeUndefined();
    });

    it.each<[Kind, Kind]>([
        ['procedure', 'repeatable'], ['procedure', 'live'], ['repeatable', 'procedure'],
        ['repeatable', 'live'], ['live', 'procedure'], ['live', 'repeatable'],
    ])('%s catches a failed %s child and releases both workflows before termination', async (parent, child) => {
        jest.spyOn(console, 'error').mockImplementation(() => undefined);
        const test = setup();
        let operation: any;
        await act(async () => {
            operation = test.create(parent, input(parent, [step(parent, 'child', names[child], input(child))]));
            if (parent === 'live' || child === 'live') due();
        });
        const terminated = jest.fn(() => [runner(parent), runner(child)]);
        operation.notifyTerminated(terminated);
        await act(async () => {
            test.waiting.reject('failed', new Error('leaf failed'));
            await expect(operation.result).rejects.toThrow('leaf failed');
        });
        expect(terminated).toHaveBeenCalledTimes(1);
        expect(terminated).toHaveBeenCalledWith({ status: 'failed', result: expect.any(Error) });
        expect(terminated).toHaveReturnedWith([undefined, undefined]);
        expect(test.query).toHaveBeenCalledTimes(1);
        expect(screen.queryByLabelText('Procedure plan')).not.toBeInTheDocument();
        expect(screen.queryByLabelText('Repeatable plan')).not.toBeInTheDocument();
    });

    it.each(['finish', 'failure', 'cancel', 'reset', 'unmount'] as const)('handles a three-type chain on %s', async (action) => {
        jest.spyOn(console, 'error').mockImplementation(() => undefined);
        const test = setup();
        const leaf = input('live');
        const middle = input('repeatable', [step('repeatable', 'live', names.live, leaf)]);
        let operation: any;
        await act(async () => {
            operation = test.create('procedure', input('procedure', [step('procedure', 'repeat', names.repeatable, middle)]));
            due();
        });
        const aborted = jest.spyOn(test.waiting.operation, 'abort');
        const terminated = jest.fn();
        operation.notifyTerminated(terminated);
        expect(terminated).not.toHaveBeenCalled();
        await act(async () => {
            if (action === 'finish') test.waiting.resolve('complete', {});
            if (action === 'failure') test.waiting.reject('failed', new Error('leaf failed'));
            if (action === 'cancel') operation.abort();
            if (action === 'reset') test.ref.current!.reset();
            if (action === 'unmount') test.unmount();
            if (action === 'finish') await operation.result;
            else await expect(operation.result).rejects.toBeInstanceOf(Error);
        });
        expect(aborted).toHaveBeenCalledTimes(action === 'finish' || action === 'failure' ? 0 : 1);
        for (const kind of Object.keys(names) as Kind[]) expect(runner(kind)).toBeUndefined();
        await act(async () => { test.waiting.resolve('complete', { late: true }); });
        expect(screen.queryByLabelText('Procedure plan')).not.toBeInTheDocument();
        expect(screen.queryByLabelText('Repeatable plan')).not.toBeInTheDocument();
    });

    it('rejects self/ancestor creation and ancestor/descendant append before changing any runner', async () => {
        const test = setup();
        await act(async () => { test.create('procedure', input('procedure', [step('procedure', 'child', names.repeatable, input('repeatable'))])); });
        const parent = runner('procedure');
        const child = runner('repeatable');
        const snapshots = [parent.getSnapshot(), child.getSnapshot()];
        for (const call of [() => test.create('procedure', input('procedure'), parent), () => test.create('procedure', input('procedure'), child),
            () => test.create('repeatable', input('repeatable'), child), () => test.append('procedure', 'bad', child), () => test.append('repeatable', 'bad', parent)]) {
            await act(async () => { await expect(call().result).rejects.toThrow(/executing|ancestor|descendant/); });
            expect([parent.getSnapshot(), child.getSnapshot()]).toEqual(snapshots);
            expect(runner('procedure')).toBe(parent);
            expect(runner('repeatable')).toBe(child);
        }
    });

    it.each<Kind>(['procedure', 'repeatable', 'live'])('permits %s self-append and independent append', async (kind) => {
        const test = setup();
        await act(async () => { test.create(kind); });
        const owner = runner(kind);
        await act(async () => {
            const appended = test.append(kind, 'self', owner);
            await expect(appended.result).resolves.toBeDefined();
            if (kind !== 'live') {
                await expect(new Promise((resolve) => appended.notifyTerminated(resolve))).resolves.toMatchObject({
                    status: kind === 'procedure' ? 'advanced' : 'ready',
                });
            }
            await expect(test.append(kind, 'independent').result).resolves.toBeDefined();
        });
        const snapshot = owner.getSnapshot();
        expect(snapshot.requests ?? snapshot.steps ?? snapshot.events).toHaveLength(3);
    });

    it.each<Kind>(['procedure', 'repeatable', 'live'])('applies the missing %s append rule without waiting', async (kind) => {
        const test = setup();
        await act(async () => {
            const appended = test.append(kind);
            if (kind === 'repeatable') await expect(appended.result).rejects.toThrow(/stop condition/);
            else await expect(appended.result).resolves.toBeDefined();
        });
        expect(Boolean(runner(kind))).toBe(kind !== 'repeatable');
        if (kind === 'procedure') expect(runner(kind).getSnapshot().goal).toBe('added');
    });

    it.each<Kind>(['procedure', 'repeatable', 'live'])('replaces only the matching hidden %s runner', async (kind) => {
        const test = setup();
        test.query.mockImplementation(() => asTool(createControlledOperation<Record<string, unknown>>().operation));
        const other: Kind = kind === 'procedure' ? 'repeatable' : 'procedure';
        let first: any;
        act(() => { first = test.create(kind); test.create(other); });
        const hidden = runner(kind);
        const visible = runner(other);
        const dispose = jest.spyOn(hidden, 'dispose');
        await act(async () => {
            test.create(kind, input(kind), visible);
            await expect(first.result).rejects.toBeInstanceOf(Error);
        });
        expect(dispose).toHaveBeenCalledTimes(1);
        expect(runner(kind)).not.toBe(hidden);
        expect(runner(other)).toBe(visible);
    });

    it('binds independently appended steps to their target runner', async () => {
        const test = setup();
        const child = createControlledOperation<Record<string, unknown>>();
        test.query.mockImplementation(({ query }) => asTool(query === 'child' ? child.operation : test.waiting.operation));
        let oldLive: any;
        let parent: any;
        act(() => { oldLive = test.create('live'); parent = test.create('procedure'); });
        const source = runner('live');
        const dispose = jest.spyOn(source, 'dispose');
        await act(async () => {
            await test.append('procedure', 'spawn', source, [step('procedure', 'spawn', names.live,
                input('live', [step('live', 'child')]))]).result;
            test.waiting.resolve('complete', {});
        });
        await expect(oldLive.result).rejects.toBeInstanceOf(Error);
        expect(dispose).toHaveBeenCalledTimes(1);
        expect(runner('live')).not.toBe(source);
        expect(runner('procedure')).toBeDefined();
        await act(async () => { due(); child.resolve('complete', {}); await expect(parent.result).resolves.toMatchObject({ status: 'complete' }); });
    });

    it.each<Kind>(['procedure', 'live'])('starts a missing %s append independently of its caller', async (kind) => {
        const test = setup();
        const independentTask = createControlledOperation<Record<string, unknown>>();
        test.query.mockImplementation(({ query }) => asTool(query === 'added' ? independentTask.operation : test.waiting.operation));
        let callerOperation: any;
        act(() => { callerOperation = test.create('repeatable'); });
        await act(async () => { await test.append(kind, 'added', runner('repeatable')).result; });
        const independent = runner(kind);
        const dispose = jest.spyOn(independent, 'dispose');
        await act(async () => { callerOperation.abort(); });
        expect(dispose).not.toHaveBeenCalled();
        expect(runner(kind)).toBe(independent);
    });

    it.each<Kind>(['procedure', 'repeatable', 'live'])('validates an entire %s append before mutating its target', async (kind) => {
        const test = setup();
        act(() => { test.create(kind); });
        const owner = runner(kind);
        const before = owner.getSnapshot();
        for (const operations of [[step(kind, 'valid'), { operation: { name: 'query_analysis_result' } }],
            ...(kind === 'procedure' ? [] : [[step(kind, 'valid'), step(kind, 'wait')]])]) {
            await act(async () => { await expect(test.append(kind, 'bad', undefined, operations as any).result).rejects.toBeInstanceOf(Error); });
            expect(owner.getSnapshot()).toEqual(before);
        }
    });

    it('releases a live telemetry subscription on reset and creates a new one for the next queue', async () => {
        const original = liveTelemetryStore.subscribeEvents.bind(liveTelemetryStore);
        const releases: jest.Mock[] = [];
        jest.spyOn(liveTelemetryStore, 'subscribeEvents').mockImplementation((listener, options) => {
            const release = jest.fn(original(listener, options));
            releases.push(release);
            return release;
        });
        const test = setup();
        act(() => { test.create('live'); });
        expect(releases).toHaveLength(1);
        act(() => { test.ref.current!.reset(); });
        expect(releases[0]).toHaveBeenCalledTimes(1);
        act(() => { test.create('live'); });
        expect(releases).toHaveLength(2);
        test.unmount();
        expect(releases[1]).toHaveBeenCalledTimes(1);
    });
});

describe('append during repeatable execution', () => {
    it('retains appended steps in later passes and clears them on completion', async () => {
        jest.useFakeTimers();
        const test = setup();
        let checks = 0;
        test.query.mockImplementation(({ query }) => query === 'stop'
            ? asTool(createOperation({ status: 'ready', data: ++checks === 2 ? 1 : 0 }, 'complete'))
            : query === 'wait' ? asTool(test.waiting.operation) : asTool(createOperation({}, 'complete')));
        let operation: any;
        act(() => { operation = test.create('repeatable'); });
        await act(async () => { await test.append('repeatable').result; test.waiting.resolve('complete', {}); });
        await act(async () => { jest.advanceTimersByTime(1000); await operation.result; });
        expect(test.query.mock.calls.map(([args]) => args.query)).toEqual(['wait', 'added', 'stop', 'wait', 'added', 'stop']);
        expect(runner('repeatable')).toBeUndefined();
        await act(async () => { await expect(test.append('repeatable').result).rejects.toThrow(/stop condition/); });
    });

    it.each(['step', 'check', 'delay'] as const)('executes steps appended during %s before a fresh stop check', async (phase) => {
        jest.useFakeTimers();
        const test = setup();
        const check = createControlledOperation<Record<string, unknown>>();
        let checks = 0;
        test.query.mockImplementation(({ query }) => query === 'stop'
            ? (++checks === 1 ? asTool(check.operation) : asTool(createOperation({ status: 'ready', data: 1 }, 'complete')))
            : query === 'wait' ? asTool(test.waiting.operation) : asTool(createOperation({}, 'complete')));
        let operation: any;
        await act(async () => { operation = test.create('repeatable'); });
        if (phase !== 'step') await act(async () => { test.waiting.resolve('complete', {}); });
        if (phase === 'delay') await act(async () => { check.resolve('complete', { status: 'ready', data: 0 }); });
        await act(async () => { await test.append('repeatable').result; });
        await act(async () => {
            test.waiting.resolve('complete', {});
            check.resolve('complete', { status: 'ready', data: 1 });
            await operation.result;
        });
        expect(test.query.mock.calls.map(([args]) => args.query)).toEqual(phase === 'step'
            ? ['wait', 'added', 'stop'] : ['wait', 'stop', 'added', 'stop']);
        expect(runner('repeatable')).toBeUndefined();
        const calls = test.query.mock.calls.length;
        act(() => { jest.runOnlyPendingTimers(); });
        expect(test.query).toHaveBeenCalledTimes(calls);
    });
});

describe('live-range creation termination', () => {
    it.each(['success', 'failure', 'telemetry reset'] as const)('waits for appended events and handles %s', async (result) => {
        const test = setup();
        const appended = createControlledOperation<Record<string, unknown>>();
        test.query.mockImplementation(({ query }) => asTool(query === 'added' ? appended.operation : test.waiting.operation));
        let operation: any;
        await act(async () => { operation = test.create('live'); due(); await test.append('live').result; });
        const owner = runner('live') as LiveRangeTodoListRunner;
        const terminated = jest.fn();
        operation.notifyTerminated(terminated);
        if (result === 'telemetry reset') {
            await act(async () => { owner.reset(); await expect(operation.result).rejects.toBeInstanceOf(Error); });
            expect(owner.get().todo_list).toMatchObject({ current_position: null, rolling_rate: null, events: [] });
            return;
        }
        jest.spyOn(console, 'error').mockImplementation(() => undefined);
        await act(async () => {
            if (result === 'failure') test.waiting.reject('failed', new Error('event failed'));
            else test.waiting.resolve('complete', {});
        });
        expect(terminated).not.toHaveBeenCalled();
        await act(async () => { due(); });
        await act(async () => {
            appended.resolve('complete', {});
            if (result === 'failure') await expect(operation.result).rejects.toThrow('event failed');
            else await expect(operation.result).resolves.toMatchObject({ status: 'empty', event_count: 0 });
        });
        expect(terminated).toHaveBeenCalledTimes(1);
        expect(runner('live')).toBeUndefined();
    });
});
