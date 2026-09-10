import { executeSubscribedFrontendOperation } from 'views/lap-analysis/ai-chat/use-voice-conversation';
import { ProcedurePlanRunner } from '../ProcedurePlan';
import { RepeatablePlanRunner } from '../RepeatablePlan';
import { LiveRangeTodoListRunner } from '../LiveRangeTodoList';
import { createControlledOperation, createOperation, type Operation } from '../operation';
import { asTool, type WorkflowDispatcher } from '../tool';

type Kind = 'procedure' | 'repeatable' | 'live';
const flush = async () => { for (let i = 0; i < 30; i += 1) await Promise.resolve(); };
const due = (runner: LiveRangeTodoListRunner, position: number) => {
    runner.acceptTelemetry({ Graphics_normalized_car_position: position });
};
const start = (kind: Kind, dispatch: WorkflowDispatcher) => {
    const operations = ['first', 'second'].map((id) => ({ operation: {
        name: 'query_analysis_result' as const, title: id, id, arguments: { query: id },
    } }));
    if (kind === 'procedure') {
        const runner = new ProcedurePlanRunner('procedure-plan', dispatch, undefined, jest.fn());
        const create = () => runner.createProcedurePlan({ workflow: {
            name: 'set_procedure_plan', goal: 'Review',
            operations: operations.map(({ operation: { id, ...operation } }) => ({ operation })),
        } });
        return { runner, operation: create(), create };
    }
    if (kind === 'repeatable') {
        const runner = new RepeatablePlanRunner('repeatable-plan', dispatch);
        const create = () => runner.createRepeatablePlan({ workflow: {
            name: 'create_repeatable_plan', goal: 'Review', operations,
            stop_when: { tool: { name: 'query_analysis_result', arguments: { query: 'stop' } }, operator: 'eq', target: 0 },
        } });
        return { runner, operation: create(), create };
    }
    const runner = new LiveRangeTodoListRunner('live-range');
    const create = () => runner.createLiveRangeTodoList({ workflow: {
        name: 'create_live_range_todo_list',
        operations: operations.map(({ operation }, index) => ({ operation: {
            name: operation.name, arguments: operation.arguments,
            event: { id: operation.id, normalized_position: (index + 1) * 0.2,
                lead_time_seconds: 0, content: { title: operation.title, description: `Details for ${operation.id}` } },
        } })),
    } }, dispatch);
    return { runner, operation: create(), create };
};
const subscribe = (operation: Operation<any, any>, signal?: AbortSignal) => {
    const frames: any[] = [];
    const finished = executeSubscribedFrontendOperation({
        call: { id: 'workflow', name: 'workflow' },
        handlers: { workflow: () => operation }, signal,
        sendText: (frame) => { frames.push(frame); },
    });
    return { frames, finished };
};

afterEach(() => { jest.restoreAllMocks(); jest.useRealTimers(); });

describe.each<Kind>(['procedure', 'repeatable', 'live'])('%s workflow progress sent to AI', (kind) => {
    it.each(['complete', 'failed', 'aborted', 'cleared', 'disposed', 'replaced'])(
        'retains completed steps when %s during the second step', async (outcome) => {
            jest.spyOn(console, 'error').mockImplementation(() => undefined);
            const second = createControlledOperation<Record<string, unknown>>();
            const dispatch = Object.assign(jest.fn((_name, args) => args?.query === 'second'
                ? asTool(second.operation)
                : asTool(createOperation({ status: 'ready', data: 0 }, 'complete'))),
            { validate: jest.fn() }) as WorkflowDispatcher;
            const { runner, operation, create } = start(kind, dispatch);
            const controller = new AbortController();
            const { frames, finished } = subscribe(operation, controller.signal);
            if (runner instanceof LiveRangeTodoListRunner) { due(runner, 0); due(runner, 0.25); }
            await flush();
            if (runner instanceof LiveRangeTodoListRunner) due(runner, 0.45);
            expect(dispatch).toHaveBeenCalledTimes(2);

            if (outcome === 'complete') second.resolve('complete', { status: 'ready', data: 0 });
            if (outcome === 'failed') second.reject('failed', new Error('offline'));
            if (outcome === 'aborted') controller.abort();
            if (outcome === 'cleared') {
                if (runner instanceof ProcedurePlanRunner) {
                    await expect(runner.clearProcedurePlan().result).resolves.toMatchObject({
                        completed_step_count: 1, stopped_at_step: { step: 2, title: 'second' },
                    });
                } else runner.clear();
            }
            if (outcome === 'disposed') runner.dispose();
            if (outcome === 'replaced') {
                const replacement = create();
                runner.dispose();
                await replacement.result.catch(() => undefined);
            }
            await finished;
            expect(frames).toHaveLength(1);
            expect(frames[0].result).toMatchObject({
                completed_step_count: outcome === 'complete' ? 2 : 1,
            });
            expect(frames[0].result.stopped_at_step).toEqual(outcome === 'complete' ? null : {
                step: 2,
                title: 'second',
                ...(kind !== 'procedure' ? { id: 'second' } : {}),
                ...(kind === 'live' ? { description: 'Details for second' } : {}),
            });
            runner.dispose();
        },
    );
});

it('identifies the repeatable stop-condition check after completed steps', async () => {
    const dispatch = Object.assign(jest.fn((_name, args) => asTool(createOperation(
        args?.query === 'stop' ? new Error('check failed') : { status: 'ready', data: 0 }, 'complete',
    ))), { validate: jest.fn() }) as WorkflowDispatcher;
    const { operation } = start('repeatable', dispatch);
    const { frames, finished } = subscribe(operation);
    await finished;
    expect(frames[0].result).toMatchObject({
        completed_step_count: 2,
        stopped_at_step: { step: 3, tool_name: 'query_analysis_result' },
    });
});

it('counts successful repeatable step executions across retries', async () => {
    jest.useFakeTimers();
    let checks = 0;
    const dispatch = Object.assign(jest.fn((_name, args) => asTool(createOperation({
        status: 'ready', data: args?.query === 'stop' && ++checks === 1 ? 1 : 0,
    }, 'complete'))), { validate: jest.fn() }) as WorkflowDispatcher;
    const { operation } = start('repeatable', dispatch);
    const { frames, finished } = subscribe(operation);
    await flush();
    expect(frames).toHaveLength(0);
    jest.advanceTimersByTime(1000);
    await finished;
    expect(frames[0].result).toMatchObject({ completed_step_count: 4, stopped_at_step: null });
});

it('uses the original live event step number when telemetry changes execution order', async () => {
    const second = createControlledOperation<Record<string, unknown>>();
    const dispatch = Object.assign(jest.fn((_name, args) => args?.query === 'first'
        ? asTool(second.operation) : asTool(createOperation({}, 'complete'))),
    { validate: jest.fn() }) as WorkflowDispatcher;
    const { runner, operation } = start('live', dispatch);
    if (!(runner instanceof LiveRangeTodoListRunner)) throw new Error('Expected live runner');
    runner.updateEvents([{ id: 'first', normalized_position: 0.6 }]);
    const { frames, finished } = subscribe(operation);
    due(runner, 0);
    due(runner, 0.45);
    await flush();
    due(runner, 0.65);
    operation.abort();
    await finished;
    expect(frames[0].result).toMatchObject({
        completed_step_count: 1,
        stopped_at_step: { step: 1, id: 'first', title: 'first', description: 'Details for first' },
    });
});

it('keeps a nested workflow failure separate from the live event that contains it', async () => {
    jest.spyOn(console, 'error').mockImplementation(() => undefined);
    const childDispatch = Object.assign(jest.fn((_name, args) => asTool(createOperation(
        args?.query === 'second' ? new Error('nested failure') : {}, 'complete',
    ))), { validate: jest.fn() }) as WorkflowDispatcher;
    let child: ReturnType<WorkflowDispatcher> | undefined;
    const parentDispatch = Object.assign(jest.fn((_name, args) => {
        if (args?.query === 'first') {
            child = start('procedure', childDispatch).operation;
            return child;
        }
        return asTool(createOperation({}, 'complete'));
    }), { validate: jest.fn() }) as WorkflowDispatcher;
    const { runner, operation } = start('live', parentDispatch);
    if (!(runner instanceof LiveRangeTodoListRunner)) throw new Error('Expected live runner');
    const { frames, finished } = subscribe(operation);
    due(runner, 0);
    due(runner, 0.25);
    await flush();
    due(runner, 0.45);
    await finished;
    await expect(child!.result).rejects.toMatchObject({
        completed_step_count: 1, stopped_at_step: { step: 2, title: 'second' },
    });
    expect(frames[0].result).toMatchObject({
        completed_step_count: 1,
        stopped_at_step: { step: 1, id: 'first', title: 'first', description: 'Details for first' },
    });
});
