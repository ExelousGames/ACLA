import {
    advanceProcedurePlan,
    buildProcedurePlan as buildProcedurePlanWithTasks,
    getProcedurePlanToolArguments,
    getProcedurePlanToolRunKey,
    getProcedurePlanUpdateKey,
    isProcedurePlanClearEvent,
    isProcedurePlanOptOutRequest,
    isProcedurePlanStartEvent,
    ProcedurePlanRunner,
    serializeProcedurePlan,
} from '../ProcedurePlan';
import type { LiveRangeTodoEventInput } from '../live-range-todo-list-types';
import type { TaskStartFunction } from '../task-start-function';
import { createAiToolComponentRefDirectory } from 'contexts/AiToolComponentRefContext';

const taskStart = jest.fn();
const buildProcedurePlan = (data: Record<string, unknown>) => (
    buildProcedurePlanWithTasks(data, () => taskStart)
);

const flushTaskStarts = async () => {
    await Promise.resolve();
    await Promise.resolve();
};

describe('standard task-start functions', () => {
    it('type-checks components with one and multiple implementations', () => {
        const start: TaskStartFunction = () => undefined;
        const startExpanded: TaskStartFunction = async () => undefined;
        const singleImplementationComponent: { start: TaskStartFunction } = { start };
        const multipleImplementationComponent: {
            start: TaskStartFunction;
            startExpanded: TaskStartFunction;
        } = { start, startExpanded };

        expect(singleImplementationComponent.start).toBe(start);
        expect(multipleImplementationComponent).toEqual({ start, startExpanded });
    });

    it('passes the same selected function unchanged to procedure and live-range items', () => {
        const selected: TaskStartFunction = jest.fn();
        const plan = buildProcedurePlanWithTasks({
            goal: 'Show one comparison.',
            requests: [{ type: 'tool_call', title: 'Show comparison', name: 'show_comparison' }],
        }, () => selected)!;
        const liveRangeEvent: LiveRangeTodoEventInput = {
            id: 'comparison',
            normalized_position: 0.5,
            content: { title: 'Show comparison' },
            data: {},
            taskStart: selected,
        };

        expect(plan.requests[0].taskStart).toBe(selected);
        expect(liveRangeEvent.taskStart).toBe(selected);
        expect(serializeProcedurePlan(plan).requests[0]).not.toHaveProperty('taskStart');
        expect(JSON.stringify(serializeProcedurePlan(plan))).not.toContain('taskStart');
    });

    it('rejects plan creation when a component cannot supply a compatible function', () => {
        expect(buildProcedurePlanWithTasks({
            goal: 'Unsupported task.',
            requests: [{ type: 'tool_call', title: 'Unsupported', name: 'unsupported' }],
        }, () => null)).toBeNull();
    });
});

describe('ProcedurePlanRunner', () => {
    it('runs synchronous and asynchronous tasks in order and removes the empty plan', async () => {
        let finishSecond!: () => void;
        const first: TaskStartFunction = jest.fn();
        const second: TaskStartFunction = jest.fn(() => new Promise<void>((resolve) => {
            finishSecond = resolve;
        }));
        const changes: Array<ReturnType<ProcedurePlanRunner['get']>> = [];
        const runner = new ProcedurePlanRunner((plan) => changes.push(plan));

        runner.replace({
            goal: 'Run both.',
            currentStep: 0,
            requests: [
                { type: 'task', title: 'First', status: 'pending', taskStart: first },
                { type: 'task', title: 'Second', status: 'pending', taskStart: second },
            ],
        });

        expect(first).toHaveBeenCalledWith(expect.any(AbortSignal));
        expect(runner.get()?.requests[0]).toMatchObject({ title: 'First', status: 'running' });
        await flushTaskStarts();
        expect(second).toHaveBeenCalledWith(expect.any(AbortSignal));
        expect(runner.get()?.requests).toEqual([
            expect.objectContaining({ title: 'Second', status: 'running' }),
        ]);

        finishSecond();
        await flushTaskStarts();
        expect(runner.get()).toBeNull();
        expect(changes.at(-1)).toBeNull();
    });

    it('reports rejection before removing the failed task and continuing', async () => {
        const order: string[] = [];
        const next = jest.fn();
        const runner = new ProcedurePlanRunner(
            (plan) => {
                if (plan === null) order.push('removed');
            },
            (_request, error) => order.push(`error:${(error as Error).message}`),
        );
        runner.replace({
            goal: 'Continue after failure.',
            currentStep: 0,
            requests: [
                { type: 'task', title: 'Failure', status: 'pending', taskStart: () => Promise.reject(new Error('failed')) },
                { type: 'task', title: 'Next', status: 'pending', taskStart: next },
            ],
        });

        await flushTaskStarts();
        await flushTaskStarts();
        expect(next).toHaveBeenCalledTimes(1);
        expect(order).toEqual(['error:failed', 'removed']);
    });

    it('deletes its registered reference after the last failed request settles', async () => {
        const directory = createAiToolComponentRefDirectory();
        const onError = jest.fn();
        const runner = new ProcedurePlanRunner('procedure-plan', undefined, onError);
        runner.addComponentRef(directory);
        runner.replace({
            goal: 'Fail once.',
            currentStep: 0,
            requests: [{
                type: 'task',
                title: 'Failure',
                status: 'pending',
                taskStart: () => Promise.reject(new Error('failed')),
            }],
        });

        await flushTaskStarts();
        expect(onError).toHaveBeenCalledWith(
            expect.objectContaining({ title: 'Failure', status: 'running' }),
            expect.objectContaining({ message: 'failed' }),
        );
        expect(directory.findBaseComponentRef()).toBeNull();
    });

    it('aborts the active function when cleared or replaced', () => {
        let clearedSignal!: AbortSignal;
        let replacedSignal!: AbortSignal;
        const pending = (capture: (signal: AbortSignal) => void): TaskStartFunction => (signal) => {
            capture(signal);
            return new Promise<void>(() => undefined);
        };
        const runner = new ProcedurePlanRunner(() => undefined);
        const planFor = (title: string, start: TaskStartFunction) => ({
            goal: title,
            currentStep: 0,
            requests: [{ type: 'task', title, status: 'pending' as const, taskStart: start }],
        });

        runner.replace(planFor('Clear me', pending((signal) => { clearedSignal = signal; })));
        runner.clear();
        expect(clearedSignal.aborted).toBe(true);

        runner.replace(planFor('Replace me', pending((signal) => { replacedSignal = signal; })));
        runner.replace(planFor('Replacement', () => undefined));
        expect(replacedSignal.aborted).toBe(true);
    });
});

describe('procedure plan tool requests', () => {
    const request = {
        type: 'tool_call',
        title: 'Show supporting context',
        name: 'show_context',
        status: 'pending' as const,
        payload: {
            arguments: { target_id: 'task-1' },
        },
    };

    it('extracts nested tool arguments', () => {
        expect(getProcedurePlanToolArguments(request)).toEqual({ target_id: 'task-1' });
    });

    it('builds a run key from the active step and request', () => {
        expect(getProcedurePlanToolRunKey({
            goal: 'Complete a delegated task.',
            currentStep: 1,
            requests: [request],
        }, request)).toBe('1:show_context:{"arguments":{"target_id":"task-1"}}');
    });
});

describe('buildProcedurePlan', () => {
    it('does not create a visible plan without AI-provided requests', () => {
        expect(buildProcedurePlan({
            event: 'procedure_plan_started',
            metadata: {
                ready: false,
            },
        })).toBeNull();
    });

    it('builds the visible plan from AI request lists', () => {
        expect(buildProcedurePlan({
            event: 'procedure_plan_started',
            goal: 'Complete a delegated task.',
            requests: [
                {
                    type: 'tool_call',
                    title: 'Show supporting context',
                    name: 'show_context',
                    payload: { tool: 'show_context', target_id: 'task-1' },
                },
                {
                    type: 'api_request',
                    title: 'Run domain task',
                },
            ],
        })).toMatchObject({
            goal: 'Complete a delegated task.',
            requests: [
                {
                    type: 'tool_call',
                    title: 'Show supporting context',
                    name: 'show_context',
                    payload: { tool: 'show_context', target_id: 'task-1' },
                    status: 'pending',
                },
                {
                    type: 'api_request',
                    title: 'Run domain task',
                    status: 'pending',
                },
            ],
            currentStep: 0,
        });
    });

    it('builds plans from non-start update events when explicit requests are present', () => {
        expect(buildProcedurePlan({
            event: 'task_ready',
            goal: 'Continue the delegated workflow.',
            requests: [
                {
                    type: 'human_action',
                    title: 'Provide an input sample',
                },
                {
                    type: 'tool_call',
                    title: 'Run workflow worker',
                    name: 'run_workflow_worker',
                    payload: { force: false },
                },
            ],
        })).toMatchObject({
            goal: 'Continue the delegated workflow.',
            requests: [
                {
                    type: 'human_action',
                    title: 'Provide an input sample',
                    status: 'pending',
                },
                {
                    type: 'tool_call',
                    title: 'Run workflow worker',
                    name: 'run_workflow_worker',
                    status: 'pending',
                    payload: { force: false },
                },
            ],
            currentStep: 0,
        });
    });

    it('forgets settled requests when creating the runtime plan', () => {
        expect(buildProcedurePlan({
            event: 'procedure_plan_started',
            goal: 'Run live analysis from a clean baseline.',
            requests: [
                {
                    type: 'tool_call',
                    title: 'Collect a clean baseline lap',
                    name: 'collect_live_baseline',
                    status: 'complete',
                },
                {
                    type: 'tool_call',
                    title: 'Request recorded-session classifier',
                    name: 'analyze_live_recorded_analysis',
                },
            ],
        })).toMatchObject({
            currentStep: 0,
            requests: [
                {
                    title: 'Request recorded-session classifier',
                    name: 'analyze_live_recorded_analysis',
                    status: 'pending',
                },
            ],
        });
    });

    it('starts at the requested pending item without retaining earlier items', () => {
        expect(buildProcedurePlan({
            event: 'baseline_classifier_request_ready',
            goal: 'Run live analysis from a clean baseline.',
            current_request: 1,
            requests: [
                {
                    type: 'tool_call',
                    title: 'Collect a clean baseline lap',
                    name: 'collect_live_baseline',
                },
                {
                    type: 'tool_call',
                    title: 'Request recorded-session classifier',
                    name: 'analyze_live_recorded_analysis',
                },
            ],
        })).toMatchObject({
            currentStep: 0,
            requests: [
                {
                    title: 'Request recorded-session classifier',
                    name: 'analyze_live_recorded_analysis',
                    status: 'pending',
                },
            ],
        });
    });

    it('rejects non-standard legacy string plan entries', () => {
        expect(buildProcedurePlan({
            event: 'procedure_plan_started',
            title: 'Next request list',
            plan: ['Show context.', 'Compare the next result.'],
            current_request: 1,
        })).toBeNull();
    });

    it('rejects nested plan objects instead of guessing the shape', () => {
        expect(buildProcedurePlan({
            event: 'procedure_plan_started',
            plan: {
                goal: 'Complete the nested task.',
                current_request: 1,
                requests: [
                    {
                        type: 'tool_call',
                        tool: 'show_context',
                        title: 'Show supporting context',
                        args: { target_id: 'task-1' },
                    },
                    {
                        type: 'human_action',
                        step: 'Provide a new sample',
                        reason: 'The assistant needs a repeat sample to compare.',
                    },
                ],
            },
        })).toBeNull();
    });

    it('rejects procedure_plan envelopes instead of treating them as the standard shape', () => {
        expect(buildProcedurePlan({
            event: 'task_ready',
            procedure_plan: {
                goal: 'Improve the current task.',
                steps: [
                    { text: 'Show supporting context' },
                    { label: 'Run the worker' },
                ],
            },
            subject: {
                name: 'Task 1',
            },
        })).toBeNull();
    });

    it('accepts AI-authored request lists without channel hints', () => {
        expect(buildProcedurePlan({
            event: 'procedure_plan_started',
            goal: 'Complete the next task.',
            requests: [
                { type: 'tool_call', title: 'Show supporting context' },
                { type: 'driver_action', title: 'Provide one clean sample' },
            ],
        })).toMatchObject({
            requests: [
                { type: 'tool_call', title: 'Show supporting context', status: 'pending' },
                { type: 'driver_action', title: 'Provide one clean sample', status: 'pending' },
            ],
        });
    });

    it('keeps unsupported output visibility fields out of requests', () => {
        const request = buildProcedurePlan({
            event: 'procedure_plan_started',
            goal: 'Run one tool.',
            requests: [
                {
                    type: 'tool_call',
                    title: 'Read context',
                    name: 'get_recorded_session_context',
                    result_visibility: 'tag',
                    output: 'tag',
                },
            ],
        })?.requests[0];

        expect(request).toMatchObject({
            type: 'tool_call',
            title: 'Read context',
            name: 'get_recorded_session_context',
            status: 'pending',
        });
        expect(request).not.toHaveProperty('result_visibility');
        expect(request).not.toHaveProperty('output');
    });

    it('rejects requests without explicit type and title', () => {
        expect(buildProcedurePlan({
            event: 'procedure_plan_started',
            goal: 'Complete the next task.',
            requests: [
                { type: 'tool_call' },
                { title: 'Provide one clean sample' },
            ],
        })).toBeNull();
    });

    it('ignores unrelated events until a request list is available', () => {
        expect(buildProcedurePlan({
            event: 'unrelated_update',
            subject: { name: 'Task 1' },
        })).toBeNull();
    });
});

describe('advanceProcedurePlan', () => {
    it('removes the active request and moves to the next pending request', () => {
        const result = advanceProcedurePlan({
            goal: 'Complete a delegated task.',
            currentStep: 0,
            requests: [
                { type: 'tool_call', title: 'Collect a clean baseline lap', name: 'collect_live_baseline', status: 'pending', taskStart },
                { type: 'tool_call', title: 'Analyze the baseline', status: 'pending', taskStart },
            ],
        }, 'baseline complete');

        expect(result).toMatchObject({
            status: 'advanced',
            current_request: 0,
            step: 'Analyze the baseline',
            reason: 'baseline complete',
            plan: {
                currentStep: 0,
                requests: [
                    { title: 'Analyze the baseline', status: 'pending' },
                ],
            },
        });
    });

    it('forgets already settled requests while retaining the next task', () => {
        const result = advanceProcedurePlan({
            goal: 'Complete a delegated task.',
            currentStep: 0,
            requests: [
                { type: 'tool_call', title: 'Collect a clean baseline lap', name: 'collect_live_baseline', status: 'complete', taskStart },
                { type: 'tool_call', title: 'Analyze the baseline', status: 'complete', taskStart },
                { type: 'driver_action', title: 'Run the next lap', status: 'pending', taskStart },
            ],
        });

        expect(result).toMatchObject({
            status: 'advanced',
            current_request: 0,
            plan: {
                currentStep: 0,
                requests: [
                    { title: 'Run the next lap', status: 'pending' },
                ],
            },
        });
    });

    it('returns an empty completed plan after the final request settles', () => {
        const result = advanceProcedurePlan({
            goal: 'Complete a delegated task.',
            currentStep: 1,
            requests: [
                { type: 'tool_call', title: 'Collect a clean baseline lap', name: 'collect_live_baseline', status: 'complete', taskStart },
                { type: 'driver_action', title: 'Run the next lap', status: 'pending', taskStart },
            ],
        });

        expect(result).toMatchObject({
            status: 'complete',
            current_request: 0,
            plan: {
                currentStep: 0,
                requests: [],
            },
        });
    });
});

describe('getProcedurePlanUpdateKey', () => {
    it('treats identical plan states as the same update', () => {
        const plan = {
            goal: 'Run live analysis.',
            currentStep: 0,
            requests: [
                {
                    type: 'tool_call',
                    title: 'Collect baseline',
                    name: 'collect_live_baseline',
                    status: 'running' as const,
                    payload: { b: 2, a: { d: 4, c: 3 } },
                },
            ],
        };

        expect(getProcedurePlanUpdateKey(plan)).toBe(getProcedurePlanUpdateKey({
            ...plan,
            requests: [
                {
                    ...plan.requests[0],
                    payload: { a: { c: 3, d: 4 }, b: 2 },
                },
            ],
        }));
    });

    it('changes when a plan step status changes', () => {
        const pendingPlan = {
            goal: 'Run live analysis.',
            currentStep: 0,
            requests: [
                { type: 'tool_call', title: 'Collect baseline', status: 'pending' as const },
            ],
        };

        expect(getProcedurePlanUpdateKey(pendingPlan)).not.toBe(getProcedurePlanUpdateKey({
            ...pendingPlan,
            requests: [
                { ...pendingPlan.requests[0], status: 'running' as const },
            ],
        }));
    });
});

describe('isProcedurePlanOptOutRequest', () => {
    it('detects explicit plan opt-out commands', () => {
        expect(isProcedurePlanOptOutRequest('skip the plan')).toBe(true);
        expect(isProcedurePlanOptOutRequest('I want to opt out of the plan')).toBe(true);
        expect(isProcedurePlanOptOutRequest("don't follow the plan anymore")).toBe(true);
        expect(isProcedurePlanOptOutRequest('clear procedure plan')).toBe(true);
    });

    it('does not treat normal plan questions as opt-out commands', () => {
        expect(isProcedurePlanOptOutRequest("what's the plan?")).toBe(false);
        expect(isProcedurePlanOptOutRequest('next step in the plan')).toBe(false);
        expect(isProcedurePlanOptOutRequest('follow the plan')).toBe(false);
    });
});

describe('isProcedurePlanStartEvent', () => {
    it('recognizes generic plan-start events', () => {
        expect(isProcedurePlanStartEvent('procedure_plan_started')).toBe(true);
        expect(isProcedurePlanStartEvent('setup_plan_started')).toBe(true);
        expect(isProcedurePlanStartEvent('worker_plan_started')).toBe(true);
    });

    it('does not treat plan updates as new plans', () => {
        expect(isProcedurePlanStartEvent('task_ready')).toBe(false);
        expect(isProcedurePlanStartEvent(undefined)).toBe(false);
    });
});

describe('isProcedurePlanClearEvent', () => {
    it('recognizes plan-clear events', () => {
        expect(isProcedurePlanClearEvent('procedure_plan_cleared')).toBe(true);
        expect(isProcedurePlanClearEvent('procedure_plan_terminated')).toBe(true);
        expect(isProcedurePlanClearEvent('live_analysis_plan_terminated')).toBe(true);
    });

    it('does not treat plan updates as clear events', () => {
        expect(isProcedurePlanClearEvent('procedure_plan_started')).toBe(false);
        expect(isProcedurePlanClearEvent('task_ready')).toBe(false);
        expect(isProcedurePlanClearEvent(undefined)).toBe(false);
    });
});
