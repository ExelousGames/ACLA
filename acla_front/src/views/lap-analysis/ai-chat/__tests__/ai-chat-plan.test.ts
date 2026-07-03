import {
    advanceProcedurePlan,
    buildProcedurePlan,
    getProcedurePlanUpdateKey,
    isProcedurePlanClearEvent,
    isProcedurePlanOptOutRequest,
    isProcedurePlanStartEvent,
} from '../ai-chat-plan';

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

    it('advances to the next request when the active request is already complete', () => {
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
            currentStep: 1,
            requests: [
                {
                    title: 'Collect a clean baseline lap',
                    name: 'collect_live_baseline',
                    status: 'complete',
                },
                {
                    title: 'Request recorded-session classifier',
                    name: 'analyze_live_recorded_analysis',
                    status: 'pending',
                },
            ],
        });
    });

    it('marks earlier requests complete when an tool status advances current_request', () => {
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
            currentStep: 1,
            requests: [
                {
                    title: 'Collect a clean baseline lap',
                    name: 'collect_live_baseline',
                    status: 'complete',
                },
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
    it('completes the active request and moves to the next pending request', () => {
        const result = advanceProcedurePlan({
            goal: 'Complete a delegated task.',
            currentStep: 0,
            requests: [
                { type: 'tool_call', title: 'Collect a clean baseline lap', name: 'collect_live_baseline', status: 'pending' },
                { type: 'tool_call', title: 'Analyze the baseline', status: 'pending' },
            ],
        }, 'baseline complete');

        expect(result).toMatchObject({
            status: 'advanced',
            current_request: 1,
            step: 'Analyze the baseline',
            reason: 'baseline complete',
            plan: {
                currentStep: 1,
                requests: [
                    { title: 'Collect a clean baseline lap', status: 'complete' },
                    { title: 'Analyze the baseline', status: 'pending' },
                ],
            },
        });
    });

    it('skips over already completed requests without marking the next pending request complete', () => {
        const result = advanceProcedurePlan({
            goal: 'Complete a delegated task.',
            currentStep: 0,
            requests: [
                { type: 'tool_call', title: 'Collect a clean baseline lap', name: 'collect_live_baseline', status: 'complete' },
                { type: 'tool_call', title: 'Analyze the baseline', status: 'complete' },
                { type: 'driver_action', title: 'Run the next lap', status: 'pending' },
            ],
        });

        expect(result).toMatchObject({
            status: 'advanced',
            current_request: 2,
            plan: {
                currentStep: 2,
                requests: [
                    { title: 'Collect a clean baseline lap', status: 'complete' },
                    { title: 'Analyze the baseline', status: 'complete' },
                    { title: 'Run the next lap', status: 'pending' },
                ],
            },
        });
    });

    it('leaves the final request active when completing the last step', () => {
        const result = advanceProcedurePlan({
            goal: 'Complete a delegated task.',
            currentStep: 1,
            requests: [
                { type: 'tool_call', title: 'Collect a clean baseline lap', name: 'collect_live_baseline', status: 'complete' },
                { type: 'driver_action', title: 'Run the next lap', status: 'pending' },
            ],
        });

        expect(result).toMatchObject({
            status: 'complete',
            current_request: 1,
            plan: {
                currentStep: 1,
                requests: [
                    { title: 'Collect a clean baseline lap', status: 'complete' },
                    { title: 'Run the next lap', status: 'complete' },
                ],
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
