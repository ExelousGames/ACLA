import {
    buildProcedurePlan,
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
                    subscriber: 'display_surface',
                    title: 'Show supporting context',
                    payload: { tool: 'show_context', target_id: 'task-1' },
                },
                {
                    type: 'api_request',
                    subscriber: 'domain_worker',
                    title: 'Run domain task',
                },
            ],
        })).toMatchObject({
            goal: 'Complete a delegated task.',
            requests: [
                {
                    type: 'tool_call',
                    subscriber: 'display_surface',
                    title: 'Show supporting context',
                    payload: { tool: 'show_context', target_id: 'task-1' },
                    status: 'pending',
                },
                {
                    type: 'api_request',
                    subscriber: 'domain_worker',
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
                    subscriber: 'human',
                    title: 'Provide an input sample',
                },
                {
                    type: 'frontend_request',
                    subscriber: 'workflow_worker',
                    title: 'Run workflow worker',
                    payload: { force: false },
                },
            ],
        })).toMatchObject({
            goal: 'Continue the delegated workflow.',
            requests: [
                {
                    type: 'human_action',
                    subscriber: 'human',
                    title: 'Provide an input sample',
                    status: 'pending',
                },
                {
                    type: 'frontend_request',
                    subscriber: 'workflow_worker',
                    title: 'Run workflow worker',
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
                    type: 'driver_action',
                    subscriber: 'driver',
                    title: 'Collect a clean baseline lap',
                    status: 'complete',
                },
                {
                    type: 'frontend_request',
                    subscriber: 'live_recorded_analysis',
                    title: 'Request recorded-session classifier',
                },
            ],
        })).toMatchObject({
            currentStep: 1,
            requests: [
                {
                    title: 'Collect a clean baseline lap',
                    status: 'complete',
                },
                {
                    title: 'Request recorded-session classifier',
                    status: 'pending',
                },
            ],
        });
    });

    it('marks earlier requests complete when an observation advances current_request', () => {
        expect(buildProcedurePlan({
            event: 'baseline_classifier_request_ready',
            goal: 'Run live analysis from a clean baseline.',
            current_request: 1,
            requests: [
                {
                    type: 'driver_action',
                    subscriber: 'driver',
                    title: 'Collect a clean baseline lap',
                },
                {
                    type: 'frontend_request',
                    subscriber: 'live_recorded_analysis',
                    title: 'Request recorded-session classifier',
                },
            ],
        })).toMatchObject({
            currentStep: 1,
            requests: [
                {
                    title: 'Collect a clean baseline lap',
                    status: 'complete',
                },
                {
                    title: 'Request recorded-session classifier',
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

    it('accepts AI-authored request lists without frontend subscribers', () => {
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
