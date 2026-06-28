import {
    buildLiveProcedurePlan,
    getProcedurePlanAdvanceBlock,
    isProcedurePlanOptOutRequest,
    isProcedurePlanStartEvent,
} from '../ai-chat-plan';

describe('buildLiveProcedurePlan', () => {
    it('does not create a visible plan without AI-provided requests', () => {
        expect(buildLiveProcedurePlan({
            event: 'live_analysis_plan_started',
            snapshot: {
                baseline_ready: false,
            },
        })).toBeNull();
        expect(buildLiveProcedurePlan({
            event: 'live_baseline_ready_for_classification',
            candidate_sections: [{ name: 'T1 Paddock Hill Bend' }],
        })).toBeNull();
    });

    it('builds the visible plan from AI request lists', () => {
        expect(buildLiveProcedurePlan({
            event: 'recorded_analysis_plan_ready',
            goal: 'Improve T2 Druids by reducing late brake.',
            requests: [
                {
                    type: 'tool_call',
                    name: 'show_map',
                    title: 'Show the focus section map',
                    payload: { section_name: 'T2 Druids' },
                },
                {
                    type: 'api_request',
                    method: 'post',
                    url: '/racing-session/imitation-learning-guidance',
                    title: 'Request imitation guidance',
                },
            ],
        })).toMatchObject({
            goal: 'Improve T2 Druids by reducing late brake.',
            requests: [
                {
                    type: 'tool_call',
                    name: 'show_map',
                    title: 'Show the focus section map',
                    payload: { section_name: 'T2 Druids' },
                },
                {
                    type: 'api_request',
                    method: 'POST',
                    url: '/racing-session/imitation-learning-guidance',
                    title: 'Request imitation guidance',
                },
            ],
            currentStep: 0,
        });
    });

    it('builds the startup live plan from explicit baseline classifier requests', () => {
        expect(buildLiveProcedurePlan({
            event: 'live_analysis_plan_started',
            goal: 'Collect a baseline and run the live section classifier.',
            requests: [
                {
                    type: 'driver_action',
                    title: 'Collect a clean baseline lap',
                },
                {
                    type: 'tool_call',
                    name: 'classify_live_section',
                    title: 'Classify the completed baseline',
                    payload: { lap: 'last' },
                },
            ],
        })).toMatchObject({
            goal: 'Collect a baseline and run the live section classifier.',
            requests: [
                {
                    type: 'driver_action',
                    title: 'Collect a clean baseline lap',
                },
                {
                    type: 'tool_call',
                    name: 'classify_live_section',
                    title: 'Classify the completed baseline',
                    payload: { lap: 'last' },
                },
            ],
            currentStep: 0,
        });
    });

    it('rejects non-standard legacy string plan entries', () => {
        expect(buildLiveProcedurePlan({
            event: 'live_analysis_window',
            title: 'Next live request list',
            plan: ['Show focus telemetry.', 'Compare the next pass.'],
            current_request: 1,
        })).toBeNull();
    });

    it('rejects nested plan objects instead of guessing the shape', () => {
        expect(buildLiveProcedurePlan({
            event: 'procedure_plan_started',
            plan: {
                goal: 'Coach the next pass through Druids.',
                current_request: 1,
                requests: [
                    {
                        type: 'tool_call',
                        tool: 'show_map',
                        title: 'Show the focus map',
                        args: { section_name: 'T2 Druids' },
                    },
                    {
                        type: 'driver_action',
                        step: 'Drive one clean pass through the focus section',
                        reason: 'The assistant needs a repeat sample to compare.',
                    },
                ],
            },
        })).toBeNull();
    });

    it('rejects procedure_plan envelopes instead of treating them as the standard shape', () => {
        expect(buildLiveProcedurePlan({
            event: 'recorded_analysis_plan_ready',
            procedure_plan: {
                goal: 'Improve Paddock Hill entry.',
                steps: [
                    { text: 'Show Paddock Hill on the map' },
                    { label: 'Coach the braking reference' },
                ],
            },
            focus: {
                section: { name: 'T1 Paddock Hill Bend' },
            },
        })).toBeNull();
    });

    it('rejects requests without explicit type and title', () => {
        expect(buildLiveProcedurePlan({
            event: 'procedure_plan_started',
            goal: 'Coach the next pass through Druids.',
            requests: [
                { type: 'tool_call', name: 'show_map' },
                { title: 'Drive one clean pass through the focus section' },
            ],
        })).toBeNull();
    });

    it('ignores unrelated live analyst events until a plan is available', () => {
        expect(buildLiveProcedurePlan({
            event: 'live_section_history_updated',
            candidate_sections: [{ name: 'T1 Paddock Hill Bend' }],
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

describe('getProcedurePlanAdvanceBlock', () => {
    const plan = {
        goal: 'Run live analysis from a clean baseline.',
        requests: [
            { type: 'request', title: 'Collect a complete baseline lap.' },
            { type: 'request', title: 'Analyze the baseline.' },
            { type: 'request', title: 'Select the focus section.' },
        ],
        currentStep: 0,
        sourceEvent: 'live_analysis_plan_started',
    };

    it('blocks live plans from advancing before baseline collection is complete', () => {
        expect(getProcedurePlanAdvanceBlock(plan, { baseline_ready: false })).toMatchObject({
            status: 'blocked',
            error: 'baseline_collection_incomplete',
        });
    });

    it('blocks live plans from advancing to focus work before a focus exists', () => {
        expect(getProcedurePlanAdvanceBlock(
            { ...plan, currentStep: 1, sourceEvent: 'live_baseline_ready_for_classification' },
            { baseline_ready: true },
            false,
        )).toMatchObject({
            status: 'blocked',
            error: 'focus_section_not_ready',
        });
    });

    it('allows the baseline analysis step after baseline collection is complete', () => {
        expect(getProcedurePlanAdvanceBlock(plan, { baseline_ready: true })).toBeNull();
    });
});

describe('isProcedurePlanStartEvent', () => {
    it('recognizes generic plan-start events', () => {
        expect(isProcedurePlanStartEvent('procedure_plan_started')).toBe(true);
        expect(isProcedurePlanStartEvent('live_analysis_plan_started')).toBe(true);
        expect(isProcedurePlanStartEvent('setup_plan_started')).toBe(true);
    });

    it('does not treat plan updates as new plans', () => {
        expect(isProcedurePlanStartEvent('live_baseline_ready_for_classification')).toBe(false);
        expect(isProcedurePlanStartEvent('recorded_analysis_plan_ready')).toBe(false);
        expect(isProcedurePlanStartEvent(undefined)).toBe(false);
    });
});
