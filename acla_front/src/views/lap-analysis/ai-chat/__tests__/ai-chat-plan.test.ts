import {
    buildLiveProcedurePlan,
    isProcedurePlanOptOutRequest,
    isProcedurePlanStartEvent,
} from '../ai-chat-plan';

describe('buildLiveProcedurePlan', () => {
    it('starts the visible plan with baseline collection', () => {
        expect(buildLiveProcedurePlan({
            event: 'live_analysis_plan_started',
            snapshot: {
                baseline_ready: false,
            },
        })).toMatchObject({
            goal: 'Run live analysis from a clean baseline.',
            steps: [
                'Collect a complete baseline lap.',
                'Analyze the baseline with recorded-session data or live section classification.',
                'Select the focus section and compare the next pass against the baseline.',
            ],
            currentStep: 0,
            sourceEvent: 'live_analysis_plan_started',
        });
    });

    it('advances the visible plan when the baseline is ready for classification', () => {
        expect(buildLiveProcedurePlan({
            event: 'live_baseline_ready_for_classification',
            candidate_sections: [{ name: 'T1 Paddock Hill Bend' }],
        })).toMatchObject({
            goal: 'Analyze the completed baseline and choose the focus section.',
            steps: [
                'Collect a complete baseline lap.',
                'Classify baseline sections from the completed lap.',
                'Select the focus section and compare the next pass against the baseline.',
            ],
            currentStep: 1,
            sourceEvent: 'live_baseline_ready_for_classification',
        });
    });

    it('builds the visible plan from recorded analysis plan events', () => {
        expect(buildLiveProcedurePlan({
            event: 'recorded_analysis_plan_ready',
            goal: 'Improve T2 Druids by reducing late brake.',
            plan: ['Focus the next approach.', 'Clean up late brake.'],
            focus: {
                section: { name: 'T2 Druids' },
            },
        })).toMatchObject({
            goal: 'Improve T2 Druids by reducing late brake.',
            steps: ['Focus the next approach.', 'Clean up late brake.'],
            focusName: 'T2 Druids',
            currentStep: 0,
        });
    });

    it('infers a visible plan from a live analysis window focus', () => {
        expect(buildLiveProcedurePlan({
            event: 'live_analysis_window',
            focus: {
                section: { name: 'T1 Paddock Hill Bend' },
                baseline: {
                    childLabels: ['late brake'],
                },
            },
        })).toMatchObject({
            goal: 'Improve T1 Paddock Hill Bend by reducing late brake.',
            steps: [
                'Use the next approach to T1 Paddock Hill Bend as the focus run.',
                'Change one thing first: clean up late brake.',
                'After the next pass, compare the focused section classification against this baseline.',
            ],
            focusName: 'T1 Paddock Hill Bend',
            sourceEvent: 'live_analysis_window',
        });
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
