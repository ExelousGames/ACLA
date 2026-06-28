import { buildLiveCoachingPlan } from '../ai-chat-plan';

describe('buildLiveCoachingPlan', () => {
    it('builds the visible plan from recorded analysis plan events', () => {
        expect(buildLiveCoachingPlan({
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

    it('infers a visible plan from a live coaching window focus', () => {
        expect(buildLiveCoachingPlan({
            event: 'coaching_window',
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
            sourceEvent: 'coaching_window',
        });
    });

    it('ignores baseline classification request events until a focus exists', () => {
        expect(buildLiveCoachingPlan({
            event: 'live_baseline_ready_for_classification',
            candidate_sections: [{ name: 'T1 Paddock Hill Bend' }],
        })).toBeNull();
    });
});
