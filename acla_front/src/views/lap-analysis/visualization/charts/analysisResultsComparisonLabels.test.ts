import { buildAnalysisResultsComparisonLabelGroups } from './analysisResultsComparisonLabels';

const categories: Record<string, string[]> = {
    MSP: ['MSP1', 'MSP2'],
    MSR: ['MSR1'],
    EA: ['EA1'],
    RM: ['RM7'],
};
const names: Record<string, string> = {
    MSP: 'Training Error',
    MSP1: 'Late braking',
    MSP2: 'Late turn-in',
    MSR1: 'Failed overtake',
    EA1: 'Matches expert line',
    RM7: 'Merge back to expert line',
};
const buildGroups = (labels: string[]) => buildAnalysisResultsComparisonLabelGroups(
    labels, (category) => categories[category] ?? [], (id) => names[id],
);

describe('comparison analysis labels', () => {
    it('groups selected IDs and display names, deduplicating aliases and ignoring unrelated labels', () => {
        expect(buildGroups([
            'RM', 'RM7', 'Merge back to expert line',
            'EA', 'Matches expert line',
            'Training Error', 'MSP1', 'Late braking', 'MSP1',
            'Mistake (Racing)', 'MSR1', 'brands_hatch2', 'O',
        ])).toEqual([
            { category: 'mistakes', subLabels: ['Late braking', 'Failed overtake'] },
            { category: 'expert', subLabels: ['Matches expert line'] },
            { category: 'recovery', subLabels: ['Merge back to expert line'] },
        ]);
    });

    it('keeps parent-only categories without inventing sublabels', () => {
        expect(buildGroups([
            'Mistake (Practice)', 'MSR', 'Expert Adherence (Training)', 'Recovery & Merge',
        ])).toEqual([
            { category: 'mistakes', subLabels: [] },
            { category: 'expert', subLabels: [] },
            { category: 'recovery', subLabels: [] },
        ]);
    });

    it('recognizes sublabels without parent labels and retains IDs before the catalog loads', () => {
        expect(buildGroups(['MSP2', 'RM7'])).toEqual([
            { category: 'mistakes', subLabels: ['Late turn-in'] },
            { category: 'recovery', subLabels: ['Merge back to expert line'] },
        ]);
        expect(buildAnalysisResultsComparisonLabelGroups(
            ['MSP2', 'EA1', 'RM7'], () => [], () => undefined,
        )).toEqual([
            { category: 'mistakes', subLabels: ['MSP2'] },
            { category: 'expert', subLabels: ['EA1'] },
            { category: 'recovery', subLabels: ['RM7'] },
        ]);
    });

    it('omits categories that do not occur in the segment', () => {
        expect(buildGroups([])).toEqual([]);
        expect(buildGroups(['O', 'brands_hatch2'])).toEqual([]);
    });
});
