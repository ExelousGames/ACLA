import type { DriverExpertComparisonLabelGroup } from 'components/driver-expert-comparison';
import type { AnalysisResultLabelResolver } from './analysisResultsModel';

const COMPARISON_LABEL_CATEGORIES = [
    { category: 'mistakes', parents: { MSP: 'Mistake (Practice)', MSR: 'Mistake (Racing)' } },
    { category: 'expert', parents: { EA: 'Expert Adherence (Training)' } },
    { category: 'recovery', parents: { RM: 'Recovery & Merge' } },
] as const;

export const buildAnalysisResultsComparisonLabelGroups = (
    labels: readonly string[],
    getCategoryLabels: (category: string) => readonly string[],
    getLabelName: AnalysisResultLabelResolver,
): DriverExpertComparisonLabelGroup[] => {
    const selectedLabels = new Set(labels);
    return COMPARISON_LABEL_CATEGORIES.flatMap(({ category, parents }) => {
        let hasParent = false;
        const subLabels = new Set<string>();
        Object.entries(parents).forEach(([parentId, fallbackName]) => {
            const parentName = getLabelName(parentId) ?? fallbackName;
            hasParent ||= [parentId, fallbackName, parentName]
                .some((label) => selectedLabels.has(label));
            const childIds = new Set([
                ...getCategoryLabels(parentId),
                ...labels.filter((label) => new RegExp(`^${parentId}\\d+$`).test(label)),
            ]);
            childIds.forEach((childId) => {
                const childName = getLabelName(childId) ?? childId;
                if (selectedLabels.has(childId) || selectedLabels.has(childName)) {
                    subLabels.add(childName);
                }
            });
        });
        return hasParent || subLabels.size > 0
            ? [{ category, subLabels: Array.from(subLabels) }]
            : [];
    });
};
