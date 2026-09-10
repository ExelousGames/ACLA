import type { DriverExpertComparisonLabelGroup, DriverExpertComparisonLabelRange } from 'components/driver-expert-comparison';
import type { AnalysisResultLabelResolver } from './analysisResultsModel';
import { normalizeSegmentLabels } from './segmentClassificationDisplay';

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

export const buildAnalysisResultsComparisonLabelRanges = (
    labels: unknown,
    getCategoryLabels: (category: string) => readonly string[],
    getLabelName: AnalysisResultLabelResolver,
): DriverExpertComparisonLabelRange[] => {
    const seen = new Set<string>();
    return normalizeSegmentLabels(labels).flatMap((label) => {
        const labelName = getLabelName(label.label_name) ?? label.label_name;
        const key = JSON.stringify([labelName, label.start_index, label.end_index]);
        if (seen.has(key)) return [];
        seen.add(key);
        const category = buildAnalysisResultsComparisonLabelGroups(
            [label.label_name], getCategoryLabels, getLabelName,
        )[0]?.category;
        return [{
            label: labelName,
            startIndex: label.start_index,
            endIndex: label.end_index,
            ...(category ? { category } : {}),
        }];
    });
};
