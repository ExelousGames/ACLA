export type SegmentLabelResolver = (labelId: string) => string | undefined;

export type SegmentTimeGap = {
    start_ms: number;
    end_ms: number;
    delta_ms: number;
};

export type SegmentClassificationSubSegment = {
    start_index: number;
    end_index: number;
    labels: string[];
    time_gap?: SegmentTimeGap;
};

export type SegmentClassificationSegment = {
    id?: string;
    parent_labels?: string[];
    start_index: number;
    end_index: number;
    child_segments?: SegmentClassificationSubSegment[];
    time_gap?: SegmentTimeGap;
};

export const getSegmentLabelText = (labelId: string, resolveLabel?: SegmentLabelResolver): string => (
    resolveLabel?.(labelId) || labelId
);

export const getSegmentParentLabelText = (
    segment: SegmentClassificationSegment,
    resolveLabel?: SegmentLabelResolver,
): string => {
    const parentLabels = getSegmentParentLabelIds(segment);
    if (parentLabels.length > 0) {
        return parentLabels.map((labelId) => getSegmentLabelText(labelId, resolveLabel)).join(', ');
    }

    return 'Unlabeled';
};

export const getSegmentChildSegments = (segment: SegmentClassificationSegment): SegmentClassificationSubSegment[] => (
    Array.isArray(segment.child_segments) ? segment.child_segments : []
);

export const getSegmentParentLabelIds = (segment: SegmentClassificationSegment): string[] => (
    Array.isArray(segment.parent_labels) ? segment.parent_labels : []
);

export const resolveSegmentParentLabelTexts = (
    segment: SegmentClassificationSegment,
    resolveLabel?: SegmentLabelResolver,
): string[] => getSegmentParentLabelIds(segment).map((labelId) => getSegmentLabelText(labelId, resolveLabel));

const dedupeTexts = (texts: string[]): string[] => {
    const seen = new Set<string>();
    return texts.filter((text) => {
        if (seen.has(text)) return false;
        seen.add(text);
        return true;
    });
};

export const getSegmentChildLabelTexts = (segment: SegmentClassificationSegment): string[] => {
    const childSegmentLabels = getSegmentChildSegments(segment)
        .flatMap((childSegment) => childSegment.labels);

    if (childSegmentLabels.length > 0) {
        return dedupeTexts(childSegmentLabels);
    }

    return [];
};

export const resolveSegmentChildLabelTexts = (
    segment: SegmentClassificationSegment,
    resolveLabel?: SegmentLabelResolver,
): string[] => getSegmentChildLabelTexts(segment).map((labelId) => getSegmentLabelText(labelId, resolveLabel));

export const getActiveChildLabelTexts = (segment: SegmentClassificationSegment, sourceIndex: number): string[] => {
    const activeChildSegment = getSegmentChildSegments(segment).find((childSegment) => (
        sourceIndex >= childSegment.start_index && sourceIndex < childSegment.end_index
    ));

    if (activeChildSegment) {
        return dedupeTexts(activeChildSegment.labels);
    }

    return getSegmentChildLabelTexts(segment);
};

export const getActiveSubLabelTexts = getActiveChildLabelTexts;

export const resolveActiveChildLabelTexts = (
    segment: SegmentClassificationSegment,
    sourceIndex: number,
    resolveLabel?: SegmentLabelResolver,
): string[] => getActiveChildLabelTexts(segment, sourceIndex).map((labelId) => getSegmentLabelText(labelId, resolveLabel));
