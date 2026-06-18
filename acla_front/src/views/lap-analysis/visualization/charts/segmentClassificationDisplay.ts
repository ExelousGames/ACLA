export type SegmentLabelResolver = (labelId: string) => string | undefined;

export type SegmentClassificationSubSegment = {
    start_index: number;
    end_index: number;
    labels: string[];
};

export type SegmentClassificationSegment = {
    id?: string;
    labels?: string[];
    parent_segment_id?: string;
    parent_label_id?: string;
    main_label_id?: string;
    start_index: number;
    end_index: number;
    sub_labels?: string[];
    sub_segments?: SegmentClassificationSubSegment[];
    child_segments?: SegmentClassificationSubSegment[];
};

export const getSegmentLabelText = (labelId: string, resolveLabel?: SegmentLabelResolver): string => (
    resolveLabel?.(labelId) || labelId
);

export const getSegmentParentLabelText = (
    segment: SegmentClassificationSegment,
    resolveLabel?: SegmentLabelResolver,
): string => {
    const labelId = segment.parent_segment_id || segment.parent_label_id || segment.main_label_id;
    if (labelId) {
        return getSegmentLabelText(labelId, resolveLabel);
    }

    if (Array.isArray(segment.labels) && segment.labels.length > 0) {
        return segment.labels.map((label) => getSegmentLabelText(label, resolveLabel)).join(', ');
    }

    return 'Unlabeled';
};

export const getSegmentMainLabelText = getSegmentParentLabelText;

export const getSegmentChildSegments = (segment: SegmentClassificationSegment): SegmentClassificationSubSegment[] => (
    Array.isArray(segment.child_segments) && segment.child_segments.length > 0
        ? segment.child_segments
        : segment.sub_segments || []
);

const getSegmentParentLabelIds = (segment: SegmentClassificationSegment): string[] => (
    [
        segment.parent_segment_id,
        segment.parent_label_id,
        segment.main_label_id
    ].filter((label): label is string => Boolean(label))
);

const dedupeTexts = (texts: string[]): string[] => {
    const seen = new Set<string>();
    return texts.filter((text) => {
        if (seen.has(text)) return false;
        seen.add(text);
        return true;
    });
};

export const getSegmentChildLabelTexts = (segment: SegmentClassificationSegment): string[] => {
    if (Array.isArray(segment.sub_labels) && segment.sub_labels.length > 0) {
        return dedupeTexts(segment.sub_labels);
    }

    const childSegmentLabels = getSegmentChildSegments(segment)
        .flatMap((childSegment) => childSegment.labels);

    if (childSegmentLabels.length > 0) {
        return dedupeTexts(childSegmentLabels);
    }

    const parentLabelIds = getSegmentParentLabelIds(segment);
    if (parentLabelIds.length > 0 && Array.isArray(segment.labels)) {
        return segment.labels.filter((label) => !parentLabelIds.includes(label));
    }

    return [];
};

export const getSegmentSubLabelTexts = getSegmentChildLabelTexts;

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

export const resolveActiveSubLabelTexts = resolveActiveChildLabelTexts;
