export type SegmentClassificationLabel = {
    label_id: string;
    label_name: string;
};

export type SegmentClassificationSubSegment = {
    start_index: number;
    end_index: number;
    labels: SegmentClassificationLabel[];
};

export type SegmentClassificationSegment = {
    id?: string;
    labels?: string[];
    parent_segment_id?: string;
    parent_segment_name?: string;
    parent_label_id?: string;
    parent_label_name?: string;
    main_label_id?: string;
    main_label_name?: string;
    start_index: number;
    end_index: number;
    sub_labels?: SegmentClassificationLabel[];
    sub_segments?: SegmentClassificationSubSegment[];
    child_segments?: SegmentClassificationSubSegment[];
};

export const getSegmentLabelText = (label: SegmentClassificationLabel): string => (
    label.label_name || label.label_id
);

export const getSegmentParentLabelText = (segment: SegmentClassificationSegment): string => (
    segment.parent_segment_name
    || segment.parent_label_name
    || segment.main_label_name
    || segment.parent_segment_id
    || segment.parent_label_id
    || segment.main_label_id
    || segment.labels?.join(', ')
    || 'Unlabeled'
);

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
        segment.main_label_id,
        segment.main_label_name,
        segment.parent_label_name,
        segment.parent_segment_name
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
        return dedupeTexts(segment.sub_labels.map(getSegmentLabelText));
    }

    const childSegmentLabels = getSegmentChildSegments(segment)
        .flatMap((childSegment) => childSegment.labels.map(getSegmentLabelText));

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

export const getActiveChildLabelTexts = (segment: SegmentClassificationSegment, sourceIndex: number): string[] => {
    const activeChildSegment = getSegmentChildSegments(segment).find((childSegment) => (
        sourceIndex >= childSegment.start_index && sourceIndex < childSegment.end_index
    ));

    if (activeChildSegment) {
        return dedupeTexts(activeChildSegment.labels.map(getSegmentLabelText));
    }

    return getSegmentChildLabelTexts(segment);
};

export const getActiveSubLabelTexts = getActiveChildLabelTexts;
